"""Conservative, mask-guided appearance adjustments for Swapio photos."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FACE_PARSER_MODEL = "bisenet_resnet_18.onnx"
PARSER_SIZE = (512, 512)
PARSER_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PARSER_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HAIR_CLASS = (17,)
SKIN_CLASSES = (1, 7, 8, 14)  # face skin, ears, and neck

# Natural-looking starting points. Values are RGB hex strings because that is
# what the UI exposes; conversion to OpenCV's BGR/LAB happens at processing time.
HAIR_PRESETS = {
    "natural_black": "#1c1714",
    "dark_brown": "#3b2416",
    "chestnut": "#6f3425",
    "auburn": "#8b3e24",
    "copper": "#b85c32",
    "golden_blonde": "#d7b56d",
    "platinum": "#d8d2c4",
    "burgundy": "#6e1f3a",
}


@dataclass(frozen=True)
class AppearanceReferences:
    hair_lab: np.ndarray | None = None
    skin_lab: np.ndarray | None = None


def _hex_to_bgr(value: str) -> np.ndarray:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Hair color must be a six-digit hex color")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return np.array([blue, green, red], dtype=np.uint8)


def _bgr_to_lab(color: np.ndarray) -> np.ndarray:
    pixel = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


class AppearanceProcessor:
    """Runs one local face parser and applies texture-preserving LAB shifts."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @staticmethod
    def _head_roi(image: np.ndarray, face) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(float, face.bbox)
        face_width = max(x2 - x1, 1.0)
        face_height = max(y2 - y1, 1.0)
        return (
            max(0, int(round(x1 - face_width * 0.75))),
            max(0, int(round(y1 - face_height * 0.90))),
            min(width, int(round(x2 + face_width * 0.75))),
            min(height, int(round(y2 + face_height * 1.25))),
        )

    def _parse_crop(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, PARSER_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = ((rgb - PARSER_MEAN) / PARSER_STD).transpose(2, 0, 1)[None]
        output = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        labels = output.argmax(0).astype(np.uint8)
        return cv2.resize(
            labels,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _selected_hair_component(
        binary: np.ndarray,
        face,
        roi: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Keep hair connected to the selected face, not a nearby person's hair."""
        left, top, _, _ = roi
        x1, y1, x2, y2 = map(float, face.bbox)
        face_width = max(x2 - x1, 1.0)
        face_height = max(y2 - y1, 1.0)
        seed = np.zeros_like(binary, dtype=np.uint8)
        sx1 = max(0, int(round(x1 - left - face_width * 0.35)))
        sx2 = min(binary.shape[1], int(round(x2 - left + face_width * 0.35)))
        sy1 = max(0, int(round(y1 - top - face_height * 0.70)))
        sy2 = min(binary.shape[0], int(round(y2 - top + face_height * 0.45)))
        seed[sy1:sy2, sx1:sx2] = 1

        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        selected = np.zeros_like(binary, dtype=np.uint8)
        minimum_area = max(20, round(face_width * face_height * 0.008))
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component = components == label
            overlap = int(np.count_nonzero(component & (seed > 0)))
            if overlap >= max(8, round(area * 0.004)):
                selected[component] = 1
        return selected

    def _soft_mask_from_labels(
        self,
        image: np.ndarray,
        face,
        roi: tuple[int, int, int, int],
        labels: np.ndarray,
        classes: tuple[int, ...],
        *,
        select_hair: bool = False,
    ) -> np.ndarray:
        left, top, right, bottom = roi
        binary = np.isin(labels, classes).astype(np.uint8)
        if select_hair:
            binary = self._selected_hair_component(binary, face, roi)

        face_width = max(float(face.bbox[2] - face.bbox[0]), 1.0)
        kernel_size = max(3, int(round(face_width * 0.045)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        feather = max(1.0, face_width * 0.018)
        local_mask = cv2.GaussianBlur(
            binary.astype(np.float32), (0, 0), feather
        ).clip(0.0, 1.0)
        full_mask = np.zeros(image.shape[:2], dtype=np.float32)
        full_mask[top:bottom, left:right] = local_mask
        return full_mask

    def region_masks(
        self,
        image: np.ndarray,
        face,
        *,
        need_hair: bool,
        need_skin: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        roi = self._head_roi(image, face)
        left, top, right, bottom = roi
        if right - left < 8 or bottom - top < 8:
            return None, None
        labels = self._parse_crop(image[top:bottom, left:right])
        hair = (
            self._soft_mask_from_labels(
                image, face, roi, labels, HAIR_CLASS, select_hair=True
            )
            if need_hair
            else None
        )
        skin = (
            self._soft_mask_from_labels(image, face, roi, labels, SKIN_CLASSES)
            if need_skin
            else None
        )
        return hair, skin

    def region_mask(
        self,
        image: np.ndarray,
        face,
        classes: tuple[int, ...],
        *,
        select_hair: bool = False,
    ) -> np.ndarray:
        roi = self._head_roi(image, face)
        left, top, right, bottom = roi
        if right - left < 8 or bottom - top < 8:
            return np.zeros(image.shape[:2], dtype=np.float32)
        labels = self._parse_crop(image[top:bottom, left:right])
        return self._soft_mask_from_labels(
            image, face, roi, labels, classes, select_hair=select_hair
        )

    def reference_lab(
        self,
        image: np.ndarray,
        face,
        classes: tuple[int, ...],
        *,
        select_hair: bool = False,
    ) -> np.ndarray | None:
        mask = self.region_mask(image, face, classes, select_hair=select_hair)
        active = mask >= 0.75
        if np.count_nonzero(active) < 25:
            return None
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return np.median(lab[active], axis=0).astype(np.float32)

    def references(
        self,
        source_image: np.ndarray,
        source_face,
        *,
        need_hair: bool,
        need_skin: bool,
    ) -> AppearanceReferences:
        hair_mask, skin_mask = self.region_masks(
            source_image,
            source_face,
            need_hair=need_hair,
            need_skin=need_skin,
        )
        lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        def median(mask: np.ndarray | None) -> np.ndarray | None:
            if mask is None:
                return None
            active = mask >= 0.75
            if np.count_nonzero(active) < 25:
                return None
            return np.median(lab[active], axis=0).astype(np.float32)

        return AppearanceReferences(hair_lab=median(hair_mask), skin_lab=median(skin_mask))

    @staticmethod
    def recolor(
        image: np.ndarray,
        mask: np.ndarray,
        desired_lab: np.ndarray,
        strength: float,
        *,
        luminance_factor: float,
    ) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        active = mask >= 0.65
        if strength <= 0 or np.count_nonzero(active) < 25:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        current = np.median(lab[active], axis=0)
        adjusted = lab.copy()
        adjusted[:, :, 0] += (
            float(desired_lab[0] - current[0]) * strength * luminance_factor
        )
        adjusted[:, :, 1] += float(desired_lab[1] - current[1]) * strength
        adjusted[:, :, 2] += float(desired_lab[2] - current[2]) * strength
        recolored = cv2.cvtColor(
            np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )
        blend = (mask * strength)[:, :, None]
        return np.clip(
            blend * recolored.astype(np.float32)
            + (1.0 - blend) * image.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def hair_color_lab(mode: str, custom_color: str) -> np.ndarray | None:
        if mode in {"off", "source"}:
            return None
        color = custom_color if mode == "custom" else HAIR_PRESETS.get(mode)
        if not color:
            return None
        try:
            return _bgr_to_lab(_hex_to_bgr(color))
        except (TypeError, ValueError):
            return _bgr_to_lab(_hex_to_bgr(HAIR_PRESETS["copper"]))

    def apply(
        self,
        image: np.ndarray,
        mask_source: np.ndarray,
        faces: list,
        *,
        hair_mode: str,
        custom_hair_color: str,
        hair_strength: float,
        skin_match: bool,
        references: AppearanceReferences,
    ) -> np.ndarray:
        result = image
        desired_hair = (
            references.hair_lab
            if hair_mode == "source"
            else self.hair_color_lab(hair_mode, custom_hair_color)
        )
        desired_skin = references.skin_lab if skin_match else None
        for face in faces:
            hair_mask, skin_mask = self.region_masks(
                mask_source,
                face,
                need_hair=desired_hair is not None,
                need_skin=desired_skin is not None,
            )
            if desired_hair is not None:
                result = self.recolor(
                    result,
                    hair_mask if hair_mask is not None else np.zeros(mask_source.shape[:2]),
                    desired_hair,
                    hair_strength,
                    luminance_factor=0.82,
                )
            if desired_skin is not None:
                result = self.recolor(
                    result,
                    skin_mask if skin_mask is not None else np.zeros(mask_source.shape[:2]),
                    desired_skin,
                    0.28,
                    luminance_factor=0.35,
                )
        return result
