#!/usr/bin/env python3
"""Offline face-swapping engine for Swapio."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REQUIRED_BUFFALO_MODELS = {"2d106det.onnx", "det_10g.onnx", "w600k_r50.onnx"}
SWAPPER_MODEL = "inswapper_128.onnx"
HYPERSWAP_MODEL = "hyperswap_1b_256.onnx"
ENHANCER_MODEL = "gpen_bfr_1024.onnx"
OCCLUDER_MODEL = "xseg_1.onnx"
FACE_PARSER_MODEL = "bisenet_resnet_18.onnx"
HISTORY_FILENAME = ".swapio-history.json"
PROCESSING_REVISION = 17
INNER_MOUTH_106_INDICES = np.array([65, 66, 62, 70, 69, 57, 60, 54])
# The 106-point model puts each eyelid contour in its own run; 43-51 and 97-105
# next to them are the eyebrows and must stay out of it.
# Exclude pupil/centre points: gaze must not influence eyelid registration.
LEFT_EYE_106_INDICES = np.array([33, 35, 36, 37, 39, 40, 41, 42])
RIGHT_EYE_106_INDICES = np.array([87, 89, 90, 91, 93, 94, 95, 96])
# Face restoration works on its own FFHQ alignment, not the swapper's ArcFace one.
ENHANCER_SIZE = 1024
# How much of the restored crop replaces the swapped one. 0.8 repaints the face
# hard enough to read as painted; the user controls this per batch.
DEFAULT_RESTORATION_STRENGTH = 0.5
FFHQ_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)
BOOSTED_QUALITIES = frozenset({"careful", "best"})

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, dict], None]
StopFn = Callable[[], bool]
FaceSelector = Callable[[Path, np.ndarray, list], int | None]


class FaceSelectionSkipped(RuntimeError):
    """The user skipped this photo without choosing a face."""


class ProcessingCancelled(RuntimeError):
    """The user stopped while a face choice was pending."""


def _noop(*_args, **_kwargs):
    return None


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def model_dir() -> Path:
    override = os.environ.get("SWAPIO_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    bundled = base_dir() / "models"
    bundled_complete = (
        all((bundled / "buffalo_l" / name).is_file() for name in REQUIRED_BUFFALO_MODELS)
        and (bundled / SWAPPER_MODEL).is_file()
        and (bundled / HYPERSWAP_MODEL).is_file()
        and (bundled / ENHANCER_MODEL).is_file()
        and (bundled / OCCLUDER_MODEL).is_file()
        and (bundled / FACE_PARSER_MODEL).is_file()
    )
    # A source checkout deliberately keeps its models beside the code. A
    # packaged public build may contain only an empty placeholder; never try
    # writing downloads into its read-only /opt application directory.
    if bundled_complete or (not getattr(sys, "frozen", False) and bundled.exists()):
        return bundled
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "swapio" / "models"


def missing_models(root: Path | None = None) -> list[str]:
    root = Path(root) if root else model_dir()
    missing = []
    buffalo = root / "buffalo_l"
    for name in sorted(REQUIRED_BUFFALO_MODELS):
        if not (buffalo / name).is_file():
            missing.append(f"buffalo_l/{name}")
    if not (root / SWAPPER_MODEL).is_file():
        missing.append(SWAPPER_MODEL)
    if not (root / HYPERSWAP_MODEL).is_file():
        missing.append(HYPERSWAP_MODEL)
    if not (root / ENHANCER_MODEL).is_file():
        missing.append(ENHANCER_MODEL)
    if not (root / OCCLUDER_MODEL).is_file():
        missing.append(OCCLUDER_MODEL)
    if not (root / FACE_PARSER_MODEL).is_file():
        missing.append(FACE_PARSER_MODEL)
    return missing


def load_image(path: Path | str) -> np.ndarray | None:
    """Load an image as oriented BGR pixels, including non-ASCII paths."""
    try:
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda face: _face_area(face))


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)


def unique_output_path(
    output_dir: Path,
    source: Path,
    suffix: str | None = None,
    date_tag: str | None = None,
    name_prefix: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix or (source.suffix.lower() if source.suffix.lower() in IMAGE_EXTS else ".jpg")
    if not suffix.startswith("."):
        suffix = "." + suffix
    date_tag = date_tag or datetime.now().strftime("%d%m%Y-%H%M%S")
    requested_stem = (name_prefix or "").strip() or source.stem
    safe_stem = "".join(
        "_" if character in "/\\" or not character.isprintable() else character
        for character in requested_stem
    ).strip(" .")
    safe_stem = safe_stem[:80].rstrip(" .") or source.stem
    output_stem = f"{safe_stem}_swapped_{date_tag}"
    candidate = output_dir / f"{output_stem}{suffix}"
    number = 2
    while candidate.exists():
        candidate = output_dir / f"{output_stem}_{number}{suffix}"
        number += 1
    return candidate


def write_image(path: Path, bgr: np.ndarray, metadata_source: Path | None = None) -> None:
    """Write pixels atomically and retain safe EXIF/ICC metadata where possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.swapio-tmp{path.suffix}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    save_args: dict = {}

    if metadata_source:
        try:
            with Image.open(metadata_source) as original:
                exif = original.getexif()
                if exif:
                    exif[274] = 1  # pixels are already oriented
                    save_args["exif"] = exif.tobytes()
                if original.info.get("icc_profile"):
                    save_args["icc_profile"] = original.info["icc_profile"]
        except Exception:
            pass

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        save_args.update(quality=98, subsampling=0, optimize=True)
    elif suffix == ".png":
        save_args.update(compress_level=6)
    elif suffix == ".webp":
        save_args.update(quality=95, method=4)

    try:
        image.save(temp, **save_args)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def image_files(folder: Path | str, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )


def suggested_output_dir(destination_folder: Path | str) -> Path:
    """Return a safe sibling output folder for a destination folder."""
    folder = Path(destination_folder).expanduser().resolve()
    if folder.name:
        return folder.parent / f"{folder.name} - Swapio Output"
    return folder / "Swapio Output"


def _file_identity(path: Path | str) -> dict[str, str | int]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def processing_key(
    source_path: Path | str,
    target_path: Path | str,
    *,
    all_faces: bool,
    quality: str,
    output_format: str,
    character_name: str,
    preserve_mouth: bool,
    destination_eyes: bool,
    restoration_strength: float,
    target_face_index: int | None = None,
    target_face_box=None,
) -> str:
    """Identify an unchanged input and the settings that produced its output."""
    payload = {
        "revision": PROCESSING_REVISION,
        "source": _file_identity(source_path),
        "target": _file_identity(target_path),
        "all_faces": all_faces,
        "quality": quality,
        "output_format": output_format,
        "character_name": character_name.strip(),
        "preserve_mouth": preserve_mouth,
        "destination_eyes": destination_eyes,
        "restoration_strength": round(float(restoration_strength), 3),
        "target_face_index": target_face_index,
        "target_face_box": np.asarray(target_face_box).round(1).tolist() if target_face_box is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def face_selection_key(target_path: Path | str, faces: list) -> str:
    """Session choices expire when a photo or its ordered detections change."""
    payload = {
        "target": _file_identity(target_path),
        "boxes": [np.asarray(face.bbox).round(1).tolist() for face in faces],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ProcessingHistory:
    """Persistent per-output-folder record used to avoid duplicate work."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.path = self.output_dir / HISTORY_FILENAME
        self.records: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text())
            if data.get("version") == 1 and isinstance(data.get("records"), dict):
                self.records = data["records"]
        except (OSError, ValueError, TypeError):
            pass

    def completed_output(self, key: str) -> Path | None:
        record = self.records.get(key)
        if not isinstance(record, dict):
            return None
        output_name = record.get("output")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            return None
        output = self.output_dir / output_name
        return output if output.is_file() else None

    def record(self, key: str, target: Path, output: Path) -> bool:
        self.records[key] = {
            "target": str(target.expanduser().resolve()),
            "output": output.name,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        return self._save()

    def _save(self) -> bool:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"version": 1, "records": self.records}, indent=2)
            )
            temporary.replace(self.path)
            return True
        except OSError:
            return False


@dataclass
class SwapSummary:
    total: int
    completed: int
    skipped: int
    failed: int
    stopped: bool
    outputs: list[str]
    errors: list[dict]
    provider: str

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "stopped": self.stopped,
            "outputs": self.outputs,
            "errors": self.errors,
            "provider": self.provider,
        }


class SwapEngine:
    """Lazily loads local InsightFace models and performs still-image swaps."""

    def __init__(self, models: Path | None = None, use_gpu: bool = True, det_size: int = 640):
        self.models = Path(models) if models else model_dir()
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.analyser = None
        self.swapper = None
        self.hyperswap = None
        self.enhancer = None
        self.occluder = None
        self.face_parser = None
        self.accelerated_providers = ["CPUExecutionProvider"]
        self.enhancer_cpu_only = False
        self.provider = "Not loaded"

    def ensure_loaded(self, on_log: LogFn = _noop) -> None:
        if self.analyser is not None and self.swapper is not None and self.hyperswap is not None:
            return
        missing = missing_models(self.models)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Required local model files are missing: {joined}. "
                "Run setup_models.py from the Swapio folder."
            )

        import onnxruntime as ort
        from insightface import model_zoo
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        want_cuda = self.use_gpu and "CUDAExecutionProvider" in available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if want_cuda
            else ["CPUExecutionProvider"]
        )
        self.accelerated_providers = providers
        on_log("Loading local face models...")
        # FaceAnalysis expects <root>/models/buffalo_l.
        analyser = FaceAnalysis(
            name="buffalo_l",
            root=str(self.models.parent),
            allowed_modules=["detection", "landmark_2d_106", "recognition"],
            providers=providers,
        )
        analyser.prepare(ctx_id=0 if want_cuda else -1, det_size=(self.det_size, self.det_size))
        swapper = model_zoo.get_model(str(self.models / SWAPPER_MODEL), providers=providers)
        # HyperSwap currently returns non-finite pixels through CUDA on some
        # consumer NVIDIA cards/drivers. Keep detection and the draft swapper
        # accelerated, but run the quality model on its reliable CPU path.
        hyperswap = ort.InferenceSession(
            str(self.models / HYPERSWAP_MODEL), providers=["CPUExecutionProvider"]
        )
        analyser_sessions = [
            getattr(model, "session", None) for model in analyser.models.values()
        ]
        accelerated_sessions = analyser_sessions + [getattr(swapper, "session", None)]
        actual_cuda = any(
            session is not None and "CUDAExecutionProvider" in session.get_providers()
            for session in accelerated_sessions
        )
        self.provider = "CUDA detection · CPU quality" if actual_cuda else "CPU"
        if actual_cuda:
            on_log("CUDA accelerates face detection; quality swapping uses the stable CPU path.")
        elif want_cuda:
            on_log("CUDA runtime was found but could not start; using CPU safely.")
        on_log(f"Models ready on {self.provider}.")
        self.analyser = analyser
        self.swapper = swapper
        self.hyperswap = hyperswap

    def _ensure_enhancer(self, on_log: LogFn = _noop):
        """Load the face restoration model the first time Best quality runs."""
        if self.enhancer is not None:
            return self.enhancer
        path = self.models / ENHANCER_MODEL
        if not path.is_file():
            raise RuntimeError(
                f"Best quality needs {ENHANCER_MODEL}. "
                "Run setup_models.py from the Swapio folder."
            )
        import onnxruntime as ort

        on_log("Loading the face restoration model...")
        # Unlike HyperSwap, restoration returns valid pixels through CUDA, so it
        # starts accelerated. _restore_face falls back permanently to CPU for the
        # rest of the session if a card cannot actually deliver a usable result.
        providers = (
            ["CPUExecutionProvider"]
            if self.enhancer_cpu_only
            else self.accelerated_providers
        )
        self.enhancer = ort.InferenceSession(str(path), providers=providers)
        active = self.enhancer.get_providers()[0]
        on_log(
            "Face restoration ready on "
            + ("CUDA." if active == "CUDAExecutionProvider" else "CPU.")
        )
        return self.enhancer

    def _visible_face_mask(self, original, face):
        """Measure visibility on the untouched photo, including protruding tongues."""
        from insightface.utils import face_align
        import onnxruntime as ort

        if self.occluder is None:
            self.occluder = ort.InferenceSession(
                str(self.models / OCCLUDER_MODEL), providers=["CPUExecutionProvider"]
            )
        matrix = face_align.estimate_norm(face.kps, 256)
        crop = cv2.warpAffine(original, matrix, (256, 256),
                              borderMode=cv2.BORDER_REPLICATE)
        mask = self.occluder.run(None, {
            "input": crop[None].astype(np.float32) / 255.0
        })[0].squeeze()
        if mask.shape != (256, 256) or not np.isfinite(mask).all():
            raise RuntimeError("Face protection returned an invalid mask; result was not saved.")
        mask = np.clip((cv2.GaussianBlur(mask, (0, 0), 1.0) - 0.4) / 0.5, 0, 1)
        return cv2.warpAffine(mask, cv2.invertAffineTransform(matrix),
                              original.shape[1::-1])

    @staticmethod
    def _apply_visibility(swapped, original, visibility):
        alpha = np.clip(visibility, 0, 1)[:, :, None]
        return np.rint(alpha * swapped + (1 - alpha) * original).clip(0, 255).astype(np.uint8)

    def _mouth_expression_mask(self, original, face, visibility):
        """Protect an open/obstructed expression in original target coordinates."""
        points = getattr(face, "landmark_2d_106", None)
        kps = getattr(face, "kps", None)
        if points is None or kps is None:
            return None
        points, kps = np.asarray(points, np.float32), np.asarray(kps, np.float32)
        if (points.shape != (106, 2) or kps.shape != (5, 2)
                or not np.isfinite(points).all() or not np.isfinite(kps).all()):
            return None
        parsed = self._parse_face(original, face)
        if parsed is None:
            return None
        _crop, labels, matrix = parsed
        landmarks = cv2.transform(points[None], matrix)[0]
        return self._expression_mask_from_labels(labels, landmarks, visibility, matrix)

    def _parse_face(self, image, face, matrix=None):
        """Parse the aligned face for expression protection."""
        import onnxruntime as ort

        if self.face_parser is None:
            self.face_parser = ort.InferenceSession(
                str(self.models / FACE_PARSER_MODEL), providers=["CPUExecutionProvider"]
            )
        if matrix is None:
            kps = np.asarray(getattr(face, "kps", None), np.float32)
            if kps.shape != (5, 2) or not np.isfinite(kps).all():
                return None
            matrix, _ = cv2.estimateAffinePartial2D(
                kps, FFHQ_TEMPLATE * 512, method=cv2.RANSAC, ransacReprojThreshold=100
            )
            if matrix is None:
                return None
        crop = cv2.warpAffine(image, matrix, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = ((rgb - np.array([0.485, 0.456, 0.406], np.float32))
                / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None]
        prediction = self.face_parser.run(None, {"input": blob})[0]
        if prediction.shape != (1, 19, 512, 512) or not np.isfinite(prediction).all():
            raise RuntimeError("Face parsing returned invalid labels; result was not saved.")
        labels = prediction[0].argmax(0).astype(np.uint8)
        return crop, labels, matrix

    @staticmethod
    def _expression_mask_from_labels(labels, landmarks, visibility, matrix):
        """Select this face's mouth components, preserving their full contour."""
        outer = cv2.convexHull(np.rint(landmarks[52:72]).astype(np.int32))
        seed = np.zeros(labels.shape, np.uint8)
        cv2.fillConvexPoly(seed, outer, 1)
        width = max(float(np.ptp(landmarks[52:72, 0])), 1.0)
        radius = max(round(width * 0.06), 1)
        seed = cv2.dilate(seed, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        ))
        binary = np.isin(labels, (11, 12, 13)).astype(np.uint8)
        count, components = cv2.connectedComponents(binary)
        selected = np.zeros_like(binary)
        for component in range(1, count):
            region = components == component
            if np.any(region & (seed > 0)):
                selected[region] = 1
        area = int(selected.sum())
        if area < 12 or area > labels.size * 0.2:
            return None
        # Closed, unobstructed lips retain generated identity. Open mouths and
        # foreground objects at the lips need complete expression protection.
        inner_area = int(np.count_nonzero((labels == 11) & (selected > 0)))
        aligned_visibility = cv2.warpAffine(visibility, matrix, labels.shape[::-1])
        visible = float(aligned_visibility[seed > 0].mean()) if np.any(seed) else 1.0
        if inner_area < max(12, width * width * 0.012) and visible >= 0.85:
            return None
        full = cv2.warpAffine(selected, cv2.invertAffineTransform(matrix),
                              visibility.shape[::-1], flags=cv2.INTER_NEAREST)
        # Keep the complete original contour with a narrow outward feather.
        # This region never moves with the generated mouth landmarks.
        distance = cv2.distanceTransform(1 - full, cv2.DIST_L2, 5)
        scale = max(float(np.linalg.norm(matrix[:, 0])), 0.001)
        feather = max(width / scale * 0.02, 1.0)
        return np.clip(1.0 - distance / feather, 0, 1)

    @staticmethod
    def _complete_mouth_protection(protection, original_face, generated_face):
        """Cover both mouth footprints so generated corners cannot remain outside the mask."""
        before = getattr(original_face, "landmark_2d_106", None)
        after = getattr(generated_face, "landmark_2d_106", None)
        if before is None:
            return protection
        before = np.asarray(before, np.float32)
        if before.shape != (106, 2) or not np.isfinite(before).all():
            return protection
        points = before[52:72]
        width, _ = SwapEngine._feature_geometry(points)
        if width < 4:
            return protection
        footprints = [points]
        displacement = 0.0
        if after is not None:
            after = np.asarray(after, np.float32)
            if after.shape == (106, 2) and np.isfinite(after).all():
                generated = after[52:72]
                displacement = float(np.linalg.norm(generated.mean(0) - points.mean(0)))
                if displacement <= width * 0.5:
                    footprints.append(generated)
        binary = (protection >= 0.95).astype(np.uint8)
        hull = cv2.convexHull(np.rint(np.vstack(footprints)).astype(np.int32))
        cv2.fillConvexPoly(binary, hull, 1)
        # Perioral context hides the old generated corners/creases. Its size
        # follows the actual mouth and its measured movement, in native pixels.
        inset = max(round(min(width * 0.15, width * 0.06 + displacement * 0.3)), 1)
        binary = cv2.dilate(binary, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
        ))
        distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        feather = max(width * 0.04, 1.0)
        soft = np.clip(1 - distance / feather, 0, 1)
        return np.maximum(protection, soft * soft * (3 - 2 * soft))

    @staticmethod
    def _detail_score(image, points):
        width, _ = SwapEngine._feature_geometry(points)
        low = np.maximum(np.floor(points.min(0) - width * 0.25).astype(int), 0)
        high = np.minimum(np.ceil(points.max(0) + width * 0.25).astype(int), image.shape[1::-1])
        if np.any(high - low < 4):
            return 0.0
        grey = cv2.cvtColor(image[low[1]:high[1], low[0]:high[0]], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.mean(np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 1.0))))

    @staticmethod
    def _eye_detail_blend(original, rendered, restored, before, after, limit):
        """Accept local detail only when generated eye geometry stays stable."""
        if before is None or after is None:
            return rendered, 0
        before, after = np.asarray(before, np.float32), np.asarray(after, np.float32)
        if (before.shape != (106, 2) or after.shape != (106, 2)
                or not np.isfinite(before).all() or not np.isfinite(after).all()):
            return rendered, 0
        result = rendered.copy()
        accepted = 0
        for indices, brow_indices in ((LEFT_EYE_106_INDICES, np.arange(33, 52)),
                                      (RIGHT_EYE_106_INDICES, np.arange(87, 106))):
            eye, placed = before[indices], after[indices]
            width, opening = SwapEngine._feature_geometry(eye)
            if width < 48 or opening / width < 0.10:
                continue
            reference = SwapEngine._detail_score(original, eye)
            existing = SwapEngine._detail_score(rendered, eye)
            rebuilt = SwapEngine._detail_score(restored, placed)
            deficit = 1.0 - existing / max(reference, 0.001)
            movement = float(np.mean(np.linalg.norm(placed - eye, axis=1)))
            if deficit < 0.25 or rebuilt < existing * 1.15 or movement > width * 0.03:
                continue
            strength = min(float(np.clip(limit, 0, 1)), deficit * 1.3)
            if strength <= 0:
                continue
            points = np.vstack([before[brow_indices], after[brow_indices]])
            points = points[np.linalg.norm(points - eye.mean(0), axis=1) <= width * 0.9]
            radius = max(round(width * 0.12), 1)
            feather = max(width * 0.06, 1)
            margin = radius + int(np.ceil(feather * 4)) + 2
            low = np.maximum(np.floor(points.min(0) - margin).astype(int), 0)
            high = np.minimum(np.ceil(points.max(0) + margin).astype(int), rendered.shape[1::-1])
            if np.any(high - low < 4):
                continue
            region = np.s_[low[1]:high[1], low[0]:high[0]]
            mask = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points - low).astype(np.int32)), 1)
            mask = cv2.dilate(mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            ))
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)
            alpha = (np.clip(alpha, 0, 1) * strength)[:, :, None]
            # Retain generated lighting/colour and transfer only fine texture.
            # Whole-patch RGB blending mixed restored low-frequency structure
            # back into already soft close-ups and could brighten dark eyes.
            sigma = max(width * 0.02, 1)
            base = rendered[region].astype(np.float32)
            rebuilt_pixels = restored[region].astype(np.float32)
            base_high = base - cv2.GaussianBlur(base, (0, 0), sigma)
            rebuilt_high = rebuilt_pixels - cv2.GaussianBlur(rebuilt_pixels, (0, 0), sigma)
            gain = float(np.clip(reference / max(rebuilt, 0.001), 1, 1.35))
            result[region] = np.rint(result[region].astype(np.float32)
                + alpha * (gain * rebuilt_high - base_high)).clip(0, 255).astype(np.uint8)
            accepted += 1
        return result, accepted

    def _eye_landmark_frame(self, original, face):
        """Build one upright analysis frame for rotated heads, reused for all eye checks."""
        models = getattr(self.analyser, "models", None) or {}
        detector = models.get("detection")
        kps, box = getattr(face, "kps", None), getattr(face, "bbox", None)
        if detector is None or kps is None or box is None:
            return None
        kps, box = np.asarray(kps, np.float32), np.asarray(box, np.float32)
        if (kps.shape != (5, 2) or box.shape != (4,)
                or not np.isfinite(kps).all() or not np.isfinite(box).all()):
            return None
        delta = kps[1] - kps[0]
        eye_distance = float(np.linalg.norm(delta))
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
        if eye_distance < 4 or abs(angle) <= 30:
            return None
        span = float(max(box[2] - box[0], box[3] - box[1]) * 2)
        if span <= 0:
            return None
        side = int(np.clip(np.ceil(span), 256, 1024))
        centre = (box[:2] + box[2:]) / 2
        matrix = cv2.getRotationMatrix2D(tuple(map(float, centre)), angle, side / span)
        matrix[:, 2] += side / 2 - centre
        roi = cv2.warpAffine(original, matrix, (side, side), borderMode=cv2.BORDER_REPLICATE)
        try:
            boxes, candidates = detector.detect(roi)
            if len(boxes) == 0 or candidates is None:
                return None
            if not np.isfinite(boxes).all() or not np.isfinite(candidates).all():
                return None
            expected = cv2.transform(kps[None], matrix)[0]
            distances = np.mean(np.linalg.norm(candidates[:, :3] - expected[None, :3], axis=2), axis=1)
            index = int(np.argmin(distances))
            if boxes[index, 4] < 0.5 or distances[index] > eye_distance * side / span * 0.25:
                return None
            return matrix, side, boxes[index, :4].copy()
        except Exception:
            return None

    def _eye_landmarks(self, image, face, frame):
        if frame is None:
            return self._landmarks_after_swap(image, face)
        models = getattr(self.analyser, "models", None) or {}
        model = models.get("landmark_2d_106")
        if model is None:
            return None
        try:
            matrix, side, box = frame
            placed = (type(face)(deepcopy(dict(face))) if isinstance(face, dict) else deepcopy(face))
            placed.bbox = box.copy()
            roi = cv2.warpAffine(image, matrix, (side, side), borderMode=cv2.BORDER_REPLICATE)
            model.get(roi, placed)
            points = np.asarray(placed.landmark_2d_106, np.float32)
            if points.shape != (106, 2) or not np.isfinite(points).all():
                return None
            placed.landmark_2d_106 = cv2.transform(points[None], cv2.invertAffineTransform(matrix))[0]
            placed.bbox = deepcopy(face.bbox)
            return placed
        except Exception:
            return None

    def _adaptive_eye_detail(self, rendered, original, face, limit, on_log):
        if limit <= 0:
            return rendered
        frame = self._eye_landmark_frame(original, face)
        placed = self._eye_landmarks(rendered, face, frame)
        points = getattr(placed, "landmark_2d_106", None)
        if points is None:
            return rendered
        reference_face = self._eye_landmarks(original, face, frame) if frame is not None else face
        needs_detail = False
        for indices in (LEFT_EYE_106_INDICES, RIGHT_EYE_106_INDICES):
            eye = points[indices]
            width, opening = self._feature_geometry(eye)
            reference_points = getattr(reference_face, "landmark_2d_106", None)
            if reference_points is not None:
                reference_width, reference_opening = self._feature_geometry(reference_points[indices])
                if reference_width <= 0 or reference_opening / reference_width < 0.10:
                    continue
            if (width >= 48 and opening / width >= 0.10
                    and self._detail_score(rendered, eye) < self._detail_score(original, eye) * 0.75):
                needs_detail = True
        if not needs_detail:
            return rendered
        restored = self._restore_face(rendered, face, 1.0, on_log)
        after = self._eye_landmarks(restored, face, frame)
        result, count = self._eye_detail_blend(
            original, rendered, restored, points,
            getattr(after, "landmark_2d_106", None), limit,
        )
        pose = " in an upright analysis frame" if frame is not None else ""
        on_log(f"Adaptive eye detail: accepted {count} eye region(s) after detail/geometry checks{pose}.")
        return result

    def source_context(self, source_path: Path | str, on_log: LogFn = _noop):
        self.ensure_loaded(on_log)
        image = load_image(source_path)
        if image is None:
            raise RuntimeError(f"Could not read source image: {source_path}")
        face = _largest_face(self.analyser.get(image))
        if face is None:
            raise RuntimeError("No face was detected in the source photo.")
        return image, face

    def source_face(self, source_path: Path | str, on_log: LogFn = _noop):
        return self.source_context(source_path, on_log)[1]

    @staticmethod
    def _implode_pixels(crop: np.ndarray, model_size: int) -> np.ndarray:
        boost_size = crop.shape[0]
        total = boost_size // model_size
        return (
            crop.reshape(model_size, total, model_size, total, 3)
            .transpose(1, 3, 0, 2, 4)
            .reshape(total**2, model_size, model_size, 3)
        )

    @staticmethod
    def _explode_pixels(frames: list[np.ndarray], model_size: int, boost_size: int) -> np.ndarray:
        total = boost_size // model_size
        return (
            np.stack(frames)
            .reshape(total, total, model_size, model_size, 3)
            .transpose(2, 0, 3, 1, 4)
            .reshape(boost_size, boost_size, 3)
        )

    @staticmethod
    def _fuse_predictions(frames: list[np.ndarray], model_size: int, boost_size: int) -> np.ndarray:
        """Combine sampling phases on one grid, without interleaving their noise."""
        total = boost_size // model_size
        if total < 1 or boost_size % model_size or len(frames) != total ** 2:
            raise RuntimeError("The swap model returned an incomplete set of samples.")
        combined = np.zeros((boost_size, boost_size, 3), np.float32)
        for index, frame in enumerate(frames):
            if frame.shape != (model_size, model_size, 3) or not np.isfinite(frame).all():
                raise RuntimeError("The swap model returned an invalid sample.")
            phase_y, phase_x = divmod(index, total)
            # Input sample (x,y) represents aligned coordinate
            # (total*x+phase_x, total*y+phase_y). Register before averaging.
            matrix = np.array([[total, 0, phase_x], [0, total, phase_y]], np.float32)
            combined += cv2.warpAffine(
                frame.astype(np.float32), matrix, (boost_size, boost_size),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            ) / len(frames)
        return np.rint(combined).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _paste_crop(target: np.ndarray, crop: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        inverse = cv2.invertAffineTransform(matrix)
        height, width = target.shape[:2]
        warped = cv2.warpAffine(
            crop, inverse, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Feather in aligned face coordinates. Eroding the rotated rectangle
        # in image space made the blend depend on roll and image clipping,
        # leaving more of the old jaw behind on tilted close-ups.
        crop_height, crop_width = crop.shape[:2]
        yy, xx = np.mgrid[:crop_height, :crop_width].astype(np.float32)
        edge = np.minimum(
            np.minimum(xx, crop_width - 1 - xx) / crop_width,
            np.minimum(yy, crop_height - 1 - yy) / crop_height,
        )
        box = np.clip((edge - 0.01) / 0.07, 0.0, 1.0)
        box = box * box * (3.0 - 2.0 * box)
        mask = cv2.warpAffine(
            box,
            inverse,
            (width, height),
            borderValue=0.0,
        )
        mask = np.clip(mask, 0.0, 1.0)[:, :, None]
        return (mask * warped + (1 - mask) * target.astype(np.float32)).astype(np.uint8)

    def _hyperswap_face(self, target: np.ndarray, target_face, source_face, boost_size: int):
        from insightface.utils import face_align

        model_size = 256
        # norm_crop2 resamples bilinearly. The aligned crop is usually an upscale
        # of the face region, and bilinear softens exactly the eyelash, lid and
        # iris detail the quality path exists to keep. Reuse the same alignment
        # matrix and resample with Lanczos instead.
        matrix = face_align.estimate_norm(target_face.kps, boost_size)
        crop = cv2.warpAffine(
            target,
            matrix,
            (boost_size, boost_size),
            flags=cv2.INTER_LANCZOS4,
            borderValue=0.0,
        )
        tiles = self._implode_pixels(crop, model_size)
        source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)
        swapped_tiles = []
        for tile in tiles:
            target_blob = ((tile[:, :, ::-1] / 255.0 - 0.5) / 0.5)
            target_blob = target_blob.transpose(2, 0, 1)[None].astype(np.float32)
            prediction = self.hyperswap.run(
                None, {"source": source_embedding, "target": target_blob}
            )[0][0]
            if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 100:
                raise RuntimeError(
                    "The quality model returned invalid pixels; the result was not saved."
                )
            prediction = prediction.transpose(1, 2, 0)
            swapped = np.clip((prediction * 0.5 + 0.5) * 255, 0, 255)
            swapped_tiles.append(swapped[:, :, ::-1])
        crop = self._fuse_predictions(swapped_tiles, model_size, boost_size)
        return self._paste_crop(target, crop, matrix)

    @staticmethod
    def _boost_size_for_face(target: np.ndarray, face, quality: str) -> int:
        """Use more model samples when a face occupies a large part of the photo."""
        if quality not in BOOSTED_QUALITIES:
            return 256
        height, width = target.shape[:2]
        x1, y1, x2, y2 = map(float, face.bbox)
        face_width = max(x2 - x1, 1.0)
        face_height = max(y2 - y1, 1.0)
        width_ratio = face_width / max(width, 1)
        height_ratio = face_height / max(height, 1)
        span = max(face_width, face_height)
        if width_ratio >= 0.40 or height_ratio >= 0.48 or span >= 700:
            return 1024
        if width_ratio >= 0.24 or height_ratio >= 0.28 or span >= 480:
            return 768
        return 512

    @staticmethod
    def _paste_restored(target: np.ndarray, crop: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Blend an FFHQ-aligned crop back with a soft box mask.

        The swapper's paste erodes a tenth of the crop away, which is right for
        an ArcFace crop that stops at the jaw. An FFHQ crop reaches into hair
        and background, so it only needs a narrow feathered border to hide the
        seam without eating the restored hairline.
        """
        size = crop.shape[0]
        inset = max(int(size * 0.06), 1)
        box = np.zeros((size, size), dtype=np.float32)
        box[inset : size - inset, inset : size - inset] = 1.0
        box = cv2.GaussianBlur(box, (0, 0), size * 0.03)
        inverse = cv2.invertAffineTransform(matrix)
        height, width = target.shape[:2]
        warped = cv2.warpAffine(
            crop, inverse, (width, height), flags=cv2.INTER_LANCZOS4, borderValue=0.0
        )
        mask = cv2.warpAffine(box, inverse, (width, height), borderValue=0.0)
        mask = np.clip(mask, 0.0, 1.0)[:, :, None]
        return (mask * warped + (1 - mask) * target.astype(np.float32)).astype(np.uint8)

    def _run_enhancer(self, session, blob: np.ndarray, on_log: LogFn = _noop) -> np.ndarray:
        """Run restoration, retiring a GPU that cannot produce usable pixels."""
        feed = {"input": blob[None].astype(np.float32)}
        accelerated = session.get_providers()[0] != "CPUExecutionProvider"
        try:
            prediction = session.run(None, feed)[0][0]
            if np.isfinite(prediction).all():
                return prediction
            reason = "returned invalid pixels"
        except Exception as exc:  # a CUDA build can also fail on memory
            if not accelerated:
                raise
            reason = f"failed with {type(exc).__name__}"
        if not accelerated:
            raise RuntimeError(
                "The face restoration model returned invalid pixels; "
                "the result was not saved."
            )
        on_log(
            f"Face restoration {reason} on the GPU; "
            "continuing on the stable CPU path."
        )
        self.enhancer_cpu_only = True
        self.enhancer = None
        session = self._ensure_enhancer(on_log)
        prediction = session.run(None, feed)[0][0]
        if not np.isfinite(prediction).all():
            raise RuntimeError(
                "The face restoration model returned invalid pixels; "
                "the result was not saved."
            )
        return prediction

    def _restore_face(
        self,
        swapped: np.ndarray,
        target_face,
        strength: float = DEFAULT_RESTORATION_STRENGTH,
        on_log: LogFn = _noop,
    ):
        """Rebuild eye, lash and tooth detail the 256px swapper cannot resolve."""
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            # Skip the model entirely rather than blending none of it, so that
            # "off" is byte-identical to Careful and costs nothing.
            return swapped
        session = self._ensure_enhancer(on_log)
        destination = FFHQ_TEMPLATE * ENHANCER_SIZE
        # A generous threshold keeps all five landmarks as inliers, so this is a
        # least-squares similarity fit rather than a robust subset fit.
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(target_face.kps, dtype=np.float32),
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )
        if matrix is None:
            return swapped
        crop = cv2.warpAffine(
            swapped,
            matrix,
            (ENHANCER_SIZE, ENHANCER_SIZE),
            flags=cv2.INTER_LANCZOS4,
            borderValue=0.0,
        )
        blob = ((crop[:, :, ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        prediction = self._run_enhancer(session, blob, on_log)
        restored = np.clip(prediction, -1.0, 1.0).transpose(1, 2, 0)
        restored = np.clip((restored + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        # Keep a little of the swapped crop so restoration sharpens the face
        # instead of repainting it.
        blended = cv2.addWeighted(crop, 1.0 - strength, restored, strength, 0.0)
        return self._paste_restored(swapped, blended, matrix)

    def _landmarks_after_swap(self, image: np.ndarray, target_face):
        """Measure the final features without mutating the original face."""
        models = getattr(self.analyser, "models", None) or {}
        model = models.get("landmark_2d_106")
        if model is None:
            return None
        try:
            # InsightFace's dict subclass returns None for missing attributes,
            # including pickle hooks, so deepcopy(Face) fails. Reconstruct it
            # from copied items and keep its landmark arrays independent.
            placed = (
                type(target_face)(deepcopy(dict(target_face)))
                if isinstance(target_face, dict) else deepcopy(target_face)
            )
            model.get(image, placed)
            points = np.asarray(placed.landmark_2d_106, dtype=np.float32)
            if points.shape != (106, 2) or not np.isfinite(points).all():
                return None
            return placed
        except Exception:
            # Preservation is optional; never paste at stale coordinates if
            # the new feature positions cannot be measured.
            return None

    @staticmethod
    def _feature_geometry(points: np.ndarray) -> tuple[float, float]:
        """Width and opening measured along the feature, independent of roll."""
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        a, b = np.unravel_index(np.argmax(distances), distances.shape)
        width = float(distances[a, b])
        if width <= 0:
            return 0.0, 0.0
        axis = (points[b] - points[a]) / width
        normal = np.array([-axis[1], axis[0]])
        return width, float(np.ptp(points @ normal))

    @staticmethod
    def _feature_mask(shape, points: np.ndarray, opening: float) -> np.ndarray:
        """Feather inward, keeping all original pixels inside the contour."""
        mask = np.zeros(shape, dtype=np.uint8)
        # Eyelid indices are not in perimeter order. Sort around the centre,
        # retaining concave lip/lid contours instead of filling their hull.
        centred = points - points.mean(axis=0)
        order = np.argsort(np.arctan2(centred[:, 1], centred[:, 0]))
        cv2.fillPoly(mask, [np.rint(points[order]).astype(np.int32)], 1)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Width-based erosion erased narrow eyes. The opening sets the blend
        # here, and clipping the distance mask avoids spilling onto lips/lids.
        feather = max(opening * 0.18, 1.0)
        soft = np.clip((distance - 0.5) / feather, 0.0, 1.0)
        return soft * soft * (3.0 - 2.0 * soft)

    @staticmethod
    def _preserve_feature(
        swapped: np.ndarray,
        original: np.ndarray,
        target_face,
        placed_face,
        indices: np.ndarray,
        minimum_opening: float,
    ) -> np.ndarray:
        if placed_face is None:
            return swapped
        before = getattr(target_face, "landmark_2d_106", None)
        after = getattr(placed_face, "landmark_2d_106", None)
        if before is None or after is None:
            return swapped
        before = np.asarray(before, dtype=np.float32)
        after = np.asarray(after, dtype=np.float32)
        if (before.shape != (106, 2) or after.shape != (106, 2)
                or not np.isfinite(before).all() or not np.isfinite(after).all()):
            return swapped
        source, destination = before[indices], after[indices]
        source_width, source_opening = SwapEngine._feature_geometry(source)
        width, opening = SwapEngine._feature_geometry(destination)
        if (min(source_width, width) < 4
                or source_opening / source_width < minimum_opening
                or opening / width < minimum_opening):
            return swapped
        matrix, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            return swapped
        scale = float(np.linalg.norm(matrix[:, 0]))
        fitted = cv2.transform(source[None], matrix)[0]
        error = float(np.sqrt(np.mean(np.sum((fitted - destination) ** 2, axis=1))))
        movement = float(np.linalg.norm(source.mean(axis=0) - destination.mean(axis=0)))
        if not 0.5 <= scale <= 2.0 or error > width * 0.15 or movement > width * 0.5:
            return swapped
        height, image_width = swapped.shape[:2]
        registered = cv2.warpAffine(
            original, matrix, (image_width, height),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
        )
        source_mask = SwapEngine._feature_mask(original.shape[:2], source, source_opening)
        source_mask = cv2.warpAffine(source_mask, matrix, (image_width, height))
        destination_mask = SwapEngine._feature_mask(swapped.shape[:2], destination, opening)
        # Both masks must agree: do not copy target lips/skin into a changed
        # opening, even when the source and generated shapes differ.
        mask = np.minimum(source_mask, destination_mask)[:, :, None]
        return np.rint(
            mask * registered.astype(np.float32)
            + (1.0 - mask) * swapped.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    def _destination_eyes(self, swapped, target, target_face, placed_face):
        """Register original eye interiors to the final swapped eyelids."""
        for indices in (LEFT_EYE_106_INDICES, RIGHT_EYE_106_INDICES):
            swapped = self._preserve_feature(
                swapped, target, target_face, placed_face, indices, 0.12
            )
        return swapped

    @staticmethod
    def _preserve_inner_mouth(swapped, original, target_face, placed_face):
        """Register target teeth/tongue inside the final swapped lip contour."""
        return SwapEngine._preserve_feature(
            swapped, original, target_face, placed_face, INNER_MOUTH_106_INDICES, 0.045
        )

    def swap_array(
        self,
        target: np.ndarray,
        source_face,
        all_faces: bool = False,
        quality: str = "careful",
        preserve_mouth: bool = True,
        destination_eyes: bool = False,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        on_log: LogFn = _noop,
        target_face_index: int | None = None,
        detected_faces: list | None = None,
    ):
        faces = sorted(self.analyser.get(target) if detected_faces is None else detected_faces,
                       key=_face_area, reverse=True)
        if not faces:
            raise RuntimeError("No destination face detected")
        if target_face_index is not None:
            if (not isinstance(target_face_index, (int, np.integer))
                    or not 0 <= target_face_index < len(faces)):
                raise RuntimeError("Selected face is unavailable; result was not saved.")
            selected = [faces[target_face_index]]
        else:
            selected = faces if all_faces else faces[:1]
        result = target.copy()
        for face in selected:
            previous = result.copy()
            visibility = self._visible_face_mask(target, face)
            mouth_protection = self._mouth_expression_mask(target, face, visibility) if preserve_mouth else None
            if quality == "fast":
                result = self.swapper.get(result, face, source_face, paste_back=True)
            else:
                boost_size = self._boost_size_for_face(target, face, quality)
                if boost_size > 512:
                    on_log(
                        f"Close-up face: combining {(boost_size // 256) ** 2} model samples "
                        "on one grid. The swap model's native resolution is 256px."
                    )
                result = self._hyperswap_face(result, face, source_face, boost_size)
            if quality == "best":
                # Restore before the mouth is put back, so the target's own
                # teeth are the last thing written into the frame.
                result = self._restore_face(result, face, restoration_strength, on_log)
            elif quality == "careful":
                result = self._adaptive_eye_detail(result, target, face, restoration_strength, on_log)
            if destination_eyes or (preserve_mouth and mouth_protection is None):
                placed_face = self._landmarks_after_swap(result, face)
                if placed_face is None:
                    on_log("Feature preservation skipped: final landmarks unavailable.")
                if destination_eyes:
                    result = self._destination_eyes(result, target, face, placed_face)
                if preserve_mouth and mouth_protection is None:
                    result = self._preserve_inner_mouth(result, target, face, placed_face)
            if mouth_protection is not None:
                placed_face = self._landmarks_after_swap(result, face)
                mouth_protection = self._complete_mouth_protection(mouth_protection, face, placed_face)
                result = self._apply_visibility(result, target, 1.0 - mouth_protection)
                on_log("Protected the target's open/obstructed mouth expression, including lips and tongue.")
            # Protect foreground after restoration and mouth registration,
            # preserving prior swapped faces too.
            result = self._apply_visibility(result, previous, visibility)
        return result, len(selected), len(faces)

    def _target_face_choice(self, target, target_path, face_selector):
        if face_selector is None:
            return None, None
        faces = sorted(self.analyser.get(target), key=_face_area, reverse=True)
        if not faces:
            raise RuntimeError("No destination face detected")
        choice = 0 if len(faces) == 1 else face_selector(Path(target_path), target, faces)
        if choice is None:
            raise FaceSelectionSkipped("Photo skipped without choosing a face.")
        if not isinstance(choice, (int, np.integer)) or not 0 <= choice < len(faces):
            raise RuntimeError("Selected face is unavailable; result was not saved.")
        return int(choice), faces

    def preview(
        self,
        source_path: Path | str,
        target_path: Path | str,
        all_faces: bool = False,
        quality: str = "careful",
        preserve_mouth: bool = True,
        destination_eyes: bool = False,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        on_log: LogFn = _noop,
        face_selector: FaceSelector | None = None,
    ) -> dict:
        source = self.source_face(source_path, on_log)
        target = load_image(target_path)
        if target is None:
            raise RuntimeError(f"Could not read destination image: {target_path}")
        choice, faces = self._target_face_choice(target, target_path, face_selector)
        result, swapped, detected = self.swap_array(
            target,
            source,
            all_faces=all_faces,
            quality=quality,
            preserve_mouth=preserve_mouth,
            destination_eyes=destination_eyes,
            restoration_strength=restoration_strength,
            on_log=on_log,
            target_face_index=choice,
            detected_faces=faces,
        )
        return {
            "image": result,
            "target": str(target_path),
            "swapped": swapped,
            "detected": detected,
            "provider": self.provider,
            "quality": quality,
            "target_face_index": choice,
        }

    def batch(
        self,
        source_path: Path | str,
        targets: Iterable[Path | str],
        output_dir: Path | str,
        all_faces: bool = False,
        quality: str = "careful",
        output_format: str = "jpg",
        character_name: str = "",
        preserve_mouth: bool = True,
        destination_eyes: bool = False,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        skip_completed: bool = True,
        on_log: LogFn = _noop,
        on_progress: ProgressFn = _noop,
        should_stop: StopFn = lambda: False,
        face_selector: FaceSelector | None = None,
    ) -> dict:
        target_paths = [Path(path) for path in targets]
        output_dir = Path(output_dir).expanduser().resolve()
        history = ProcessingHistory(output_dir)
        keyed_targets: list[tuple[Path, str]] = []
        for target_path in target_paths:
            key = processing_key(
                source_path,
                target_path,
                all_faces=all_faces,
                quality=quality,
                output_format=output_format,
                character_name=character_name,
                preserve_mouth=preserve_mouth,
                destination_eyes=destination_eyes,
                restoration_strength=restoration_strength,
            )
            keyed_targets.append((target_path, key))
        pending = [
            (target, key)
            for target, key in keyed_targets
            if face_selector is not None or not skip_completed or history.completed_output(key) is None
        ]
        source = None
        if pending:
            source = self.source_face(source_path, on_log)
        completed = 0
        skipped = 0
        errors: list[dict] = []
        outputs: list[str] = []
        total = len(target_paths)
        cancelled = False
        batch_date_tag = datetime.now().strftime("%d%m%Y-%H%M%S")
        detail = {
            "best": "Best clean sampling with face restoration",
            "careful": "Careful clean sampling",
            "balanced": "Balanced 256px",
            "fast": "Fast 128px",
        }.get(quality, quality)
        on_log(
            f"Processing {total} destination photo(s) at {detail}. "
            "Originals stay untouched."
        )
        for index, (target_path, key) in enumerate(keyed_targets, start=1):
            if should_stop():
                on_log("Stopped by user.")
                break
            try:
                target = None
                choice, faces = None, None
                if face_selector is not None:
                    target = load_image(target_path)
                    if target is None:
                        raise RuntimeError("Unreadable image")
                    choice, faces = self._target_face_choice(target, target_path, face_selector)
                    key = processing_key(
                        source_path, target_path, all_faces=all_faces, quality=quality,
                        output_format=output_format, character_name=character_name,
                        preserve_mouth=preserve_mouth, destination_eyes=destination_eyes,
                        restoration_strength=restoration_strength, target_face_index=choice,
                        target_face_box=faces[choice].bbox,
                    )
                if should_stop():
                    raise ProcessingCancelled("Stopped by user.")
                previous_output = history.completed_output(key) if skip_completed else None
                if previous_output is not None:
                    skipped += 1
                    on_log(f"↷ {target_path.name}: unchanged; already saved as {previous_output.name}")
                    on_progress(index, total,
                                {"completed": completed, "skipped": skipped, "failed": len(errors)})
                    continue
                if target is None:
                    target = load_image(target_path)
                if target is None:
                    raise RuntimeError("Unreadable image")
                result, swapped, detected = self.swap_array(
                    target,
                    source,
                    all_faces=all_faces,
                    quality=quality,
                    preserve_mouth=preserve_mouth,
                    destination_eyes=destination_eyes,
                    restoration_strength=restoration_strength,
                    on_log=on_log,
                    target_face_index=choice,
                    detected_faces=faces,
                )
                suffix = ".png" if output_format == "png" else ".jpg"
                output_path = unique_output_path(
                    output_dir,
                    target_path,
                    suffix=suffix,
                    date_tag=batch_date_tag,
                    name_prefix=character_name,
                )
                write_image(output_path, result, metadata_source=target_path)
                if not history.record(key, target_path, output_path):
                    on_log("! Could not update batch history; this photo may run again later.")
                outputs.append(str(output_path))
                completed += 1
                detail = (f"face {choice + 1} of {detected}" if choice is not None
                          else f"{swapped} of {detected} face(s)" if detected > 1 else "1 face")
                on_log(f"✓ {target_path.name} → {output_path.name} ({detail})")
            except ProcessingCancelled:
                cancelled = True
                on_log("Stopped by user.")
                break
            except FaceSelectionSkipped:
                skipped += 1
                on_log(f"↷ {target_path.name}: skipped by choice")
            except Exception as exc:  # continue a batch when one photo is unsuitable
                message = f"{type(exc).__name__}: {exc}"
                errors.append({"path": str(target_path), "error": message})
                on_log(f"! {target_path.name}: {message}")
            on_progress(
                index,
                total,
                {"completed": completed, "skipped": skipped, "failed": len(errors)},
            )

        stopped = cancelled or (should_stop() and completed + skipped + len(errors) < total)
        return SwapSummary(
            total=total,
            completed=completed,
            skipped=skipped,
            failed=len(errors),
            stopped=stopped,
            outputs=outputs,
            errors=errors,
            provider=self.provider,
        ).as_dict()
