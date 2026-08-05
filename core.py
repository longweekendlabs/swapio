#!/usr/bin/env python3
"""Offline face-swapping engine for Swapio."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REQUIRED_BUFFALO_MODELS = {"det_10g.onnx", "w600k_r50.onnx"}
SWAPPER_MODEL = "inswapper_128.onnx"
HYPERSWAP_MODEL = "hyperswap_1a_256.onnx"

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, dict], None]
StopFn = Callable[[], bool]


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


def unique_output_path(output_dir: Path, source: Path, suffix: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix or (source.suffix.lower() if source.suffix.lower() in IMAGE_EXTS else ".jpg")
    if not suffix.startswith("."):
        suffix = "." + suffix
    candidate = output_dir / f"{source.stem}_swapped{suffix}"
    number = 2
    while candidate.exists():
        candidate = output_dir / f"{source.stem}_swapped_{number}{suffix}"
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


@dataclass
class SwapSummary:
    total: int
    completed: int
    failed: int
    stopped: bool
    outputs: list[str]
    errors: list[dict]
    provider: str

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
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
        on_log("Loading local face models...")
        # FaceAnalysis expects <root>/models/buffalo_l.
        analyser = FaceAnalysis(
            name="buffalo_l",
            root=str(self.models.parent),
            allowed_modules=["detection", "recognition"],
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

        self.provider = "CUDA detection · CPU quality" if want_cuda else "CPU"
        if want_cuda:
            on_log("CUDA accelerates face detection; quality swapping uses the stable CPU path.")
        on_log(f"Models ready on {self.provider}.")
        self.analyser = analyser
        self.swapper = swapper
        self.hyperswap = hyperswap

    def source_face(self, source_path: Path | str, on_log: LogFn = _noop):
        self.ensure_loaded(on_log)
        image = load_image(source_path)
        if image is None:
            raise RuntimeError(f"Could not read source image: {source_path}")
        face = _largest_face(self.analyser.get(image))
        if face is None:
            raise RuntimeError("No face was detected in the source photo.")
        return face

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
    def _paste_crop(target: np.ndarray, crop: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        inverse = cv2.invertAffineTransform(matrix)
        height, width = target.shape[:2]
        warped = cv2.warpAffine(crop, inverse, (width, height), borderValue=0.0)
        mask = cv2.warpAffine(
            np.full(crop.shape[:2], 255, dtype=np.float32),
            inverse,
            (width, height),
            borderValue=0.0,
        )
        mask[mask > 20] = 255
        ys, xs = np.where(mask == 255)
        if len(xs) == 0 or len(ys) == 0:
            return target
        mask_size = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
        erode = max(mask_size // 10, 10)
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=1)
        blur = max(mask_size // 20, 5)
        mask = cv2.GaussianBlur(mask, (2 * blur + 1, 2 * blur + 1), 0) / 255.0
        mask = mask[:, :, None]
        return (mask * warped + (1 - mask) * target.astype(np.float32)).astype(np.uint8)

    def _hyperswap_face(self, target: np.ndarray, target_face, source_face, boost_size: int):
        from insightface.utils import face_align

        model_size = 256
        crop, matrix = face_align.norm_crop2(target, target_face.kps, boost_size)
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
            swapped = np.clip((prediction * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            swapped_tiles.append(swapped[:, :, ::-1])
        crop = self._explode_pixels(swapped_tiles, model_size, boost_size)
        return self._paste_crop(target, crop, matrix)

    def swap_array(
        self,
        target: np.ndarray,
        source_face,
        all_faces: bool = False,
        quality: str = "careful",
    ):
        faces = sorted(self.analyser.get(target), key=_face_area, reverse=True)
        if not faces:
            raise RuntimeError("No destination face detected")
        selected = faces if all_faces else faces[:1]
        result = target.copy()
        for face in selected:
            if quality == "fast":
                result = self.swapper.get(result, face, source_face, paste_back=True)
            else:
                boost_size = 512 if quality == "careful" else 256
                result = self._hyperswap_face(result, face, source_face, boost_size)
        return result, len(selected), len(faces)

    def preview(
        self,
        source_path: Path | str,
        target_path: Path | str,
        all_faces: bool = False,
        quality: str = "careful",
        on_log: LogFn = _noop,
    ) -> dict:
        source = self.source_face(source_path, on_log)
        target = load_image(target_path)
        if target is None:
            raise RuntimeError(f"Could not read destination image: {target_path}")
        result, swapped, detected = self.swap_array(target, source, all_faces, quality)
        return {
            "image": result,
            "target": str(target_path),
            "swapped": swapped,
            "detected": detected,
            "provider": self.provider,
            "quality": quality,
        }

    def batch(
        self,
        source_path: Path | str,
        targets: Iterable[Path | str],
        output_dir: Path | str,
        all_faces: bool = False,
        quality: str = "careful",
        output_format: str = "png",
        on_log: LogFn = _noop,
        on_progress: ProgressFn = _noop,
        should_stop: StopFn = lambda: False,
    ) -> dict:
        target_paths = [Path(path) for path in targets]
        output_dir = Path(output_dir)
        source = self.source_face(source_path, on_log)
        completed = 0
        errors: list[dict] = []
        outputs: list[str] = []
        total = len(target_paths)
        detail = {"careful": "Careful 512px", "balanced": "Balanced 256px", "fast": "Fast 128px"}.get(
            quality, quality
        )
        on_log(
            f"Processing {total} destination photo(s) at {detail}. "
            "Originals stay untouched."
        )

        for index, target_path in enumerate(target_paths, start=1):
            if should_stop():
                on_log("Stopped by user.")
                break
            try:
                target = load_image(target_path)
                if target is None:
                    raise RuntimeError("Unreadable image")
                result, swapped, detected = self.swap_array(target, source, all_faces, quality)
                suffix = ".png" if output_format == "png" else ".jpg"
                output_path = unique_output_path(output_dir, target_path, suffix=suffix)
                write_image(output_path, result, metadata_source=target_path)
                outputs.append(str(output_path))
                completed += 1
                detail = f"{swapped} of {detected} face(s)" if detected > 1 else "1 face"
                on_log(f"✓ {target_path.name} → {output_path.name} ({detail})")
            except Exception as exc:  # continue a batch when one photo is unsuitable
                message = f"{type(exc).__name__}: {exc}"
                errors.append({"path": str(target_path), "error": message})
                on_log(f"! {target_path.name}: {message}")
            on_progress(index, total, {"completed": completed, "failed": len(errors)})

        stopped = should_stop() and completed + len(errors) < total
        return SwapSummary(
            total=total,
            completed=completed,
            failed=len(errors),
            stopped=stopped,
            outputs=outputs,
            errors=errors,
            provider=self.provider,
        ).as_dict()
