#!/usr/bin/env python3
"""Offline face-swapping engine for Swapio."""
from __future__ import annotations

import hashlib
import json
import os
import sys
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
HYPERSWAP_MODEL = "hyperswap_1a_256.onnx"
ENHANCER_MODEL = "gpen_bfr_1024.onnx"
HISTORY_FILENAME = ".swapio-history.json"
PROCESSING_REVISION = 7
INNER_MOUTH_106_INDICES = np.array([65, 66, 62, 70, 69, 57, 60, 54])
# The 106-point model puts each eyelid contour in its own run; 43-51 and 97-105
# next to them are the eyebrows and must stay out of it.
LEFT_EYE_106_INDICES = np.arange(33, 43)
RIGHT_EYE_106_INDICES = np.arange(87, 97)
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
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    def _paste_crop(target: np.ndarray, crop: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        inverse = cv2.invertAffineTransform(matrix)
        height, width = target.shape[:2]
        warped = cv2.warpAffine(
            crop, inverse, (width, height), flags=cv2.INTER_LANCZOS4, borderValue=0.0
        )
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
            swapped = np.clip((prediction * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            swapped_tiles.append(swapped[:, :, ::-1])
        crop = self._explode_pixels(swapped_tiles, model_size, boost_size)
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

    def _destination_eyes(self, swapped: np.ndarray, target: np.ndarray, target_face):
        """Put the destination's own eyeballs back, registered to the swapped lids.

        The swapper renders each eye at 256px and the result reads wrong: the
        highlight glares and the iris loses its colour. The destination's eyes
        are real photographic pixels, so use them.

        The earlier attempt at this pasted them at the destination's coordinates
        and made large heads worse, because the swap moves the eyes and that
        offset grows with head size. Here each eye is warped from where the
        destination has it onto where the swap actually put it, measured by
        re-running the landmark model on the swapped pixels, so the eyeball
        lands inside the lids that are really there.
        """
        models = getattr(self.analyser, "models", None) or {}
        model = models.get("landmark_2d_106")
        if model is None or getattr(target_face, "landmark_2d_106", None) is None:
            return swapped
        try:
            moved = type(target_face)(dict(target_face))
            model.get(swapped, moved)
        except Exception:
            return swapped
        after = getattr(moved, "landmark_2d_106", None)
        if after is None or len(after) < 106:
            return swapped

        before = np.asarray(target_face.landmark_2d_106, dtype=np.float32)
        after = np.asarray(after, dtype=np.float32)
        height, width = swapped.shape[:2]
        result = swapped
        for indices in (LEFT_EYE_106_INDICES, RIGHT_EYE_106_INDICES):
            source_eye, placed_eye = before[indices], after[indices]
            eye_width = float(np.ptp(placed_eye[:, 0]))
            eye_height = float(np.ptp(placed_eye[:, 1]))
            # A closed eye has no eyeball worth carrying over.
            if eye_width < 6 or eye_height / eye_width < 0.12:
                continue
            matrix, _ = cv2.estimateAffinePartial2D(source_eye, placed_eye, method=cv2.LMEDS)
            if matrix is None:
                continue
            registered = cv2.warpAffine(
                target, matrix, (width, height),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            )
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(
                mask, cv2.convexHull(np.rint(placed_eye).astype(np.int32)), 255
            )
            # Stay inside the lid line so the swapped lashes and liner survive.
            inset = max(1, round(eye_width * 0.10))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
            )
            mask = cv2.erode(mask, kernel, iterations=1)
            soft = (
                cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(eye_width * 0.05, 1.0))
                / 255.0
            )[:, :, None]
            result = (
                soft * registered.astype(np.float32)
                + (1.0 - soft) * result.astype(np.float32)
            ).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _preserve_inner_mouth(
        swapped: np.ndarray,
        original: np.ndarray,
        target_face,
    ) -> np.ndarray:
        """Restore only the target pixels inside an open inner-lip contour."""
        landmarks = getattr(target_face, "landmark_2d_106", None)
        if landmarks is None or len(landmarks) < 106:
            return swapped
        points = np.asarray(landmarks, dtype=np.float32)[INNER_MOUTH_106_INDICES]
        left, right = points[0], points[4]
        top, bottom = points[2], points[6]
        mouth_width = float(np.linalg.norm(right - left))
        mouth_opening = float(np.linalg.norm(bottom - top))
        if mouth_width < 4 or mouth_opening / mouth_width < 0.045:
            return swapped

        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        # The contour already follows the inner lip. Only a very small inset is
        # needed; the old 1.8% erosion clipped tooth edges on large close-ups.
        inset = max(1, round(mouth_width * 0.008))
        kernel_size = inset * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, kernel, iterations=1)
        feather = max(mouth_width * 0.022, 0.9)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0
        mask = mask[:, :, None]
        return (
            mask * original.astype(np.float32)
            + (1.0 - mask) * swapped.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    def swap_array(
        self,
        target: np.ndarray,
        source_face,
        all_faces: bool = False,
        quality: str = "careful",
        preserve_mouth: bool = True,
        destination_eyes: bool = True,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        on_log: LogFn = _noop,
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
                boost_size = self._boost_size_for_face(target, face, quality)
                if boost_size > 512:
                    on_log(
                        f"Close-up face detected: using adaptive {boost_size}px detail "
                        "for cleaner eyes, lips and teeth."
                    )
                result = self._hyperswap_face(result, face, source_face, boost_size)
            if quality == "best":
                # Restore before the mouth is put back, so the target's own
                # teeth are the last thing written into the frame.
                result = self._restore_face(result, face, restoration_strength, on_log)
            if destination_eyes:
                result = self._destination_eyes(result, target, face)
            if preserve_mouth:
                result = self._preserve_inner_mouth(result, target, face)
        return result, len(selected), len(faces)

    def preview(
        self,
        source_path: Path | str,
        target_path: Path | str,
        all_faces: bool = False,
        quality: str = "careful",
        preserve_mouth: bool = True,
        destination_eyes: bool = True,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        on_log: LogFn = _noop,
    ) -> dict:
        source = self.source_face(source_path, on_log)
        target = load_image(target_path)
        if target is None:
            raise RuntimeError(f"Could not read destination image: {target_path}")
        result, swapped, detected = self.swap_array(
            target,
            source,
            all_faces=all_faces,
            quality=quality,
            preserve_mouth=preserve_mouth,
            destination_eyes=destination_eyes,
            restoration_strength=restoration_strength,
            on_log=on_log,
        )
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
        output_format: str = "jpg",
        character_name: str = "",
        preserve_mouth: bool = True,
        destination_eyes: bool = True,
        restoration_strength: float = DEFAULT_RESTORATION_STRENGTH,
        skip_completed: bool = True,
        on_log: LogFn = _noop,
        on_progress: ProgressFn = _noop,
        should_stop: StopFn = lambda: False,
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
            if not skip_completed or history.completed_output(key) is None
        ]
        source = None
        if pending:
            source = self.source_face(source_path, on_log)
        completed = 0
        skipped = 0
        errors: list[dict] = []
        outputs: list[str] = []
        total = len(target_paths)
        batch_date_tag = datetime.now().strftime("%d%m%Y-%H%M%S")
        detail = {
            "best": "Best adaptive 512–1024px with face restoration",
            "careful": "Careful adaptive 512–1024px",
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
            previous_output = history.completed_output(key) if skip_completed else None
            if previous_output is not None:
                skipped += 1
                on_log(f"↷ {target_path.name}: unchanged; already saved as {previous_output.name}")
                on_progress(
                    index,
                    total,
                    {"completed": completed, "skipped": skipped, "failed": len(errors)},
                )
                continue
            try:
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
                detail = f"{swapped} of {detected} face(s)" if detected > 1 else "1 face"
                on_log(f"✓ {target_path.name} → {output_path.name} ({detail})")
            except Exception as exc:  # continue a batch when one photo is unsuitable
                message = f"{type(exc).__name__}: {exc}"
                errors.append({"path": str(target_path), "error": message})
                on_log(f"! {target_path.name}: {message}")
            on_progress(
                index,
                total,
                {"completed": completed, "skipped": skipped, "failed": len(errors)},
            )

        stopped = should_stop() and completed + skipped + len(errors) < total
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
