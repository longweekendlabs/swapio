#!/usr/bin/env python3
"""Install Swapio's local models after an explicit license acknowledgement."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from version import APP_NAME, VERSION

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
BUFFALO_FILES = ("det_10g.onnx", "w600k_r50.onnx")
BUFFALO_SHA256 = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
CASTIVO_MODELS = PROJECT_DIR.parent / "app-person-identifier" / "models" / "buffalo_l"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SWAPPER_URL = "https://huggingface.co/mikestealth/inswapper/resolve/main/inswapper_128.onnx"
SWAPPER_SHA256 = "0fa95f167682b4f61edf24f8d66c46b4ab130e8be058f00c8150e6d0170ca72f"
HYPERSWAP_URL = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/"
    "hyperswap_1a_256.onnx"
)
HYPERSWAP_SHA256 = "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"

ProgressFn = Callable[[str, str, int, int], None]
StopFn = Callable[[], bool]


class DownloadCancelled(RuntimeError):
    """Raised when the user cancels model setup."""


def _noop_progress(_component: str, _message: str, _done: int, _total: int) -> None:
    pass


def download(
    url: str,
    destination: Path,
    component: str = "download",
    on_progress: ProgressFn = _noop_progress,
    should_stop: StopFn = lambda: False,
) -> None:
    print(f"Downloading {destination.name}...")
    request = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{VERSION}"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        done = 0
        last_percent = -1
        while True:
            if should_stop():
                raise DownloadCancelled("Model download cancelled.")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            done += len(chunk)
            on_progress(component, "Downloading", done, total)
            if total:
                percent = done * 100 // total
                if percent == last_percent:
                    continue
                last_percent = percent
                sys.stdout.write(f"\r  {percent:3d}%  {done / 1024 / 1024:7.1f}/{total / 1024 / 1024:.1f} MB")
            else:
                sys.stdout.write(f"\r  {done / 1024 / 1024:7.1f} MB")
            sys.stdout.flush()
    print()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_buffalo(
    model_root: Path = MODEL_DIR,
    on_progress: ProgressFn = _noop_progress,
    should_stop: StopFn = lambda: False,
) -> None:
    component = "Face detector and identity encoder"
    destination = model_root / "buffalo_l"
    destination.mkdir(parents=True, exist_ok=True)
    missing = [
        name
        for name in BUFFALO_FILES
        if not (destination / name).is_file()
        or sha256(destination / name) != BUFFALO_SHA256[name]
    ]
    if not missing:
        print("Face detector and identity encoder are already installed.")
        on_progress(component, "Installed and verified", 1, 1)
        return
    for name in missing:
        (destination / name).unlink(missing_ok=True)
    if all(
        (CASTIVO_MODELS / name).is_file()
        and sha256(CASTIVO_MODELS / name) == BUFFALO_SHA256[name]
        for name in missing
    ):
        print("Reusing the compatible local Castivo face models.")
        for name in missing:
            shutil.copy2(CASTIVO_MODELS / name, destination / name)
        on_progress(component, "Reused compatible local models", 1, 1)
        return

    with tempfile.TemporaryDirectory(prefix="swapio-models-") as temp_dir:
        archive = Path(temp_dir) / "buffalo_l.zip"
        download(BUFFALO_URL, archive, component, on_progress, should_stop)
        if should_stop():
            raise DownloadCancelled("Model download cancelled.")
        on_progress(component, "Installing", 0, 0)
        with zipfile.ZipFile(archive) as bundle:
            entries = {Path(name).name: name for name in bundle.namelist()}
            for name in missing:
                if name not in entries:
                    raise RuntimeError(f"The model archive did not contain {name}")
                with bundle.open(entries[name]) as source, (destination / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
        invalid = [
            name
            for name in BUFFALO_FILES
            if sha256(destination / name) != BUFFALO_SHA256[name]
        ]
        if invalid:
            raise RuntimeError("Detector model checksum did not match: " + ", ".join(invalid))
        on_progress(component, "Installed and verified", 1, 1)


def install_swapper(
    model_root: Path = MODEL_DIR,
    on_progress: ProgressFn = _noop_progress,
    should_stop: StopFn = lambda: False,
) -> None:
    component = "Fast face swapper"
    destination = model_root / "inswapper_128.onnx"
    if destination.is_file() and sha256(destination) == SWAPPER_SHA256:
        print("Face swapper is already installed and verified.")
        on_progress(component, "Installed and verified", 1, 1)
        return
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(".onnx.part")
    partial.unlink(missing_ok=True)
    try:
        download(SWAPPER_URL, partial, component, on_progress, should_stop)
        on_progress(component, "Verifying checksum", 0, 0)
        actual = sha256(partial)
        if actual != SWAPPER_SHA256:
            raise RuntimeError(
                "Downloaded swapper checksum did not match. "
                f"Expected {SWAPPER_SHA256}, got {actual}."
            )
        partial.replace(destination)
        on_progress(component, "Installed and verified", 1, 1)
    finally:
        partial.unlink(missing_ok=True)


def install_hyperswap(
    model_root: Path = MODEL_DIR,
    on_progress: ProgressFn = _noop_progress,
    should_stop: StopFn = lambda: False,
) -> None:
    component = "High-quality face swapper"
    destination = model_root / "hyperswap_1a_256.onnx"
    if destination.is_file() and sha256(destination) == HYPERSWAP_SHA256:
        print("High-quality face swapper is already installed and verified.")
        on_progress(component, "Installed and verified", 1, 1)
        return
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(".onnx.part")
    partial.unlink(missing_ok=True)
    try:
        download(HYPERSWAP_URL, partial, component, on_progress, should_stop)
        on_progress(component, "Verifying checksum", 0, 0)
        actual = sha256(partial)
        if actual != HYPERSWAP_SHA256:
            raise RuntimeError(
                "Downloaded HyperSwap checksum did not match. "
                f"Expected {HYPERSWAP_SHA256}, got {actual}."
            )
        partial.replace(destination)
        on_progress(component, "Installed and verified", 1, 1)
    finally:
        partial.unlink(missing_ok=True)


def install_all(
    model_root: Path = MODEL_DIR,
    on_progress: ProgressFn = _noop_progress,
    should_stop: StopFn = lambda: False,
) -> None:
    """Install every required model, skipping files already verified."""
    model_root.mkdir(parents=True, exist_ok=True)
    install_buffalo(model_root, on_progress, should_stop)
    if should_stop():
        raise DownloadCancelled("Model download cancelled.")
    install_swapper(model_root, on_progress, should_stop)
    if should_stop():
        raise DownloadCancelled("Model download cancelled.")
    install_hyperswap(model_root, on_progress, should_stop)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Swapio's offline face models")
    parser.add_argument(
        "--acknowledge-noncommercial",
        action="store_true",
        help="confirm that the pretrained face models will only be used as licensed",
    )
    args = parser.parse_args()
    if not args.acknowledge_noncommercial:
        print(
            "Swapio's pretrained models have separate research/non-commercial terms.\n"
            "Commercial use may require separate permission from the model owners.\n\n"
            "If that is acceptable for your use, rerun with:\n"
            "  python setup_models.py --acknowledge-noncommercial",
            file=sys.stderr,
        )
        return 2
    install_all(MODEL_DIR)
    print(f"Swapio models are ready in {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
