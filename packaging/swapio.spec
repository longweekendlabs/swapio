# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

project_root = Path(SPECPATH).parent
datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
]
if os.environ.get("SWAPIO_BUNDLE_MODELS") == "1":
    required = [
        project_root / "models/buffalo_l/2d106det.onnx",
        project_root / "models/buffalo_l/det_10g.onnx",
        project_root / "models/buffalo_l/w600k_r50.onnx",
        project_root / "models/inswapper_128.onnx",
        project_root / "models/hyperswap_1a_256.onnx",
        project_root / "models/gpen_bfr_1024.onnx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Cannot bundle models; missing: " + ", ".join(missing))
    datas.append((str(project_root / "models"), "models"))
binaries = []
hiddenimports = [
    # Pulls in certifi's hook so cacert.pem is collected; setup_models verifies
    # downloads against it rather than the build machine's OpenSSL paths.
    "certifi",
    "insightface.model_zoo.arcface_onnx",
    "insightface.model_zoo.attribute",
    "insightface.model_zoo.inswapper",
    "insightface.model_zoo.landmark",
    "insightface.model_zoo.retinaface",
    "insightface.model_zoo.scrfd",
]

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # These packages belonged to the retired appearance experiments.  Some
    # optional import paths in third-party libraries can make PyInstaller find
    # them when they happen to be installed in a reused build environment.
    # Swapio's face pipeline is ONNX-only and must not ship that stale runtime.
    excludes=[
        "torch",
        "transformers",
        "safetensors",
        "tokenizers",
        "huggingface_hub",
        "hf_xet",
    ],
    noarchive=False,
    optimize=0,
)

# The CUDA build must retain ONNX Runtime's CUDA provider, but bundling the
# machine's entire CUDA/cuDNN installation adds roughly 1.4 GB and is brittle
# across drivers. These ABI-versioned libraries are normal system prerequisites
# for the GPU profile; ONNX Runtime falls back to CPU when they are unavailable.
system_cuda_libraries = {
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudart.so.12",
    "libcudnn.so.9",
    "libcufft.so.11",
    "libcurand.so.10",
    "libnvrtc.so.12",
}
a.binaries = [
    entry for entry in a.binaries if Path(entry[0]).name not in system_cuda_libraries
]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="swapio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="swapio",
)
