# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

project_root = Path(SPECPATH).parent
datas = [(str(project_root / "assets"), "assets")]
if os.environ.get("SWAPIO_BUNDLE_MODELS") == "1":
    required = [
        project_root / "models/buffalo_l/det_10g.onnx",
        project_root / "models/buffalo_l/w600k_r50.onnx",
        project_root / "models/inswapper_128.onnx",
        project_root / "models/hyperswap_1a_256.onnx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Cannot bundle models; missing: " + ", ".join(missing))
    datas.append((str(project_root / "models"), "models"))
binaries = []
hiddenimports = [
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
    excludes=[],
    noarchive=False,
    optimize=0,
)
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
