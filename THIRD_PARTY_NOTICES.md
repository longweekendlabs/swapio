# Third-Party Notices

Swapio's own source code is MIT licensed. See LICENSE. Two categories of
third-party material are not covered by it.

## Pretrained face models

Swapio downloads these on first run from their original publishers. They are not
in this repository and not inside the released packages, and each remains under
its owner's terms:

- **InsightFace buffalo_l** (detection, landmarks, identity) and **InSwapper**:
  the published pretrained models are stated to be for non-commercial research use.
- **HyperSwap 1A 256**: published under ResearchRAIL.
- **GPEN-BFR 1024** (face restoration): research and non-commercial terms.

Review those terms before use. Commercial use may require separate permission
from the respective model owner.

## Libraries redistributed inside the packages

The RPM, DEB and AppImage bundle a Python runtime and its dependencies:

- **Qt via PySide6**, under the **LGPL-3.0**. Qt is bundled as separate shared
  libraries rather than statically linked, so they can be replaced with
  compatible builds. Qt's source and licence are available from
  https://www.qt.io/licensing and https://download.qt.io.
- **ONNX Runtime**, MIT.
- **OpenCV** (opencv-python-headless), Apache-2.0.
- **NumPy**, BSD-3-Clause. **Pillow**, MIT-CMU. **InsightFace** (library code), MIT.

Running Swapio from source pulls these from PyPI instead, under the same terms.
