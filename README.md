# Swapio

Swap one source face into a batch of destination photos without sending anything
to the cloud. Swapio is a focused Linux desktop app from Long Weekend Labs.

## What it does

- One source portrait per run
- Individual destination photos or a whole folder tree
- Preview before batch processing
- Largest-face or all-faces mode
- Careful 512px, balanced 256px, and fast 128px processing modes
- HyperSwap high-detail engine with InSwapper available for fast drafts
- Lossless PNG output by default, with optional JPEG 98 output
- CUDA-accelerated detection and fast drafts; stable CPU execution for HyperSwap quality modes
- Original files are never modified
- Failed photos are skipped and reported instead of aborting the batch
- Oriented pixels and safe EXIF/ICC metadata are retained where possible

Swapio changes the facial identity while retaining the destination pose,
expression, hair, lighting, body, and image dimensions. Lossless PNG keeps every
pixel outside the composited face exactly as decoded from the destination.

## Run from source

Python 3.11 is recommended. The default install runs on CPU and works across
Linux systems. An optional CUDA 12/cuDNN 9 profile accelerates detection and
fast draft swaps on compatible NVIDIA systems.

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python setup_models.py --acknowledge-noncommercial
./run.sh
```

For NVIDIA CUDA acceleration, install `requirements-gpu.txt` instead of
`requirements.txt`.

On first launch, the app explains the model licenses and offers to download the
missing files with live progress and checksum verification. The setup reuses
Castivo's compatible local detector/encoder files when available. The command-line
setup shown above remains available, and the application does not use the network
after setup.

## Build a local Linux bundle or RPM

```bash
./packaging/build_linux.sh
./dist/swapio/swapio
./packaging/build_rpm.sh
```

Public bundles and RPMs stay small by downloading verified models on first run.
For a private local test RPM containing models already installed on your machine,
run `./packaging/build_rpm.sh --bundle-models`. Do not redistribute that package
without appropriate permission for the pretrained model files.

## Model licensing

The application code and pretrained models have different licenses. InsightFace
states that its downloaded pretrained models are for non-commercial research
use, while HyperSwap is published under ResearchRAIL. Review the model terms
before use; commercial use may require separate permission. The setup script
keeps models out of Git and requires an explicit acknowledgement before
installation.

Only alter photos you own or have permission to modify.

## Current scope

Version 0.2.0 processes still images (`jpg`, `jpeg`, `png`, `webp`, `bmp`, `tif`,
and `tiff`). Video, character libraries, and face enhancement are intentionally
outside the first focused release.

Made with ♥ by **[Long Weekend Labs](https://github.com/longweekendlabs)**.

© 2026 Long Weekend Labs. All rights reserved.
