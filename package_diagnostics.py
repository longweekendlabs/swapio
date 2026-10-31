"""Explicit installed-package checks, without GUI prompts or saved preferences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import core
from version import VERSION


def self_test_swap(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a real offline packaged face swap.")
    parser.add_argument("--self-test-swap", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Diagnostic output folder")
    parser.add_argument("--face-index", type=int, default=0, help="Zero-based target face index")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(arguments)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine = core.SwapEngine(use_gpu=not args.cpu)
    report = {
        "version": VERSION,
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "models": str(engine.models.resolve()),
        "face_index": args.face_index,
        "passed": False,
    }
    try:
        if args.face_index < 0:
            raise ValueError("Face index must be nonnegative")
        choices = []

        def choose(_path, _image, faces):
            choices.append({"detected": len(faces), "selected": args.face_index})
            return args.face_index

        summary = engine.batch(
            args.source, [args.target], output, quality="careful", output_format="png",
            face_selector=choose, skip_completed=False,
            on_log=lambda message: print(message, flush=True),
        )
        report.update(summary=summary, choices=choices)
        if summary["completed"] != 1 or summary["failed"]:
            raise RuntimeError("Diagnostic swap failed; see summary.errors")
        original = core.load_image(args.target)
        saved = core.load_image(summary["outputs"][0])
        if saved is None or original is None or saved.shape != original.shape:
            raise RuntimeError("Export did not decode with the original image dimensions")
        if np.array_equal(saved, original):
            raise RuntimeError("Export is identical to the target; no changed pixels")
        report["providers"] = {
            "identity": engine.swapper.session.get_providers(),
            "detail": engine.hyperswap.get_providers(),
            "restoration": engine.enhancer.get_providers() if engine.enhancer else [],
        }
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (output / "swapio-self-test.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1
