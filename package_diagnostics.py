"""Explicit installed-package checks, without GUI prompts or saved preferences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import core
from version import VERSION


def self_test_swap(arguments: list[str], worker_class) -> int:
    from PySide6.QtCore import QEventLoop, QTimer

    parser = argparse.ArgumentParser(description="Run a real offline packaged face swap.")
    parser.add_argument("--self-test-swap", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Diagnostic output folder")
    parser.add_argument("--face-index", type=int, default=0, help="Zero-based target face index")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Check unloading and reloading across jobs")
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
        if args.face_index < 0 or args.repeat < 1:
            raise ValueError("Face index must be nonnegative and repeat must be positive")
        choices = []
        jobs = []
        release = engine.release_models

        def capture_release():
            report["providers"] = {
                "identity": engine.swapper.session.get_providers() if engine.swapper else [],
                "detail": engine.hyperswap.get_providers() if engine.hyperswap else [],
                "restoration": engine.enhancer.get_providers() if engine.enhancer else [],
            }
            return release()

        engine.release_models = capture_release

        def choose(request):
            choices.append({"detected": len(request["boxes"]), "selected": args.face_index})
            request["worker"].submit_face_choice(args.face_index)

        for _ in range(args.repeat):
            report["phase"] = "processing"
            (output / "swapio-self-test.json").write_text(json.dumps(report, indent=2) + "\n")
            results, errors = [], []
            worker = worker_class(
                engine, "batch", ask_face=True, source_path=args.source, targets=[args.target],
                output_dir=output, quality="careful", output_format="png", skip_completed=False,
            )
            worker.face_choice_requested.connect(choose)
            worker.log.connect(lambda message: print(message, flush=True))
            worker.batch_ready.connect(results.append)
            worker.failed.connect(errors.append)
            worker.cancelled.connect(lambda: errors.append("Diagnostic was cancelled"))
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            worker.start()
            loop.exec()
            worker.wait()
            if errors or not results:
                raise RuntimeError("; ".join(errors) or "Worker returned no result")
            summary = results[0]
            report.update(summary=summary, choices=choices)
            if summary["completed"] != 1 or summary["failed"]:
                raise RuntimeError("Diagnostic swap failed; see summary.errors")
            original = core.load_image(args.target)
            saved = core.load_image(summary["outputs"][0])
            if saved is None or original is None or saved.shape != original.shape:
                raise RuntimeError("Export did not decode with the original image dimensions")
            if np.array_equal(saved, original):
                raise RuntimeError("Export is identical to the target; no changed pixels")
            unloaded = all(getattr(engine, name) is None for name in (
                "analyser", "swapper", "hyperswap", "enhancer", "occluder", "face_parser"))
            if not unloaded:
                raise RuntimeError("Worker retained processing models after completion")
            jobs.append({"summary": summary, "providers": report["providers"], "models_unloaded": unloaded})
            report.update(jobs=jobs, phase="idle")
            (output / "swapio-self-test.json").write_text(json.dumps(report, indent=2) + "\n")
            # Leave an observable idle interval for external device measurements.
            idle = QEventLoop()
            QTimer.singleShot(500, idle.quit)
            idle.exec()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (output / "swapio-self-test.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1
