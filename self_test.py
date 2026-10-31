"""Explicit, preference-isolated Preview/export check for a frozen executable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

import core
from version import VERSION


def run(module, application, arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(arguments)
    originals = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (args.source, args.target)}
    missing = core.missing_models()
    if missing:
        raise RuntimeError("Missing local models: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="swapio-check-", dir=args.output))
    previous_read, previous_write = module.read_state, module.write_state
    module.read_state = lambda: {"prefer_gpu": True, "quality": "best", "output_format": "png"}
    module.write_state = lambda _state: None
    window = None
    failures, previews, batches, logs = [], [], [], []
    ticker = QTimer()

    def dismiss():
        for widget in application.topLevelWidgets():
            if isinstance(widget, QMessageBox):
                if widget.windowTitle() != "Batch complete":
                    failures.append(widget.text())
                widget.accept()

    def click_and_wait(button):
        if not button.isEnabled():
            raise RuntimeError("Processing controls are not ready.")
        button.click()
        worker = window.worker
        worker.failed.connect(failures.append)
        worker.preview_ready.connect(previews.append)
        worker.batch_ready.connect(batches.append)
        worker.log.connect(logs.append)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        timeout = QTimer(); timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit); timeout.start(180000)
        loop.exec(); timeout.stop(); application.processEvents()
        if worker.isRunning():
            worker.stop()
            raise RuntimeError("Preview/export check timed out.")
        if failures:
            raise RuntimeError("; ".join(failures))

    try:
        window = module.MainWindow()
        window.resize(1400, 1050); window.show()
        window._set_source(args.source)
        window._add_targets([args.target])
        window.permission.setChecked(True)
        window.output_edit.setText(str(output / "exports"))
        ticker.timeout.connect(dismiss); ticker.start(100)
        click_and_wait(window.preview_button)
        preview = previews[-1]
        if window.engine.active_backend != "source":
            raise RuntimeError("Preview did not use the source-appearance renderer.")
        providers = {name: session.get_providers()[0]
                     for name, session in window.engine.source_renderer.sessions.items()}
        if args.require_cuda and (preview["provider"] != "CUDA" or any(
                providers.get(name) != "CUDAExecutionProvider"
                for name in ("appearance", "motion", "generator"))):
            raise RuntimeError(f"CUDA required; actual providers: {providers}")
        core.write_image(output / "preview.png", preview["image"])
        window.grab().save(str(output / "preview-window.png"))
        click_and_wait(window.run_button)
        summary = batches[-1]
        if summary["completed"] != 1 or summary["failed"]:
            raise RuntimeError(f"PNG export failed: {summary}")
        exported = core.load_image(summary["outputs"][0])
        if exported is None or exported.shape != preview["image"].shape:
            raise RuntimeError("PNG export dimensions do not match Preview.")
        delta = np.abs(exported.astype(np.int16) - preview["image"].astype(np.int16))
        if float(delta.mean()) > .05:
            raise RuntimeError("PNG export differs materially from Preview.")
        click_and_wait(window.run_button)
        if batches[-1]["skipped"] != 1:
            raise RuntimeError("Repeat batch did not skip its unchanged output.")
        for path, checksum in originals.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"Original file changed: {path}")
        report = {"version": VERSION, "provider": preview["provider"], "models": providers,
                  "target_eyes": window.destination_eyes.isChecked(),
                  "reference_detail": window.restoration_strength.value(),
                  "preview_png_max_delta": int(delta.max()), "mean_delta": float(delta.mean()),
                  "original_checksums": originals, "output": str(output),
                  "note": "Processing check; inspect the output images for visible quality."}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        (output / "processing.log").write_text("\n".join(logs))
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        ticker.stop()
        if window is not None:
            window.close()
        module.read_state, module.write_state = previous_read, previous_write
