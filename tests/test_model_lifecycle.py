from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

import app
import core


QT_APP = QApplication.instance() or QApplication([])


class ModelLifecycleTests(unittest.TestCase):
    def test_unloading_releases_all_wrappers_including_cycles_without_resetting_preferences(self):
        class Wrapper:
            def __init__(self):
                self.cycle = self

        engine = core.SwapEngine(use_gpu=True, det_size=320)
        names = ("analyser", "swapper", "hyperswap", "enhancer", "occluder", "face_parser")
        for name in names:
            setattr(engine, name, Wrapper())
        refs = [weakref.ref(getattr(engine, name)) for name in names]
        engine.enhancer_cpu_only = True
        engine.hyperswap_cpu_only = True
        self.assertTrue(engine.release_models())
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertTrue(all(getattr(engine, name) is None for name in names))
        self.assertEqual(engine.provider, "Not loaded")
        self.assertTrue(engine.models_released)
        self.assertTrue(engine.use_gpu)
        self.assertEqual(engine.det_size, 320)
        self.assertTrue(engine.enhancer_cpu_only)
        self.assertTrue(engine.hyperswap_cpu_only)
        self.assertFalse(engine.release_models())

    def test_every_worker_outcome_unloads_on_worker_thread_before_notifying_gui(self):
        for mode, failure, stopped in [
            ("preview", None, False), ("batch", None, False), ("batch", None, True),
            ("preview", RuntimeError("inference failed"), False),
            ("batch", RuntimeError("write failed"), False),
            ("preview", core.ProcessingCancelled(), False),
            ("preview", core.FaceSelectionSkipped(), False),
            ("preview", None, True),
        ]:
            with self.subTest(mode=mode, failure=failure, stopped=stopped):
                state = {"loaded": False, "released": 0}
                image = np.full((3, 3, 3), 42, np.uint8)
                def process(**_kwargs):
                    state["loaded"] = True
                    if failure is not None:
                        raise failure
                    if stopped and mode == "preview":
                        worker.stop()
                    return {"image": image, "provider": "CUDA", "stopped": stopped}
                def release():
                    self.assertIsNot(QThread.currentThread(), QT_APP.thread())
                    state["loaded"] = False
                    state["released"] += 1
                    return True
                engine = SimpleNamespace(preview=process, batch=process, release_models=release)
                worker = app.TaskWorker(engine, mode)
                events = []
                for name in ("preview_ready", "batch_ready", "cancelled", "failed"):
                    getattr(worker, name).connect(
                        lambda *payload, name=name: events.append((name, payload, state["loaded"]))
                    )
                worker.start()
                self.assertTrue(worker.wait(3000))
                QT_APP.processEvents()
                self.assertEqual(state["released"], 1)
                self.assertEqual(len(events), 1)
                expected = ("cancelled" if isinstance(failure, (core.ProcessingCancelled, core.FaceSelectionSkipped))
                            or (mode == "preview" and stopped)
                            else "failed" if failure else f"{mode}_ready")
                self.assertEqual(events[0][0], expected)
                self.assertFalse(events[0][2], "GUI saw completion before cleanup")
                if expected == "preview_ready":
                    self.assertIs(events[0][1][0]["image"], image)
                    self.assertEqual(events[0][1][0]["provider"], "CUDA")

    @patch("app.core.missing_models", return_value=[])
    @patch("app.read_state", return_value={})
    @patch("app.write_state")
    def test_idle_status_explains_automatic_model_reload(self, *_mocks):
        window = app.MainWindow()
        window.engine.models_released = True
        window._refresh_model_status()
        self.assertIn("Idle", window.model_status.text())
        self.assertIn("reload automatically", window.model_status.toolTip())
        window._set_running(True)
        self.assertIn("Processing", window.model_status.text())
        window._preview_cancelled()
        self.assertIn("Idle", window.model_status.text())
        window.close()

