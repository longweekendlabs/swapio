import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import time
import unittest
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
import app
import core

QT_APP = QApplication.instance() or QApplication([])


class FaceSelectionTests(unittest.TestCase):
    def make_engine(self):
        faces = [SimpleNamespace(bbox=np.array([5, 5, 45, 45])),
                 SimpleNamespace(bbox=np.array([65, 10, 90, 35]))]
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(get=Mock(return_value=faces), models={})
        engine.source_face = Mock(return_value=object())
        engine.swapper = SimpleNamespace(get=Mock(side_effect=lambda image, *_args, **_kw: image + 50))
        def visibility(image, face):
            mask = np.zeros(image.shape[:2], np.float32)
            l, t, r, b = face.bbox
            mask[t:b, l:r] = 1
            return mask
        engine._visible_face_mask = visibility
        return engine, faces

    def inputs(self, root):
        source, target = Path(root) / 'source.png', Path(root) / 'target.png'
        core.write_image(source, np.zeros((60, 100, 3), np.uint8))
        core.write_image(target, np.zeros((60, 100, 3), np.uint8))
        return source, target

    def test_smaller_selected_face_swaps_and_detection_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = self.inputs(root)
            engine, faces = self.make_engine()
            choice = Mock(return_value=1)
            result = engine.preview(source, target, quality='fast', all_faces=True, face_selector=choice)
            self.assertEqual((result['swapped'], result['detected']), (1, 2))
            self.assertIs(engine.swapper.get.call_args.args[1], faces[1])
            self.assertEqual(int(result['image'][20, 75, 0]), 50)
            self.assertEqual(int(result['image'][20, 25, 0]), 0)
            engine.analyser.get.assert_called_once()
            choice.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, 'Selected face'):
                engine.swap_array(np.zeros((60, 100, 3), np.uint8), object(),
                                  target_face_index=2, quality='fast')

    def test_single_face_does_not_prompt(self):
        engine, faces = self.make_engine()
        engine.analyser.get.return_value = faces[:1]
        callback = Mock(side_effect=AssertionError('single face must not prompt'))
        choice, measured = engine._target_face_choice(np.zeros((60, 100, 3), np.uint8), Path('p'), callback)
        self.assertEqual(choice, 0)
        self.assertEqual(len(measured), 1)
        callback.assert_not_called()

    def test_batch_history_distinguishes_face_and_changed_detection(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = self.inputs(root)
            engine, faces = self.make_engine()
            output = Path(root) / 'out'
            def batch(choice):
                return engine.batch(source, [target], output, quality='fast',
                                    face_selector=lambda *_: choice)
            first, repeat, different = batch(1), batch(1), batch(0)
            self.assertEqual((first['completed'], repeat['skipped'], different['completed']), (1, 1, 1))
            faces[1].bbox = np.array([64, 10, 90, 35])
            changed = batch(1)
            self.assertEqual(changed['completed'], 1)

    def test_skip_and_stop_do_not_swap_or_count_as_failures(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = self.inputs(root)
            engine, _ = self.make_engine()
            skipped = engine.batch(source, [target], Path(root) / 'out', face_selector=lambda *_: None)
            self.assertEqual((skipped['completed'], skipped['skipped'], skipped['failed']), (0, 1, 0))
            def stop(*_):
                raise core.ProcessingCancelled()
            stopped = engine.batch(source, [target], Path(root) / 'out', face_selector=stop)
            self.assertTrue(stopped['stopped'])
            self.assertEqual((stopped['completed'], stopped['failed']), (0, 0))
            engine.swapper.get.assert_not_called()

    def test_session_cache_expires_when_candidates_change(self):
        with tempfile.TemporaryDirectory() as root:
            _, target = self.inputs(root)
            _, faces = self.make_engine()
            key = core.face_selection_key(target, faces)
            worker = app.TaskWorker(None, 'preview', face_choices={key: 1})
            requested = Mock()
            worker.face_choice_requested.connect(requested)
            self.assertEqual(worker._select_face(target, np.zeros((60, 100, 3), np.uint8), faces), 1)
            requested.assert_not_called()
            faces[1].bbox += [1, 0, 1, 0]
            self.assertNotEqual(key, core.face_selection_key(target, faces))

    @patch('app.write_state')
    @patch('app.read_state', return_value={'ask_face': True, 'face_mode': 1})
    @patch('app.core.missing_models', return_value=[])
    def test_option_overrides_all_faces_and_is_persisted(self, _missing, _read, write):
        window = app.MainWindow()
        self.assertTrue(window.ask_face.isChecked())
        self.assertFalse(window._all_faces())
        self.assertFalse(window.face_mode.isEnabled())
        window.ask_face.setChecked(False)
        self.assertTrue(window._all_faces())
        self.assertTrue(window.face_mode.isEnabled())
        self.assertFalse(write.call_args.args[0]['ask_face'])
        window.close()

    def test_dialog_requires_explicit_selection_and_accepts_smaller_face(self):
        dialog = app.FaceSelectionDialog('portrait.png', np.zeros((60, 100, 3), np.uint8),
                                         [[5, 5, 45, 45], [65, 10, 90, 35]])
        dialog.show(); QT_APP.processEvents()
        self.assertFalse(dialog.swap_button.isEnabled())
        self.assertIsNone(dialog.selected_index)
        rect = dialog.faces.visualItemRect(dialog.faces.item(1))
        QTest.mouseClick(dialog.faces.viewport(), Qt.LeftButton, pos=rect.center())
        QTest.mouseClick(dialog.swap_button, Qt.LeftButton)
        self.assertEqual(dialog.selected_index, 1)
        self.assertEqual(dialog.result(), app.QDialog.Accepted)
        dialog.close()

    def test_stopping_worker_while_waiting_for_choice_unblocks(self):
        class Engine:
            def preview(self, target_path, face_selector, **_kw):
                face_selector(target_path, np.zeros((60, 100, 3), np.uint8),
                              [SimpleNamespace(bbox=np.array([5, 5, 45, 45])),
                               SimpleNamespace(bbox=np.array([65, 10, 90, 35]))])
                raise AssertionError('cancel must not fall back to largest face')
        with tempfile.TemporaryDirectory() as root:
            _, target = self.inputs(root)
            worker = app.TaskWorker(Engine(), 'preview', ask_face=True, target_path=target)
            cancelled, failed = [], []
            worker.face_choice_requested.connect(lambda _: worker.stop())
            worker.cancelled.connect(lambda: cancelled.append(True))
            worker.failed.connect(failed.append)
            worker.start()
            deadline = time.monotonic() + 3
            while worker.isRunning() and time.monotonic() < deadline:
                QT_APP.processEvents(); QTest.qWait(10)
            worker.stop(); worker.wait(1000); QT_APP.processEvents()
            self.assertFalse(worker.isRunning())
            self.assertEqual(cancelled, [True])
            self.assertEqual(failed, [])


if __name__ == '__main__':
    unittest.main()
