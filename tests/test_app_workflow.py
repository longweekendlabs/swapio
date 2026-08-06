from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import app


QT_APP = QApplication.instance() or QApplication([])


class AppWorkflowTests(unittest.TestCase):
    @patch("app.read_state", return_value={})
    @patch("app.core.missing_models", return_value=[])
    def test_workflow_canvases_do_not_overlap_card_controls(self, _missing, _state):
        window = app.MainWindow()
        window.resize(1180, 820)
        window.show()
        QT_APP.processEvents()

        self.assertLess(window.source_view.geometry().bottom(), window.source_name.geometry().top())
        self.assertLess(
            window.preview_view.geometry().bottom(),
            window.preview_caption.geometry().top(),
        )
        self.assertLess(
            window.source_name.geometry().bottom(),
            window.character_name.geometry().top(),
        )
        window.close()

    @patch("app.read_state", return_value={})
    @patch("app.core.missing_models", return_value=[])
    def test_adding_folder_replaces_stale_output_suggestion(self, _missing, _state):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "New destination"
            destination.mkdir()
            photo = destination / "photo.jpg"
            photo.touch()
            window = app.MainWindow()
            window.output_edit.setText("/old/destination-output")

            with patch("app.native_directory", return_value=str(destination)), patch(
                "app.core.image_files", return_value=[photo]
            ):
                window._add_folder()

            self.assertEqual(
                window.output_edit.text(),
                str(Path(temp) / "New destination - Swapio Output"),
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
