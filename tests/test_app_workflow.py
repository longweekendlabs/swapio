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
