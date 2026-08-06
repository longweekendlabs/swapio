from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
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

        source_bottom = window.source_view.mapTo(
            window, QPoint(0, window.source_view.height() - 1)
        ).y()
        source_name_top = window.source_name.mapTo(window, QPoint(0, 0)).y()
        preview_bottom = window.preview_view.mapTo(
            window, QPoint(0, window.preview_view.height() - 1)
        ).y()
        preview_caption_top = window.preview_caption.mapTo(window, QPoint(0, 0)).y()
        source_name_bottom = window.source_name.mapTo(
            window, QPoint(0, window.source_name.height() - 1)
        ).y()
        character_top = window.character_name.mapTo(window, QPoint(0, 0)).y()

        self.assertLess(source_bottom, source_name_top)
        self.assertLess(
            preview_bottom,
            preview_caption_top,
        )
        self.assertLess(
            source_name_bottom,
            character_top,
        )
        self.assertLessEqual(
            max(
                window.source_view.height(),
                window.target_list.height(),
                window.preview_view.height(),
            )
            - min(
                window.source_view.height(),
                window.target_list.height(),
                window.preview_view.height(),
            ),
            1,
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
