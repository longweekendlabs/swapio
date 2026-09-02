from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from model_setup_dialog import ModelSetupDialog
import setup_models
from version import VERSION


QT_APP = QApplication.instance() or QApplication([])


class ModelSetupTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(VERSION, "0.4.3")

    def test_download_requires_license_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"SWAPIO_MODEL_DIR": temp}
        ):
            dialog = ModelSetupDialog()
            self.assertFalse(dialog.download_button.isEnabled())
            dialog.acknowledge.setChecked(True)
            self.assertTrue(dialog.download_button.isEnabled())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
