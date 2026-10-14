from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from model_setup_dialog import ModelSetupDialog
import core
import setup_models
from version import VERSION


QT_APP = QApplication.instance() or QApplication([])


class ModelSetupTests(unittest.TestCase):
    def test_release_version(self):
        self.assertRegex(VERSION, r"^\d+\.\d+\.\d+$")

    def test_downloads_verify_against_the_bundled_certificates(self):
        """A frozen build cannot rely on the build machine's OpenSSL paths."""
        import certifi

        context = setup_models.certificate_context()
        self.assertTrue(context.get_ca_certs(), "no certificates loaded")
        with patch.object(setup_models.urllib.request, "urlopen") as urlopen:
            urlopen.side_effect = RuntimeError("stop before any network use")
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(RuntimeError):
                    setup_models.download("https://example.invalid/m.onnx", Path(temp) / "m")
        self.assertIn("context", urlopen.call_args.kwargs)
        self.assertIsInstance(urlopen.call_args.kwargs["context"], setup_models.ssl.SSLContext)
        self.assertEqual(
            setup_models.certificate_context().get_ca_certs(),
            setup_models.ssl.create_default_context(cafile=certifi.where()).get_ca_certs(),
        )

    def test_download_requires_license_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"SWAPIO_MODEL_DIR": temp}
        ):
            dialog = ModelSetupDialog()
            self.assertFalse(dialog.download_button.isEnabled())
            dialog.acknowledge.setChecked(True)
            self.assertTrue(dialog.download_button.isEnabled())
            dialog.close()

    def test_previous_model_collection_requests_the_new_mouth_parser(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"SWAPIO_MODEL_DIR": temp}
        ):
            root = Path(temp)
            for name in core.missing_models(root):
                if name != core.FACE_PARSER_MODEL:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            dialog = ModelSetupDialog()
            self.assertIn("Not installed", dialog.labels["Mouth expression protection"].text())
            self.assertFalse(dialog.download_button.isHidden())
            dialog.close()
            (root / core.FACE_PARSER_MODEL).touch()
            complete = ModelSetupDialog()
            self.assertIn("Installed", complete.labels["Mouth expression protection"].text())
            self.assertTrue(complete.download_button.isHidden())
            complete.close()


if __name__ == "__main__":
    unittest.main()
