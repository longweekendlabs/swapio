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
        self.assertEqual(VERSION, "0.4.4")

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


if __name__ == "__main__":
    unittest.main()
