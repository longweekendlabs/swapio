from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class NativeDialogTests(unittest.TestCase):
    @patch.object(sys, "frozen", True, create=True)
    @patch.dict(
        os.environ,
        {
            "LD_LIBRARY_PATH": "/opt/swapio/_internal",
            "LD_LIBRARY_PATH_ORIG": "/usr/lib64",
            "QT_PLUGIN_PATH": "/opt/swapio/_internal/PySide6/Qt/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/opt/swapio/_internal/cv2/qt/plugins",
            "QT_QPA_FONTDIR": "/opt/swapio/_internal/cv2/qt/fonts",
        },
    )
    def test_packaged_dialog_restores_system_library_environment(self):
        environment = app.native_process_environment()

        self.assertEqual(environment["LD_LIBRARY_PATH"], "/usr/lib64")
        self.assertNotIn("QT_PLUGIN_PATH", environment)
        self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", environment)
        self.assertNotIn("QT_QPA_FONTDIR", environment)

    @patch.object(sys, "frozen", True, create=True)
    @patch.dict(
        os.environ,
        {"LD_LIBRARY_PATH": "/opt/swapio/_internal", "LD_LIBRARY_PATH_ORIG": ""},
    )
    def test_packaged_dialog_removes_an_originally_absent_library_path(self):
        environment = app.native_process_environment()

        self.assertNotIn("LD_LIBRARY_PATH", environment)

    @patch("app.subprocess.run")
    @patch("app.shutil.which")
    def test_kdialog_multiple_images_uses_requested_start_folder(self, which, run):
        which.return_value = "/usr/bin/kdialog"
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="/photos/one.jpg\n/photos/two.png\n",
        )

        paths = app.native_image_files(None, "Add photos", "/photos", multiple=True)

        self.assertEqual(paths, ["/photos/one.jpg", "/photos/two.png"])
        command = run.call_args.args[0]
        self.assertEqual(command[0], "kdialog")
        self.assertIn("--multiple", command)
        self.assertEqual(command[command.index("--getopenfilename") + 1], "/photos")
        self.assertIn("env", run.call_args.kwargs)

    @patch("app.subprocess.run")
    @patch("app.shutil.which")
    def test_kdialog_directory_returns_native_selection(self, which, run):
        which.return_value = "/usr/bin/kdialog"
        run.return_value = SimpleNamespace(returncode=0, stdout="/output/folder\n")

        path = app.native_directory(None, "Output", "/last/output")

        self.assertEqual(path, "/output/folder")
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["--getexistingdirectory", "/last/output"])

    @patch("app.QFileDialog.getOpenFileName")
    @patch("app.subprocess.run")
    @patch("app.shutil.which")
    def test_crashed_native_picker_falls_back_to_qt(self, which, run, qt_picker):
        which.side_effect = lambda name: "/usr/bin/kdialog" if name == "kdialog" else None
        run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="Qt mismatch")
        qt_picker.return_value = ("/photos/fallback.jpg", "Images")

        paths = app.native_image_files(None, "Choose", "/photos")

        self.assertEqual(paths, ["/photos/fallback.jpg"])


if __name__ == "__main__":
    unittest.main()
