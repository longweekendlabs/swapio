from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class NativeDialogTests(unittest.TestCase):
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

    @patch("app.subprocess.run")
    @patch("app.shutil.which")
    def test_kdialog_directory_returns_native_selection(self, which, run):
        which.return_value = "/usr/bin/kdialog"
        run.return_value = SimpleNamespace(returncode=0, stdout="/output/folder\n")

        path = app.native_directory(None, "Output", "/last/output")

        self.assertEqual(path, "/output/folder")
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["--getexistingdirectory", "/last/output"])


if __name__ == "__main__":
    unittest.main()
