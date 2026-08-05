from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import core


class FakeAnalyser:
    def __init__(self, faces):
        self.faces = faces

    def get(self, _image):
        return self.faces


class FakeSwapper:
    def __init__(self):
        self.calls = []

    def get(self, image, face, _source_face, paste_back=True):
        self.calls.append((face, paste_back))
        return image + 1


class CoreTests(unittest.TestCase):
    def test_image_files_filters_and_sorts_recursively(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 8)).save(root / "b.png")
            Image.new("RGB", (8, 8)).save(root / "nested" / "a.jpg")
            (root / "ignore.txt").write_text("not an image")
            self.assertEqual(
                [path.name for path in core.image_files(root)],
                ["b.png", "a.jpg"],
            )

    def test_unique_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "photo.jpg"
            first = core.unique_output_path(root, source, date_tag="05082026-143012")
            self.assertEqual(first.name, "photo_swapped_05082026-143012.jpg")
            first.touch()
            self.assertEqual(
                core.unique_output_path(root, source, date_tag="05082026-143012").name,
                "photo_swapped_05082026-143012_2.jpg",
            )

    def test_character_name_replaces_original_name_and_is_path_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            output = core.unique_output_path(
                Path(temp),
                Path("opaque-camera-name.jpg"),
                suffix=".png",
                date_tag="05082026-143012",
                name_prefix="Ada/Lovelace",
            )

            self.assertEqual(output.name, "Ada_Lovelace_swapped_05082026-143012.png")

    def test_swap_largest_face_only(self):
        small = SimpleNamespace(bbox=np.array([0, 0, 10, 10]))
        large = SimpleNamespace(bbox=np.array([0, 0, 20, 20]))
        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser([small, large])
        engine.swapper = FakeSwapper()
        result, swapped, detected = engine.swap_array(
            np.zeros((32, 32, 3), dtype=np.uint8),
            source_face=small,
            all_faces=False,
            quality="fast",
        )
        self.assertEqual((swapped, detected), (1, 2))
        self.assertIs(engine.swapper.calls[0][0], large)
        self.assertTrue(np.all(result == 1))

    def test_swap_all_faces(self):
        faces = [
            SimpleNamespace(bbox=np.array([0, 0, 10, 10])),
            SimpleNamespace(bbox=np.array([0, 0, 20, 20])),
        ]
        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser(faces)
        engine.swapper = FakeSwapper()
        result, swapped, detected = engine.swap_array(
            np.zeros((32, 32, 3), dtype=np.uint8),
            source_face=faces[0],
            all_faces=True,
            quality="fast",
        )
        self.assertEqual((swapped, detected, len(engine.swapper.calls)), (2, 2, 2))
        self.assertTrue(np.all(result == 2))

    def test_pixel_boost_round_trip(self):
        image = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
        tiles = core.SwapEngine._implode_pixels(image, 256)
        restored = core.SwapEngine._explode_pixels(list(tiles), 256, 512)
        np.testing.assert_array_equal(restored, image)


if __name__ == "__main__":
    unittest.main()
