from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import appearance


class FakeSession:
    def __init__(self):
        self.calls = 0
        self.last_tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _outputs, inputs):
        self.calls += 1
        self.last_tensor = inputs["input"]
        logits = np.zeros((1, 19, 32, 32), dtype=np.float32)
        logits[:, 0] = 1
        logits[:, 17, 2:14, 5:27] = 4
        logits[:, 1, 14:27, 8:24] = 4
        return [logits]


class AppearanceTests(unittest.TestCase):
    def test_parser_normalizes_input_and_returns_full_image_masks(self):
        session = FakeSession()
        processor = appearance.AppearanceProcessor(session)
        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        face = SimpleNamespace(bbox=np.array([25, 25, 55, 55], dtype=np.float32))

        hair, skin = processor.region_masks(
            image, face, need_hair=True, need_skin=True
        )

        self.assertEqual(session.calls, 1)
        self.assertEqual(session.last_tensor.shape, (1, 3, 512, 512))
        self.assertEqual(hair.shape, image.shape[:2])
        self.assertEqual(skin.shape, image.shape[:2])
        self.assertGreater(float(hair.max()), 0.9)
        self.assertGreater(float(skin.max()), 0.9)

    def test_hair_selection_rejects_a_nearby_disconnected_person(self):
        binary = np.zeros((80, 120), dtype=np.uint8)
        binary[8:45, 12:50] = 1
        binary[5:60, 82:116] = 1
        face = SimpleNamespace(bbox=np.array([20, 25, 45, 55], dtype=np.float32))

        selected = appearance.AppearanceProcessor._selected_hair_component(
            binary, face, (0, 0, 120, 80)
        )

        self.assertGreater(int(selected[20, 20]), 0)
        self.assertEqual(int(selected[20, 100]), 0)

    def test_recolor_keeps_unmasked_pixels_exact_and_preserves_texture(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(30, dtype=np.uint8)[:, None] * 3 + 30
        image[:, :, 1] = 45
        image[:, :, 2] = 65
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[5:25, 5:25] = 1
        desired = appearance._bgr_to_lab(np.array([45, 95, 185], dtype=np.uint8))

        result = appearance.AppearanceProcessor.recolor(
            image, mask, desired, 0.8, luminance_factor=0.8
        )

        np.testing.assert_array_equal(result[:4], image[:4])
        self.assertFalse(np.array_equal(result[10, 10], image[10, 10]))
        self.assertGreater(float(result[5:25, 5:25, 0].std()), 1.0)

    def test_hair_and_skin_share_one_parser_pass_per_face(self):
        session = FakeSession()
        processor = appearance.AppearanceProcessor(session)
        image = np.full((80, 80, 3), 100, dtype=np.uint8)
        face = SimpleNamespace(bbox=np.array([25, 25, 55, 55], dtype=np.float32))
        references = appearance.AppearanceReferences(
            hair_lab=np.array([100, 150, 160], dtype=np.float32),
            skin_lab=np.array([150, 135, 145], dtype=np.float32),
        )

        processor.apply(
            image,
            image,
            [face],
            hair_mode="source",
            custom_hair_color="#b85c32",
            hair_strength=0.65,
            skin_match=True,
            references=references,
        )

        self.assertEqual(session.calls, 1)


if __name__ == "__main__":
    unittest.main()
