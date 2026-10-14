from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import core
import setup_models


class FaceProtectionTests(unittest.TestCase):
    def test_complete_mouth_mask_removes_generated_corner_outside_original_lips(self):
        points = np.zeros((106, 2), np.float32)
        points[52:72] = np.tile([[40, 45], [80, 45], [80, 60], [40, 60]], (5, 1))
        moved = points.copy()
        moved[52:72] += [-8, 0]
        original = np.full((110, 130, 3), 100, np.uint8)
        rendered = original.copy()
        rendered[50, 32] = 10  # old generated corner, outside target lip mask
        mask = np.zeros(original.shape[:2], np.float32)
        mask[45:61, 40:81] = 1
        expanded = core.SwapEngine._complete_mouth_protection(
            mask, SimpleNamespace(landmark_2d_106=points),
            SimpleNamespace(landmark_2d_106=moved),
        )
        result = core.SwapEngine._apply_visibility(rendered, original, 1 - expanded)
        np.testing.assert_array_equal(result[50, 32], original[50, 32])
        self.assertEqual(float(expanded[5, 5]), 0)

    def test_eye_detail_is_local_and_rejects_moved_geometry(self):
        points = np.zeros((106, 2), np.float32)
        contour = np.array([[-30, 0], [-20, -8], [0, -10], [20, -8],
                            [30, 0], [20, 8], [0, 10], [-20, 8]], np.float32)
        points[core.LEFT_EYE_106_INDICES] = contour + [65, 80]
        points[core.RIGHT_EYE_106_INDICES] = contour + [185, 80]
        yy, xx = np.mgrid[:170, :260]
        pattern = ((xx + yy) % 2 * 60).astype(np.uint8)
        original = np.repeat((pattern + 80)[:, :, None], 3, axis=2)
        rendered = np.full_like(original, 100)
        restored = np.stack([pattern + 30, pattern + 60, pattern + 100], axis=2)
        result, count = core.SwapEngine._eye_detail_blend(
            original, rendered, restored, points, points, 0.5
        )
        self.assertEqual(count, 2)
        self.assertFalse(np.array_equal(result[80, 65], rendered[80, 65]))
        # The restored model's colour offset must not repaint the eye region.
        np.testing.assert_allclose(result[70:90, 55:75].mean(axis=(0, 1)), [100, 100, 100], atol=1)
        # Rendering only eyes must not repaint the mouth, nose, or background.
        np.testing.assert_array_equal(result[145:], rendered[145:])
        np.testing.assert_array_equal(result[:20], rendered[:20])
        moved = points.copy()
        moved[np.r_[core.LEFT_EYE_106_INDICES, core.RIGHT_EYE_106_INDICES]] += [12, 0]
        rejected, count = core.SwapEngine._eye_detail_blend(
            original, rendered, restored, points, moved, 0.5
        )
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(rejected, rendered)

    def test_upright_eye_landmarks_return_to_image_coordinates(self):
        engine = core.SwapEngine()
        original_points = np.zeros((106, 2), np.float32)
        face = SimpleNamespace(bbox=np.array([10, 10, 190, 190], np.float32),
                               landmark_2d_106=original_points.copy())
        matrix = np.array([[0, 1, 0], [-1, 0, 200]], np.float32)
        local_box = np.array([20, 40, 160, 170], np.float32)
        def measure(image, placed):
            self.assertEqual(image.shape, (200, 200, 3))
            np.testing.assert_array_equal(placed.bbox, local_box)
            placed.landmark_2d_106 = np.tile([40, 80], (106, 1)).astype(np.float32)
        engine.analyser = SimpleNamespace(models={'landmark_2d_106': SimpleNamespace(get=measure)})
        placed = engine._eye_landmarks(np.zeros((200, 200, 3), np.uint8), face,
                                       (matrix, 200, local_box))
        np.testing.assert_allclose(placed.landmark_2d_106, np.tile([120, 40], (106, 1)))
        np.testing.assert_array_equal(face.landmark_2d_106, original_points)
        np.testing.assert_array_equal(placed.bbox, face.bbox)

    def test_eye_detail_skips_small_eyes_and_zero_strength(self):
        points = self.eye_landmarks()
        image = np.full((100, 120, 3), 100, np.uint8)
        for limit in (0, 0.5):
            result, count = core.SwapEngine._eye_detail_blend(
                image, image, image, points, points, limit
            )
            self.assertEqual(count, 0)
            np.testing.assert_array_equal(result, image)

    def test_foreground_stays_fixed_after_restoration_and_mouth_processing(self):
        original = np.full((100, 100, 3), 20, np.uint8)
        original[55:90, 42:58] = (230, 40, 10)  # bottle crossing the mouth
        visibility = np.ones((100, 100), np.float32)
        visibility[55:90, 42:58] = 0
        face = SimpleNamespace(bbox=np.array([10, 10, 90, 90]))
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(get=lambda _: [face], models={})
        engine._visible_face_mask = lambda *_: visibility
        engine._hyperswap_face = lambda image, *_: np.full_like(image, 90)
        engine._restore_face = lambda image, *_: np.full_like(image, 110)
        engine._preserve_inner_mouth = lambda image, *_: np.full_like(image, 130)
        result, _, _ = engine.swap_array(original, face, quality='best')
        np.testing.assert_array_equal(result[visibility == 0], original[visibility == 0])
        self.assertTrue(np.all(result[visibility == 1] == 130))

    def test_protecting_second_face_keeps_first_swapped_face(self):
        faces = [SimpleNamespace(bbox=np.array([0, 0, 50, 50]), x=0),
                 SimpleNamespace(bbox=np.array([50, 0, 100, 50]), x=50)]
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(get=lambda _: faces, models={})
        def mask(image, face):
            result = np.zeros(image.shape[:2], np.float32)
            result[:, face.x:face.x+50] = 1
            return result
        engine._visible_face_mask = mask
        engine._hyperswap_face = lambda image, *_: image + 20
        result, count, _ = engine.swap_array(np.zeros((50, 100, 3), np.uint8),
                                             faces[0], all_faces=True)
        self.assertEqual(count, 2)
        self.assertTrue(np.all(result == 20))

    def test_invalid_model_mask_aborts_instead_of_saving_unprotected_face(self):
        engine = core.SwapEngine()
        engine.occluder = SimpleNamespace(run=lambda *_: [np.full((1, 256, 256, 1), np.nan)])
        from insightface.utils.face_align import arcface_dst
        face = SimpleNamespace(kps=arcface_dst.copy())
        with self.assertRaisesRegex(RuntimeError, 'invalid mask'):
            engine._visible_face_mask(np.zeros((256, 256, 3), np.uint8), face)

    @staticmethod
    def eye_landmarks():
        before = np.zeros((106, 2), np.float32)
        before[core.LEFT_EYE_106_INDICES] = [
            [30, 50], [40, 43], [50, 42], [60, 43],
            [70, 50], [60, 57], [50, 58], [40, 57]]
        before[34] = [50, 50]
        return before

    def test_careful_does_not_postwarp_generated_eye_pixels(self):
        before = self.eye_landmarks()
        after = before.copy()
        after[np.r_[core.LEFT_EYE_106_INDICES, 34]] += [4, 3]
        image = np.full((100, 120, 3), 150, np.uint8)
        cv2.circle(image, (54, 53), 3, (10, 30, 80), -1)
        face = SimpleNamespace(bbox=np.array([0, 0, 100, 100]), landmark_2d_106=before)
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(get=lambda _: [face], models={})
        engine._visible_face_mask = lambda *_: np.ones(image.shape[:2], np.float32)
        engine._hyperswap_face = lambda *_: image.copy()
        engine._landmarks_after_swap = lambda *_: SimpleNamespace(landmark_2d_106=after)
        result, _, _ = engine.swap_array(image * 0, face, quality='careful')
        np.testing.assert_array_equal(result, image)

    def test_open_mouth_protection_includes_lips_and_extended_tongue_only(self):
        labels = np.zeros((128, 128), np.uint8)
        labels[20:25, 20:60] = 12
        labels[25:40, 20:60] = 11
        labels[40:65, 33:48] = 11
        labels[90:100, 100:110] = 11  # unrelated mouth/segmentation component
        landmarks = np.zeros((106, 2), np.float32)
        landmarks[52:72] = np.tile([[20, 20], [60, 20], [60, 40], [20, 40]], (5, 1))
        visibility = np.ones(labels.shape, np.float32)
        matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        mask = core.SwapEngine._expression_mask_from_labels(labels, landmarks, visibility, matrix)
        self.assertIsNotNone(mask)
        self.assertEqual(float(mask[23, 40]), 1)
        self.assertEqual(float(mask[60, 40]), 1)
        self.assertEqual(float(mask[95, 105]), 0)
        original = np.full((128, 128, 3), 80, np.uint8)
        swapped = np.full_like(original, 150)
        result = core.SwapEngine._apply_visibility(swapped, original, 1 - mask)
        np.testing.assert_array_equal(result[mask == 1], original[mask == 1])
        np.testing.assert_array_equal(result[mask == 0], swapped[mask == 0])

    def test_closed_unobstructed_lips_keep_generated_identity(self):
        labels = np.zeros((128, 128), np.uint8)
        labels[25:28, 20:60] = 12
        landmarks = np.zeros((106, 2), np.float32)
        landmarks[52:72] = np.tile([[20, 20], [60, 20], [60, 40], [20, 40]], (5, 1))
        matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        mask = core.SwapEngine._expression_mask_from_labels(
            labels, landmarks, np.ones(labels.shape, np.float32), matrix
        )
        self.assertIsNone(mask)
        obscured = core.SwapEngine._expression_mask_from_labels(
            labels, landmarks, np.full(labels.shape, 0.4, np.float32), matrix
        )
        self.assertIsNotNone(obscured)

    def test_mouth_parser_rejects_invalid_output_before_compositing(self):
        engine = core.SwapEngine()
        engine.face_parser = SimpleNamespace(run=lambda *_: [np.full((1, 19, 512, 512), np.nan)])
        face = SimpleNamespace(kps=core.FFHQ_TEMPLATE * 128,
                               landmark_2d_106=np.ones((106, 2), np.float32))
        with self.assertRaisesRegex(RuntimeError, 'invalid labels'):
            engine._mouth_expression_mask(np.zeros((128, 128, 3), np.uint8), face,
                                           np.ones((128, 128), np.float32))

    def test_checksum_failure_keeps_existing_model_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            installed = root / core.OCCLUDER_MODEL
            installed.write_bytes(b'existing')
            def download(_url, destination, *_args):
                destination.write_bytes(b'corrupt download')
            with patch.object(setup_models, 'download', side_effect=download):
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    setup_models.install_occluder(root)
            self.assertEqual(installed.read_bytes(), b'existing')
            self.assertFalse(installed.with_suffix('.onnx.part').exists())
            self.assertIn(core.OCCLUDER_MODEL, core.missing_models(root / 'empty'))
