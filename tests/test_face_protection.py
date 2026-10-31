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
    def test_forehead_mask_covers_bare_skin_between_hair_patches_and_spares_high_generated_brows(self):
        points = np.zeros((106, 2), np.float32)
        points[43:52] = np.column_stack((np.linspace(160, 220, 9), np.full(9, 200)))
        points[97:106] = np.column_stack((np.linspace(290, 350, 9), np.full(9, 200)))
        face = SimpleNamespace(landmark_2d_106=points,
                               kps=np.array([[190, 240], [320, 240], [255, 290],
                                             [220, 340], [290, 340]], np.float32))
        before = np.ones((512, 512), np.uint8)
        before[:60] = 17
        after = before.copy()
        after[65:95, 160:220] = 17
        after[155:165, 165:215] = 2  # generated eyebrow above target brow
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        mask = core.SwapEngine._forehead_protection(before, after, identity, face, (512, 512, 3))
        np.testing.assert_array_equal(mask[60:110, 140:360], 1)
        np.testing.assert_array_equal(mask[175:190, 240:280], 1)
        self.assertEqual(float(mask[160, 190]), 0)
        np.testing.assert_array_equal(mask[250:], 0)
        # A skin label between invented hair islands must receive the same
        # protection; otherwise restoring islands leaves a false curved seam.
        self.assertEqual(float(mask[80, 240]), float(mask[80, 190]))

    def test_forehead_mask_requires_valid_landmarks_and_follows_alignment(self):
        labels = np.ones((512, 512), np.uint8)
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        self.assertIsNone(core.SwapEngine._forehead_protection(
            labels, labels, identity, SimpleNamespace(), (600, 600, 3)))
        points = np.zeros((106, 2), np.float32)
        points[43:52] = [190, 200]
        points[97:106] = [320, 200]
        face = SimpleNamespace(landmark_2d_106=points,
                               kps=np.array([[190, 240], [320, 240], [255, 290],
                                             [220, 340], [290, 340]], np.float32))
        base = core.SwapEngine._forehead_protection(labels, labels, identity, face, (600, 600, 3))
        # Translate input coordinates with the face while keeping its aligned
        # crop fixed; protection must translate back to the image as well.
        face.landmark_2d_106 += [60, 70]
        face.kps += [60, 70]
        matrix = np.array([[1, 0, -60], [0, 1, -70]], np.float32)
        moved = core.SwapEngine._forehead_protection(labels, labels, matrix, face, (600, 600, 3))
        np.testing.assert_array_equal(moved[70:582, 60:572], base[:512, :512])

    def test_hairline_protection_removes_invented_hair_without_replacing_brows_or_face(self):
        original = np.full((512, 512, 3), 160, np.uint8)
        original[:80] = 20
        generated = np.full_like(original, 190)
        generated[:110] = 35  # new hair drawn below the real hairline
        generated[220:235, 140:220] = 50  # source eyebrow must remain
        original_labels = np.ones((512, 512), np.uint8)
        original_labels[:80] = 17
        generated_labels = original_labels.copy()
        generated_labels[:110] = 17
        generated_labels[220:235, 140:220] = 2
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        engine = core.SwapEngine()
        calls = []
        def parse(image, _face, matrix=None):
            calls.append(matrix)
            return image, original_labels if image is original else generated_labels, identity
        engine._parse_face = parse
        face = SimpleNamespace(kps=np.ones((5, 2), np.float32))
        fixed = engine._protect_hairline(generated, original, face)
        np.testing.assert_array_equal(fixed[:110], original[:110])
        np.testing.assert_array_equal(fixed[125:], generated[125:])
        self.assertIsNone(calls[0])
        self.assertIs(calls[1], identity)

    def test_hairline_protection_warps_to_target_and_leaves_absent_hair_unchanged(self):
        before = np.ones((512, 512), np.uint8)
        after = before.copy()
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        self.assertIsNone(core.SwapEngine._hairline_protection(before, after, identity, (700, 700, 3)))
        after[20:40, 20:40] = 17
        matrix = np.array([[1, 0, -100], [0, 1, -200]], np.float32)
        mask = core.SwapEngine._hairline_protection(before, after, matrix, (700, 700, 3))
        self.assertEqual(mask.shape, (700, 700))
        self.assertEqual(float(mask[230, 130]), 1)
        self.assertEqual(float(mask[30, 30]), 0)

    def test_hairline_protection_excludes_features_even_beside_classified_hair(self):
        before = np.ones((512, 512), np.uint8)
        after = before.copy()
        after[20:100, 20:100] = 17
        before[50:70, 80:120] = 4
        before[30:40, 80:120] = 2
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        mask = core.SwapEngine._hairline_protection(before, after, identity, (512, 512, 3))
        self.assertEqual(float(mask[55, 90]), 0)
        self.assertEqual(float(mask[35, 90]), 0)
        self.assertEqual(float(mask[55, 50]), 1)

    def test_landmark_guard_preserves_features_when_parser_mistakes_lashes_for_hair(self):
        original = np.full((512, 512, 3), 160, np.uint8)
        generated = np.full_like(original, 90)
        before = np.ones((512, 512), np.uint8)
        after = before.copy()
        after[:250] = 17  # incorrect parser boundary overlaps central features
        identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        engine = core.SwapEngine()
        engine._parse_face = lambda image, _face, **_kw: (
            image, before if image is original else after, identity)
        points = np.resize(np.array([[150, 180], [350, 180], [350, 400], [150, 400]],
                                    np.float32), (106, 2))
        face = SimpleNamespace(kps=np.ones((5, 2), np.float32), landmark_2d_106=points)
        fixed = engine._protect_hairline(generated, original, face)
        np.testing.assert_array_equal(fixed[180:401, 150:351], generated[180:401, 150:351])
        np.testing.assert_array_equal(fixed[60:100, 60:100], original[60:100, 60:100])

    def test_hairline_protection_runs_after_model_choice_with_previous_face_results(self):
        faces = [SimpleNamespace(bbox=np.array([0, 0, 80, 80])),
                 SimpleNamespace(bbox=np.array([80, 0, 120, 40]))]
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(get=lambda _: faces)
        engine._visible_face_mask = lambda image, _: np.ones(image.shape[:2], np.float32)
        engine._identity_checked_face = lambda image, *_args, **_kwargs: (image + 1, True)
        received = []
        def protect(rendered, previous, face, _log):
            received.append((int(rendered[0, 0, 0]), int(previous[0, 0, 0]), face))
            return rendered
        engine._protect_hairline = protect
        engine.swap_array(np.zeros((128, 128, 3), np.uint8), object(), all_faces=True,
                          preserve_mouth=False)
        self.assertEqual([(a, b) for a, b, _ in received], [(1, 0), (2, 1)])
        self.assertIs(received[1][2], faces[1])

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

    def test_eye_detail_skips_missing_texture_and_zero_strength(self):
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
