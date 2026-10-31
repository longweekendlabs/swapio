from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import core


class EyeTextureTests(unittest.TestCase):
    @staticmethod
    def points(width):
        points = np.zeros((106, 2), np.float32)
        contour = np.array([[-.5, 0], [-.33, -.13], [0, -.17], [.33, -.13],
                            [.5, 0], [.33, .13], [0, .17], [-.33, .13]]) * width
        points[core.LEFT_EYE_106_INDICES] = contour + [width, width]
        points[core.RIGHT_EYE_106_INDICES] = contour + [width * 3, width]
        return points

    def test_generated_eye_recovery_scales_below_and_above_old_cutoff(self):
        for width in (16, 44, 80, 160):
            with self.subTest(width=width):
                points = self.points(width)
                yy, xx = np.mgrid[:width * 3, :width * 4]
                texture = ((xx + yy) % 2 * 60).astype(np.uint8)
                reference = np.repeat((texture + 80)[:, :, None], 3, axis=2)
                generated = np.full_like(reference, 100)
                rebuilt = np.repeat((texture + 50)[:, :, None], 3, axis=2)
                result, count = core.SwapEngine._eye_detail_blend(
                    reference, generated, rebuilt, points, points, .5,
                )
                self.assertEqual(count, 2)
                self.assertGreater(core.SwapEngine._detail_score(result, points[core.LEFT_EYE_106_INDICES]), 0)
                np.testing.assert_array_equal(result[width * 2:], generated[width * 2:])
                rejected, count = core.SwapEngine._eye_detail_blend(
                    reference, generated, rebuilt, points, points + [width * .1, 0], .5,
                )
                self.assertEqual(count, 0)
                np.testing.assert_array_equal(rejected, generated)

    def test_adaptive_pass_actually_runs_for_reported_eye_size(self):
        engine = core.SwapEngine()
        points = self.points(44)
        yy, xx = np.mgrid[:132, :176]
        reference = np.repeat((((xx + yy) % 2 * 60) + 80)[:, :, None], 3, axis=2).astype(np.uint8)
        generated = np.full_like(reference, 100)
        face = SimpleNamespace(landmark_2d_106=points)
        engine._eye_landmark_frame = Mock(return_value=None)
        engine._eye_landmarks = Mock(return_value=face)
        engine._restore_face = Mock(return_value=reference)
        result = engine._adaptive_eye_detail(generated, reference, face, .5, lambda _: None)
        engine._restore_face.assert_called_once()
        self.assertFalse(np.array_equal(result, generated))

    def test_restoration_reuses_prediction_for_eye_detail_and_respects_off(self):
        engine = core.SwapEngine()
        image = np.full((128, 128, 3), 90, np.uint8)
        face = SimpleNamespace(kps=core.FFHQ_TEMPLATE * 128)
        engine._ensure_enhancer = Mock(return_value=object())
        engine._run_enhancer = Mock(return_value=np.zeros((3, 1024, 1024), np.float32))
        engine._eye_landmark_frame = Mock(return_value=None)
        measured = SimpleNamespace(landmark_2d_106=self.points(16))
        engine._eye_landmarks = Mock(return_value=measured)
        engine._eye_detail_blend = Mock(side_effect=lambda reference, rendered, *_: (rendered, 2))
        engine._restore_face(image, face, .5, detail_reference=image)
        engine._run_enhancer.assert_called_once()
        self.assertIs(engine._eye_detail_blend.call_args.args[0], image)
        self.assertEqual(engine._eye_detail_blend.call_args.args[-1], .5)
        engine._run_enhancer.reset_mock()
        self.assertIs(engine._restore_face(image, face, 0, detail_reference=image), image)
        engine._run_enhancer.assert_not_called()

    def test_target_preservation_keeps_lashes_without_repainting_brows_or_mouth(self):
        for width in (16, 44, 80):
            with self.subTest(width=width):
                points = self.points(width)
                face = SimpleNamespace(landmark_2d_106=points)
                original = np.full((width * 3, width * 4, 3), 220, np.uint8)
                generated = np.full_like(original, 80)
                result = core.SwapEngine()._destination_eyes(generated, original, face, face)
                # Above the eye opening, where the old mask left blurred lashes.
                self.assertGreater(int(result[round(width * .8), width, 0]), 160)
                np.testing.assert_array_equal(result[:round(width * .5)], generated[:round(width * .5)])
                np.testing.assert_array_equal(result[width * 2:], generated[width * 2:])
                moved = SimpleNamespace(landmark_2d_106=points + [width, 0])
                np.testing.assert_array_equal(
                    core.SwapEngine()._destination_eyes(generated, original, face, moved), generated,
                )


if __name__ == '__main__':
    unittest.main()
