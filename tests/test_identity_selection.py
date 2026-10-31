from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import core


class IdentitySelectionTests(unittest.TestCase):
    def engine(self):
        engine = core.SwapEngine()
        self.target = np.full((32, 32, 3), 10, np.uint8)
        self.primary = np.full_like(self.target, 30)
        self.alternate = np.full_like(self.target, 60)
        self.restored = np.full_like(self.target, 90)
        self.face = SimpleNamespace(bbox=np.array([0, 0, 32, 32]))
        engine._hyperswap_face = Mock(return_value=self.primary)
        engine._inswapper_face = Mock(return_value=self.alternate)
        engine._restore_face = Mock(return_value=self.restored)
        return engine

    def choose(self, engine, quality="careful", strength=0.5, size=512):
        return engine._identity_checked_face(
            self.target, self.face, object(), size, quality, strength, lambda _: None,
        )

    def test_clear_identity_gain_selects_alternate_and_limits_restoration(self):
        for limit in (0.2, 0.8):
            engine = self.engine()
            engine._source_similarity = Mock(side_effect=[0.8, 0.91, 0.90])
            result, selected = self.choose(engine, strength=limit)
            self.assertTrue(selected)
            self.assertIs(result, self.restored)
            self.assertEqual(engine._restore_face.call_args.args[2], min(limit, 0.35))
            self.assertIs(engine._restore_face.call_args.args[4], self.target)
            self.assertEqual(engine._inswapper_face.call_args.args[3], 256)

    def test_close_scores_keep_existing_detailed_rendering(self):
        engine = self.engine()
        engine._source_similarity = Mock(side_effect=[0.80, 0.82])
        result, selected = self.choose(engine)
        self.assertFalse(selected)
        self.assertIs(result, self.primary)
        engine._restore_face.assert_not_called()

    def test_large_native_face_keeps_detail_path_without_trying_coarse_model(self):
        engine = self.engine()
        self.face.bbox = np.array([10, 10, 450, 510])
        engine._source_similarity = Mock()
        result, selected = self.choose(engine)
        self.assertFalse(selected)
        self.assertIs(result, self.primary)
        engine._source_similarity.assert_not_called()
        engine._inswapper_face.assert_not_called()

    def test_restoration_that_repaints_identity_is_rejected(self):
        engine = self.engine()
        engine._source_similarity = Mock(side_effect=[0.80, 0.91, 0.81])
        result, selected = self.choose(engine)
        self.assertTrue(selected)
        self.assertIs(result, self.alternate)

    def test_zero_strength_and_best_do_not_restore_inside_selection(self):
        for quality, strength in (("careful", 0), ("best", 0.8)):
            engine = self.engine()
            engine._source_similarity = Mock(side_effect=[0.80, 0.91])
            result, selected = self.choose(engine, quality, strength, size=1024)
            self.assertTrue(selected)
            self.assertIs(result, self.alternate)
            engine._restore_face.assert_not_called()
            self.assertEqual(engine._inswapper_face.call_args.args[3], 512)

    def test_unmeasurable_candidates_and_draft_mode_keep_primary(self):
        for scores in ([None], [0.8, None]):
            engine = self.engine()
            engine._source_similarity = Mock(side_effect=scores)
            result, selected = self.choose(engine)
            self.assertFalse(selected)
            self.assertIs(result, self.primary)
        engine = self.engine()
        engine._source_similarity = Mock()
        self.choose(engine, quality="balanced")
        engine._source_similarity.assert_not_called()
        engine._inswapper_face.assert_not_called()

    def test_scoring_uses_fixed_alignment_without_mutating_target_or_redetecting(self):
        engine = self.engine()
        self.face.kps = np.ones((5, 2), np.float32)
        def recognize(_image, placed):
            placed.normed_embedding = np.array([0.6, 0.8], np.float32)
            placed.kps[:] = 9
        engine.analyser = SimpleNamespace(
            models={"recognition": SimpleNamespace(get=recognize)},
            get=Mock(side_effect=AssertionError("must reuse the selected face")),
        )
        source = SimpleNamespace(normed_embedding=np.array([1, 0], np.float32))
        self.assertAlmostEqual(engine._source_similarity(self.target, self.face, source), 0.6)
        np.testing.assert_array_equal(self.face.kps, np.ones((5, 2)))
        for vector in ([np.nan, 0], [0, 0], [1, 0, 0]):
            engine.analyser.models["recognition"].get = lambda _, placed: setattr(
                placed, "normed_embedding", np.array(vector, np.float32))
            self.assertIsNone(engine._source_similarity(self.target, self.face, source))

    def test_alternate_still_obeys_final_foreground_protection(self):
        engine = self.engine()
        engine.analyser = SimpleNamespace(get=lambda _: [self.face], models={})
        visibility = np.ones(self.target.shape[:2], np.float32)
        visibility[10:20, 12:18] = 0
        engine._visible_face_mask = lambda *_: visibility
        engine._source_similarity = lambda image, *_: {30: 0.80, 60: 0.91, 90: 0.90}[int(image[0, 0, 0])]
        result, count, _ = engine.swap_array(self.target, object(), preserve_mouth=False)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(result[visibility == 0], self.target[visibility == 0])
        np.testing.assert_array_equal(result[visibility == 1], self.restored[visibility == 1])

    def test_gain_erased_by_final_blend_keeps_finished_primary(self):
        engine = self.engine()
        engine._source_similarity = Mock(side_effect=[0.80, 0.91, 0.80, 0.82])
        result, selected = engine._identity_checked_face(
            self.target, self.face, object(), 512, "careful", 0, lambda _: None,
            finish=lambda image, _identity: image + 1,
        )
        self.assertFalse(selected)
        np.testing.assert_array_equal(result, self.primary + 1)

    def test_identity_model_input_mapping_colour_and_bad_output(self):
        from insightface.utils.face_align import arcface_dst
        engine = core.SwapEngine()
        face = SimpleNamespace(kps=arcface_dst + [8, 0])
        source = SimpleNamespace(normed_embedding=np.array([0.6, 0.8], np.float32))
        prediction = np.broadcast_to(np.array([0.1, 0.2, 0.3], np.float32)[None, :, None, None],
                                     (1, 3, 128, 128)).copy()
        run = Mock(return_value=[prediction])
        engine.swapper = SimpleNamespace(
            emap=np.eye(2, dtype=np.float32), session=SimpleNamespace(
                run=run, get_providers=lambda: ["CPUExecutionProvider"]),
            input_names=["target", "source"], output_names=["output"],
        )
        result = engine._inswapper_face(np.zeros((128, 128, 3), np.uint8), face, source, 256)
        self.assertEqual(run.call_count, 4)
        inputs = run.call_args.args[1]
        self.assertEqual(inputs["target"].shape, (1, 3, 128, 128))
        np.testing.assert_allclose(inputs["source"], [[0.6, 0.8]])
        np.testing.assert_allclose(result[64, 64], [76.5, 51, 25.5], atol=1)
        run.return_value = [np.full_like(prediction, np.nan)]
        with self.assertRaisesRegex(RuntimeError, "identity model returned invalid pixels"):
            engine._inswapper_face(np.zeros((128, 128, 3), np.uint8), face, source, 256)


if __name__ == "__main__":
    unittest.main()
