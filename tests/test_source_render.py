from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import core
import source_render as sr
import setup_models


def landmarks(width=44, opening=.32):
    points = np.zeros((106, 2), np.float32)
    # Semantic order: bottom, left corner, lower-left/right, right corner,
    # top, upper-left/right. The donor must work even across a 90-degree roll.
    contour = np.array([[0, .5], [-.5, 0], [-.3, .4], [.3, .4],
                        [.5, 0], [0, -.5], [-.3, -.4], [.3, -.4]], np.float32)
    contour *= [width, width * opening]
    for ids, pupil, centre in ((core.LEFT_EYE_106_INDICES, 34, [width, width]),
                               (core.RIGHT_EYE_106_INDICES, 88, [width * 3, width])):
        points[ids] = contour + centre
        points[pupil] = centre
    return points


class ReferenceDetailTests(unittest.TestCase):
    def test_opening_eyelids_does_not_leave_a_stretched_pupil_behind_the_iris_layer(self):
        for source_opening, placed_opening in ((.18, .50), (.50, .18)):
            with self.subTest(source=source_opening, placed=placed_opening):
                image = np.full((132, 176, 3), 220, np.uint8)
                for centre in ((44, 44), (132, 44)):
                    cv2.circle(image, centre, 3, (0, 0, 0), -1)
                result, count = sr.reference_eyes(image, image, landmarks(opening=source_opening), landmarks(opening=placed_opening))
                self.assertEqual(count, 2)
                yy, xx = np.where(result[30:59, 35:54].max(2) < 20)
                self.assertGreaterEqual(int(np.ptp(yy)), 5)
                self.assertLessEqual(int(np.ptp(yy)), 7)
                self.assertLessEqual(abs(int(np.ptp(yy)) - int(np.ptp(xx))), 1)

    def test_actual_source_pixels_work_at_small_and_large_sizes_without_target_copy(self):
        for width in (16, 44, 80, 160):
            with self.subTest(width=width):
                points = landmarks(width)
                source = np.full((width * 3, width * 4, 3), [20, 80, 160], np.uint8)
                base = np.full_like(source, [180, 100, 40])
                result, count = sr.reference_eyes(base, source, points, points)
                self.assertEqual(count, 2)
                np.testing.assert_array_equal(result[width, width], source[width, width])
                np.testing.assert_array_equal(result[width * 2:], base[width * 2:])
                np.testing.assert_array_equal(base[width, width], [180, 100, 40])

    def test_donor_keeps_semantic_orientation_after_head_roll(self):
        width = 44
        points = landmarks(width)
        yy, xx = np.mgrid[:132, :176]
        source = np.stack([xx, yy, xx // 2 + yy // 2], axis=-1).astype(np.uint8)
        base = np.full_like(source, 90)
        upright, _ = sr.reference_eyes(base, source, points, points)
        # Rotate the output canvas 180 degrees; source itself stays upright.
        matrix = np.array([[-1, 0, 175], [0, -1, 131]], np.float32)
        rotated = cv2.transform(points[None], matrix)[0]
        result, count = sr.reference_eyes(base, source, points, rotated)
        self.assertEqual(count, 2)
        expected = cv2.warpAffine(upright, matrix, (176, 132))
        self.assertLess(float(np.abs(result.astype(float) - expected).mean()), .1)

    def test_closed_eye_and_invalid_measurements_do_not_invent_a_donor(self):
        base = np.full((132, 176, 3), 90, np.uint8)
        source = np.full_like(base, 220)
        for placed in (landmarks(opening=.02), np.full((106, 2), np.nan), np.zeros((5, 2))):
            result, count = sr.reference_eyes(base, source, landmarks(), placed)
            self.assertEqual(count, 0)
            np.testing.assert_array_equal(result, base)

    def test_illumination_ignores_non_skin_features_and_does_not_resample_texture(self):
        source = np.full((512, 512, 3), 100, np.uint8)
        scene = np.full_like(source, 130)
        mask = np.ones((512, 512), bool)
        mask[150:350, 150:350] = False
        field = sr.fit_illumination(source, scene, mask)
        scene[150:350, 150:350] = [0, 0, 255]
        second = sr.fit_illumination(source, scene, mask)
        np.testing.assert_allclose(field[0], second[0], atol=1e-5)
        yy, xx = np.mgrid[:700, :700]
        texture = np.repeat((80 + (xx % 2) * 80).astype(np.uint8)[:, :, None], 3, axis=2)
        identity = np.array([[512 / 700, 0, 0], [0, 512 / 700, 0]], np.float32)
        result = sr.apply_illumination(texture, identity, field, np.ones((700, 700), np.float32))
        self.assertEqual(result.shape, texture.shape)
        self.assertGreater(float(result[:, 1::2].mean() - result[:, ::2].mean()), 60)


class RuntimeSafetyTests(unittest.TestCase):
    def renderer(self):
        return sr.SourceRenderer(SimpleNamespace(_refresh_provider=Mock()))

    def test_bad_gpu_output_retries_once_and_retires_only_that_session(self):
        renderer = self.renderer()
        gpu, cpu = Mock(), Mock()
        gpu.get_providers.return_value = ["CUDAExecutionProvider"]
        gpu.run.return_value = [np.full((1, 21, 3), np.nan)]
        cpu.get_providers.return_value = ["CPUExecutionProvider"]
        cpu.run.return_value = [np.zeros((1, 21, 3), np.float32)]
        renderer.sessions["eyes"] = gpu
        renderer.sessions["motion"] = object()
        with patch.object(renderer, "_session", side_effect=[gpu, cpu]):
            output = renderer._run("eyes", {}, lambda _: None)
        self.assertTrue(np.isfinite(output[0]).all())
        self.assertIn("eyes", renderer.cpu_only)
        self.assertIn("motion", renderer.sessions)
        self.assertNotIn("eyes", renderer.sessions)
        cpu.run.assert_called_once()

    def test_invalid_cpu_output_stops_instead_of_exporting_bad_pixels(self):
        renderer = self.renderer()
        cpu = Mock()
        cpu.get_providers.return_value = ["CPUExecutionProvider"]
        cpu.run.return_value = [np.zeros((1, 3, 128, 128), np.float32)]
        with patch.object(renderer, "_session", return_value=cpu):
            with self.assertRaisesRegex(RuntimeError, "invalid shape"):
                renderer._run("generator", {}, lambda _: None)
        cpu.run.assert_called_once()

    def test_flat_or_out_of_range_generator_pixels_are_not_exportable(self):
        renderer = self.renderer()
        cpu = Mock(); cpu.get_providers.return_value = ["CPUExecutionProvider"]
        for value in (0, 10):
            cpu.run.return_value = [np.full((1, 3, 512, 512), value, np.float32)]
            with patch.object(renderer, "_session", return_value=cpu), self.assertRaisesRegex(RuntimeError, "invalid pixels"):
                renderer._run("generator", {}, lambda _: None)

    def test_inconsistent_five_point_alignment_is_rejected(self):
        image = np.full((512, 512, 3), 100, np.uint8)
        points = sr.ARC_TEMPLATE * 512
        points[2] += [200, -200]
        with self.assertRaisesRegex(RuntimeError, "unreliable"):
            sr.SourceRenderer._prepare(image, SimpleNamespace(kps=points))

    def test_missing_model_does_not_reach_onnx_or_retry_as_a_cuda_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            owner = SimpleNamespace(models=Path(temp), use_gpu=False, _refresh_provider=Mock())
            renderer = sr.SourceRenderer(owner)
            with patch("onnxruntime.InferenceSession") as constructor:
                with self.assertRaisesRegex(RuntimeError, "Missing or incomplete"):
                    renderer._session("appearance", lambda _: None)
                constructor.assert_not_called()

    def test_reference_cache_tracks_native_pixels_even_with_identical_model_input(self):
        owner = SimpleNamespace(_eye_landmark_frame=Mock(return_value=None),
            _eye_landmarks=Mock(return_value=SimpleNamespace(landmark_2d_106=landmarks())))
        renderer = sr.SourceRenderer(owner)
        renderer._prepare = Mock(return_value=(np.zeros((1, 3, 256, 256), np.float32), None))
        motion = [np.array(0, np.float32)] * 3 + [np.ones((1, 1), np.float32),
            np.zeros((1, 3), np.float32), np.zeros((1, 21, 3), np.float32), np.zeros((1, 21, 3), np.float32)]
        renderer._run = Mock(side_effect=lambda name, *_: [np.zeros(1)] if name == "appearance" else motion)
        face = SimpleNamespace(kps=sr.ARC_TEMPLATE * 100)
        image = np.full((100, 100, 3), 90, np.uint8)
        renderer._reference(image, face, lambda _: None)
        renderer._reference(image.copy(), face, lambda _: None)
        self.assertEqual(renderer._run.call_count, 2)
        renderer._reference(cv2.resize(image, (200, 200)), face, lambda _: None)
        self.assertEqual(renderer._run.call_count, 4)

    def test_gpu_constructor_failure_uses_visible_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b"verified model fixture"
            (root / "model.onnx").write_bytes(payload)
            owner = SimpleNamespace(models=root, use_gpu=True, _refresh_provider=Mock())
            renderer = sr.SourceRenderer(owner)
            cpu = Mock()
            cpu.get_providers.return_value = ["CPUExecutionProvider"]
            manifest = {"appearance": ("model.onnx", len(payload), hashlib.sha256(payload).hexdigest())}
            with patch.object(sr, "SOURCE_MODELS", manifest), patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), patch("onnxruntime.InferenceSession", side_effect=[RuntimeError("CUDA allocation"), cpu]) as ctor:
                log = Mock()
                self.assertIs(renderer._session("appearance", log), cpu)
                self.assertEqual(ctor.call_args.kwargs["providers"], ["CPUExecutionProvider"])
            self.assertIn("appearance", renderer.cpu_only)
            self.assertTrue(any("retrying on CPU" in call.args[0] for call in log.call_args_list))


class MeasuredRefinementTests(unittest.TestCase):
    def exercise(self, quality, openings, target=(.32, .32)):
        image = np.full((100, 180, 3), 100, np.uint8)
        face = SimpleNamespace(kps=sr.ARC_TEMPLATE * 100, landmark_2d_106=landmarks())
        # Change each eye's opening independently, keeping semantic contours.
        for ids, ratio in ((core.LEFT_EYE_106_INDICES, target[0]), (core.RIGHT_EYE_106_INDICES, target[1])):
            eye = face.landmark_2d_106[ids]
            face.landmark_2d_106[ids, 1] = eye[:, 1].mean() + (eye[:, 1] - eye[:, 1].mean()) * ratio / .32
        owner = SimpleNamespace(_eye_landmark_frame=Mock(return_value=None),
            _eye_landmarks=Mock(side_effect=[SimpleNamespace(landmark_2d_106=landmarks(opening=o)) for o in openings]),
            _paste_crop=Mock(return_value=image),
            _parse_face=Mock(return_value=(np.full((512, 512, 3), 100, np.uint8), np.ones((512, 512), np.uint8), np.eye(2, 3, dtype=np.float32))))
        renderer = sr.SourceRenderer(owner)
        motion = [np.array(0, np.float32)] * 3 + [np.ones((1, 1), np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 21, 3), np.float32), np.zeros((1, 21, 3), np.float32)]
        renderer._reference = Mock(return_value=(None, motion, np.zeros((1, 21, 3), np.float32), None))
        renderer._prepare = Mock(return_value=(None, np.eye(2, 3, dtype=np.float32)))
        def run(name, feed, log):
            if name == "motion": return motion
            if name == "generator": return [np.zeros((1, 3, 512, 512), np.float32)]
            return [np.ones((1, 21, 3), np.float32)]
        renderer._run = Mock(side_effect=run)
        logs = []
        with patch.object(sr, "reference_eyes", side_effect=lambda base, *_: (base, 0)), patch.object(sr, "fit_illumination", return_value=None):
            renderer.render(image, image, face, image, face, quality, 1, np.ones(image.shape[:2], np.float32), logs.append)
        return renderer, logs

    def test_best_permits_a_third_measured_pass_while_careful_stops_at_two(self):
        for quality, expected in (("careful", 2), ("best", 3)):
            renderer, logs = self.exercise(quality, [.10, .20, .31])
            generators = [c for c in renderer._run.call_args_list if c.args[0] == "generator"]
            self.assertEqual(len(generators), expected)
            self.assertTrue(any(f"{expected} generation pass" in log for log in logs))

    def test_a_worse_correction_is_discarded_instead_of_repeated(self):
        renderer, logs = self.exercise("best", [.20, .10])
        self.assertEqual(len([c for c in renderer._run.call_args_list if c.args[0] == "generator"]), 2)
        self.assertTrue(any("1 generation pass" in log for log in logs))

    def test_asymmetric_eye_expression_does_not_trigger_symmetric_eye_retargeting(self):
        renderer, _ = self.exercise("best", [.20], target=(.45, .08))
        self.assertFalse(any(c.args[0] == "eyes" for c in renderer._run.call_args_list))


class SourceMouthVisibilityTests(unittest.TestCase):
    def fixture(self):
        points = np.zeros((106, 2), np.float32)
        points[52:72] = np.tile([[20, 20], [60, 20], [60, 45], [20, 45]], (5, 1))
        points[core.INNER_MOUTH_106_INDICES] = np.tile([[25, 27], [55, 27], [55, 38], [25, 38]], (2, 1))
        labels = np.zeros((128, 128), np.uint8)
        labels[20:25, 20:60] = 12; labels[25:40, 20:60] = 11; labels[40:45, 20:60] = 13
        visibility = np.ones(labels.shape, np.float32); visibility[labels == 11] = 0
        engine = core.SwapEngine()
        engine._parse_face = Mock(return_value=(None, labels, np.eye(2, 3, dtype=np.float32)))
        return engine, SimpleNamespace(landmark_2d_106=points), labels, visibility

    def test_clear_cavity_can_use_generated_source_pixels_but_other_protected_pixels_cannot(self):
        engine, face, labels, visibility = self.fixture()
        visibility[:10] = 0
        result = engine._source_mouth_visibility(np.zeros((128, 128, 3), np.uint8), face, visibility)
        self.assertEqual(result[32, 40], 1)
        np.testing.assert_array_equal(result[labels != 11], visibility[labels != 11])
        np.testing.assert_array_equal(result[:10], visibility[:10])
        _, protection, unobstructed = engine._source_mouth_context(np.zeros((128, 128, 3), np.uint8), face, visibility)
        self.assertIsNone(protection)
        self.assertTrue(unobstructed)

    def test_obstructed_lips_and_extended_tongue_keep_the_foreground_mask(self):
        for obstruction in ("lips", "tongue"):
            with self.subTest(obstruction=obstruction):
                engine, face, labels, visibility = self.fixture()
                if obstruction == "lips": visibility[20:45, 20:40] = 0
                else: labels[40:65, 33:48] = 11
                result = engine._source_mouth_visibility(np.zeros((128, 128, 3), np.uint8), face, visibility)
                np.testing.assert_array_equal(result, visibility)
                _, protection, unobstructed = engine._source_mouth_context(np.zeros((128, 128, 3), np.uint8), face, visibility)
                self.assertIsNotNone(protection)
                self.assertFalse(unobstructed)


class UprightDetectionTests(unittest.TestCase):
    def test_high_confidence_upside_down_bad_landmarks_are_redetected(self):
        bad = SimpleNamespace(bbox=np.array([40, 40, 140, 180], np.float32),
            kps=np.array([[110, 140], [70, 138], [90, 110], [100, -20], [80, -20]], np.float32),
            det_score=.99)
        upright = SimpleNamespace(bbox=np.array([90, 70, 190, 210], np.float32),
            kps=np.array([[115, 110], [165, 110], [140, 140], [120, 175], [160, 175]], np.float32),
            landmark_2d_106=np.tile([140, 140], (106, 1)).astype(np.float32),
            embedding=np.ones(512), det_score=.9)
        original = bad.kps.copy()
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(models={"detection": object()}, get=Mock(side_effect=[[bad], [upright]]))
        faces = engine.detect_faces(np.zeros((220, 180, 3), np.uint8))
        self.assertEqual(len(faces), 1)
        self.assertTrue(engine._reliable_keypoints(faces[0]))
        self.assertAlmostEqual(float(np.linalg.norm(faces[0].kps[1] - faces[0].kps[0])), 50, places=3)
        np.testing.assert_array_equal(bad.kps, original)
        np.testing.assert_array_equal(faces[0].embedding, upright.embedding)
        self.assertEqual(faces[0].landmark_2d_106.shape, (106, 2))

    def test_unrecoverable_rotated_face_cannot_be_exported_with_wrong_landmarks(self):
        bad = SimpleNamespace(bbox=np.array([40, 40, 140, 180], np.float32),
            kps=np.array([[110, 140], [70, 140], [90, 110], [100, -20], [80, -20]], np.float32))
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(models={"detection": object()}, get=Mock(side_effect=[[bad], [], [], [], []]))
        with self.assertRaisesRegex(RuntimeError, "reliable landmarks"):
            engine.detect_faces(np.zeros((220, 180, 3), np.uint8))

    def test_inside_bounds_but_malformed_points_trigger_quarter_turn_recovery(self):
        bad = SimpleNamespace(bbox=np.array([37, 42, 90, 105], np.float32),
            kps=np.array([[60, 73], [74, 79], [66, 76], [55, 78], [71, 83]], np.float32))
        upright = SimpleNamespace(bbox=np.array([90, 70, 190, 210], np.float32),
            kps=np.array([[115, 110], [165, 110], [140, 140], [120, 175], [160, 175]], np.float32))
        self.assertFalse(core.SwapEngine._reliable_keypoints(bad))
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(models={"detection": object()}, get=Mock(side_effect=[[bad], [], [upright]]))
        faces = engine.detect_faces(np.zeros((128, 128, 3), np.uint8))
        self.assertEqual(len(faces), 1)
        self.assertTrue(engine._reliable_keypoints(faces[0]))
        self.assertEqual(engine.analyser.get.call_count, 3)

    def test_recovery_rejects_a_sideways_candidate_even_with_plausible_geometry(self):
        bad = SimpleNamespace(bbox=np.array([40, 40, 140, 180], np.float32),
            kps=np.array([[110, 140], [70, 140], [90, 110], [100, -20], [80, -20]], np.float32))
        upright = SimpleNamespace(bbox=np.array([90, 70, 190, 210], np.float32),
            kps=np.array([[115, 110], [165, 110], [140, 140], [120, 175], [160, 175]], np.float32))
        sideways = core.SwapEngine._map_face(upright, np.array([[-1, 0, 256], [0, -1, 256]], np.float32))
        self.assertTrue(core.SwapEngine._reliable_keypoints(sideways))
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(models={"detection": object()}, get=Mock(side_effect=[[bad], [sideways], [upright]]))
        self.assertEqual(len(engine.detect_faces(np.zeros((220, 180, 3), np.uint8))), 1)
        self.assertEqual(engine.analyser.get.call_count, 3)


class SourceModelInstallTests(unittest.TestCase):
    def test_verified_shared_copy_is_linked_without_another_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); shared = root / "another-app/models"; shared.mkdir(parents=True)
            target = root / "swapio/models"
            payload = b"correct model fixture"
            (shared / "test.onnx").write_bytes(payload)
            manifest = {"appearance": ("test.onnx", len(payload), hashlib.sha256(payload).hexdigest())}
            with patch.object(setup_models, "SOURCE_MODELS", manifest), patch.object(setup_models, "source_model_candidates", return_value=[target, shared]), patch.object(setup_models, "download") as download:
                setup_models.install_source_renderer(target)
                download.assert_not_called()
            self.assertTrue((target / "test.onnx").is_symlink())
            self.assertEqual((target / "test.onnx").resolve(), (shared / "test.onnx").resolve())

    def test_bad_download_and_cancellation_cannot_replace_a_shared_model(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); shared = root / "shared.onnx"; shared.write_bytes(b"old shared model")
                target = root / "models"; target.mkdir(); (target / "test.onnx").symlink_to(shared)
                manifest = {"appearance": ("test.onnx", 5, hashlib.sha256(b"right").hexdigest())}
                def download(_url, partial, *_args):
                    partial.write_bytes(b"wrong")
                stop = Mock(side_effect=[False, True]) if cancel else lambda: False
                error = setup_models.DownloadCancelled if cancel else RuntimeError
                with patch.object(setup_models, "SOURCE_MODELS", manifest), patch.object(setup_models, "source_model_candidates", return_value=[target]), patch.object(setup_models, "download", side_effect=download):
                    with self.assertRaises(error):
                        setup_models.install_source_renderer(target, should_stop=stop)
                self.assertEqual(shared.read_bytes(), b"old shared model")
                self.assertFalse((target / "test.onnx.part").exists())


if __name__ == "__main__":
    unittest.main()
