from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import core


class FakeAnalyser:
    def __init__(self, faces):
        self.faces = faces
        self.models = {}

    def get(self, _image):
        return self.faces


class FakeSwapper:
    def __init__(self):
        self.calls = []

    def get(self, image, face, _source_face, paste_back=True):
        self.calls.append((face, paste_back))
        return image + 1


class CoreTests(unittest.TestCase):
    def setUp(self):
        # These tests isolate the swap/feature pipeline from ONNX inference.
        from unittest.mock import patch
        visibility = patch.object(core.SwapEngine, "_visible_face_mask",
            side_effect=lambda image, _face: np.ones(image.shape[:2], np.float32))
        visibility.start()
        self.addCleanup(visibility.stop)

    @staticmethod
    def feature_face(points, indices=core.INNER_MOUTH_106_INDICES):
        landmarks = np.zeros((106, 2), dtype=np.float32)
        landmarks[indices] = points
        return SimpleNamespace(landmark_2d_106=landmarks)

    def test_mouth_pixels_follow_moved_lips_and_do_not_replace_new_lips(self):
        points = np.array([[30, 50], [40, 43], [50, 41], [60, 43],
                           [70, 50], [60, 58], [50, 60], [40, 58]], np.float32)
        original = np.full((120, 120, 3), 30, np.uint8)
        cv2.fillConvexPoly(original, points.astype(np.int32), (240, 240, 240))
        swapped = np.full_like(original, 100)
        before = self.feature_face(points)
        after = self.feature_face(points + [0, 16])
        result = core.SwapEngine._preserve_inner_mouth(swapped, original, before, after)
        self.assertGreater(int(result[66, 50, 0]), 230)
        # Old mouth coordinates are now outside the opening, on the new lip.
        self.assertEqual(int(result[50, 50, 0]), 100)
        mask = np.zeros(original.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, (points + [0, 16]).astype(np.int32), 1)
        np.testing.assert_array_equal(result[mask == 0], swapped[mask == 0])

    def test_eye_interior_survives_narrow_opening_at_multiple_scales(self):
        for scale in (1, 3):
            with self.subTest(scale=scale):
                # 20%-open eyes used to disappear under width-based erosion.
                points = np.array([[20, 40], [30, 36], [40, 36], [50, 38],
                                   [60, 40], [50, 42], [40, 44], [30, 44]], np.float32) * scale
                face = self.feature_face(points, core.LEFT_EYE_106_INDICES)
                swapped = np.full((80 * scale, 80 * scale, 3), 100, np.uint8)
                original = np.full_like(swapped, 240)
                result = core.SwapEngine()._destination_eyes(swapped, original, face, face)
                self.assertGreater(int(result[40 * scale, 40 * scale, 0]), 225)
                self.assertEqual(int(result[34 * scale, 40 * scale, 0]), 100)

    def test_feature_geometry_does_not_treat_tilted_closed_eye_as_open(self):
        points = np.array([[10, 20], [20, 20], [30, 20], [40, 20]], np.float32)
        for angle in (0, 35, 75):
            matrix = cv2.getRotationMatrix2D((25, 20), angle, 1)
            rotated = cv2.transform(points[None], matrix)[0]
            width, opening = core.SwapEngine._feature_geometry(rotated)
            self.assertAlmostEqual(width, 30, places=4)
            self.assertLess(opening, 0.0001)

    def test_feature_mask_preserves_a_concave_lip_boundary(self):
        points = np.array([[10, 20], [20, 28], [30, 20], [30, 40], [10, 40]], np.float32)
        mask = core.SwapEngine._feature_mask((50, 50), points, 20)
        self.assertEqual(float(mask[23, 20]), 0.0)
        self.assertGreater(float(mask[34, 20]), 0.9)

    def test_preservation_skips_missing_invalid_or_unrelated_landmarks(self):
        points = np.array([[30, 50], [40, 43], [50, 41], [60, 43],
                           [70, 50], [60, 58], [50, 60], [40, 58]], np.float32)
        original = np.full((120, 120, 3), 240, np.uint8)
        swapped = np.full_like(original, 100)
        before = self.feature_face(points)
        for after in (None, self.feature_face(points + [50, 0]),
                      self.feature_face(points * np.nan), SimpleNamespace()):
            result = core.SwapEngine._preserve_inner_mouth(swapped, original, before, after)
            np.testing.assert_array_equal(result, swapped)

    def test_final_landmarks_are_measured_once_without_mutating_target(self):
        class Face(dict):
            # Match InsightFace's missing-attribute semantics, which break
            # deepcopy(face) through its absent pickle hooks.
            def __getattr__(self, name):
                return self.get(name)

        original_points = np.ones((106, 2), np.float32)
        face = Face(bbox=np.array([0, 0, 30, 30]),
                    landmark_2d_106=original_points.copy())
        calls = []

        def measure(image, placed):
            calls.append(int(image[0, 0, 0]))
            placed.landmark_2d_106[:] = 9

        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser([face])
        engine.analyser.models['landmark_2d_106'] = SimpleNamespace(get=measure)
        engine._hyperswap_face = lambda target, *_: target + 1
        engine._restore_face = lambda target, *_: target + 10
        received = []

        def preserve(image, original, before, after):
            received.append(after)
            self.assertIs(before, face)
            self.assertTrue(np.all(after.landmark_2d_106 == 9))
            return image

        engine._destination_eyes = preserve
        engine._preserve_inner_mouth = preserve
        engine.swap_array(
            np.zeros((64, 64, 3), np.uint8), face, quality='best', destination_eyes=True
        )
        self.assertEqual(calls, [11])
        self.assertIs(received[0], received[1])
        np.testing.assert_array_equal(face.landmark_2d_106, original_points)

    def test_default_swap_does_not_copy_target_eyes(self):
        face = SimpleNamespace(bbox=np.array([0, 0, 30, 30]))
        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser([face])
        engine._hyperswap_face = lambda image, *_: image + 1
        engine._destination_eyes = lambda *_: self.fail("Target eyes must be opt-in")
        result, _, _ = engine.swap_array(
            np.zeros((64, 64, 3), np.uint8), face, preserve_mouth=False
        )
        self.assertTrue(np.all(result == 1))

    def test_crop_blend_is_consistent_under_rotation_and_image_clipping(self):
        crop = np.full((256, 256, 3), 240, np.uint8)
        target = np.zeros((600, 600, 3), np.uint8)
        sample_points = np.array([[128, 240], [20, 128], [128, 128]], np.float32)
        samples = []
        for angle in (0, 35, 70):
            inverse = cv2.getRotationMatrix2D((128, 128), angle, 1.3)
            inverse[:, 2] += 170
            placed = cv2.transform(sample_points[None], inverse)[0].round().astype(int)
            result = core.SwapEngine._paste_crop(target, crop, cv2.invertAffineTransform(inverse))
            samples.append(result[placed[:, 1], placed[:, 0], 0].astype(float))
            self.assertEqual(int(result[0, 0, 0]), 0)
        np.testing.assert_allclose(samples, np.broadcast_to(samples[0], (3, 3)), atol=8)
        # A clipped image must see the same mask as the corresponding region
        # of a larger canvas, including where the crop crosses the photo edge.
        inverse = np.array([[1, 0, -35], [0, 1, -20]], np.float32)
        small = core.SwapEngine._paste_crop(target[:200, :200], crop, cv2.invertAffineTransform(inverse))
        inverse[:, 2] += 100
        large = core.SwapEngine._paste_crop(target, crop, cv2.invertAffineTransform(inverse))
        np.testing.assert_array_equal(small, large[100:300, 100:300])

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

    def test_suggested_output_is_a_sibling_of_destination_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "Incoming"
            destination.mkdir()

            self.assertEqual(
                core.suggested_output_dir(destination),
                Path(temp) / "Incoming - Swapio Output",
            )

    def test_repeat_batch_skips_unchanged_completed_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            target = root / "target.png"
            output = root / "output"
            Image.new("RGB", (16, 16), "white").save(source)
            Image.new("RGB", (16, 16), "navy").save(target)
            engine = core.SwapEngine()
            engine.source_face = lambda *_args, **_kwargs: object()
            engine.swap_array = lambda image, *_args, **_kwargs: (image, 1, 1)

            first = engine.batch(source, [target], output, quality="fast")
            second = engine.batch(source, [target], output, quality="fast")

            self.assertEqual((first["completed"], first["skipped"]), (1, 0))
            self.assertEqual((second["completed"], second["skipped"]), (0, 1))
            self.assertEqual(len(list(output.glob("*_swapped_*.jpg"))), 1)

            Path(first["outputs"][0]).unlink()
            third = engine.batch(source, [target], output, quality="fast")
            self.assertEqual((third["completed"], third["skipped"]), (1, 0))

    def test_changing_mouth_preservation_reruns_completed_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            target = root / "target.png"
            output = root / "output"
            Image.new("RGB", (16, 16), "white").save(source)
            Image.new("RGB", (16, 16), "navy").save(target)
            engine = core.SwapEngine()
            engine.source_face = lambda *_args, **_kwargs: object()
            engine.swap_array = lambda image, *_args, **_kwargs: (image, 1, 1)

            preserved = engine.batch(source, [target], output, preserve_mouth=True)
            fully_swapped = engine.batch(source, [target], output, preserve_mouth=False)

            self.assertEqual((preserved["completed"], preserved["skipped"]), (1, 0))
            self.assertEqual((fully_swapped["completed"], fully_swapped["skipped"]), (1, 0))
            self.assertEqual(len(list(output.glob("*_swapped_*.jpg"))), 2)

    def test_careful_quality_adapts_to_closeup_face_size(self):
        distant = SimpleNamespace(bbox=np.array([100, 100, 260, 300]))
        closeup = SimpleNamespace(bbox=np.array([100, 80, 500, 650]))
        extreme = SimpleNamespace(bbox=np.array([20, 20, 760, 780]))
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        self.assertEqual(core.SwapEngine._boost_size_for_face(image, distant, "careful"), 512)
        self.assertEqual(core.SwapEngine._boost_size_for_face(image, closeup, "careful"), 1024)
        self.assertEqual(core.SwapEngine._boost_size_for_face(image, extreme, "careful"), 1024)
        self.assertEqual(core.SwapEngine._boost_size_for_face(image, closeup, "balanced"), 256)
        self.assertEqual(core.SwapEngine._boost_size_for_face(image, distant, "best"), 512)
        self.assertEqual(core.SwapEngine._boost_size_for_face(image, closeup, "best"), 1024)

    def test_best_quality_restores_after_swapping_and_before_the_mouth(self):
        face = SimpleNamespace(bbox=np.array([0, 0, 20, 20]))
        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser([face])
        order: list[str] = []

        def fake_hyperswap(target, _face, _source, _boost):
            order.append("swap")
            return target + 1

        def fake_restore(swapped, _face, _strength=0.5, _on_log=core._noop):
            order.append("restore")
            return swapped + 10

        def fake_mouth(swapped, _original, _face, _placed_face):
            order.append("mouth")
            return swapped + 100

        engine._hyperswap_face = fake_hyperswap
        engine._restore_face = fake_restore
        engine._preserve_inner_mouth = staticmethod(fake_mouth).__func__
        result, _, _ = engine.swap_array(
            np.zeros((64, 64, 3), dtype=np.uint8),
            source_face=face,
            quality="best",
            preserve_mouth=True,
        )
        self.assertEqual(order, ["swap", "restore", "mouth"])
        self.assertTrue(np.all(result == 111))

    def test_careful_quality_never_loads_the_restoration_model(self):
        face = SimpleNamespace(bbox=np.array([0, 0, 20, 20]))
        engine = core.SwapEngine()
        engine.analyser = FakeAnalyser([face])
        engine._hyperswap_face = lambda target, *_: target + 1
        engine._restore_face = lambda *_: self.fail("Careful must not restore")
        engine.swap_array(
            np.zeros((64, 64, 3), dtype=np.uint8),
            source_face=face,
            quality="careful",
            preserve_mouth=False,
        )
        self.assertIsNone(engine.enhancer)

    def test_restoration_retires_a_gpu_that_returns_invalid_pixels(self):
        class FakeSession:
            def __init__(self, provider, output):
                self.provider = provider
                self.output = output

            def get_providers(self):
                return [self.provider]

            def run(self, _outputs, _feed):
                return [self.output]

        engine = core.SwapEngine()
        engine.accelerated_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        engine.enhancer = FakeSession(
            "CUDAExecutionProvider", np.full((1, 3, 4, 4), np.nan, dtype=np.float32)
        )
        engine._ensure_enhancer = lambda on_log=core._noop: FakeSession(
            "CPUExecutionProvider", np.zeros((1, 3, 4, 4), dtype=np.float32)
        )
        messages: list[str] = []
        result = engine._run_enhancer(
            engine.enhancer, np.zeros((3, 4, 4), dtype=np.float32), messages.append
        )

        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(engine.enhancer_cpu_only)
        self.assertTrue(any("CPU" in message for message in messages))

    def test_restoration_failing_on_cpu_is_reported_not_silently_saved(self):
        class FakeSession:
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _outputs, _feed):
                return [np.full((1, 3, 4, 4), np.nan, dtype=np.float32)]

        engine = core.SwapEngine()
        with self.assertRaises(RuntimeError):
            engine._run_enhancer(FakeSession(), np.zeros((3, 4, 4), dtype=np.float32))

    def test_restoration_model_is_required_for_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIn(core.ENHANCER_MODEL, core.missing_models(Path(temp)))

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

    def test_phase_fusion_removes_alternating_prediction_bias(self):
        frames = [np.full((8, 8, 3), value, np.float32) for value in (20, 80, 80, 20)]
        interleaved = core.SwapEngine._explode_pixels(frames, 8, 16)
        self.assertGreater(float(interleaved.std()), 20)
        fused = core.SwapEngine._fuse_predictions(frames, 8, 16)
        np.testing.assert_array_equal(fused, np.full((16, 16, 3), 50, np.uint8))

    def test_phase_fusion_registers_sample_coordinates(self):
        yy, xx = np.mgrid[:32, :32].astype(np.float32)
        image = np.stack([xx * 3, yy * 3, (xx + yy)], axis=2)
        frames = list(core.SwapEngine._implode_pixels(image, 8))
        result = core.SwapEngine._fuse_predictions(frames, 8, 32)
        np.testing.assert_allclose(result[8:-8, 8:-8], image[8:-8, 8:-8], atol=1)

    def test_phase_fusion_rejects_incomplete_or_nonfinite_predictions(self):
        frames = [np.ones((8, 8, 3), np.float32) for _ in range(4)]
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            core.SwapEngine._fuse_predictions(frames[:-1], 8, 16)
        frames[2][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'invalid sample'):
            core.SwapEngine._fuse_predictions(frames, 8, 16)

    def test_preserve_inner_mouth_restores_only_landmark_polygon(self):
        original = np.full((48, 48, 3), 20, dtype=np.uint8)
        swapped = np.full((48, 48, 3), 200, dtype=np.uint8)
        landmarks = np.zeros((106, 2), dtype=np.float32)
        mouth = np.array(
            [
                [10, 22],
                [15, 19],
                [20, 18],
                [25, 19],
                [30, 22],
                [25, 27],
                [20, 29],
                [15, 27],
            ],
            dtype=np.float32,
        )
        landmarks[core.INNER_MOUTH_106_INDICES] = mouth
        face = SimpleNamespace(landmark_2d_106=landmarks)

        result = core.SwapEngine._preserve_inner_mouth(swapped, original, face, face)

        self.assertLess(int(result[23, 20, 0]), 30)
        self.assertEqual(int(result[2, 2, 0]), 200)

    def test_changing_eye_restoration_reruns_a_completed_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            target = root / "target.png"
            Image.new("RGB", (8, 8)).save(source)
            Image.new("RGB", (8, 8)).save(target)
            common = dict(
                all_faces=False,
                quality="best",
                output_format="png",
                character_name="",
                preserve_mouth=True,
                restoration_strength=0.5,
            )
            self.assertNotEqual(
                core.processing_key(source, target, destination_eyes=False, **common),
                core.processing_key(source, target, destination_eyes=True, **common),
            )

    def test_changing_restoration_strength_reruns_a_completed_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            target = root / "target.png"
            Image.new("RGB", (8, 8)).save(source)
            Image.new("RGB", (8, 8)).save(target)
            common = dict(
                all_faces=False,
                quality="best",
                output_format="png",
                character_name="",
                preserve_mouth=True,
                destination_eyes=True,
            )
            self.assertNotEqual(
                core.processing_key(source, target, restoration_strength=0.5, **common),
                core.processing_key(source, target, restoration_strength=0.2, **common),
            )

    def test_destination_eyes_is_a_no_op_without_a_landmark_model(self):
        engine = core.SwapEngine()
        engine.analyser = SimpleNamespace(models={})
        swapped = np.full((32, 32, 3), 7, dtype=np.uint8)
        np.testing.assert_array_equal(
            engine._destination_eyes(swapped, swapped * 0, SimpleNamespace(), None), swapped
        )

    def test_preserve_inner_mouth_ignores_closed_lips(self):
        original = np.zeros((40, 40, 3), dtype=np.uint8)
        swapped = np.full((40, 40, 3), 200, dtype=np.uint8)
        landmarks = np.zeros((106, 2), dtype=np.float32)
        landmarks[core.INNER_MOUTH_106_INDICES] = np.array(
            [[10, 20], [15, 20], [20, 20], [25, 20], [30, 20], [25, 20], [20, 20], [15, 20]],
            dtype=np.float32,
        )
        face = SimpleNamespace(landmark_2d_106=landmarks)

        result = core.SwapEngine._preserve_inner_mouth(swapped, original, face, face)

        np.testing.assert_array_equal(result, swapped)


if __name__ == "__main__":
    unittest.main()
