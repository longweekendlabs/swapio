import unittest
from unittest.mock import patch
import numpy as np
import core


class Session:
    def __init__(self, provider, pixels=None, error=None):
        self.provider = provider
        self.pixels = pixels
        self.error = error
        self.calls = 0

    def get_providers(self):
        return [self.provider]

    def run(self, _outputs, _feed):
        self.calls += 1
        if self.error:
            raise self.error
        return [self.pixels[None]]


class CudaQualityTests(unittest.TestCase):
    def test_identity_cuda_allocation_failure_retries_same_face_and_reuses_cpu(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        engine = core.SwapEngine()
        session = Mock()
        session.get_providers.return_value = ["CUDAExecutionProvider"]
        session.set_providers.side_effect = lambda providers: setattr(
            session.get_providers, "return_value", providers)
        engine.swapper = SimpleNamespace(session=session)
        target, selected_face, source = object(), object(), object()
        calls = []
        def operation():
            calls.append((target, selected_face, source))
            if session.get_providers()[0] == "CUDAExecutionProvider":
                raise RuntimeError("Conv_612 Failed to allocate memory for requested buffer of size 270607872")
            return "swapped"
        logs = []
        for _ in range(2):
            self.assertEqual(engine._identity_cpu_retry(operation, logs.append), "swapped")
        self.assertEqual(calls, [(target, selected_face, source)] * 3)
        session.set_providers.assert_called_once_with(["CPUExecutionProvider"])
        self.assertEqual(len(logs), 1)
        self.assertIn("CPU", logs[0])

    def test_identity_cpu_failure_propagates_without_retries(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        engine = core.SwapEngine()
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        engine.swapper = SimpleNamespace(session=session)
        operation = Mock(side_effect=RuntimeError("CPU inference failed"))
        with self.assertRaisesRegex(RuntimeError, "CPU inference failed"):
            engine._identity_cpu_retry(operation, lambda _: None)
        session.set_providers.assert_not_called()
        operation.assert_called_once()

    def test_valid_cuda_output_does_not_reload_or_use_cpu(self):
        engine = core.SwapEngine()
        pixels = np.zeros((3, 256, 256), np.float32)
        engine.hyperswap = Session("CUDAExecutionProvider", pixels)
        with patch("onnxruntime.InferenceSession") as create:
            result = engine._run_hyperswap({})
            np.testing.assert_array_equal(result, pixels)
            self.assertTrue(np.shares_memory(result, pixels))
            create.assert_not_called()
        self.assertFalse(engine.hyperswap_cpu_only)

    def test_nonfinite_cuda_retries_once_and_keeps_cpu_for_later_samples(self):
        engine = core.SwapEngine()
        gpu = Session("CUDAExecutionProvider", np.full((3, 256, 256), np.nan))
        cpu = Session("CPUExecutionProvider", np.zeros((3, 256, 256), np.float32))
        engine.hyperswap = gpu
        messages = []
        with patch("onnxruntime.InferenceSession", return_value=cpu) as create:
            for _ in range(2):
                self.assertTrue(np.isfinite(engine._run_hyperswap({}, messages.append)).all())
            create.assert_called_once()
            self.assertEqual(create.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual((gpu.calls, cpu.calls), (1, 2))
        self.assertTrue(engine.hyperswap_cpu_only)
        self.assertEqual(engine.provider, "CPU")
        self.assertTrue(any("CUDA" in message and "CPU" in message for message in messages))

    def test_cuda_runtime_failure_uses_valid_cpu_result(self):
        engine = core.SwapEngine()
        engine.hyperswap = Session("CUDAExecutionProvider", error=RuntimeError("device failure"))
        cpu = Session("CPUExecutionProvider", np.ones((3, 256, 256), np.float32))
        with patch("onnxruntime.InferenceSession", return_value=cpu):
            np.testing.assert_array_equal(engine._run_hyperswap({}), cpu.pixels)

    def test_invalid_cpu_output_aborts_without_another_retry(self):
        engine = core.SwapEngine()
        for shape in ((3, 256, 256), (3, 128, 128)):
            engine.hyperswap = Session("CPUExecutionProvider", np.full(shape, np.nan))
            with patch("onnxruntime.InferenceSession") as create:
                with self.assertRaisesRegex(RuntimeError, "invalid pixels"):
                    engine._run_hyperswap({})
                create.assert_not_called()

    def test_invalid_cpu_fallback_is_also_rejected(self):
        engine = core.SwapEngine()
        invalid = np.full((3, 256, 256), np.inf)
        engine.hyperswap = Session("CUDAExecutionProvider", invalid)
        with patch("onnxruntime.InferenceSession", return_value=Session("CPUExecutionProvider", invalid)):
            with self.assertRaisesRegex(RuntimeError, "invalid pixels"):
                engine._run_hyperswap({})


if __name__ == "__main__":
    unittest.main()
