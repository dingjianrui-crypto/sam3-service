from __future__ import annotations

import unittest
from pathlib import Path

from sam3_service.errors import ServiceError
from sam3_service.segmenter import (
    Sam3Segmenter,
    _configure_sam3_batches,
    _patch_sam3_init_state,
)


class _ModelWithoutOffloadOptions:
    def __init__(self) -> None:
        self.resource_path = ""

    def init_state(self, resource_path: str) -> dict[str, str]:
        self.resource_path = resource_path
        return {"resource_path": resource_path}


class _ModelWithOffloadOptions:
    def __init__(self) -> None:
        self.offload_state_to_cpu = False

    def init_state(
        self,
        resource_path: str,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        self.offload_state_to_cpu = offload_state_to_cpu
        return {"resource_path": resource_path}


class _Predictor:
    def __init__(self, model: object) -> None:
        self.model = model


class _Cuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _Torch:
    class OutOfMemoryError(Exception):
        pass

    def __init__(self) -> None:
        self.cuda = _Cuda()


class _SessionPredictor:
    def __init__(self, *, out_of_memory: bool = False) -> None:
        self.out_of_memory = out_of_memory
        self.requests: list[dict[str, object]] = []

    def handle_request(self, request: dict[str, object]) -> dict[str, str]:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session"}
        if request["type"] == "add_prompt" and self.out_of_memory:
            raise _Torch.OutOfMemoryError("out of memory")
        return {"is_success": "true"}

    def handle_stream_request(self, request: dict[str, object]):
        self.requests.append(request)
        return iter(())


class Sam3CompatibilityTest(unittest.TestCase):
    def test_removes_session_options_rejected_by_model(self) -> None:
        model = _ModelWithoutOffloadOptions()
        predictor = _Predictor(model)
        _patch_sam3_init_state(predictor)

        state = predictor.model.init_state(
            resource_path="/tmp/video.mp4",
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )

        self.assertEqual(state, {"resource_path": "/tmp/video.mp4"})
        self.assertEqual(model.resource_path, "/tmp/video.mp4")

    def test_preserves_session_options_supported_by_model(self) -> None:
        model = _ModelWithOffloadOptions()
        predictor = _Predictor(model)
        _patch_sam3_init_state(predictor)

        predictor.model.init_state(
            resource_path="/tmp/video.mp4",
            offload_state_to_cpu=True,
        )

        self.assertTrue(model.offload_state_to_cpu)

    def test_configures_single_frame_memory_mode(self) -> None:
        model = type("Model", (), {})()
        predictor = _Predictor(model)

        _configure_sam3_batches(
            predictor,
            grounding_batch_size=1,
            postprocess_batch_size=1,
        )

        self.assertFalse(model.use_batched_grounding)
        self.assertEqual(model.batched_grounding_batch_size, 1)
        self.assertEqual(model.postprocess_batch_size, 1)

    def test_offloads_video_frames_and_closes_session(self) -> None:
        segmenter = Sam3Segmenter.__new__(Sam3Segmenter)
        segmenter.torch = _Torch()
        segmenter.predictor = _SessionPredictor()
        segmenter.offload_video_to_cpu = True

        frames = segmenter.segment(
            Path("/tmp/video.mp4"),
            {"frame_count": 10, "fps": 10},
            "prompt",
            "paddle",
            0.5,
            lambda _done, _total: None,
            lambda: False,
        )

        self.assertEqual(list(frames), [])
        self.assertTrue(segmenter.predictor.requests[0]["offload_video_to_cpu"])
        self.assertEqual(segmenter.predictor.requests[-1]["type"], "close_session")
        self.assertEqual(segmenter.torch.cuda.empty_cache_calls, 1)

    def test_reports_gpu_out_of_memory_and_recovers_cache(self) -> None:
        segmenter = Sam3Segmenter.__new__(Sam3Segmenter)
        segmenter.torch = _Torch()
        segmenter.predictor = _SessionPredictor(out_of_memory=True)
        segmenter.offload_video_to_cpu = True

        frames = segmenter.segment(
            Path("/tmp/video.mp4"),
            {"frame_count": 10, "fps": 10},
            "prompt",
            "paddle",
            0.5,
            lambda _done, _total: None,
            lambda: False,
        )

        with self.assertRaisesRegex(ServiceError, "shorter video") as raised:
            list(frames)

        self.assertEqual(raised.exception.code, "GPU_OUT_OF_MEMORY")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(segmenter.predictor.requests[-1]["type"], "close_session")
        self.assertEqual(segmenter.torch.cuda.empty_cache_calls, 1)


if __name__ == "__main__":
    unittest.main()
