from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from sam3_service.errors import ServiceError
from sam3_service.segmenter import (
    Sam3Segmenter,
    _configure_sam3_batches,
    _fit_centerline_mask,
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

    def test_fits_thick_centerline_mask_from_full_paddle_mask(self) -> None:
        mask = [[False for _ in range(130)] for _ in range(80)]
        for y in range(37, 44):
            for x in range(10, 120):
                mask[y][x] = True
        for y in range(24, 57):
            for x in range(3, 24):
                mask[y][x] = True
            for x in range(106, 127):
                mask[y][x] = True
        for y in range(5, 13):
            for x in range(70, 82):
                mask[y][x] = True

        centerline = _fit_centerline_mask(mask)

        self.assertIsNotNone(centerline)
        assert centerline is not None
        self.assertEqual(centerline.segmentation["type"], "rle")
        self.assertGreaterEqual(centerline.box_xywh[2], 123)
        self.assertLessEqual(centerline.box_xywh[2], 125)
        self.assertLess(centerline.box_xywh[3], 20)
        self.assertEqual(len(centerline.line_xyxy), 4)
        self.assertGreater(centerline.line_xyxy[2] - centerline.line_xyxy[0], 120)

        original_area = sum(1 for row in mask for value in row if value)
        centerline_area = _rle_area(centerline.segmentation)
        self.assertLess(centerline_area, original_area * 0.65)
        self.assertGreater(centerline_area, 500)

    def test_uses_fixed_configured_centerline_thickness(self) -> None:
        mask = [[False for _ in range(80)] for _ in range(50)]
        for y in range(23, 28):
            for x in range(5, 75):
                mask[y][x] = True
        for y in range(12, 39):
            for x in range(5, 16):
                mask[y][x] = True
            for x in range(64, 75):
                mask[y][x] = True

        with patch.dict("os.environ", {"SAM3_CENTERLINE_THICKNESS_PIXELS": "12"}):
            centerline = _fit_centerline_mask(mask)

        self.assertIsNotNone(centerline)
        assert centerline is not None
        self.assertGreaterEqual(centerline.box_xywh[3], 11)
        self.assertLessEqual(centerline.box_xywh[3], 13)
        self.assertEqual(len(centerline.line_xyxy), 4)


def _rle_area(rle: dict[str, object]) -> int:
    area = 0
    value = 0
    for count in rle["counts"]:  # type: ignore[index]
        if value == 1:
            area += int(count)
        value = 1 - value
    return area


if __name__ == "__main__":
    unittest.main()
