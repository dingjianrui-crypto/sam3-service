from __future__ import annotations

import unittest

from sam3_service.segmenter import _patch_sam3_init_state


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


if __name__ == "__main__":
    unittest.main()
