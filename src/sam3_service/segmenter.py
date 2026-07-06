from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Protocol

from .errors import JobCancelled, ServiceError

_SAM3_SESSION_OPTIONS = {
    "async_loading_frames",
    "offload_state_to_cpu",
    "offload_video_to_cpu",
    "video_loader_type",
}


def _patch_sam3_init_state(predictor: Any) -> None:
    """Keep the upstream session wrapper compatible with its concrete model."""
    original = predictor.model.init_state
    parameters = inspect.signature(original).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return

    unsupported = _SAM3_SESSION_OPTIONS.difference(parameters)
    if not unsupported:
        return

    @wraps(original)
    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        for option in unsupported:
            kwargs.pop(option, None)
        return original(*args, **kwargs)

    predictor.model.init_state = compatible_init_state


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: int
    prompt_id: str
    instance_id: str
    box_xywh: list[float]
    score: float | None
    segmentation: dict[str, Any]


class VideoSegmenter(Protocol):
    model_name: str

    def segment(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        prompt_id: str,
        prompt: str,
        score_threshold: float,
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[FrameResult]: ...


class MockSegmenter:
    model_name = "mock-moving-paddle-v1"

    def segment(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        prompt_id: str,
        prompt: str,
        score_threshold: float,
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[FrameResult]:
        del video_path, prompt, score_threshold
        width = int(metadata["width"])
        height = int(metadata["height"])
        fps = float(metadata["fps"])
        total = int(metadata["frame_count"])
        for frame_index in range(total):
            if cancelled():
                raise JobCancelled()
            phase = frame_index / max(total - 1, 1)
            center_x = width * (0.2 + 0.6 * phase)
            center_y = height * (0.52 + 0.12 * math.sin(phase * math.tau))
            shaft_length = min(width, height) * 0.42
            shaft_width = max(8.0, min(width, height) * 0.025)
            angle = -0.65 + 1.3 * phase
            dx = math.cos(angle) * shaft_length / 2
            dy = math.sin(angle) * shaft_length / 2
            px = -math.sin(angle) * shaft_width / 2
            py = math.cos(angle) * shaft_width / 2
            polygon = [
                [center_x - dx + px, center_y - dy + py],
                [center_x + dx + px, center_y + dy + py],
                [center_x + dx - px, center_y + dy - py],
                [center_x - dx - px, center_y - dy - py],
            ]
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            if frame_index % max(1, round(fps / 2)) == 0:
                progress(frame_index + 1, total)
            yield FrameResult(
                frame_index=frame_index,
                timestamp_ms=round(frame_index * 1000 / fps),
                prompt_id=prompt_id,
                instance_id=f"{prompt_id}:1",
                box_xywh=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                score=0.96,
                segmentation={"type": "polygon", "points": polygon},
            )
        progress(total, total)


class Sam3Segmenter:
    model_name = "sam3.1-object-multiplex"

    def __init__(self, checkpoint_path: Path | None = None, *, offline: bool = False) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise ServiceError(
                "MODEL_UNAVAILABLE",
                f"SAM 3.1 checkpoint does not exist: {checkpoint_path}",
                status_code=503,
            )
        if offline and checkpoint_path is None:
            raise ServiceError(
                "MODEL_UNAVAILABLE",
                "SAM3_OFFLINE requires SAM3_CHECKPOINT_PATH to reference a local checkpoint.",
                status_code=503,
            )
        try:
            import torch
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "sam3" or exc.name.startswith("sam3.")):
                message = "SAM 3.1 is not installed in the worker environment."
            else:
                message = (
                    "A SAM 3.1 runtime dependency is missing from the worker "
                    f"environment: {exc.name or exc}."
                )
            raise ServiceError(
                "MODEL_UNAVAILABLE",
                message,
                status_code=503,
            ) from exc
        except ImportError as exc:
            raise ServiceError(
                "MODEL_UNAVAILABLE",
                f"SAM 3.1 could not be imported: {exc}.",
                status_code=503,
            ) from exc
        if not torch.cuda.is_available():
            raise ServiceError(
                "MODEL_UNAVAILABLE", "CUDA is unavailable to the worker.", status_code=503
            )
        if not torch.cuda.is_bf16_supported() and os.getenv(
            "SAM3_ALLOW_UNSUPPORTED_BF16", "0"
        ) != "1":
            raise ServiceError(
                "MODEL_UNSUPPORTED_GPU",
                "Current upstream SAM 3.1 video code uses BF16 autocast, but this GPU "
                "does not report BF16 support. Validate an FP16 compatibility patch "
                "before enabling SAM3_ALLOW_UNSUPPORTED_BF16.",
                status_code=503,
            )
        self.torch = torch
        self.offload_video_to_cpu = os.getenv(
            "SAM3_OFFLOAD_VIDEO_TO_CPU", "1"
        ).lower() in {"1", "true", "yes"}
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            use_fa3=False,
            compile=False,
        )
        _patch_sam3_init_state(self.predictor)

    def segment(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        prompt_id: str,
        prompt: str,
        score_threshold: float,
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[FrameResult]:
        session_id: str | None = None
        total = int(metadata["frame_count"])
        fps = float(metadata["fps"])
        try:
            response = self.predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": self.offload_video_to_cpu,
                }
            )
            session_id = response["session_id"]
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "output_prob_thresh": score_threshold,
                }
            )
            for response in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "output_prob_thresh": score_threshold,
                }
            ):
                if cancelled():
                    self.predictor.handle_request(
                        {"type": "cancel_propagation", "session_id": session_id}
                    )
                    raise JobCancelled()
                frame_index = int(response["frame_index"])
                outputs = response["outputs"]
                object_ids = _as_list(outputs.get("out_obj_ids"))
                boxes = _as_list(outputs.get("out_boxes_xywh"))
                scores = _as_list(outputs.get("out_probs"))
                masks = outputs.get("out_binary_masks")
                for index, object_id in enumerate(object_ids):
                    mask = masks[index]
                    box = [float(value) for value in boxes[index]]
                    if box and max(abs(value) for value in box) <= 2:
                        box = [
                            box[0] * float(metadata["width"]),
                            box[1] * float(metadata["height"]),
                            box[2] * float(metadata["width"]),
                            box[3] * float(metadata["height"]),
                        ]
                    yield FrameResult(
                        frame_index=frame_index,
                        timestamp_ms=round(frame_index * 1000 / fps),
                        prompt_id=prompt_id,
                        instance_id=f"{prompt_id}:{int(object_id)}",
                        box_xywh=box,
                        score=float(scores[index]) if index < len(scores) else None,
                        segmentation=_encode_uncompressed_rle(mask),
                    )
                progress(min(frame_index + 1, total), total)
        except self.torch.OutOfMemoryError as exc:
            raise ServiceError(
                "GPU_OUT_OF_MEMORY",
                "The GPU could not fit this video even with video frames offloaded to CPU. "
                "Try a shorter video; if the problem persists, lower the inference resolution.",
                retryable=False,
            ) from exc
        finally:
            if session_id is not None:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            self.torch.cuda.empty_cache()


def create_segmenter(
    name: str,
    checkpoint_path: Path | None = None,
    *,
    offline: bool = False,
) -> VideoSegmenter:
    if name == "mock":
        return MockSegmenter()
    if name in {"sam3", "sam3.1"}:
        return Sam3Segmenter(checkpoint_path, offline=offline)
    raise ServiceError("INVALID_CONFIGURATION", f"Unknown segmenter: {name}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _encode_uncompressed_rle(mask: Any) -> dict[str, Any]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    height, width = mask.shape[-2:]
    counts: list[int] = []
    current = 0
    run = 0
    # COCO RLE uses column-major order.
    for x in range(width):
        for y in range(height):
            value = 1 if bool(mask[y][x]) else 0
            if value == current:
                run += 1
            else:
                counts.append(run)
                run = 1
                current = value
    counts.append(run)
    return {"type": "rle", "size": [height, width], "counts": counts}
