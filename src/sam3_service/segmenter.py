from __future__ import annotations

import inspect
import math
import os
import random
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
_DEFAULT_CENTERLINE_THICKNESS_PIXELS = 8.0


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


def _configure_sam3_batches(
    predictor: Any,
    *,
    grounding_batch_size: int,
    postprocess_batch_size: int,
) -> None:
    predictor.model.use_batched_grounding = grounding_batch_size > 1
    predictor.model.batched_grounding_batch_size = grounding_batch_size
    predictor.model.postprocess_batch_size = postprocess_batch_size


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: int
    prompt_id: str
    instance_id: str
    box_xywh: list[float]
    score: float | None
    segmentation: dict[str, Any]
    centerline_segmentation: dict[str, Any] | None = None
    centerline_box_xywh: list[float] | None = None
    centerline_line_xyxy: list[float] | None = None


@dataclass(frozen=True)
class CenterlineMask:
    segmentation: dict[str, Any]
    box_xywh: list[float]
    line_xyxy: list[float]


@dataclass(frozen=True)
class DetectionSettings:
    redetect_interval_frames: int
    max_detections_per_frame: int
    dedupe_iou_threshold: float


class VideoSegmenter(Protocol):
    model_name: str

    def segment(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        prompt_id: str,
        prompt: str,
        score_threshold: float,
        job_settings: dict[str, Any],
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
        job_settings: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[FrameResult]:
        del video_path, prompt, score_threshold, job_settings
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
                centerline_segmentation={"type": "polygon", "points": polygon},
                centerline_box_xywh=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                centerline_line_xyxy=[
                    center_x - dx,
                    center_y - dy,
                    center_x + dx,
                    center_y + dy,
                ],
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
            from sam3.model import decoder as sam3_decoder
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
        bf16_supported = torch.cuda.is_bf16_supported()
        allow_legacy_gpu = os.getenv("SAM3_ALLOW_UNSUPPORTED_BF16", "0") == "1"
        if not bf16_supported and not allow_legacy_gpu:
            raise ServiceError(
                "MODEL_UNSUPPORTED_GPU",
                "Current upstream SAM 3.1 video code uses BF16 autocast, but this GPU "
                "does not report BF16 support. Validate an FP16 compatibility patch "
                "before enabling SAM3_ALLOW_UNSUPPORTED_BF16.",
                status_code=503,
            )
        if not bf16_supported:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def legacy_gpu_sdpa_kernel(_backend: Any):
                return sdpa_kernel(SDPBackend.MATH)

            sam3_decoder.sdpa_kernel = legacy_gpu_sdpa_kernel
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        self.torch = torch
        self.offload_video_to_cpu = os.getenv(
            "SAM3_OFFLOAD_VIDEO_TO_CPU", "1"
        ).lower() in {"1", "true", "yes"}
        max_tracked_objects = max(1, int(os.getenv("SAM3_MAX_TRACKED_OBJECTS", "16")))
        grounding_batch_size = max(1, int(os.getenv("SAM3_GROUNDING_BATCH_SIZE", "1")))
        postprocess_batch_size = max(1, int(os.getenv("SAM3_POSTPROCESS_BATCH_SIZE", "1")))
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            max_num_objects=max_tracked_objects,
            use_fa3=False,
            compile=False,
        )
        _patch_sam3_init_state(self.predictor)
        _configure_sam3_batches(
            self.predictor,
            grounding_batch_size=grounding_batch_size,
            postprocess_batch_size=postprocess_batch_size,
        )

    def segment(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        prompt_id: str,
        prompt: str,
        score_threshold: float,
        job_settings: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[FrameResult]:
        session_id: str | None = None
        total = int(metadata["frame_count"])
        fps = float(metadata["fps"])
        detection_settings = _detection_settings(job_settings)
        try:
            response = self.predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": self.offload_video_to_cpu,
                }
            )
            session_id = response["session_id"]
            emitted_prompt_frames: set[int] = set()
            for frame_index in _redetect_anchor_frames(total, detection_settings):
                if cancelled():
                    raise JobCancelled()
                prompt_response = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": prompt,
                        "output_prob_thresh": score_threshold,
                    }
                )
                prompt_results = _frame_results_from_response(
                    prompt_response,
                    metadata,
                    prompt_id,
                    frame_index,
                    fps,
                    detection_settings,
                )
                if prompt_results:
                    emitted_prompt_frames.add(frame_index)
                    yield from prompt_results
                progress(min(frame_index + 1, total), total)
            if len(emitted_prompt_frames) == total:
                progress(total, total)
                return
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
                if frame_index in emitted_prompt_frames:
                    progress(min(frame_index + 1, total), total)
                    continue
                yield from _frame_results_from_response(
                    response,
                    metadata,
                    prompt_id,
                    frame_index,
                    fps,
                    detection_settings,
                )
                progress(min(frame_index + 1, total), total)
        except self.torch.OutOfMemoryError as exc:
            raise ServiceError(
                "GPU_OUT_OF_MEMORY",
                "The GPU could not fit this video in minimum-memory mode. Try a shorter "
                "video; this SAM 3.1 multiplex build may require a GPU with more VRAM.",
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


def _detection_settings(job_settings: dict[str, Any]) -> DetectionSettings:
    return DetectionSettings(
        redetect_interval_frames=max(0, int(job_settings.get("redetect_interval_frames", 0))),
        max_detections_per_frame=max(1, int(job_settings.get("max_detections_per_frame", 13))),
        dedupe_iou_threshold=max(
            0.0, min(1.0, float(job_settings.get("dedupe_iou_threshold", 0.6)))
        ),
    )


def _redetect_anchor_frames(total: int, settings: DetectionSettings) -> list[int]:
    if total <= 0:
        return [0]
    if settings.redetect_interval_frames <= 0:
        return [0]
    anchors = list(range(0, total, settings.redetect_interval_frames))
    if not anchors:
        return [0]
    return anchors


def _frame_results_from_response(
    response: dict[str, Any],
    metadata: dict[str, Any],
    prompt_id: str,
    fallback_frame_index: int,
    fps: float,
    settings: DetectionSettings,
) -> list[FrameResult]:
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        return []

    frame_index = int(response.get("frame_index", fallback_frame_index))
    object_ids = _as_list(outputs.get("out_obj_ids"))
    boxes = _as_list(outputs.get("out_boxes_xywh"))
    scores = _as_list(outputs.get("out_probs"))
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return []

    results: list[FrameResult] = []
    for index, object_id in enumerate(object_ids):
        if index >= len(masks):
            continue
        mask = masks[index]
        centerline = _fit_centerline_mask(mask)
        box = _scaled_box(boxes[index], metadata) if index < len(boxes) else _box_from_any_mask(mask)
        if box is None:
            continue
        results.append(
            FrameResult(
                frame_index=frame_index,
                timestamp_ms=round(frame_index * 1000 / fps),
                prompt_id=prompt_id,
                instance_id=f"{prompt_id}:{int(object_id)}",
                box_xywh=box,
                score=float(scores[index]) if index < len(scores) else None,
                segmentation=_encode_uncompressed_rle(mask),
                centerline_segmentation=centerline.segmentation if centerline else None,
                centerline_box_xywh=centerline.box_xywh if centerline else None,
                centerline_line_xyxy=centerline.line_xyxy if centerline else None,
            )
        )
    return _dedupe_frame_results(results, settings)


def _scaled_box(raw_box: Any, metadata: dict[str, Any]) -> list[float] | None:
    box = [float(value) for value in raw_box]
    if len(box) != 4:
        return None
    if max(abs(value) for value in box) <= 2:
        return [
            box[0] * float(metadata["width"]),
            box[1] * float(metadata["height"]),
            box[2] * float(metadata["width"]),
            box[3] * float(metadata["height"]),
        ]
    return box


def _box_from_any_mask(mask: Any) -> list[float] | None:
    mask, height, width = _coerce_2d_mask(mask)
    rows = [[bool(mask[y][x]) for x in range(width)] for y in range(height)]
    return _box_from_mask(rows)


def _dedupe_frame_results(
    results: list[FrameResult], settings: DetectionSettings
) -> list[FrameResult]:
    ordered = sorted(
        results,
        key=lambda result: (
            result.score if result.score is not None else 0.0,
            _box_area(result.box_xywh),
        ),
        reverse=True,
    )
    kept: list[FrameResult] = []
    for result in ordered:
        if any(
            _box_iou(result.box_xywh, existing.box_xywh) > settings.dedupe_iou_threshold
            for existing in kept
        ):
            continue
        kept.append(result)
        if len(kept) >= settings.max_detections_per_frame:
            break
    return sorted(kept, key=lambda result: result.instance_id)


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3]) if len(box) == 4 else 0.0


def _box_iou(first: list[float], second: list[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _encode_uncompressed_rle(mask: Any) -> dict[str, Any]:
    mask, height, width = _coerce_2d_mask(mask)
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


def _fit_centerline_mask(mask: Any) -> CenterlineMask | None:
    mask, height, width = _coerce_2d_mask(mask)
    points = _mask_points(mask, height, width)
    if len(points) < 8:
        return None

    line = _ransac_line(points, width, height) or _principal_line(points)
    if line is None:
        return None

    threshold = _initial_centerline_threshold(points, width, height)
    inliers = [point for point in points if _line_distance(point, line) <= threshold]
    if len(inliers) >= max(8, len(points) // 20):
        refined = _principal_line(inliers)
        if refined is not None:
            line = refined
    else:
        inliers = points

    inlier_projections = sorted(_line_projection(point, line) for point in inliers)
    if len(inlier_projections) < 2:
        return None

    # Fit direction from long, thin inliers, but keep the output length equal to
    # the full detected object extent along that direction: paddle blade-tip to
    # blade-tip, or boat head to tail.
    object_projections = [_line_projection(point, line) for point in points]
    start = min(object_projections)
    end = max(object_projections)
    if end - start < 2:
        return None

    half_width = _fixed_centerline_half_width()

    rows = _line_band_mask(height, width, line, start, end, half_width)
    box = _box_from_mask(rows)
    if box is None:
        return None
    center_x, center_y, unit_x, unit_y = line
    line_xyxy = [
        center_x + start * unit_x,
        center_y + start * unit_y,
        center_x + end * unit_x,
        center_y + end * unit_y,
    ]
    return CenterlineMask(
        segmentation=_encode_uncompressed_rle(rows),
        box_xywh=box,
        line_xyxy=line_xyxy,
    )


def _coerce_2d_mask(mask: Any) -> tuple[Any, int, int]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()

    shape = getattr(mask, "shape", None)
    if shape is not None:
        shape_values = [int(value) for value in shape]
        while len(shape_values) > 2:
            mask = mask[0]
            shape_values = shape_values[1:]
        return mask, shape_values[-2], shape_values[-1]

    height = len(mask)
    width = len(mask[0]) if height else 0
    return mask, height, width


def _mask_points(mask: Any, height: int, width: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for y in range(height):
        row = mask[y]
        for x in range(width):
            if bool(row[x]):
                points.append((float(x), float(y)))
    return points


def _initial_centerline_threshold(
    points: list[tuple[float, float]], width: int, height: int
) -> float:
    area_scale = math.sqrt(len(points)) * 0.08
    frame_scale = min(width, height) * 0.025
    return max(3.0, min(16.0, max(area_scale, frame_scale)))


def _fixed_centerline_half_width() -> float:
    raw = os.getenv(
        "SAM3_CENTERLINE_THICKNESS_PIXELS",
        os.getenv("SAM3_SHAFT_THICKNESS_PIXELS", str(_DEFAULT_CENTERLINE_THICKNESS_PIXELS)),
    )
    try:
        thickness = float(raw)
    except ValueError:
        thickness = _DEFAULT_CENTERLINE_THICKNESS_PIXELS
    thickness = max(2.0, min(128.0, thickness))
    return thickness / 2.0


def _ransac_line(
    points: list[tuple[float, float]], width: int, height: int
) -> tuple[float, float, float, float] | None:
    sample = _subsample(points, 3000)
    if len(sample) < 2:
        return None

    threshold = _initial_centerline_threshold(points, width, height)
    rng = random.Random(17)
    best_line: tuple[float, float, float, float] | None = None
    best_score = -1.0
    iterations = min(160, max(32, len(sample) // 6))

    for _ in range(iterations):
        first, second = rng.sample(sample, 2)
        line = _line_from_points(first, second)
        if line is None:
            continue
        projections: list[float] = []
        for point in sample:
            if _line_distance(point, line) <= threshold:
                projections.append(_line_projection(point, line))
        if len(projections) < 2:
            continue
        projections.sort()
        span = _percentile(projections, 95) - _percentile(projections, 5)
        score = len(projections) * max(span, 1.0)
        if score > best_score:
            best_score = score
            best_line = line

    return best_line


def _subsample(points: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    return points[::step]


def _principal_line(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    xx = sum((point[0] - center_x) ** 2 for point in points)
    yy = sum((point[1] - center_y) ** 2 for point in points)
    xy = sum((point[0] - center_x) * (point[1] - center_y) for point in points)
    if xx == 0 and yy == 0:
        return None
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    return center_x, center_y, math.cos(angle), math.sin(angle)


def _line_from_points(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float, float, float] | None:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return None
    return first[0], first[1], dx / length, dy / length


def _line_projection(
    point: tuple[float, float], line: tuple[float, float, float, float]
) -> float:
    center_x, center_y, unit_x, unit_y = line
    return (point[0] - center_x) * unit_x + (point[1] - center_y) * unit_y


def _line_distance(
    point: tuple[float, float], line: tuple[float, float, float, float]
) -> float:
    center_x, center_y, unit_x, unit_y = line
    return abs((point[0] - center_x) * unit_y - (point[1] - center_y) * unit_x)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _line_band_mask(
    height: int,
    width: int,
    line: tuple[float, float, float, float],
    start: float,
    end: float,
    half_width: float,
) -> list[list[bool]]:
    center_x, center_y, unit_x, unit_y = line
    rows: list[list[bool]] = []
    half_width_squared = half_width * half_width
    for y in range(height):
        row: list[bool] = []
        for x in range(width):
            projection = (x - center_x) * unit_x + (y - center_y) * unit_y
            distance = (x - center_x) * unit_y - (y - center_y) * unit_x
            row.append(start <= projection <= end and distance * distance <= half_width_squared)
        rows.append(row)
    return rows


def _box_from_mask(mask: list[list[bool]]) -> list[float] | None:
    min_x: int | None = None
    min_y: int | None = None
    max_x: int | None = None
    max_y: int | None = None
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if not value:
                continue
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = x if max_x is None else max(max_x, x)
            max_y = y if max_y is None else max(max_y, y)
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return [float(min_x), float(min_y), float(max_x - min_x + 1), float(max_y - min_y + 1)]
