from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import zlib
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ServiceError
from .media import probe_video

Color = tuple[int, int, int, int]
Line = tuple[float, float, float, float]
LabelPosition = str
DEGREE_LABEL_TITLE = "桨叶角度"
SPM_LABEL_TITLE = "桨频"
SPM_INSTANT_LABEL = "瞬时桨频"
SPM_AVERAGE_LABEL = "平均桨频"
LANDSCAPE_METRIC_CENTER_OFFSET_PERCENT = 5.5
PORTRAIT_METRIC_CENTER_OFFSET_PERCENT = 16.0
SPM_MIN_INTERVAL_MS = 500
SPM_MAX_INTERVAL_MS = 3500
SPM_MIN_INTERVALS = 3
SPM_MIN_PROMINENCE_DEGREES = 10


@dataclass(frozen=True)
class ExportOptions:
    angle_label_position: LabelPosition = "top"
    angle_label_font_size: int | None = None
    include_spm: bool = False
    metric_center_offset_percent: float | None = None
    reference_prompt_id: str | None = None
    target_prompt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Centerline:
    record: dict[str, Any]
    line: Line
    color: Color


@dataclass(frozen=True)
class DegreeLabel:
    instance_id: str
    degree: int
    line: Line
    color: Color


@dataclass(frozen=True)
class DegreeLabelEntry:
    text: str
    label: DegreeLabel
    text_color: Color


@dataclass(frozen=True)
class SpmEstimate:
    instantaneous: float | None
    average: float | None


class SpmEstimator:
    def __init__(self, *, window_ms: int = 8000) -> None:
        self.window_ms = window_ms
        self._tracks: dict[int, _SpmTrack] = {}

    def update(self, timestamp_ms: int, labels: list[DegreeLabel]) -> SpmEstimate:
        for index, label in enumerate(labels):
            self._tracks.setdefault(index, _SpmTrack()).update(timestamp_ms, float(label.degree))
        recent_values: list[float] = []
        average_values: list[float] = []
        for track in self._tracks.values():
            recent = track.instantaneous_spm(timestamp_ms, self.window_ms)
            average = track.average_spm()
            if recent is not None:
                recent_values.append(recent)
            if average is not None:
                average_values.append(average)
        return SpmEstimate(
            instantaneous=_mean(recent_values),
            average=_mean(average_values),
        )


class _SpmTrack:
    def __init__(self) -> None:
        self.samples: list[tuple[int, float]] = []
        self.events_ms: list[int] = []
        self._last_slope = 0
        self._last_extreme_ms: int | None = None
        self._last_extreme_degree: float | None = None

    def update(self, timestamp_ms: int, degree: float) -> None:
        if self.samples and timestamp_ms <= self.samples[-1][0]:
            return
        self.samples.append((timestamp_ms, degree))
        if len(self.samples) > 5:
            self.samples = self.samples[-5:]
        if len(self.samples) < 3:
            return
        previous_degree = self.samples[-2][1]
        slope = _sign(degree - previous_degree, epsilon=1.5)
        if slope == 0:
            return
        if self._last_slope and slope != self._last_slope:
            event_ms, event_degree = self.samples[-2]
            self._record_event(event_ms, event_degree)
        self._last_slope = slope

    def _record_event(self, timestamp_ms: int, degree: float) -> None:
        if self._last_extreme_ms is not None:
            interval_ms = timestamp_ms - self._last_extreme_ms
            previous_degree = (
                degree if self._last_extreme_degree is None else self._last_extreme_degree
            )
            prominence = abs(degree - previous_degree)
            if (
                interval_ms < SPM_MIN_INTERVAL_MS
                or interval_ms > SPM_MAX_INTERVAL_MS
                or prominence < SPM_MIN_PROMINENCE_DEGREES
            ):
                return
        self.events_ms.append(timestamp_ms)
        self._last_extreme_ms = timestamp_ms
        self._last_extreme_degree = degree

    def instantaneous_spm(self, timestamp_ms: int, window_ms: int) -> float | None:
        events = [
            event
            for event in self.events_ms
            if timestamp_ms - window_ms <= event <= timestamp_ms
        ]
        return _spm_from_events(events)

    def average_spm(self) -> float | None:
        return _spm_from_events(self.events_ms)


def export_centerline_video(
    *,
    video_path: Path,
    output_path: Path,
    temporary_dir: Path,
    manifest: dict[str, Any],
    chunk_paths: list[Path],
    options: ExportOptions | None = None,
) -> Path:
    if not video_path.is_file():
        raise ServiceError("NOT_FOUND", "Video content is unavailable.", status_code=404)
    if not chunk_paths:
        raise ServiceError("NOT_FOUND", "Result chunks are unavailable.", status_code=404)

    manifest_video = manifest["video"]
    manifest_width = int(manifest_video["width"])
    manifest_height = int(manifest_video["height"])
    manifest_fps = float(manifest_video["fps"] or 30)
    video_metadata = probe_video(video_path)
    width = int(video_metadata["width"] or manifest_width)
    height = int(video_metadata["height"] or manifest_height)
    fps = float(video_metadata["fps"] or manifest_fps or 30)
    frame_count = max(
        int(video_metadata["frame_count"] or 0),
        math.ceil(float(video_metadata["duration_ms"] or 0) * fps / 1000),
    )
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ServiceError("EXPORT_FAILED", "Result manifest has invalid video metadata.")

    export_options = _normalize_export_options(options, manifest, width, height)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    frames_dir = temporary_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".tmp.mp4")

    colors = {
        prompt["id"]: _parse_hex(prompt.get("color", "#35C2FF"))
        for prompt in manifest.get("prompts", [])
    }
    frames = _load_frames_by_timestamp(chunk_paths)
    frame_timestamps = sorted(frames)
    scale_x = width / manifest_width if manifest_width > 0 else 1.0
    scale_y = height / manifest_height if manifest_height > 0 else 1.0
    result_tolerance_ms = max(1000 / max(fps, 1), 500 / max(manifest_fps, 1), 40)
    carried_labels: list[DegreeLabel] = []
    spm_estimator = SpmEstimator()
    for frame_index in range(frame_count):
        image = _transparent_image(width, height)
        timestamp_ms = round(frame_index * 1000 / fps)
        records = _records_for_timestamp(
            frames, frame_timestamps, timestamp_ms, result_tolerance_ms
        )
        scaled_records = [_scale_record(record, scale_x, scale_y) for record in records]
        carried_labels = _draw_frame_overlay(
            image,
            width,
            height,
            scaled_records,
            colors,
            export_options=export_options,
            fallback_labels=carried_labels,
            timestamp_ms=timestamp_ms,
            spm_estimator=spm_estimator,
        )
        _write_png_rgba(frames_dir / f"{frame_index:06d}.png", width, height, image)

    filter_complex = "[0:v][1:v]overlay=0:0:format=auto[ov]"
    filter_complex += ";[ov]null[v]"

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "%06d.png"),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        temporary_output.replace(output_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        temporary_output.unlink(missing_ok=True)
        detail = exc.stderr[-1000:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ServiceError(
            "EXPORT_FAILED",
            f"Video export failed: {detail}",
            retryable=True,
            status_code=500,
        ) from exc
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return output_path


def _load_frames_by_timestamp(chunk_paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("frames", []):
            frames.setdefault(int(record["timestamp_ms"]), []).append(record)
    return frames


def _normalize_export_options(
    options: ExportOptions | None,
    manifest: dict[str, Any],
    width: int,
    height: int,
) -> ExportOptions:
    prompts = manifest.get("prompts", [])
    prompt_ids = tuple(prompt["id"] for prompt in prompts if prompt.get("id"))
    requested = options or ExportOptions()
    reference_prompt_id = requested.reference_prompt_id
    if reference_prompt_id not in prompt_ids:
        reference_prompt_id = _default_reference_prompt_id(prompts)
    target_prompt_ids = tuple(
        prompt_id for prompt_id in requested.target_prompt_ids if prompt_id in prompt_ids
    )
    if not target_prompt_ids:
        target_prompt_ids = _default_target_prompt_ids(prompts, reference_prompt_id)
    position = (
        requested.angle_label_position
        if requested.angle_label_position in {"top", "bottom"}
        else "top"
    )
    default_font_size = max(18, round(min(height * 0.045, width * 0.038)))
    font_size = requested.angle_label_font_size or default_font_size
    metric_center_offset_percent = requested.metric_center_offset_percent
    if metric_center_offset_percent is None:
        metric_center_offset_percent = _default_metric_center_offset_percent(width, height)
    return ExportOptions(
        angle_label_position=position,
        angle_label_font_size=max(12, min(96, int(font_size))),
        include_spm=bool(requested.include_spm),
        metric_center_offset_percent=max(0.0, min(45.0, float(metric_center_offset_percent))),
        reference_prompt_id=reference_prompt_id,
        target_prompt_ids=target_prompt_ids,
    )


def _default_reference_prompt_id(prompts: list[dict[str, Any]]) -> str | None:
    for prompt in prompts:
        if "boat" in str(prompt.get("text", "")).lower():
            return prompt.get("id")
    if len(prompts) > 1:
        return prompts[1].get("id")
    return prompts[0].get("id") if prompts else None


def _default_target_prompt_ids(
    prompts: list[dict[str, Any]], reference_prompt_id: str | None
) -> tuple[str, ...]:
    preferred = [
        prompt["id"]
        for prompt in prompts
        if prompt.get("id") != reference_prompt_id
        and "paddle" in str(prompt.get("text", "")).lower()
    ]
    if preferred:
        return tuple(preferred)
    return tuple(prompt["id"] for prompt in prompts if prompt.get("id") != reference_prompt_id)


def _records_for_timestamp(
    frames: dict[int, list[dict[str, Any]]],
    timestamps: list[int],
    timestamp_ms: int,
    tolerance_ms: float,
) -> list[dict[str, Any]]:
    if not timestamps:
        return []
    insertion_index = bisect_left(timestamps, timestamp_ms)
    candidates = []
    if insertion_index < len(timestamps):
        candidates.append(timestamps[insertion_index])
    if insertion_index > 0:
        candidates.append(timestamps[insertion_index - 1])
    nearest_timestamp = min(candidates, key=lambda value: abs(value - timestamp_ms))
    if abs(nearest_timestamp - timestamp_ms) > tolerance_ms:
        return []
    return frames[nearest_timestamp]


def _scale_record(record: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
    if abs(scale_x - 1) < 1e-6 and abs(scale_y - 1) < 1e-6:
        return record
    scaled = dict(record)
    values = record.get("centerline_line_xyxy")
    if values and len(values) == 4 and _centerline_rle_size(record) is None:
        scaled["centerline_line_xyxy"] = [
            float(values[0]) * scale_x,
            float(values[1]) * scale_y,
            float(values[2]) * scale_x,
            float(values[3]) * scale_y,
        ]
    return scaled


def _draw_frame_overlay(
    image: bytearray,
    width: int,
    height: int,
    records: list[dict[str, Any]],
    colors: dict[str, Color],
    *,
    export_options: ExportOptions,
    fallback_labels: list[DegreeLabel],
    timestamp_ms: int,
    spm_estimator: SpmEstimator,
) -> list[DegreeLabel]:
    centerlines: list[Centerline] = []
    for record in records:
        line = _record_line(record, width, height)
        if line is None:
            continue
        color = colors.get(record["prompt_id"], (53, 194, 255, 255))
        centerlines.append(Centerline(record=record, line=line, color=color))
        _draw_line(image, width, height, line, color, max(3, round(min(width, height) * 0.006)))

    labels = _degree_labels(centerlines, export_options)
    displayed_labels = labels or fallback_labels
    active_entries = _degree_label_entries(labels)
    for entry in active_entries:
        _draw_target_degree_marker(image, width, height, entry)
    if displayed_labels:
        _draw_degree_label_block(image, width, height, displayed_labels, export_options)
    if export_options.include_spm:
        estimate = spm_estimator.update(timestamp_ms, labels)
        _draw_spm_label(image, width, height, estimate, export_options)
    return labels or fallback_labels


def _record_line(record: dict[str, Any], width: int, height: int) -> Line | None:
    values = record.get("centerline_line_xyxy")
    if not values or len(values) != 4:
        return None
    try:
        scale_x, scale_y = _centerline_coordinate_scale(record, width, height)
        line = (
            float(values[0]) * scale_x,
            float(values[1]) * scale_y,
            float(values[2]) * scale_x,
            float(values[3]) * scale_y,
        )
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in line):
        return None
    return line  # type: ignore[return-value]


def _centerline_coordinate_scale(
    record: dict[str, Any],
    width: int,
    height: int,
) -> tuple[float, float]:
    size = _centerline_rle_size(record)
    if size is None:
        return (1.0, 1.0)
    mask_height, mask_width = size
    if mask_width <= 0 or mask_height <= 0:
        return (1.0, 1.0)
    return (width / mask_width, height / mask_height)


def _centerline_rle_size(record: dict[str, Any]) -> tuple[int, int] | None:
    segmentation = record.get("centerline_segmentation")
    if not isinstance(segmentation, dict) or segmentation.get("type") != "rle":
        return None
    size = segmentation.get("size")
    if not isinstance(size, list | tuple) or len(size) != 2:
        return None
    try:
        return (int(size[0]), int(size[1]))
    except (TypeError, ValueError):
        return None


def _degree_labels(centerlines: list[Centerline], options: ExportOptions) -> list[DegreeLabel]:
    if not options.reference_prompt_id or not options.target_prompt_ids:
        return []
    references = [
        centerline
        for centerline in centerlines
        if centerline.record.get("prompt_id") == options.reference_prompt_id
    ]
    if not references:
        return []
    target_prompt_ids = set(options.target_prompt_ids)
    targets = [
        centerline
        for centerline in centerlines
        if centerline.record.get("prompt_id") in target_prompt_ids
    ]
    labels: list[DegreeLabel] = []
    for target in sorted(
        targets,
        key=lambda item: (_line_center(item.line)[0], item.record.get("instance_id", "")),
    ):
        reference = _nearest_centerline(target, references)
        degrees = _line_angle_degrees(target.line, reference.line)
        if degrees is None:
            continue
        labels.append(
            DegreeLabel(
                instance_id=str(target.record.get("instance_id", "")),
                degree=round(degrees),
                line=target.line,
                color=target.color,
            )
        )
    return labels


def _nearest_centerline(target: Centerline, references: list[Centerline]) -> Centerline:
    target_x, target_y = _line_center(target.line)
    return min(
        references,
        key=lambda reference: math.hypot(
            target_x - _line_center(reference.line)[0],
            target_y - _line_center(reference.line)[1],
        ),
    )


def _line_center(line: Line) -> tuple[float, float]:
    return (line[0] + line[2]) / 2, (line[1] + line[3]) / 2


def _line_angle_degrees(first: Line, second: Line) -> float | None:
    first_vector = _normalize((first[2] - first[0], first[3] - first[1]))
    second_vector = _normalize((second[2] - second[0], second[3] - second[1]))
    if first_vector is None or second_vector is None:
        return None
    if _dot(first_vector, second_vector) < 0:
        second_vector = (-second_vector[0], -second_vector[1])
    radians = math.acos(_clamp(_dot(first_vector, second_vector), -1, 1))
    degrees = math.degrees(radians)
    if not math.isfinite(degrees) or degrees < 2:
        return None
    return min(degrees, 180 - degrees)


def _draw_target_degree_marker(
    image: bytearray,
    width: int,
    height: int,
    entry: DegreeLabelEntry,
) -> None:
    label = entry.label
    center_x, center_y = _line_center(label.line)
    dx = label.line[2] - label.line[0]
    dy = label.line[3] - label.line[1]
    length = max(math.hypot(dx, dy), 1.0)
    offset = max(18, min(width, height) * 0.035)
    label_x = center_x + (-dy / length) * offset
    label_y = center_y + (dx / length) * offset
    _draw_line(
        image,
        width,
        height,
        (center_x, center_y, label_x, label_y),
        label.color,
        max(2, round(min(width, height) * 0.003)),
    )
    _draw_small_degree_label(
        image,
        width,
        height,
        label_x,
        label_y,
        entry.text,
        entry.text_color,
    )


def _draw_degree_label_block(
    image: bytearray,
    width: int,
    height: int,
    labels: list[DegreeLabel],
    options: ExportOptions,
) -> None:
    if _draw_degree_label_block_with_pillow(image, width, height, labels, options):
        return
    _draw_degree_label_block_bitmap(image, width, height, labels, options)


def _degree_label_entries(labels: list[DegreeLabel]) -> list[DegreeLabelEntry]:
    highlight_index = _highlighted_degree_index(labels)
    entries: list[DegreeLabelEntry] = []
    show_index = len(labels) > 1
    for index, label in enumerate(labels, start=1):
        text_color = (
            (255, 82, 96, 255)
            if highlight_index is not None and index - 1 == highlight_index
            else (255, 242, 168, 255)
        )
        entries.append(
            DegreeLabelEntry(
                text=f"{index}: {label.degree}°" if show_index else f"{label.degree}°",
                label=label,
                text_color=text_color,
            )
        )
    return entries


def _highlighted_degree_index(labels: list[DegreeLabel]) -> int | None:
    if len(labels) <= 3:
        return None
    average = sum(label.degree for label in labels) / len(labels)
    return max(
        range(len(labels)),
        key=lambda index: abs(labels[index].degree - average),
    )


def _draw_degree_label_block_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    labels: list[DegreeLabel],
    options: ExportOptions,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    font_path = _find_export_font()
    if font_path is None:
        return False

    font_size = int(options.angle_label_font_size or max(18, round(height * 0.045)))
    title_font_size = max(12, round(font_size * 0.72))
    try:
        font = ImageFont.truetype(str(font_path), font_size)
        title_font = ImageFont.truetype(str(font_path), title_font_size)
    except OSError:
        return False

    entries = _degree_label_entries(labels)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_width = max(1, round(font_size * 0.05))
    text_boxes = [
        draw.textbbox((0, 0), entry.text, font=font, stroke_width=stroke_width)
        for entry in entries
    ]
    item_gap = max(12, round(font_size * 0.75))
    item_widths = [box[2] - box[0] for box in text_boxes]
    item_heights = [box[3] - box[1] for box in text_boxes]
    value_width = sum(item_widths) + item_gap * max(0, len(entries) - 1)
    value_height = max(item_heights)
    title_bbox = draw.textbbox(
        (0, 0),
        DEGREE_LABEL_TITLE,
        font=title_font,
        stroke_width=stroke_width,
    )
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_gap = max(4, round(font_size * 0.2))
    block_width = max(value_width, title_width)
    block_height = title_height + title_gap + value_height
    left = round(width / 2 - block_width / 2)
    top = _metric_label_top(
        width,
        height,
        block_height,
        font_size,
        options.angle_label_position,
        options.metric_center_offset_percent,
    )
    title_left = left + round((block_width - title_width) / 2)
    draw.text(
        (title_left - title_bbox[0], top - title_bbox[1]),
        DEGREE_LABEL_TITLE,
        font=title_font,
        fill=(235, 245, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 9, 255),
    )
    value_top = top + title_height + title_gap
    x = left + round((block_width - value_width) / 2)
    for entry, bbox, item_width, item_height in zip(
        entries, text_boxes, item_widths, item_heights
    ):
        y = value_top + (value_height - item_height) / 2
        draw.text(
            (x - bbox[0], y - bbox[1]),
            entry.text,
            font=font,
            fill=entry.text_color,
            stroke_width=stroke_width,
            stroke_fill=(2, 5, 9, 255),
        )
        x += item_width + item_gap
    _blend_overlay(image, width, overlay.tobytes())
    return True


def _draw_degree_label_block_bitmap(
    image: bytearray,
    width: int,
    height: int,
    labels: list[DegreeLabel],
    options: ExportOptions,
) -> None:
    entries = _degree_label_entries(labels)
    font_size = int(options.angle_label_font_size or max(18, round(height * 0.045)))
    scale = max(2, round(font_size / 7))
    gap = max(1, round(scale * 0.75))
    item_gap = max(3 * scale, round(font_size * 0.75))
    glyph_items = [[_glyph(character) for character in entry.text] for entry in entries]
    item_widths = [
        sum(len(glyph[0]) * scale for glyph in glyphs) + gap * max(0, len(glyphs) - 1)
        for glyphs in glyph_items
    ]
    text_width = sum(item_widths) + item_gap * max(0, len(entries) - 1)
    text_height = 7 * scale
    left = round(width / 2 - text_width / 2)
    top = _metric_label_top(
        width,
        height,
        text_height,
        font_size,
        options.angle_label_position,
        options.metric_center_offset_percent,
    )
    x = left
    for entry, glyphs, item_width in zip(entries, glyph_items, item_widths):
        for glyph in glyphs:
            _draw_bitmap(
                image,
                width,
                height,
                x + scale,
                top + scale,
                glyph,
                scale,
                (2, 5, 9, 255),
            )
            _draw_bitmap(image, width, height, x, top, glyph, scale, entry.text_color)
            x += len(glyph[0]) * scale + gap
        x += item_gap - gap


def _draw_spm_label(
    image: bytearray,
    width: int,
    height: int,
    estimate: SpmEstimate,
    options: ExportOptions,
) -> None:
    text = (
        f"{SPM_INSTANT_LABEL}: {_format_spm(estimate.instantaneous)}   "
        f"{SPM_AVERAGE_LABEL}: {_format_spm(estimate.average)}"
    )
    if _draw_spm_label_with_pillow(image, width, height, text, options):
        return
    _draw_spm_label_bitmap(image, width, height, text, options)


def _format_spm(value: float | None) -> str:
    return "--" if value is None else str(round(value))


def _default_metric_center_offset_percent(width: int, height: int) -> float:
    if height > width:
        return PORTRAIT_METRIC_CENTER_OFFSET_PERCENT
    return LANDSCAPE_METRIC_CENTER_OFFSET_PERCENT


def _metric_label_margin(
    width: int,
    height: int,
    font_size: int,
    center_offset_percent: float | None = None,
) -> int:
    percent = (
        _default_metric_center_offset_percent(width, height)
        if center_offset_percent is None
        else center_offset_percent
    )
    return max(round(height * max(0.0, min(45.0, percent)) / 100), font_size)


def _metric_label_top(
    width: int,
    height: int,
    text_height: int,
    font_size: int,
    position: LabelPosition,
    center_offset_percent: float | None = None,
) -> int:
    margin = _metric_label_margin(width, height, font_size, center_offset_percent)
    if position == "top":
        return margin
    return max(margin, height - margin - text_height)


def _spm_label_top(
    width: int,
    height: int,
    text_height: int,
    font_size: int,
    options: ExportOptions,
) -> int:
    position = "bottom" if options.angle_label_position == "top" else "top"
    return _metric_label_top(
        width,
        height,
        text_height,
        font_size,
        position,
        options.metric_center_offset_percent,
    )


def _draw_spm_label_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    text: str,
    options: ExportOptions,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    font_path = _find_export_font()
    if font_path is None:
        return False

    font_size = int(options.angle_label_font_size or max(18, round(height * 0.045)))
    title_font_size = max(12, round(font_size * 0.72))
    try:
        font = ImageFont.truetype(str(font_path), font_size)
        title_font = ImageFont.truetype(str(font_path), title_font_size)
    except OSError:
        return False

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_width = max(1, round(font_size * 0.06))
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    title_bbox = draw.textbbox(
        (0, 0),
        SPM_LABEL_TITLE,
        font=title_font,
        stroke_width=stroke_width,
    )
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    title_gap = max(4, round(font_size * 0.2))
    block_height = title_height + title_gap + text_height
    top = _spm_label_top(width, height, block_height, font_size, options)
    title_left = round(width / 2 - title_width / 2)
    draw.text(
        (title_left - title_bbox[0], top - title_bbox[1]),
        SPM_LABEL_TITLE,
        font=title_font,
        fill=(235, 245, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 9, 255),
    )
    text_left = round(width / 2 - text_width / 2)
    text_top = top + title_height + title_gap
    draw.text(
        (text_left - bbox[0], text_top - bbox[1]),
        text,
        font=font,
        fill=(235, 245, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 9, 255),
    )
    _blend_overlay(image, width, overlay.tobytes())
    return True


def _draw_spm_label_bitmap(
    image: bytearray,
    width: int,
    height: int,
    text: str,
    options: ExportOptions,
) -> None:
    font_size = int(options.angle_label_font_size or max(18, round(height * 0.045)))
    scale = max(2, round(font_size / 7))
    gap = max(1, round(scale * 0.75))
    glyphs = [_glyph(character) for character in text]
    text_width = sum(len(glyph[0]) * scale for glyph in glyphs) + gap * max(0, len(glyphs) - 1)
    text_height = 7 * scale
    x = round(width / 2 - text_width / 2)
    y = _spm_label_top(width, height, text_height, font_size, options)
    for glyph in glyphs:
        _draw_bitmap(image, width, height, x + scale, y + scale, glyph, scale, (2, 5, 9, 255))
        _draw_bitmap(image, width, height, x, y, glyph, scale, (235, 245, 255, 255))
        x += len(glyph[0]) * scale + gap


def _find_export_font() -> Path | None:
    configured = os.getenv("SAM3_EXPORT_FONT_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def _draw_small_degree_label(
    image: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    text: str,
    text_color: Color = (255, 242, 168, 255),
) -> None:
    scale = max(2, round(width / 700))
    glyphs = [_glyph(character) for character in text]
    gap = max(1, round(scale * 0.4))
    text_width = sum(len(glyph[0]) * scale for glyph in glyphs) + gap * (len(glyphs) - 1)
    text_height = max(len(glyph) for glyph in glyphs) * scale
    padding_x = 3 * scale
    padding_y = 2 * scale
    left = round(center_x - text_width / 2)
    top = round(center_y - text_height / 2)
    _fill_rect(
        image,
        width,
        height,
        left - padding_x,
        top - padding_y,
        text_width + padding_x * 2,
        text_height + padding_y * 2,
        (2, 5, 9, 200),
    )
    x = left
    for glyph in glyphs:
        _draw_bitmap(image, width, height, x, top, glyph, scale, text_color)
        x += len(glyph[0]) * scale + gap


def _draw_line(
    image: bytearray,
    width: int,
    height: int,
    line: Line,
    color: Color,
    thickness: int,
) -> None:
    x1, y1, x2, y2 = line
    radius = thickness / 2
    min_x = max(0, math.floor(min(x1, x2) - radius - 1))
    max_x = min(width - 1, math.ceil(max(x1, x2) + radius + 1))
    min_y = max(0, math.floor(min(y1, y2) - radius - 1))
    max_y = min(height - 1, math.ceil(max(y1, y2) + radius + 1))
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        return
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = _clamp(((x - x1) * dx + (y - y1) * dy) / length_sq, 0, 1)
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            if (x - closest_x) ** 2 + (y - closest_y) ** 2 <= radius * radius:
                _blend_pixel(image, width, x, y, color)


_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "d": ("00001", "00001", "00001", "01111", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "n": ("00000", "10110", "11001", "10001", "10001", "10001", "10001"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "11110", "00000"),
    "t": ("00100", "00100", "11111", "00100", "00100", "00101", "00010"),
    "v": ("00000", "10001", "10001", "10001", "01010", "01010", "00100"),
    ":": ("000", "010", "010", "000", "010", "010", "000"),
    "-": ("000", "000", "000", "111", "000", "000", "000"),
    "°": ("01100", "10010", "10010", "01100", "00000", "00000", "00000"),
    " ": ("000", "000", "000", "000", "000", "000", "000"),
}


def _glyph(character: str) -> tuple[str, ...]:
    return _GLYPHS.get(character, _GLYPHS[" "])


def _draw_bitmap(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    bitmap: tuple[str, ...],
    scale: int,
    color: Color,
) -> None:
    for row_index, row in enumerate(bitmap):
        for col_index, value in enumerate(row):
            if value == "1":
                _fill_rect(
                    image,
                    width,
                    height,
                    left + col_index * scale,
                    top + row_index * scale,
                    scale,
                    scale,
                    color,
                )


def _fill_rect(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    rect_width: int,
    rect_height: int,
    color: Color,
) -> None:
    for y in range(max(0, top), min(height, top + rect_height)):
        for x in range(max(0, left), min(width, left + rect_width)):
            _blend_pixel(image, width, x, y, color)


def _transparent_image(width: int, height: int) -> bytearray:
    return bytearray(width * height * 4)


def _blend_pixel(image: bytearray, width: int, x: int, y: int, color: Color) -> None:
    if color[3] <= 0:
        return
    index = (y * width + x) * 4
    source_alpha = color[3] / 255
    inverse = 1 - source_alpha
    image[index] = round(color[0] * source_alpha + image[index] * inverse)
    image[index + 1] = round(color[1] * source_alpha + image[index + 1] * inverse)
    image[index + 2] = round(color[2] * source_alpha + image[index + 2] * inverse)
    image[index + 3] = min(255, round(color[3] + image[index + 3] * inverse))


def _blend_overlay(image: bytearray, width: int, overlay: bytes) -> None:
    for index in range(0, len(overlay), 4):
        alpha = overlay[index + 3]
        if alpha <= 0:
            continue
        pixel = index // 4
        _blend_pixel(
            image,
            width,
            pixel % width,
            pixel // width,
            (overlay[index], overlay[index + 1], overlay[index + 2], alpha),
        )


def _write_png_rgba(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(
                b"IHDR",
                width.to_bytes(4, "big")
                + height.to_bytes(4, "big")
                + b"\x08\x06\x00\x00\x00",
            ),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=1)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return len(data).to_bytes(4, "big") + kind + data + checksum.to_bytes(4, "big")


def _parse_hex(value: str) -> Color:
    stripped = value.lstrip("#")
    if len(stripped) != 6:
        return (53, 194, 255, 255)
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
        255,
    )


def _sign(value: float, *, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _spm_from_events(events_ms: list[int]) -> float | None:
    if len(events_ms) < SPM_MIN_INTERVALS + 1:
        return None
    intervals = [
        (later - earlier) / 1000
        for earlier, later in zip(events_ms, events_ms[1:])
        if SPM_MIN_INTERVAL_MS <= later - earlier <= SPM_MAX_INTERVAL_MS
    ]
    if len(intervals) < SPM_MIN_INTERVALS:
        return None
    return 60 / _median(intervals)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _normalize(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector[0], vector[1])
    if length < 1e-6:
        return None
    return vector[0] / length, vector[1] / length


def _dot(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
