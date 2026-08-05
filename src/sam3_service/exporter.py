from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import zlib
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any

from .errors import ServiceError
from .media import probe_video

Color = tuple[int, int, int, int]
Line = tuple[float, float, float, float]
SelectionRect = tuple[float, float, float, float]
SelectionKeyframe = tuple[int, float, float, float, float]
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
PADDLE_EVENT_CONFIRM_SAMPLES = 2
PADDLE_CATCH_DEPTH_RATIO = 0.015
PADDLE_EXIT_DEPTH_RATIO = 0.005
PADDLE_EVENT_DEDUPE_MS = 250
PADDLE_EVENT_MAX_CONFIRM_GAP_MS = 400
PADDLE_EVENT_TRACK_GAP_MS = 1500
PADDLE_BLADE_ZONE_RATIO = 0.28
PADDLE_FRAGMENT_ANGLE_DEGREES = 14.0
PADDLE_PHASE_LOW_DEGREES = 20.0
PADDLE_PHASE_HIGH_DEGREES = 70.0
PADDLE_EVENT_PHASE_GAP = 4
PADDLE_EVENT_PADDLE_COLOR = (0, 229, 255, 255)
PADDLE_EVENT_REFERENCE_COLOR = (255, 196, 61, 255)
PADDLE_EVENT_ANGLE_COLOR = (255, 255, 255, 255)


@dataclass(frozen=True)
class ExportOptions:
    angle_label_position: LabelPosition = "top"
    angle_label_font_size: int | None = None
    include_angles: bool = True
    include_spm: bool = False
    include_catch: bool = False
    include_exit: bool = False
    event_hold_seconds: float = 1.5
    metric_center_offset_percent: float | None = None
    reference_prompt_id: str | None = None
    reference_line_mode: str | None = None
    target_prompt_ids: tuple[str, ...] = ()
    reference_track_ids: tuple[str, ...] = ()
    target_track_ids: tuple[str, ...] = ()
    selection_rect: SelectionRect | None = None
    selection_keyframes: tuple[SelectionKeyframe, ...] = ()
    target_slot_count: int = 0


@dataclass(frozen=True)
class Centerline:
    record: dict[str, Any]
    line: Line
    color: Color


@dataclass(frozen=True)
class DegreeLabel:
    instance_id: str
    degree: int | None
    line: Line | None
    color: Color


@dataclass(frozen=True)
class DegreeLabelEntry:
    text: str
    layout_text: str
    label: DegreeLabel
    text_color: Color


@dataclass(frozen=True)
class SpmEstimate:
    instantaneous: float | None
    average: float | None


@dataclass(frozen=True)
class PaddleEvent:
    kind: str
    timestamp_ms: int
    instance_id: str
    line: Line
    confidence: float
    reference_line: Line | None = None
    degree: float | None = None


@dataclass(frozen=True)
class FreezeMoment:
    frame_index: int
    events: tuple[PaddleEvent, ...]


@dataclass(frozen=True)
class _PaddleObservation:
    source_ids: tuple[str, ...]
    reference_id: str
    line: Line
    reference_line: Line


@dataclass
class _PaddleEventState:
    physical_id: str = ""
    source_ids: set[str] = field(default_factory=set)
    reference_id: str = ""
    last_seen_ms: int = -1
    last_line: Line | None = None
    stable_lengths: list[float] = field(default_factory=list)
    angle_samples: list[float] = field(default_factory=list)
    last_phase_angle: float | None = None
    phase_direction: int = 0
    pending_phase_direction: int = 0
    pending_phase_count: int = 0
    phase_min_angle: float = 90.0
    phase_max_angle: float = 0.0
    phase_index: int = 0
    blade_overlaps: tuple[bool, bool] | None = None
    endpoint_depths: tuple[float, float] | None = None
    last_catch_phase: int | None = None
    last_exit_phase: int | None = None
    immersed: bool | None = None
    candidate_kind: str | None = None
    candidate_blade: int | None = None
    candidate_phase_index: int = 0
    candidate_count: int = 0
    candidate_timestamp_ms: int = 0
    candidate_line: Line | None = None
    candidate_reference_line: Line | None = None
    candidate_degree: float | None = None
    candidate_confidence: float = 0.0


class SpmEstimator:
    def __init__(self, *, window_ms: int = 8000) -> None:
        self.window_ms = window_ms
        self._tracks: dict[str, _SpmTrack] = {}

    def update(self, timestamp_ms: int, labels: list[DegreeLabel]) -> SpmEstimate:
        for label in labels:
            if label.degree is None:
                continue
            self._tracks.setdefault(label.instance_id, _SpmTrack()).update(
                timestamp_ms, float(label.degree)
            )
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
    progress: Callable[[str, float, str], None] | None = None,
) -> Path:
    if not video_path.is_file():
        raise ServiceError("NOT_FOUND", "Video content is unavailable.", status_code=404)
    if not chunk_paths:
        raise ServiceError("NOT_FOUND", "Result chunks are unavailable.", status_code=404)

    _report_progress(progress, "preparing", 2, "Preparing export")
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
    if (
        export_options.target_slot_count == 0
        and (export_options.selection_rect is not None or export_options.selection_keyframes)
    ):
        export_options = replace(
            export_options,
            target_slot_count=_maximum_target_count_in_selection(
                frames,
                export_options,
                width,
                height,
                scale_x,
                scale_y,
            ),
        )
    events: list[PaddleEvent] = []
    if export_options.include_catch or export_options.include_exit:
        _report_progress(progress, "analyzing_events", 5, "Analyzing paddle events")
        events = _detect_paddle_events(
            frames,
            export_options,
            width,
            height,
            scale_x,
            scale_y,
            progress,
        )
    freeze_moments = _freeze_moments(events, fps, frame_count)
    freeze_by_frame = {moment.frame_index: moment for moment in freeze_moments}
    freeze_frame_count = max(1, round(export_options.event_hold_seconds * fps))
    output_frame_count = frame_count + freeze_frame_count * len(freeze_moments)
    _report_progress(progress, "rendering", 15, "Rendering overlay frames")
    result_tolerance_ms = max(1000 / max(fps, 1), 500 / max(manifest_fps, 1), 40)
    spm_estimator = SpmEstimator()
    output_frame_index = 0
    for frame_index in range(frame_count):
        image = _transparent_image(width, height)
        timestamp_ms = round(frame_index * 1000 / fps)
        freeze_moment = freeze_by_frame.get(frame_index)
        records = _records_for_timestamp(
            frames, frame_timestamps, timestamp_ms, result_tolerance_ms
        )
        scaled_records = [_scale_record(record, scale_x, scale_y) for record in records]
        scaled_records = [
            record
            for record in scaled_records
            if _record_selected_for_export(
                record, export_options, width, height, timestamp_ms
            )
        ]
        _draw_frame_overlay(
            image,
            width,
            height,
            scaled_records,
            colors,
            export_options=export_options,
            timestamp_ms=timestamp_ms,
            spm_estimator=spm_estimator,
            paddle_events=(),
        )
        if freeze_moment is not None:
            event_image = bytearray(image)
            for event in freeze_moment.events:
                _draw_paddle_event_label(
                    event_image, width, height, event, export_options
                )
            event_frame_path = frames_dir / f"{output_frame_index:06d}.png"
            _write_png_rgba(event_frame_path, width, height, event_image)
            output_frame_index += 1
            for _ in range(freeze_frame_count - 1):
                shutil.copyfile(
                    event_frame_path,
                    frames_dir / f"{output_frame_index:06d}.png",
                )
                output_frame_index += 1
        _write_png_rgba(
            frames_dir / f"{output_frame_index:06d}.png", width, height, image
        )
        output_frame_index += 1
        if frame_index == frame_count - 1 or frame_index % max(1, frame_count // 100) == 0:
            percent = 15 + 75 * (frame_index + 1) / frame_count
            _report_progress(
                progress,
                "rendering",
                percent,
                f"Rendering frame {frame_index + 1} of {frame_count}",
            )

    has_audio = bool(freeze_moments) and _has_audio_stream(video_path)
    filter_parts = [
        _freeze_video_filter(freeze_moments, freeze_frame_count, fps, frame_count)
    ]
    if has_audio:
        filter_parts.append(
            _freeze_audio_filter(freeze_moments, freeze_frame_count, fps, frame_count)
        )
    filter_parts.append("[base][1:v]overlay=0:0:format=auto:shortest=1[ov]")
    filter_complex = ";".join(filter_parts)
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
        "-frames:v",
        str(output_frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        command.extend(["-map", "[a]", "-c:a", "aac"])
    else:
        command.extend(["-map", "0:a?", "-c:a", "copy"])
    command.extend(["-movflags", "+faststart", str(temporary_output)])
    _report_progress(progress, "encoding", 92, "Encoding MP4")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        temporary_output.replace(output_path)
        _report_progress(progress, "finalizing", 99, "Finalizing export")
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


def _report_progress(
    progress: Callable[[str, float, str], None] | None,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if progress is not None:
        progress(stage, max(0.0, min(100.0, percent)), message)


def _freeze_moments(
    events: list[PaddleEvent], fps: float, frame_count: int
) -> tuple[FreezeMoment, ...]:
    if not events or frame_count <= 0:
        return ()
    groups: list[list[PaddleEvent]] = []
    for event in sorted(events, key=lambda item: item.timestamp_ms):
        if (
            groups
            and event.timestamp_ms - groups[-1][0].timestamp_ms
            <= PADDLE_EVENT_DEDUPE_MS
        ):
            groups[-1].append(event)
        else:
            groups.append([event])
    events_by_frame: dict[int, list[PaddleEvent]] = {}
    for group in groups:
        frame_index = max(
            0,
            min(frame_count - 1, round(group[0].timestamp_ms * fps / 1000)),
        )
        events_by_frame.setdefault(frame_index, []).extend(group)
    return tuple(
        FreezeMoment(frame_index=frame_index, events=tuple(group))
        for frame_index, group in sorted(events_by_frame.items())
    )


def _freeze_video_filter(
    moments: tuple[FreezeMoment, ...],
    freeze_frames: int,
    fps: float,
    frame_count: int,
) -> str:
    if not moments or freeze_frames <= 0:
        return "[0:v]null[base]"
    segments = _freeze_segments(moments, frame_count)
    branches = "".join(f"[vpart{index}]" for index in range(len(segments)))
    graph = [f"[0:v]fps=fps={fps:.6f}:start_time=0[vpart0]"]
    if len(segments) > 1:
        graph[0] = (
            f"[0:v]fps=fps={fps:.6f}:start_time=0,"
            f"split={len(segments)}{branches}"
        )
    hold_seconds = freeze_frames / fps
    padding_seconds = max(0, freeze_frames - 1) / fps
    for index, (kind, start_frame, end_frame) in enumerate(segments):
        if kind == "freeze":
            graph.append(
                f"[vpart{index}]trim=start_frame={start_frame}:end_frame={end_frame},"
                f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={padding_seconds:.9f},"
                f"trim=duration={hold_seconds:.9f}[vseg{index}]"
            )
        else:
            graph.append(
                f"[vpart{index}]trim=start_frame={start_frame}:end_frame={end_frame},"
                f"setpts=PTS-STARTPTS[vseg{index}]"
            )
    if len(segments) == 1:
        graph.append("[vseg0]null[base]")
    else:
        inputs = "".join(f"[vseg{index}]" for index in range(len(segments)))
        graph.append(f"{inputs}concat=n={len(segments)}:v=1:a=0[base]")
    return ";".join(graph)


def _freeze_audio_filter(
    moments: tuple[FreezeMoment, ...],
    freeze_frames: int,
    fps: float,
    frame_count: int,
) -> str:
    segments = _freeze_segments(moments, frame_count)
    branches = "".join(f"[apart{index}]" for index in range(len(segments)))
    graph = ["[0:a]anull[apart0]"]
    if len(segments) > 1:
        graph[0] = f"[0:a]asplit={len(segments)}{branches}"
    hold_seconds = freeze_frames / fps
    for index, (kind, start_frame, end_frame) in enumerate(segments):
        start_seconds = start_frame / fps
        end_seconds = end_frame / fps
        if kind == "freeze":
            graph.append(
                f"[apart{index}]atrim=start={start_seconds:.9f}:end={end_seconds:.9f},"
                f"volume=0,asetpts=PTS-STARTPTS,apad=pad_dur={hold_seconds:.9f},"
                f"atrim=duration={hold_seconds:.9f}[aseg{index}]"
            )
        else:
            graph.append(
                f"[apart{index}]atrim=start={start_seconds:.9f}:end={end_seconds:.9f},"
                f"asetpts=PTS-STARTPTS[aseg{index}]"
            )
    if len(segments) == 1:
        graph.append("[aseg0]anull[a]")
    else:
        inputs = "".join(f"[aseg{index}]" for index in range(len(segments)))
        graph.append(f"{inputs}concat=n={len(segments)}:v=0:a=1[a]")
    return ";".join(graph)


def _freeze_segments(
    moments: tuple[FreezeMoment, ...], frame_count: int
) -> tuple[tuple[str, int, int], ...]:
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for frame_index in sorted({moment.frame_index for moment in moments}):
        if cursor < frame_index:
            segments.append(("normal", cursor, frame_index))
        segments.append(("freeze", frame_index, frame_index + 1))
        cursor = frame_index
    if cursor < frame_count:
        segments.append(("normal", cursor, frame_count))
    return tuple(segments)


def _has_audio_stream(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


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
    tracks = manifest.get("tracks", [])
    reference_tracks = [
        track for track in tracks if track.get("prompt_id") == reference_prompt_id
    ]
    target_tracks = [
        track for track in tracks if track.get("prompt_id") in target_prompt_ids
    ]
    reference_track_ids = _resolve_requested_track_ids(
        requested.reference_track_ids, reference_tracks
    )
    target_track_ids = _resolve_requested_track_ids(
        requested.target_track_ids, target_tracks
    )
    if (
        not requested.target_track_ids
        and requested.target_slot_count == 0
        and requested.selection_rect is None
        and not requested.selection_keyframes
    ):
        target_track_ids = tuple(
            str(track["id"]) for track in target_tracks if track.get("id")
        )
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
    manifest_line_mode = manifest.get("settings", {}).get(
        "boat_reference_line", "centerline"
    )
    reference_line_mode = (
        requested.reference_line_mode
        if requested.reference_line_mode in {"centerline", "waterline"}
        else manifest_line_mode
    )
    if reference_line_mode not in {"centerline", "waterline"}:
        reference_line_mode = "centerline"
    return ExportOptions(
        angle_label_position=position,
        angle_label_font_size=max(12, min(96, int(font_size))),
        include_angles=bool(requested.include_angles),
        include_spm=bool(requested.include_spm),
        include_catch=bool(requested.include_catch),
        include_exit=bool(requested.include_exit),
        event_hold_seconds=max(0.1, min(10.0, float(requested.event_hold_seconds))),
        metric_center_offset_percent=max(0.0, min(45.0, float(metric_center_offset_percent))),
        reference_prompt_id=reference_prompt_id,
        reference_line_mode=reference_line_mode,
        target_prompt_ids=target_prompt_ids,
        reference_track_ids=reference_track_ids,
        target_track_ids=target_track_ids,
        selection_rect=requested.selection_rect,
        selection_keyframes=tuple(
            sorted(requested.selection_keyframes, key=lambda keyframe: keyframe[0])
        ),
        target_slot_count=max(0, requested.target_slot_count),
    )


def _resolve_requested_track_ids(
    requested_ids: tuple[str, ...], tracks: list[dict[str, Any]]
) -> tuple[str, ...]:
    resolved: list[str] = []
    for requested_id in requested_ids:
        for track in tracks:
            track_id = str(track.get("id", ""))
            source_ids = {str(value) for value in track.get("instance_ids", [])}
            if requested_id == track_id or requested_id in source_ids:
                if track_id and track_id not in resolved:
                    resolved.append(track_id)
                break
    return tuple(resolved)


def _record_selected_for_export(
    record: dict[str, Any],
    options: ExportOptions,
    width: int | None = None,
    height: int | None = None,
    timestamp_ms: int = 0,
) -> bool:
    selection_rect = _selection_rect_at(options, timestamp_ms)
    if selection_rect is not None and width and height:
        line = _record_line(record, width, height)
        if line is None:
            return False
        center_x, center_y = _line_center(line)
        left, top, rect_width, rect_height = selection_rect
        if not (
            left <= center_x / width <= left + rect_width
            and top <= center_y / height <= top + rect_height
        ):
            return False
    prompt_id = record.get("prompt_id")
    track_id = _record_track_id(record)
    if prompt_id == options.reference_prompt_id and options.reference_track_ids:
        return track_id in options.reference_track_ids
    if prompt_id in options.target_prompt_ids and options.target_track_ids:
        return track_id in options.target_track_ids
    return True


def _selection_rect_at(options: ExportOptions, timestamp_ms: int) -> SelectionRect | None:
    keyframes = options.selection_keyframes
    if not keyframes:
        return options.selection_rect
    active = keyframes[0]
    for keyframe in keyframes:
        if keyframe[0] > timestamp_ms:
            break
        active = keyframe
    return active[1:]


def _maximum_target_count_in_selection(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> int:
    target_prompt_ids = set(options.target_prompt_ids)
    maximum = 0
    for timestamp_ms, records in frames.items():
        count = sum(
            1
            for raw_record in records
            if raw_record.get("prompt_id") in target_prompt_ids
            and _record_selected_for_export(
                _scale_record(raw_record, scale_x, scale_y),
                options,
                width,
                height,
                timestamp_ms,
            )
        )
        maximum = max(maximum, count)
    return maximum


def _record_track_id(record: dict[str, Any]) -> str:
    return str(record.get("track_id") or record.get("instance_id", ""))


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
    for geometry in ("centerline", "waterline"):
        field = f"{geometry}_line_xyxy"
        values = record.get(field)
        if values and len(values) == 4 and _line_rle_size(record, geometry) is None:
            scaled[field] = [
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
    timestamp_ms: int,
    spm_estimator: SpmEstimator,
    paddle_events: tuple[PaddleEvent, ...] = (),
) -> None:
    centerlines: list[Centerline] = []
    for record in records:
        use_waterline = (
            export_options.reference_line_mode == "waterline"
            and record.get("prompt_id") == export_options.reference_prompt_id
        )
        line = _record_line(record, width, height, use_waterline=use_waterline)
        if line is None:
            continue
        color = colors.get(record["prompt_id"], (53, 194, 255, 255))
        centerlines.append(Centerline(record=record, line=line, color=color))
        _draw_line(image, width, height, line, color, max(3, round(min(width, height) * 0.006)))

    labels = _degree_labels(centerlines, export_options)
    displayed_labels = _degree_slots(labels, export_options)
    if export_options.include_angles:
        for entry in _degree_label_entries(displayed_labels):
            if entry.label.degree is not None and entry.label.line is not None:
                _draw_target_degree_marker(image, width, height, entry)
        if displayed_labels:
            _draw_degree_label_block(image, width, height, displayed_labels, export_options)
    if export_options.include_spm:
        has_fixed_metric_slots = export_options.target_slot_count > 0
        has_selection = (
            export_options.selection_rect is not None
            or bool(export_options.selection_keyframes)
        )
        spm_labels = displayed_labels if has_selection or has_fixed_metric_slots else labels
        estimate = spm_estimator.update(timestamp_ms, spm_labels)
        _draw_spm_label(image, width, height, estimate, export_options)
    for event in paddle_events:
        _draw_paddle_event_label(image, width, height, event, export_options)


def _record_line(
    record: dict[str, Any],
    width: int,
    height: int,
    *,
    use_waterline: bool = False,
) -> Line | None:
    geometry = "waterline" if use_waterline else "centerline"
    values = record.get(f"{geometry}_line_xyxy")
    if use_waterline and (not values or len(values) != 4):
        geometry = "centerline"
        values = record.get("centerline_line_xyxy")
    if not values or len(values) != 4:
        return None
    try:
        scale_x, scale_y = _line_coordinate_scale(record, width, height, geometry)
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
    return _line_coordinate_scale(record, width, height, "centerline")


def _line_coordinate_scale(
    record: dict[str, Any],
    width: int,
    height: int,
    geometry: str,
) -> tuple[float, float]:
    size = _line_rle_size(record, geometry)
    if size is None:
        return (1.0, 1.0)
    mask_height, mask_width = size
    if mask_width <= 0 or mask_height <= 0:
        return (1.0, 1.0)
    return (width / mask_width, height / mask_height)


def _centerline_rle_size(record: dict[str, Any]) -> tuple[int, int] | None:
    return _line_rle_size(record, "centerline")


def _line_rle_size(
    record: dict[str, Any], geometry: str
) -> tuple[int, int] | None:
    segmentation = record.get(f"{geometry}_segmentation")
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
        and (
            not options.reference_track_ids
            or _record_track_id(centerline.record) in options.reference_track_ids
        )
    ]
    if not references:
        return []
    target_prompt_ids = set(options.target_prompt_ids)
    targets = [
        centerline
        for centerline in centerlines
        if centerline.record.get("prompt_id") in target_prompt_ids
        and (
            not options.target_track_ids
            or _record_track_id(centerline.record) in options.target_track_ids
        )
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
                instance_id=_record_track_id(target.record),
                degree=round(degrees),
                line=target.line,
                color=target.color,
            )
        )
    return labels


def _degree_slots(labels: list[DegreeLabel], options: ExportOptions) -> list[DegreeLabel]:
    if options.target_slot_count > 0:
        slots = [
            replace(label, instance_id=f"slot:{index + 1}")
            for index, label in enumerate(labels[: options.target_slot_count])
        ]
        slots.extend(
            DegreeLabel(
                instance_id=f"slot:{index + 1}",
                degree=None,
                line=None,
                color=(53, 194, 255, 255),
            )
            for index in range(len(slots), options.target_slot_count)
        )
        return slots
    if not options.target_track_ids:
        return labels
    labels_by_track = {label.instance_id: label for label in labels}
    return [
        labels_by_track.get(
            track_id,
            DegreeLabel(
                instance_id=track_id,
                degree=None,
                line=None,
                color=(53, 194, 255, 255),
            ),
        )
        for track_id in options.target_track_ids
    ]


def _detect_paddle_events(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    progress: Callable[[str, float, str], None] | None = None,
) -> list[PaddleEvent]:
    states: list[_PaddleEventState] = []
    detected: list[PaddleEvent] = []
    timestamps = sorted(frames)
    target_prompt_ids = set(options.target_prompt_ids)
    band_half_width = _event_band_half_width(scale_x, scale_y)
    for index, timestamp_ms in enumerate(timestamps):
        records = [_scale_record(record, scale_x, scale_y) for record in frames[timestamp_ms]]
        records = [
            record
            for record in records
            if _record_selected_for_export(record, options, width, height, timestamp_ms)
        ]
        references: list[Centerline] = []
        targets: list[Centerline] = []
        for record in records:
            is_reference = record.get("prompt_id") == options.reference_prompt_id
            line = _record_line(
                record,
                width,
                height,
                use_waterline=is_reference and options.reference_line_mode == "waterline",
            )
            if line is None:
                continue
            centerline = Centerline(record=record, line=line, color=(255, 255, 255, 255))
            if is_reference:
                references.append(centerline)
            elif record.get("prompt_id") in target_prompt_ids:
                targets.append(centerline)
        observations = _consolidate_paddle_observations(
            targets, references, width, height
        )
        assignments = _assign_paddle_observations(
            observations, states, timestamp_ms, width, height
        )
        for observation, state in assignments:
            event = _update_phase_aware_paddle_state(
                state,
                observation,
                timestamp_ms,
                band_half_width,
            )
            if event is not None:
                detected.append(event)
        if index == len(timestamps) - 1 or index % max(1, len(timestamps) // 20) == 0:
            _report_progress(
                progress,
                "analyzing_events",
                5 + 8 * (index + 1) / max(len(timestamps), 1),
                f"Analyzing events {index + 1} of {len(timestamps)}",
            )
    deduplicated = _dedupe_paddle_events(detected, width, height)
    return [
        event
        for event in deduplicated
        if (event.kind == "catch" and options.include_catch)
        or (event.kind == "exit" and options.include_exit)
    ]


def _event_band_half_width(scale_x: float, scale_y: float) -> float:
    raw = os.getenv(
        "SAM3_CENTERLINE_THICKNESS_PIXELS",
        os.getenv("SAM3_SHAFT_THICKNESS_PIXELS", "8"),
    )
    try:
        thickness = float(raw)
    except ValueError:
        thickness = 8.0
    return max(1.0, min(64.0, thickness / 2)) * max(scale_x, scale_y)


def _consolidate_paddle_observations(
    targets: list[Centerline],
    references: list[Centerline],
    width: int,
    height: int,
) -> list[_PaddleObservation]:
    if not references:
        return []
    grouped: dict[str, tuple[Centerline, list[list[Centerline]]]] = {}
    for target in sorted(targets, key=lambda item: _line_length(item.line), reverse=True):
        reference = _nearest_centerline(target, references)
        reference_id = _record_track_id(reference.record)
        entry = grouped.setdefault(reference_id, (reference, []))
        clusters = entry[1]
        cluster = next(
            (
                candidate
                for candidate in clusters
                if _paddle_fragments_compatible(
                    target.line,
                    _merge_paddle_lines([item.line for item in candidate]),
                    width,
                    height,
                )
            ),
            None,
        )
        if cluster is None:
            clusters.append([target])
        else:
            cluster.append(target)
    observations: list[_PaddleObservation] = []
    for reference_id, (reference, clusters) in grouped.items():
        for cluster in clusters:
            observations.append(
                _PaddleObservation(
                    source_ids=tuple(
                        sorted({_record_track_id(item.record) for item in cluster})
                    ),
                    reference_id=reference_id,
                    line=_merge_paddle_lines([item.line for item in cluster]),
                    reference_line=reference.line,
                )
            )
    return observations


def _paddle_fragments_compatible(
    first: Line,
    second: Line,
    width: int,
    height: int,
) -> bool:
    angle = _acute_line_angle_degrees(first, second)
    if angle > PADDLE_FRAGMENT_ANGLE_DEGREES:
        return False
    first_center = _line_center(first)
    second_center = _line_center(second)
    axis = _normalize((first[2] - first[0], first[3] - first[1]))
    if axis is None:
        return False
    perpendicular = (-axis[1], axis[0])
    center_delta = (
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    perpendicular_gap = abs(_dot(center_delta, perpendicular))
    perpendicular_limit = max(12.0, min(width, height) * 0.025)
    if perpendicular_gap > perpendicular_limit:
        return False
    first_interval = sorted(
        _dot((point[0] - first_center[0], point[1] - first_center[1]), axis)
        for point in ((first[0], first[1]), (first[2], first[3]))
    )
    second_interval = sorted(
        _dot((point[0] - first_center[0], point[1] - first_center[1]), axis)
        for point in ((second[0], second[1]), (second[2], second[3]))
    )
    longitudinal_gap = max(
        0.0,
        max(first_interval[0], second_interval[0])
        - min(first_interval[1], second_interval[1]),
    )
    return longitudinal_gap <= max(
        20.0, max(_line_length(first), _line_length(second)) * 0.3
    )


def _merge_paddle_lines(lines: list[Line]) -> Line:
    anchor = max(lines, key=_line_length)
    axis = _normalize((anchor[2] - anchor[0], anchor[3] - anchor[1]))
    if axis is None:
        return anchor
    normal = (-axis[1], axis[0])
    origin = _line_center(anchor)
    projections: list[float] = []
    offsets: list[float] = []
    for line in lines:
        for point in ((line[0], line[1]), (line[2], line[3])):
            delta = (point[0] - origin[0], point[1] - origin[1])
            projections.append(_dot(delta, axis))
            offsets.append(_dot(delta, normal))
    offset = _median(offsets)
    start = min(projections)
    end = max(projections)
    return (
        origin[0] + axis[0] * start + normal[0] * offset,
        origin[1] + axis[1] * start + normal[1] * offset,
        origin[0] + axis[0] * end + normal[0] * offset,
        origin[1] + axis[1] * end + normal[1] * offset,
    )


def _assign_paddle_observations(
    observations: list[_PaddleObservation],
    states: list[_PaddleEventState],
    timestamp_ms: int,
    width: int,
    height: int,
) -> list[tuple[_PaddleObservation, _PaddleEventState]]:
    available = {
        index
        for index, state in enumerate(states)
        if timestamp_ms - state.last_seen_ms <= PADDLE_EVENT_TRACK_GAP_MS
    }
    assigned: list[tuple[_PaddleObservation, _PaddleEventState]] = []
    for observation in sorted(observations, key=lambda item: _line_center(item.line)[0]):
        best_index: int | None = None
        best_cost = math.inf
        for state_index in available:
            state = states[state_index]
            if state.last_line is None:
                continue
            angle = _acute_line_angle_degrees(observation.line, state.last_line)
            center_distance = math.dist(
                _line_center(observation.line), _line_center(state.last_line)
            )
            shared_source = bool(state.source_ids.intersection(observation.source_ids))
            if angle > 40 or center_distance > max(
                min(width, height) * 0.22,
                max(_line_length(observation.line), _line_length(state.last_line)) * 0.55,
            ):
                continue
            reference_penalty = 0 if observation.reference_id == state.reference_id else 0.25
            source_bonus = -0.35 if shared_source else 0
            cost = (
                center_distance / max(min(width, height), 1)
                + angle / 180
                + reference_penalty
                + source_bonus
            )
            if cost < best_cost:
                best_cost = cost
                best_index = state_index
        if best_index is None:
            state = _PaddleEventState(physical_id=f"paddle:physical:{len(states) + 1}")
            states.append(state)
        else:
            state = states[best_index]
            available.remove(best_index)
        state.source_ids.update(observation.source_ids)
        state.reference_id = observation.reference_id
        assigned.append((observation, state))
    return assigned


def _update_phase_aware_paddle_state(
    state: _PaddleEventState,
    observation: _PaddleObservation,
    timestamp_ms: int,
    band_half_width: float,
) -> PaddleEvent | None:
    line = _stabilize_paddle_line(state, observation.line)
    angle = _acute_line_angle_degrees(line, observation.reference_line)
    _update_paddle_phase(state, angle)
    overlaps = _blade_waterline_overlaps(
        line, observation.reference_line, band_half_width
    )
    depths = _endpoint_signed_depths(line, observation.reference_line)
    previous_overlaps = state.blade_overlaps
    previous_depths = state.endpoint_depths
    previous_seen_ms = state.last_seen_ms
    state.last_seen_ms = timestamp_ms
    state.last_line = line

    event: PaddleEvent | None = None
    if (
        state.candidate_kind is not None
        and state.candidate_blade is not None
        and timestamp_ms - state.candidate_timestamp_ms <= PADDLE_EVENT_MAX_CONFIRM_GAP_MS
    ):
        blade = state.candidate_blade
        condition_holds = (
            overlaps[blade]
            if state.candidate_kind == "catch"
            else not overlaps[blade]
        )
        if condition_holds:
            state.candidate_count += 1
            if state.candidate_count >= PADDLE_EVENT_CONFIRM_SAMPLES:
                event = _confirm_phase_event(state)
        else:
            _clear_paddle_event_candidate(state)
    elif state.candidate_kind is not None:
        _clear_paddle_event_candidate(state)

    if event is None and previous_overlaps is not None:
        catch_candidates: list[tuple[float, int]] = []
        exit_candidates: list[tuple[float, int]] = []
        for blade in range(2):
            depth_delta = (
                0.0
                if previous_depths is None
                else depths[blade] - previous_depths[blade]
            )
            if (
                not previous_overlaps[blade]
                and overlaps[blade]
                and _event_phase_allowed(state, "catch")
                and depth_delta >= -band_half_width * 0.25
            ):
                catch_candidates.append((depth_delta, blade))
            if (
                previous_overlaps[blade]
                and not overlaps[blade]
                and _event_phase_allowed(state, "exit")
                and depth_delta <= band_half_width * 0.25
            ):
                exit_candidates.append((depth_delta, blade))
        if catch_candidates:
            _, blade = max(catch_candidates)
            _start_paddle_event_candidate(
                state,
                "catch",
                blade,
                timestamp_ms,
                line,
                observation.reference_line,
                0.9,
            )
        elif exit_candidates:
            _, blade = min(exit_candidates)
            _start_paddle_event_candidate(
                state,
                "exit",
                blade,
                timestamp_ms,
                line,
                observation.reference_line,
                0.9,
            )
    if previous_seen_ms >= 0 and timestamp_ms - previous_seen_ms > PADDLE_EVENT_TRACK_GAP_MS:
        _clear_paddle_event_candidate(state)
    state.blade_overlaps = overlaps
    state.endpoint_depths = depths
    return event


def _start_paddle_event_candidate(
    state: _PaddleEventState,
    kind: str,
    blade: int,
    timestamp_ms: int,
    line: Line,
    reference_line: Line,
    confidence: float,
) -> None:
    state.candidate_kind = kind
    state.candidate_blade = blade
    state.candidate_phase_index = state.phase_index
    state.candidate_count = 1
    state.candidate_timestamp_ms = timestamp_ms
    state.candidate_line = line
    state.candidate_reference_line = reference_line
    state.candidate_degree = _line_angle_degrees(line, reference_line)
    state.candidate_confidence = confidence


def _confirm_phase_event(state: _PaddleEventState) -> PaddleEvent | None:
    if (
        state.candidate_kind is None
        or state.candidate_blade is None
        or state.candidate_line is None
    ):
        return None
    kind = state.candidate_kind
    event = PaddleEvent(
        kind=kind,
        timestamp_ms=state.candidate_timestamp_ms,
        instance_id=state.physical_id,
        line=state.candidate_line,
        confidence=round(state.candidate_confidence, 3),
        reference_line=state.candidate_reference_line,
        degree=state.candidate_degree,
    )
    if kind == "catch":
        state.last_catch_phase = state.candidate_phase_index
    else:
        state.last_exit_phase = state.candidate_phase_index
    _clear_paddle_event_candidate(state)
    return event


def _clear_paddle_event_candidate(state: _PaddleEventState) -> None:
    state.candidate_kind = None
    state.candidate_blade = None
    state.candidate_phase_index = 0
    state.candidate_count = 0
    state.candidate_line = None
    state.candidate_reference_line = None
    state.candidate_degree = None


def _event_phase_allowed(state: _PaddleEventState, kind: str) -> bool:
    last_phase = (
        state.last_catch_phase if kind == "catch" else state.last_exit_phase
    )
    if last_phase is None:
        return True
    return state.phase_index - last_phase >= PADDLE_EVENT_PHASE_GAP


def _catch_phase_allowed(state: _PaddleEventState, blade: int = 0) -> bool:
    del blade
    return _event_phase_allowed(state, "catch")


def _update_paddle_phase(state: _PaddleEventState, angle: float) -> None:
    state.angle_samples.append(angle)
    state.angle_samples = state.angle_samples[-3:]
    smoothed = _median(state.angle_samples)
    if state.last_phase_angle is None:
        state.last_phase_angle = smoothed
        state.phase_min_angle = smoothed
        state.phase_max_angle = smoothed
        return
    slope = _sign(smoothed - state.last_phase_angle, epsilon=1.0)
    state.phase_min_angle = min(state.phase_min_angle, smoothed)
    state.phase_max_angle = max(state.phase_max_angle, smoothed)
    if slope != 0:
        if state.phase_direction == 0:
            if state.pending_phase_direction == slope:
                state.pending_phase_count += 1
            else:
                state.pending_phase_direction = slope
                state.pending_phase_count = 1
            if state.pending_phase_count >= 2:
                state.phase_direction = slope
                state.pending_phase_count = 0
        elif slope == state.phase_direction:
            state.pending_phase_direction = 0
            state.pending_phase_count = 0
        else:
            reached_extreme = (
                state.phase_direction > 0
                and state.phase_max_angle >= PADDLE_PHASE_HIGH_DEGREES
            ) or (
                state.phase_direction < 0
                and state.phase_min_angle <= PADDLE_PHASE_LOW_DEGREES
            )
            if reached_extreme:
                if state.pending_phase_direction == slope:
                    state.pending_phase_count += 1
                else:
                    state.pending_phase_direction = slope
                    state.pending_phase_count = 1
                if state.pending_phase_count >= 2:
                    state.phase_index += 1
                    state.phase_direction = slope
                    state.pending_phase_direction = 0
                    state.pending_phase_count = 0
                    state.phase_min_angle = smoothed
                    state.phase_max_angle = smoothed
    state.last_phase_angle = smoothed


def _stabilize_paddle_line(state: _PaddleEventState, observed: Line) -> Line:
    line = _orient_line_like(observed, state.last_line)
    observed_length = _line_length(line)
    if observed_length < 2:
        return line
    state.stable_lengths.append(observed_length)
    state.stable_lengths = state.stable_lengths[-7:]
    stable_length = sorted(state.stable_lengths)[
        min(len(state.stable_lengths) - 1, round((len(state.stable_lengths) - 1) * 0.75))
    ]
    if stable_length <= observed_length * 1.08:
        return line
    unit = _normalize((line[2] - line[0], line[3] - line[1]))
    if unit is None:
        return line
    if state.last_line is not None:
        start_distance = math.dist((line[0], line[1]), (state.last_line[0], state.last_line[1]))
        end_distance = math.dist((line[2], line[3]), (state.last_line[2], state.last_line[3]))
        if start_distance <= end_distance:
            return (
                line[0],
                line[1],
                line[0] + unit[0] * stable_length,
                line[1] + unit[1] * stable_length,
            )
        return (
            line[2] - unit[0] * stable_length,
            line[3] - unit[1] * stable_length,
            line[2],
            line[3],
        )
    center = _line_center(line)
    return (
        center[0] - unit[0] * stable_length / 2,
        center[1] - unit[1] * stable_length / 2,
        center[0] + unit[0] * stable_length / 2,
        center[1] + unit[1] * stable_length / 2,
    )


def _orient_line_like(line: Line, previous: Line | None) -> Line:
    if previous is None:
        return line
    same_cost = math.dist((line[0], line[1]), (previous[0], previous[1])) + math.dist(
        (line[2], line[3]), (previous[2], previous[3])
    )
    flipped_cost = math.dist(
        (line[2], line[3]), (previous[0], previous[1])
    ) + math.dist((line[0], line[1]), (previous[2], previous[3]))
    return line if same_cost <= flipped_cost else (line[2], line[3], line[0], line[1])


def _blade_waterline_overlaps(
    paddle: Line, reference: Line, band_half_width: float
) -> tuple[bool, bool]:
    first_end = (
        paddle[0],
        paddle[1],
        paddle[0] + (paddle[2] - paddle[0]) * PADDLE_BLADE_ZONE_RATIO,
        paddle[1] + (paddle[3] - paddle[1]) * PADDLE_BLADE_ZONE_RATIO,
    )
    second_end = (
        paddle[2] - (paddle[2] - paddle[0]) * PADDLE_BLADE_ZONE_RATIO,
        paddle[3] - (paddle[3] - paddle[1]) * PADDLE_BLADE_ZONE_RATIO,
        paddle[2],
        paddle[3],
    )
    overlap_distance = band_half_width * 2
    return (
        _segment_distance(first_end, reference) <= overlap_distance,
        _segment_distance(second_end, reference) <= overlap_distance,
    )


def _endpoint_signed_depths(paddle: Line, reference: Line) -> tuple[float, float]:
    reference_dx = reference[2] - reference[0]
    reference_dy = reference[3] - reference[1]
    length = max(math.hypot(reference_dx, reference_dy), 1.0)
    normal = (-reference_dy / length, reference_dx / length)
    if normal[1] < 0:
        normal = (-normal[0], -normal[1])
    origin = _line_center(reference)
    return tuple(
        (point[0] - origin[0]) * normal[0] + (point[1] - origin[1]) * normal[1]
        for point in ((paddle[0], paddle[1]), (paddle[2], paddle[3]))
    )  # type: ignore[return-value]


def _segment_distance(first: Line, second: Line) -> float:
    if _line_intersection_within_segments(first, second):
        return 0.0
    return min(
        _point_segment_distance((first[0], first[1]), second),
        _point_segment_distance((first[2], first[3]), second),
        _point_segment_distance((second[0], second[1]), first),
        _point_segment_distance((second[2], second[3]), first),
    )


def _line_intersection_within_segments(first: Line, second: Line) -> bool:
    point = _line_intersection(first, second)
    if point is None:
        return False
    tolerance = 1e-6
    return all(
        min(line[0], line[2]) - tolerance <= point[0] <= max(line[0], line[2]) + tolerance
        and min(line[1], line[3]) - tolerance
        <= point[1]
        <= max(line[1], line[3]) + tolerance
        for line in (first, second)
    )


def _point_segment_distance(point: tuple[float, float], segment: Line) -> float:
    dx = segment[2] - segment[0]
    dy = segment[3] - segment[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-9:
        return math.dist(point, (segment[0], segment[1]))
    projection = _clamp(
        ((point[0] - segment[0]) * dx + (point[1] - segment[1]) * dy)
        / length_squared,
        0,
        1,
    )
    closest = (segment[0] + projection * dx, segment[1] + projection * dy)
    return math.dist(point, closest)


def _acute_line_angle_degrees(first: Line, second: Line) -> float:
    first_vector = _normalize((first[2] - first[0], first[3] - first[1]))
    second_vector = _normalize((second[2] - second[0], second[3] - second[1]))
    if first_vector is None or second_vector is None:
        return 0.0
    dot = abs(_clamp(_dot(first_vector, second_vector), -1, 1))
    return math.degrees(math.acos(dot))


def _line_length(line: Line) -> float:
    return math.hypot(line[2] - line[0], line[3] - line[1])


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _paddle_water_depth_ratio(paddle: Line, waterline: Line) -> float | None:
    water_dx = waterline[2] - waterline[0]
    water_dy = waterline[3] - waterline[1]
    water_length = math.hypot(water_dx, water_dy)
    if water_length < 2:
        return None
    normal_x = -water_dy / water_length
    normal_y = water_dx / water_length
    if normal_y < 0:
        normal_x, normal_y = -normal_x, -normal_y
    origin_x, origin_y = _line_center(waterline)
    depths = [
        (x - origin_x) * normal_x + (y - origin_y) * normal_y
        for x, y in ((paddle[0], paddle[1]), (paddle[2], paddle[3]))
    ]
    return max(depths) / water_length


def _update_paddle_event_state(
    state: _PaddleEventState,
    instance_id: str,
    timestamp_ms: int,
    line: Line,
    depth_ratio: float,
    reference_line: Line | None = None,
) -> PaddleEvent | None:
    if state.immersed is None:
        state.immersed = depth_ratio >= PADDLE_CATCH_DEPTH_RATIO
        return None
    candidate_kind: str | None = None
    threshold = PADDLE_CATCH_DEPTH_RATIO
    if not state.immersed and depth_ratio >= PADDLE_CATCH_DEPTH_RATIO:
        candidate_kind = "catch"
        threshold = PADDLE_CATCH_DEPTH_RATIO
    elif state.immersed and depth_ratio <= PADDLE_EXIT_DEPTH_RATIO:
        candidate_kind = "exit"
        threshold = PADDLE_EXIT_DEPTH_RATIO
    if candidate_kind is None:
        state.candidate_kind = None
        state.candidate_count = 0
        state.candidate_line = None
        state.candidate_reference_line = None
        state.candidate_degree = None
        return None
    confidence = min(1.0, 0.55 + abs(depth_ratio - threshold) / 0.04)
    if state.candidate_kind != candidate_kind:
        state.candidate_kind = candidate_kind
        state.candidate_count = 1
        state.candidate_timestamp_ms = timestamp_ms
        state.candidate_line = line
        state.candidate_reference_line = reference_line
        state.candidate_degree = (
            _line_angle_degrees(line, reference_line) if reference_line is not None else None
        )
        state.candidate_confidence = confidence
        return None
    state.candidate_count += 1
    state.candidate_confidence = max(state.candidate_confidence, confidence)
    if state.candidate_count < PADDLE_EVENT_CONFIRM_SAMPLES or state.candidate_line is None:
        return None
    event = PaddleEvent(
        kind=candidate_kind,
        timestamp_ms=state.candidate_timestamp_ms,
        instance_id=instance_id,
        line=state.candidate_line,
        confidence=round(state.candidate_confidence, 3),
        reference_line=state.candidate_reference_line,
        degree=state.candidate_degree,
    )
    state.immersed = candidate_kind == "catch"
    state.candidate_kind = None
    state.candidate_count = 0
    state.candidate_line = None
    state.candidate_reference_line = None
    state.candidate_degree = None
    return event


def _dedupe_paddle_events(
    events: list[PaddleEvent], width: int, height: int
) -> list[PaddleEvent]:
    kept: list[PaddleEvent] = []
    for event in sorted(events, key=lambda item: (item.timestamp_ms, item.kind)):
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(kept)
                if previous.kind == event.kind
                and abs(previous.timestamp_ms - event.timestamp_ms) <= PADDLE_EVENT_DEDUPE_MS
                and _paddle_fragments_compatible(
                    previous.line,
                    event.line,
                    width,
                    height,
                )
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(event)
        elif (event.confidence, _line_length(event.line)) > (
            kept[duplicate_index].confidence,
            _line_length(kept[duplicate_index].line),
        ):
            kept[duplicate_index] = event
    return kept


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
    if label.line is None:
        return
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


def _draw_paddle_event_label(
    image: bytearray,
    width: int,
    height: int,
    event: PaddleEvent,
    options: ExportOptions,
) -> None:
    line_width = max(3, round(min(width, height) * 0.007))
    _draw_line(
        image,
        width,
        height,
        event.line,
        PADDLE_EVENT_PADDLE_COLOR,
        line_width,
    )
    if event.reference_line is not None:
        _draw_line(
            image,
            width,
            height,
            event.reference_line,
            PADDLE_EVENT_REFERENCE_COLOR,
            line_width,
        )
    if event.reference_line is not None and event.degree is not None:
        vertex = _line_intersection(event.line, event.reference_line)
        if vertex is not None and (
            -width * 0.1 <= vertex[0] <= width * 1.1
            and -height * 0.1 <= vertex[1] <= height * 1.1
        ):
            _draw_event_angle_marker(image, width, height, event, vertex, options)
            return
    center_x, center_y = _line_center(event.line)
    dx = event.line[2] - event.line[0]
    dy = event.line[3] - event.line[1]
    length = max(math.hypot(dx, dy), 1.0)
    offset = max(28, min(width, height) * 0.06)
    label_x = center_x + (-dy / length) * offset
    label_y = center_y + (dx / length) * offset
    _draw_line(
        image,
        width,
        height,
        (center_x, center_y, label_x, label_y),
        PADDLE_EVENT_PADDLE_COLOR,
        max(2, round(min(width, height) * 0.004)),
    )
    text = _event_label_text(event)
    if not text:
        return
    if _draw_paddle_event_label_with_pillow(
        image,
        width,
        height,
        label_x,
        label_y,
        text,
        PADDLE_EVENT_ANGLE_COLOR,
        options,
    ):
        return
    _draw_small_degree_label(
        image,
        width,
        height,
        label_x,
        label_y,
        text,
        PADDLE_EVENT_ANGLE_COLOR,
    )


def _draw_event_angle_marker(
    image: bytearray,
    width: int,
    height: int,
    event: PaddleEvent,
    vertex: tuple[float, float],
    options: ExportOptions,
) -> None:
    assert event.reference_line is not None
    reference_vector = _normalize(
        (
            event.reference_line[2] - event.reference_line[0],
            event.reference_line[3] - event.reference_line[1],
        )
    )
    paddle_vector = _normalize(
        (event.line[2] - event.line[0], event.line[3] - event.line[1])
    )
    if reference_vector is None or paddle_vector is None:
        return
    if _dot(reference_vector, paddle_vector) < 0:
        paddle_vector = (-paddle_vector[0], -paddle_vector[1])
    start_angle = math.atan2(reference_vector[1], reference_vector[0])
    end_angle = math.atan2(paddle_vector[1], paddle_vector[0])
    delta = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
    radius = max(24.0, min(width, height) * 0.065)
    line_width = max(2, round(min(width, height) * 0.004))
    for vector, color in (
        (reference_vector, PADDLE_EVENT_REFERENCE_COLOR),
        (paddle_vector, PADDLE_EVENT_PADDLE_COLOR),
    ):
        _draw_line(
            image,
            width,
            height,
            (
                vertex[0],
                vertex[1],
                vertex[0] + vector[0] * radius * 1.25,
                vertex[1] + vector[1] * radius * 1.25,
            ),
            color,
            line_width,
        )
    previous = (
        vertex[0] + math.cos(start_angle) * radius,
        vertex[1] + math.sin(start_angle) * radius,
    )
    segment_count = max(6, round(abs(delta) * radius / 8))
    for index in range(1, segment_count + 1):
        angle = start_angle + delta * index / segment_count
        point = (
            vertex[0] + math.cos(angle) * radius,
            vertex[1] + math.sin(angle) * radius,
        )
        _draw_line(
            image,
            width,
            height,
            (previous[0], previous[1], point[0], point[1]),
            PADDLE_EVENT_ANGLE_COLOR,
            line_width,
        )
        previous = point
    middle_angle = start_angle + delta / 2
    label_radius = radius + max(18, (options.angle_label_font_size or 32) * 0.65)
    label_x = vertex[0] + math.cos(middle_angle) * label_radius
    label_y = vertex[1] + math.sin(middle_angle) * label_radius
    text = _event_label_text(event)
    if not _draw_paddle_event_label_with_pillow(
        image,
        width,
        height,
        label_x,
        label_y,
        text,
        PADDLE_EVENT_ANGLE_COLOR,
        options,
    ):
        _draw_small_degree_label(
            image,
            width,
            height,
            label_x,
            label_y,
            text,
            PADDLE_EVENT_ANGLE_COLOR,
        )


def _event_label_text(event: PaddleEvent) -> str:
    if event.degree is None:
        return ""
    return f"{round(event.degree)}°"


def _line_intersection(first: Line, second: Line) -> tuple[float, float] | None:
    first_dx = first[2] - first[0]
    first_dy = first[3] - first[1]
    second_dx = second[2] - second[0]
    second_dy = second[3] - second[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) < 1e-6:
        return None
    offset_x = second[0] - first[0]
    offset_y = second[1] - first[1]
    distance = (offset_x * second_dy - offset_y * second_dx) / denominator
    return (first[0] + distance * first_dx, first[1] + distance * first_dy)


def _draw_paddle_event_label_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    text: str,
    color: Color,
    options: ExportOptions,
) -> bool:
    if not text:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    font_path = _find_export_font()
    if font_path is None:
        return False
    font_size = max(14, round((options.angle_label_font_size or 32) * 0.8))
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        return False
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_width = max(1, round(font_size * 0.08))
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    left = max(2, min(width - text_width - 2, round(center_x - text_width / 2)))
    top = max(2, min(height - text_height - 2, round(center_y - text_height / 2)))
    draw.text(
        (left - box[0], top - box[1]),
        text,
        font=font,
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 9, 255),
    )
    _blend_overlay(image, width, overlay.tobytes())
    return True


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
        if label.degree is None:
            text = ""
            text_color = (148, 163, 184, 255)
        else:
            text = f"{label.degree}°"
            text_color = (
                (255, 82, 96, 255)
                if highlight_index is not None and index - 1 == highlight_index
                else (255, 242, 168, 255)
            )
        entries.append(
            DegreeLabelEntry(
                text=f"{index}: {text}" if show_index and text else text,
                layout_text=f"{index}: 000°" if show_index else "000°",
                label=label,
                text_color=text_color,
            )
        )
    return entries


def _highlighted_degree_index(labels: list[DegreeLabel]) -> int | None:
    measured = [
        (index, label.degree)
        for index, label in enumerate(labels)
        if label.degree is not None
    ]
    if len(measured) <= 3:
        return None
    average = sum(degree for _, degree in measured) / len(measured)
    return max(
        measured,
        key=lambda item: abs(item[1] - average),
    )[0]


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
    layout_boxes = [
        draw.textbbox((0, 0), entry.layout_text, font=font, stroke_width=stroke_width)
        for entry in entries
    ]
    item_gap = max(12, round(font_size * 0.75))
    item_widths = [box[2] - box[0] for box in layout_boxes]
    item_heights = [box[3] - box[1] for box in layout_boxes]
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
        if not entry.text:
            x += item_width + item_gap
            continue
        actual_width = bbox[2] - bbox[0]
        y = value_top + (value_height - item_height) / 2
        draw.text(
            (x + (item_width - actual_width) / 2 - bbox[0], y - bbox[1]),
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
    layout_glyph_items = [
        [_glyph(character) for character in entry.layout_text] for entry in entries
    ]
    item_widths = [
        sum(len(glyph[0]) * scale for glyph in glyphs) + gap * max(0, len(glyphs) - 1)
        for glyphs in layout_glyph_items
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
        glyph_width = sum(len(glyph[0]) * scale for glyph in glyphs) + gap * max(
            0, len(glyphs) - 1
        )
        glyph_x = x + round((item_width - glyph_width) / 2)
        for glyph in glyphs:
            _draw_bitmap(
                image,
                width,
                height,
                glyph_x + scale,
                top + scale,
                glyph,
                scale,
                (2, 5, 9, 255),
            )
            _draw_bitmap(image, width, height, glyph_x, top, glyph, scale, entry.text_color)
            glyph_x += len(glyph[0]) * scale + gap
        x += item_width + item_gap


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
