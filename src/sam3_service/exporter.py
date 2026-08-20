from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator

from .body_motion import MIN_LANDMARK_CONFIDENCE, load_body_motion_frames_by_index
from .errors import ServiceError
from .media import probe_video
from .paddle_completeness import build_paddle_completeness_predictor

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
PADDLE_FRAGMENT_MIN_PERPENDICULAR_PIXELS = 24.0
PADDLE_FRAGMENT_PERPENDICULAR_FRAME_RATIO = 0.04
PADDLE_REFLECTION_CLOSE_FRAME_RATIO = 0.25
PADDLE_REFLECTION_MIN_CLOSE_PIXELS = 24.0
PADDLE_EVENT_SLOT_MAX_SPACING_RATIO = 0.4
PADDLE_EVENT_SLOT_MIN_GATE_PIXELS = 24.0
BOAT_REFERENCE_LENGTH_MAX_INNOVATION_RATIO = 0.15
BOAT_REFERENCE_LENGTH_PROCESS_NOISE_RATIO = 0.01
BOAT_REFERENCE_LENGTH_MEASUREMENT_NOISE_RATIO = 0.03
BOAT_REFERENCE_LENGTH_MIN_NOISE_PIXELS = 2.0
PADDLE_DIRECTION_MIN_DELTAS = 5
PADDLE_DIRECTION_MIN_DISPLACEMENT_DEGREES = 45.0
PADDLE_DIRECTION_MIN_CONSENSUS = 0.75
PADDLE_DIRECTION_MAX_SAMPLE_GAP_MS = 400
CANOE_DIRECTION_MIN_VOTES = 5
CANOE_DIRECTION_MIN_CONSENSUS = 0.70
CANOE_DIRECTION_RIGHT_MAX_DEGREES = 80.0
CANOE_DIRECTION_LEFT_MIN_DEGREES = 100.0
CANOE_DIRECTION_MIN_AXIS_SPAN_DEGREES = 8.0
CANOE_PHASE_MIN_TREND_DEGREES = 3.0
CANOE_PHASE_EXIT_CONFIRM_SAMPLES = 2
CANOE_CATCH_LENGTH_RESTORE_TOLERANCE = 0.15
CANOE_CONTACT_LENGTH_RATIO = 0.86
CANOE_RELEASE_LENGTH_RATIO = 0.92
CANOE_MIN_LENGTH_CHANGE_RATIO = 0.06
CANOE_CONTACT_MIN_SAMPLES = 2
PADDLE_PHASE_BACKTRACK_TOLERANCE_DEGREES = 15.0
PADDLE_PHASE_ANCHOR_AGREEMENT_RELATIVE_TOLERANCE = 0.15
PADDLE_PHASE_CANDIDATE_RELATIVE_TOLERANCE = 0.10
PADDLE_DEPTH_MOTION_EPSILON_PIXELS = 0.5
WATERLINE_BOAT_AXIS_MAX_ANGLE_DEGREES = 20.0
EXPORT_END_GUARD_FRAMES = 2
PADDLE_EVENT_PADDLE_COLOR = (0, 229, 255, 255)
PADDLE_EVENT_REFERENCE_COLOR = (255, 196, 61, 255)
PADDLE_EVENT_CATCH_ANGLE_COLOR = (255, 82, 96, 255)
PADDLE_EVENT_EXIT_ANGLE_COLOR = (46, 204, 113, 255)
PADDLE_ANGLE_TEXT_COLOR = (255, 242, 168, 255)
BODY_LEFT_COLOR = (52, 211, 153, 235)
BODY_RIGHT_COLOR = (244, 114, 182, 235)
BODY_LEFT_ELBOW_COLOR = (52, 211, 153, 255)
BODY_RIGHT_ELBOW_COLOR = (244, 114, 182, 255)
BODY_TORSO_COLOR = PADDLE_ANGLE_TEXT_COLOR
BODY_LEFT_SHOULDER_COLOR = (56, 189, 248, 255)
BODY_RIGHT_SHOULDER_COLOR = (251, 146, 60, 255)
BODY_LEFT_KNEE_COLOR = (167, 139, 250, 255)
BODY_RIGHT_KNEE_COLOR = (250, 204, 21, 255)


@dataclass(frozen=True)
class ExportOptions:
    angle_label_position: LabelPosition = "top"
    angle_label_font_size: int | None = None
    include_angles: bool = True
    include_spm: bool = False
    include_catch: bool = False
    include_exit: bool = False
    include_event_freeze: bool = False
    event_hold_seconds: float = 1.2
    include_event_metrics: bool = False
    include_body_motion: bool = False
    event_metric_center_offset_percent: float = 5.5
    metric_center_offset_percent: float | None = None
    reference_prompt_id: str | None = None
    reference_line_mode: str | None = None
    target_prompt_ids: tuple[str, ...] = ()
    reference_track_ids: tuple[str, ...] = ()
    target_track_ids: tuple[str, ...] = ()
    selection_rect: SelectionRect | None = None
    selection_keyframes: tuple[SelectionKeyframe, ...] = ()
    target_slot_count: int = 0
    event_paddle_index: int | None = None


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
    phase_angle: float | None = None
    cycle_index: int | None = None
    active_blade: int | None = None
    rotation_direction: str | None = None
    travel_direction: str | None = None
    discipline: str = "kayak"


@dataclass(frozen=True)
class FreezeMoment:
    frame_index: int
    events: tuple[PaddleEvent, ...]


@dataclass(frozen=True)
class EventMetricColumn:
    """One selected-paddle event and the aligned values for every metric slot."""

    event: PaddleEvent
    values: tuple[int | None, ...]


@dataclass(frozen=True)
class _PaddleObservation:
    source_ids: tuple[str, ...]
    reference_id: str
    line: Line
    reference_line: Line
    raw_line: Line | None = None
    phase_length_restored: bool = False
    phase_length_verified: bool = True


@dataclass(frozen=True)
class _TimedPaddleObservation:
    timestamp_ms: int
    physical_id: str
    observation: _PaddleObservation


@dataclass(frozen=True)
class _CanoeSample:
    timed: _TimedPaddleObservation
    blade: int
    blade_position: float
    air_length_ratio: float
    contact: bool


@dataclass(frozen=True)
class _CanoePullInterval:
    physical_id: str
    reference_id: str
    entry: _CanoeSample
    exit: _CanoeSample
    release: _CanoeSample
    minimum_air_length_ratio: float


@dataclass(frozen=True)
class _CanoePhaseSample:
    timed: _TimedPaddleObservation
    line: Line
    restored_line: Line
    blade: int
    angle: float


@dataclass(frozen=True)
class _CanoePhase:
    start: _CanoePhaseSample
    peak: _CanoePhaseSample
    end: _CanoePhaseSample
    complete: bool


@dataclass
class _BoatReferenceLengthFilter:
    length: float
    variance: float
    last_timestamp_ms: int | None = None


@dataclass
class _PaddleEventCandidate:
    kind: str
    blade: int
    cycle_index: int
    timestamp_ms: int
    line: Line
    reference_line: Line
    phase_angle: float
    count: int = 1
    confidence: float = 0.0


@dataclass
class _PaddleEventState:
    physical_id: str = ""
    source_ids: set[str] = field(default_factory=set)
    reference_id: str = ""
    # All selected observations maintain physical endpoint continuity. The
    # legacy last_seen/line/depth fields below contain event-eligible evidence.
    last_observation_ms: int = -1
    last_orientation_line: Line | None = None
    last_seen_ms: int = -1
    last_line: Line | None = None
    last_reference_line: Line | None = None
    stable_lengths: list[float] = field(default_factory=list)
    stroke_length: float | None = None
    stroke_blade: int | None = None
    stroke_cycle_index: int | None = None
    stroke_phase_half: int | None = None
    rotation_direction: str | None = None
    travel_direction: str | None = None
    direction_confidence: float = 0.0
    active_blade: int | None = None
    last_directed_angle: float | None = None
    unwrapped_angle: float | None = None
    cycle_index: int = 0
    phase_confident: bool = False
    endpoint_depths: tuple[float, float] | None = None
    emitted_events: set[tuple[int, str]] = field(default_factory=set)
    candidates: dict[str, _PaddleEventCandidate] = field(default_factory=dict)
    # Legacy depth-only state is retained for the compatibility helper below.
    immersed: bool | None = None
    candidate_kind: str | None = None
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
    body_motion_chunk_paths: list[Path] | None = None,
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
    frame_count = _resolved_export_frame_count(video_metadata, fps)
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ServiceError("EXPORT_FAILED", "Result manifest has invalid video metadata.")

    export_options = _normalize_export_options(options, manifest, width, height)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".tmp.mp4")

    colors = {
        prompt["id"]: _parse_hex(prompt.get("color", "#35C2FF"))
        for prompt in manifest.get("prompts", [])
    }
    frames = _load_frames_by_timestamp(chunk_paths)
    frames_by_index = _load_frames_by_index(chunk_paths)
    body_motion_frames = load_body_motion_frames_by_index(body_motion_chunk_paths or [])
    body_motion_discipline = str(
        manifest.get("body_motion", {}).get("discipline", "kayak")
    )
    frame_timestamps = sorted(frames)
    scale_x = width / manifest_width if manifest_width > 0 else 1.0
    scale_y = height / manifest_height if manifest_height > 0 else 1.0
    discipline = str(
        manifest.get("settings", {}).get("paddling_discipline", "kayak")
    )
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
        events = _detect_paddle_events_for_discipline(
            frames,
            export_options,
            width,
            height,
            scale_x,
            scale_y,
            discipline=discipline,
            body_motion_frames=body_motion_frames,
            progress=progress,
        )
    freeze_moments = _freeze_moments(
        events,
        fps,
        frame_count,
        max_events_per_moment=export_options.target_slot_count,
    )
    canoe_travel_direction = _canoe_export_travel_direction(
        events,
        body_motion_frames=body_motion_frames,
        frames=frames,
        options=export_options,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
        discipline=discipline,
        needs_body_motion=export_options.include_body_motion,
        progress=progress,
    )
    overlay_reference_lines = (
        _canoe_overlay_reference_lines(
            frames,
            export_options,
            width,
            height,
            scale_x,
            scale_y,
            body_motion_frames,
            progress,
        )
        if discipline == "canoe"
        else {}
    )
    result_tolerance_ms = max(1000 / max(fps, 1), 500 / max(manifest_fps, 1), 40)
    freeze_by_frame = {moment.frame_index: moment for moment in freeze_moments}
    freeze_frame_count = _event_freeze_frame_count(export_options, fps)
    has_audio = (
        bool(freeze_moments)
        and freeze_frame_count > 0
        and _has_audio_stream(video_path)
    )
    filter_parts: list[str] = []
    if has_audio:
        filter_parts.append(
            _freeze_audio_filter(
                freeze_moments,
                freeze_frame_count,
                fps,
                frame_count,
                input_label="1:a",
            )
        )
    filter_parts.append("[0:v]null[v]")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(video_path),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[v]",
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
        command.extend(["-map", "1:a?", "-c:a", "copy"])
    command.extend(["-movflags", "+faststart", str(temporary_output)])

    _report_progress(progress, "rendering", 15, "Rendering overlay frames")
    spm_estimator = SpmEstimator()
    output_frame_index = 0
    decoded_frame_count = 0
    final_image: bytearray | None = None
    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    assert encoder.stderr is not None
    try:
        for frame_index, image in enumerate(
            _decode_rgba_video_frames(video_path, width, height, fps)
        ):
            decoded_frame_count += 1
            timestamp_ms = round(frame_index * 1000 / fps)
            freeze_moment = freeze_by_frame.get(frame_index)
            body_motion_record = body_motion_frames.get(frame_index)
            records = (
                frames_by_index.get(frame_index, [])
                if frames_by_index is not None
                else _records_for_timestamp(
                    frames, frame_timestamps, timestamp_ms, result_tolerance_ms
                )
            )
            scaled_records = [_scale_record(record, scale_x, scale_y) for record in records]
            scaled_records = [
                record
                for record in scaled_records
                if _record_selected_for_export(
                    record, export_options, width, height, timestamp_ms
                )
            ]
            # Event geometry is backdated to the actual waterline crossing and can
            # differ from the regular geometry stored for this source frame. Keep
            # a clean copy so a held event frame does not show both line sets.
            event_image = bytearray(image) if freeze_moment is not None else None
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
                reference_lines_by_track=overlay_reference_lines,
            )
            if export_options.include_body_motion and body_motion_record is not None:
                _draw_body_motion_overlay(
                    image,
                    width,
                    height,
                    body_motion_record,
                    discipline=body_motion_discipline,
                    canoe_travel_direction=canoe_travel_direction,
                    angle_label_font_size=export_options.angle_label_font_size,
                    metric_offset_percent=(
                        export_options.event_metric_center_offset_percent
                    ),
                )
            if freeze_moment is not None:
                assert event_image is not None
                if export_options.include_body_motion and body_motion_record is not None:
                    _draw_body_motion_overlay(
                        event_image,
                        width,
                        height,
                        body_motion_record,
                        discipline=body_motion_discipline,
                        canoe_travel_direction=canoe_travel_direction,
                        angle_label_font_size=export_options.angle_label_font_size,
                        metric_offset_percent=(
                            export_options.event_metric_center_offset_percent
                        ),
                    )
                if export_options.include_angles:
                    _draw_event_companion_angles(
                        event_image,
                        width,
                        height,
                        scaled_records,
                        colors,
                        export_options,
                        freeze_moment.events,
                        timestamp_ms=timestamp_ms,
                        reference_lines_by_track=overlay_reference_lines,
                    )
                for event in freeze_moment.events:
                    _draw_paddle_event_label(
                        event_image, width, height, event, export_options
                    )
                for _ in range(freeze_frame_count):
                    encoder.stdin.write(event_image)
                    output_frame_index += 1
            output_image = (
                event_image
                if freeze_moment is not None
                and freeze_frame_count == 0
                else image
            )
            encoder.stdin.write(output_image)
            output_frame_index += 1
            final_image = output_image
            if frame_index == frame_count - 1 or frame_index % max(1, frame_count // 100) == 0:
                percent = 15 + 75 * (frame_index + 1) / frame_count
                _report_progress(
                    progress,
                    "rendering",
                    percent,
                    f"Rendering frame {frame_index + 1} of {frame_count}",
                )

        if final_image is None:
            raise ServiceError(
                "EXPORT_FAILED",
                "Video decoding produced no frames.",
                retryable=True,
                status_code=500,
            )
        for _ in range(EXPORT_END_GUARD_FRAMES):
            encoder.stdin.write(final_image)
            output_frame_index += 1
        _report_progress(progress, "encoding", 92, "Finalizing MP4")
        encoder.stdin.close()
        detail = encoder.stderr.read().decode("utf-8", errors="replace")[-1000:]
        return_code = encoder.wait(timeout=3600)
        if return_code != 0:
            raise ServiceError(
                "EXPORT_FAILED",
                f"Video encoding failed: {detail}",
                retryable=True,
                status_code=500,
            )
        temporary_output.replace(output_path)
        _report_progress(progress, "finalizing", 99, "Finalizing export")
    except ServiceError:
        temporary_output.unlink(missing_ok=True)
        raise
    except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
        temporary_output.unlink(missing_ok=True)
        detail = encoder.stderr.read().decode("utf-8", errors="replace")[-1000:]
        raise ServiceError(
            "EXPORT_FAILED",
            f"Video export failed: {detail}",
            retryable=True,
            status_code=500,
        ) from exc
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return output_path


def _resolved_export_frame_count(video_metadata: dict[str, Any], fps: float) -> int:
    detected_frame_count = int(video_metadata.get("frame_count") or 0)
    if detected_frame_count > 0:
        return detected_frame_count
    duration_ms = float(video_metadata.get("duration_ms") or 0)
    return max(0, round(duration_ms * fps / 1000))


def _decode_rgba_video_frames(
    video_path: Path,
    width: int,
    height: int,
    fps: float,
) -> Iterator[bytearray]:
    frame_size = width * height * 4
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=fps={fps:.6f}:start_time=0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        frame_index = 0
        while True:
            chunks: list[bytes] = []
            remaining = frame_size
            while remaining > 0:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == frame_size:
                break
            if remaining:
                detail = process.stderr.read().decode("utf-8", errors="replace")[-1000:]
                raise ServiceError(
                    "EXPORT_FAILED",
                    f"Video decoding stopped at frame {frame_index}: {detail}",
                    retryable=True,
                    status_code=500,
                )
            yield bytearray(b"".join(chunks))
            frame_index += 1
        process.stdout.close()
        detail = process.stderr.read().decode("utf-8", errors="replace")[-1000:]
        return_code = process.wait(timeout=60)
        if return_code != 0:
            raise ServiceError(
                "EXPORT_FAILED",
                f"Video decoding failed: {detail}",
                retryable=True,
                status_code=500,
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _report_progress(
    progress: Callable[[str, float, str], None] | None,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if progress is not None:
        progress(stage, max(0.0, min(100.0, percent)), message)


def _freeze_moments(
    events: list[PaddleEvent],
    fps: float,
    frame_count: int,
    max_events_per_moment: int = 0,
) -> tuple[FreezeMoment, ...]:
    if not events or frame_count <= 0:
        return ()
    groups: list[list[PaddleEvent]] = []
    for event in sorted(events, key=lambda item: item.timestamp_ms):
        if (
            groups
            and event.kind == groups[-1][0].kind
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
        FreezeMoment(
            frame_index=frame_index,
            events=tuple(
                sorted(
                    group,
                    key=lambda event: (event.confidence, _line_length(event.line)),
                    reverse=True,
                )[:max_events_per_moment]
                if max_events_per_moment > 0
                else group
            ),
        )
        for frame_index, group in sorted(events_by_frame.items())
    )


def _event_freeze_frame_count(options: ExportOptions, fps: float) -> int:
    if not options.include_event_freeze:
        return 0
    return max(1, round(options.event_hold_seconds * fps))


def _freeze_audio_filter(
    moments: tuple[FreezeMoment, ...],
    freeze_frames: int,
    fps: float,
    frame_count: int,
    input_label: str = "0:a",
) -> str:
    segments = _freeze_segments(moments, frame_count)
    branches = "".join(f"[apart{index}]" for index in range(len(segments)))
    graph = [f"[{input_label}]anull[apart0]"]
    if len(segments) > 1:
        graph[0] = f"[{input_label}]asplit={len(segments)}{branches}"
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


def _load_frames_by_index(
    chunk_paths: list[Path],
) -> dict[int, list[dict[str, Any]]] | None:
    """Load exact source-frame mappings, or None for legacy result chunks.

    An empty list for a known frame is meaningful: it means that frame has no
    detections and must not inherit geometry from a neighboring timestamp.
    """
    frames: dict[int, list[dict[str, Any]]] = {}
    found_frame_index = False
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("frames", []):
            frame_index = record.get("frame_index")
            if frame_index is None:
                continue
            try:
                index = int(frame_index)
            except (TypeError, ValueError):
                continue
            found_frame_index = True
            frames.setdefault(index, []).append(record)
    return frames if found_frame_index else None


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
        include_event_freeze=bool(requested.include_event_freeze),
        event_hold_seconds=max(0.1, min(10.0, float(requested.event_hold_seconds))),
        include_event_metrics=bool(requested.include_event_metrics),
        include_body_motion=bool(requested.include_body_motion),
        event_metric_center_offset_percent=max(
            -45.0,
            min(45.0, float(requested.event_metric_center_offset_percent)),
        ),
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
        event_paddle_index=(
            requested.event_paddle_index
            if requested.event_paddle_index is not None
            and requested.event_paddle_index > 0
            else None
        ),
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


def _canoe_export_travel_direction(
    events: list[PaddleEvent],
    *,
    body_motion_frames: dict[int, dict[str, Any]] | None = None,
    frames: dict[int, list[dict[str, Any]]] | None = None,
    options: ExportOptions | None = None,
    width: int | None = None,
    height: int | None = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    discipline: str = "kayak",
    needs_body_motion: bool = False,
    progress: Callable[[str, float, str], None] | None = None,
) -> str | None:
    for event in events:
        if event.discipline == "canoe" and event.travel_direction in {"left", "right"}:
            return event.travel_direction
    if discipline == "canoe":
        body_direction, _confidence = _canoe_body_travel_direction(
            body_motion_frames or {}
        )
        if body_direction in {"left", "right"}:
            return body_direction
    if (
        not needs_body_motion
        or discipline != "canoe"
        or frames is None
        or options is None
        or width is None
        or height is None
    ):
        return None
    tracks = _track_paddle_observations(
        frames,
        options,
        width,
        height,
        scale_x,
        scale_y,
        progress,
    )
    direction_votes: dict[str, float] = {}
    for _reference_id, (_source, direction, confidence) in _canoe_directions(
        tracks
    ).items():
        if direction in {"left", "right"}:
            direction_votes[direction] = direction_votes.get(direction, 0.0) + confidence
    if not direction_votes:
        return None
    return max(direction_votes, key=direction_votes.get)


def _canoe_overlay_reference_lines(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    body_motion_frames: dict[int, dict[str, Any]],
    progress: Callable[[str, float, str], None] | None = None,
) -> dict[tuple[str, int], Line]:
    tracks = _track_paddle_observations(
        frames,
        options,
        width,
        height,
        scale_x,
        scale_y,
        progress,
    )
    directions = _canoe_directions(tracks)
    body_direction = _canoe_body_travel_direction(body_motion_frames)
    if body_direction[0] in {"left", "right"}:
        directions = {
            reference_id: ("body_motion", body_direction[0], body_direction[1])
            for reference_id in {
                timed.observation.reference_id
                for observations in tracks.values()
                for timed in observations
            }
        }
    return _stabilized_event_reference_lines(
        _event_boat_reference_lines(
            frames,
            options,
            width,
            height,
            scale_x,
            scale_y,
        ),
        directions,
    )


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
    reference_lines_by_track: dict[tuple[str, int], Line] | None = None,
) -> None:
    centerlines: list[Centerline] = []
    for record in records:
        line = _display_record_line(
            record,
            export_options,
            width,
            height,
            timestamp_ms,
            reference_lines_by_track,
        )
        if line is None:
            continue
        color = colors.get(record["prompt_id"], (53, 194, 255, 255))
        centerlines.append(Centerline(record=record, line=line, color=color))
        _draw_line(image, width, height, line, color, max(3, round(min(width, height) * 0.006)))

    labels = _degree_labels(centerlines, export_options)
    displayed_labels = _degree_slots(labels, export_options)
    if export_options.include_angles:
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


def _draw_event_companion_angles(
    image: bytearray,
    width: int,
    height: int,
    records: list[dict[str, Any]],
    colors: dict[str, Color],
    options: ExportOptions,
    events: tuple[PaddleEvent, ...],
    *,
    timestamp_ms: int,
    reference_lines_by_track: dict[tuple[str, int], Line] | None = None,
) -> None:
    """Draw all paddle angles in the top/bottom block on an event frame."""
    if not events:
        return
    centerlines: list[Centerline] = []
    for record in records:
        line = _display_record_line(
            record,
            options,
            width,
            height,
            timestamp_ms,
            reference_lines_by_track,
        )
        if line is None:
            continue
        centerlines.append(
            Centerline(
                record=record,
                line=line,
                color=colors.get(record.get("prompt_id", ""), (53, 194, 255, 255)),
            )
        )
    slots = _event_frame_degree_slots(centerlines, options, events)
    if not any(slot.degree is not None for slot in slots):
        return
    line_width = max(3, round(min(width, height) * 0.006))
    for entry in _degree_label_entries(slots):
        if entry.label.degree is None or entry.label.line is None:
            continue
        _draw_line(
            image,
            width,
            height,
            entry.label.line,
            entry.label.color,
            line_width,
        )
    event_text_colors = {
        index: slot.color
        for index, slot in enumerate(slots)
        if slot.degree is not None and slot.line is None
    }
    _draw_degree_label_block(
        image,
        width,
        height,
        slots,
        options,
        text_color_overrides=event_text_colors,
    )


def _event_companion_degree_slots(
    centerlines: list[Centerline],
    options: ExportOptions,
    event: PaddleEvent,
) -> list[DegreeLabel]:
    """Return directional metric slots with the event angle in its selected slot."""
    if options.target_slot_count <= 0 or options.event_paddle_index is None:
        return []
    return _event_frame_degree_slots(centerlines, options, (event,))


def _event_frame_degree_slots(
    centerlines: list[Centerline],
    options: ExportOptions,
    events: tuple[PaddleEvent, ...],
) -> list[DegreeLabel]:
    """Use event-time angles in the ordinary top/bottom paddle metric slots."""
    labels = _degree_labels(centerlines, options)
    if options.event_paddle_index is not None and events[0].travel_direction == "right":
        labels.reverse()
    slots = _degree_slots(labels, options)
    if options.event_paddle_index is not None:
        selected_indexes = [options.event_paddle_index - 1]
    else:
        available_indexes = {
            index for index, slot in enumerate(slots) if slot.line is not None
        }
        selected_indexes = []
        for event in events:
            if not available_indexes:
                break
            event_center = _line_center(event.line)
            selected_index = min(
                available_indexes,
                key=lambda index: math.hypot(
                    _line_center(slots[index].line)[0] - event_center[0],
                    _line_center(slots[index].line)[1] - event_center[1],
                ),
            )
            available_indexes.remove(selected_index)
            selected_indexes.append(selected_index)
    for selected_index, event in zip(selected_indexes, events):
        if not 0 <= selected_index < len(slots):
            continue
        selected = slots[selected_index]
        display_angle = _event_display_angle(event)
        slots[selected_index] = replace(
            selected,
            degree=round(display_angle) % 360 if display_angle is not None else None,
            line=None,
            color=_paddle_event_angle_color(event),
        )
    return slots


def _build_event_metric_columns(
    events: list[PaddleEvent],
    frames: dict[int, list[dict[str, Any]]],
    frames_by_index: dict[int, list[dict[str, Any]]] | None,
    frame_timestamps: list[int],
    colors: dict[str, Color],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    fps: float,
    result_tolerance_ms: float,
    reference_lines_by_track: dict[tuple[str, int], Line] | None = None,
) -> tuple[EventMetricColumn, ...]:
    """Precompute the fixed event-table columns before rendering any frames."""
    columns: list[EventMetricColumn] = []
    for event in sorted(events, key=lambda item: item.timestamp_ms):
        frame_index = max(0, round(event.timestamp_ms * fps / 1000))
        records = (
            frames_by_index.get(frame_index, [])
            if frames_by_index is not None
            else _records_for_timestamp(
                frames,
                frame_timestamps,
                event.timestamp_ms,
                result_tolerance_ms,
            )
        )
        scaled_records = [_scale_record(record, scale_x, scale_y) for record in records]
        scaled_records = [
            record
            for record in scaled_records
            if _record_selected_for_export(
                record,
                options,
                width,
                height,
                event.timestamp_ms,
            )
        ]
        centerlines = _centerlines_for_records(
            scaled_records,
            colors,
            options,
            width,
            height,
            event.timestamp_ms,
            reference_lines_by_track,
        )
        columns.append(
            EventMetricColumn(
                event=event,
                values=_event_metric_values(centerlines, options, event),
            )
        )
    return tuple(columns)


def _centerlines_for_records(
    records: list[dict[str, Any]],
    colors: dict[str, Color],
    options: ExportOptions,
    width: int,
    height: int,
    timestamp_ms: int | None = None,
    reference_lines_by_track: dict[tuple[str, int], Line] | None = None,
) -> list[Centerline]:
    centerlines: list[Centerline] = []
    for record in records:
        line = _display_record_line(
            record,
            options,
            width,
            height,
            timestamp_ms,
            reference_lines_by_track,
        )
        if line is None:
            continue
        centerlines.append(
            Centerline(
                record=record,
                line=line,
                color=colors.get(record.get("prompt_id", ""), (53, 194, 255, 255)),
            )
        )
    return centerlines


def _display_record_line(
    record: dict[str, Any],
    options: ExportOptions,
    width: int,
    height: int,
    timestamp_ms: int | None = None,
    reference_lines_by_track: dict[tuple[str, int], Line] | None = None,
) -> Line | None:
    is_reference = record.get("prompt_id") == options.reference_prompt_id
    if is_reference and timestamp_ms is not None and reference_lines_by_track:
        line = reference_lines_by_track.get((_record_track_id(record), timestamp_ms))
        if line is not None:
            return line
    return _record_line(
        record,
        width,
        height,
        use_waterline=is_reference and options.reference_line_mode == "waterline",
    )


def _event_metric_values(
    centerlines: list[Centerline],
    options: ExportOptions,
    event: PaddleEvent,
) -> tuple[int | None, ...]:
    """Return the selected raw angle and signed companion differences."""
    slot_count = max(1, options.target_slot_count)
    selected_index = (options.event_paddle_index or 1) - 1
    selected_angle = _event_display_angle(event)
    selected_degree = (
        round(selected_angle) % 360 if selected_angle is not None else None
    )
    labels = _degree_labels(centerlines, options)
    if event.travel_direction == "right":
        labels.reverse()
    slots = _degree_slots(labels, replace(options, target_slot_count=slot_count))
    values: list[int | None] = []
    for index in range(slot_count):
        if index == selected_index:
            values.append(selected_degree)
            continue
        degree = slots[index].degree if index < len(slots) else None
        if degree is None or selected_degree is None:
            values.append(None)
            continue
        aligned_degree = _aligned_companion_degree(degree, selected_degree)
        values.append(_signed_degree_difference(aligned_degree, selected_degree))
    return tuple(values)


def _aligned_companion_degree(acute_degree: int, reference_degree: int) -> int:
    """Choose the directed equivalent of an acute metric nearest the event angle."""
    candidates = {
        acute_degree % 360,
        (180 - acute_degree) % 360,
        (180 + acute_degree) % 360,
        (360 - acute_degree) % 360,
    }
    return min(
        candidates,
        key=lambda degree: abs(_signed_degree_difference(degree, reference_degree)),
    )


def _signed_degree_difference(degree: int, reference_degree: int) -> int:
    difference = (degree - reference_degree + 180) % 360 - 180
    return int(difference)


def _event_metric_text(
    column: EventMetricColumn,
    row_index: int,
    selected_index: int,
) -> str:
    value = column.values[row_index] if row_index < len(column.values) else None
    if value is None:
        return "--"
    if row_index == selected_index:
        return f"{value}°"
    return f"{value:+d}°" if value else "0°"


def _draw_event_metric_table(
    image: bytearray,
    width: int,
    height: int,
    columns: tuple[EventMetricColumn, ...],
    timestamp_ms: int,
    options: ExportOptions,
) -> None:
    if not columns or options.event_paddle_index is None:
        return
    visible_count = sum(column.event.timestamp_ms <= timestamp_ms for column in columns)
    if visible_count <= 0:
        return
    if _draw_event_metric_table_with_pillow(
        image,
        width,
        height,
        columns,
        visible_count,
        options,
    ):
        return
    _draw_event_metric_table_bitmap(
        image,
        width,
        height,
        columns,
        visible_count,
        options,
    )


def _draw_event_metric_table_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    columns: tuple[EventMetricColumn, ...],
    visible_count: int,
    options: ExportOptions,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    font_path = _find_export_font()
    if font_path is None:
        return False

    row_count = max(1, options.target_slot_count)
    selected_index = options.event_paddle_index - 1
    max_font_size = int(options.angle_label_font_size or max(18, round(height * 0.045)))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    layout: tuple[Any, int, int, int, int, int] | None = None
    for font_size in range(max_font_size, 5, -1):
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except OSError:
            return False
        stroke_width = max(1, round(font_size * 0.05))
        label_width = max(
            draw.textbbox((0, 0), f"P{row_count}", font=font, stroke_width=stroke_width)[2],
            draw.textbbox((0, 0), "P1", font=font, stroke_width=stroke_width)[2],
        )
        cell_bbox = draw.textbbox(
            (0, 0), "-000°", font=font, stroke_width=stroke_width
        )
        cell_width = cell_bbox[2] - cell_bbox[0] + max(4, round(font_size * 0.35))
        row_height = cell_bbox[3] - cell_bbox[1] + max(3, round(font_size * 0.3))
        label_gap = max(5, round(font_size * 0.45))
        column_gap = max(2, round(font_size * 0.15))
        table_width = (
            label_width
            + label_gap
            + len(columns) * cell_width
            + max(0, len(columns) - 1) * column_gap
        )
        table_height = row_count * row_height
        if table_width <= width * 0.96 and table_height <= height * 0.45:
            layout = (
                font,
                font_size,
                stroke_width,
                label_width,
                cell_width,
                row_height,
            )
            break
    if layout is None:
        return False
    font, font_size, stroke_width, label_width, cell_width, row_height = layout
    label_gap = max(5, round(font_size * 0.45))
    column_gap = max(2, round(font_size * 0.15))
    table_width = (
        label_width
        + label_gap
        + len(columns) * cell_width
        + max(0, len(columns) - 1) * column_gap
    )
    table_height = row_count * row_height
    left = round((width - table_width) / 2)
    top = _event_metric_table_top(
        height,
        table_height,
        options.event_metric_center_offset_percent,
    )
    for row_index in range(row_count):
        row_top = top + row_index * row_height
        row_label = f"P{row_index + 1}"
        label_bbox = draw.textbbox(
            (0, 0), row_label, font=font, stroke_width=stroke_width
        )
        draw.text(
            (
                left + label_width - (label_bbox[2] - label_bbox[0]) - label_bbox[0],
                row_top + (row_height - (label_bbox[3] - label_bbox[1])) / 2 - label_bbox[1],
            ),
            row_label,
            font=font,
            fill=(235, 245, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(2, 5, 9, 255),
        )
        cell_left = left + label_width + label_gap
        for column_index, column in enumerate(columns):
            if column_index >= visible_count:
                break
            text = _event_metric_text(column, row_index, selected_index)
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
            actual_width = bbox[2] - bbox[0]
            actual_height = bbox[3] - bbox[1]
            x = cell_left + column_index * (cell_width + column_gap)
            draw.text(
                (
                    x + (cell_width - actual_width) / 2 - bbox[0],
                    row_top + (row_height - actual_height) / 2 - bbox[1],
                ),
                text,
                font=font,
                fill=_paddle_event_angle_color(column.event),
                stroke_width=stroke_width,
                stroke_fill=(2, 5, 9, 255),
            )
    _blend_overlay(image, width, overlay.tobytes())
    return True


def _draw_event_metric_table_bitmap(
    image: bytearray,
    width: int,
    height: int,
    columns: tuple[EventMetricColumn, ...],
    visible_count: int,
    options: ExportOptions,
) -> None:
    row_count = max(1, options.target_slot_count)
    selected_index = (options.event_paddle_index or 1) - 1
    max_scale = max(1, round((options.angle_label_font_size or 32) / 7))
    scale = max_scale
    while scale > 1:
        cell_width = 6 * 5 * scale
        label_width = 8 * scale
        table_width = label_width + len(columns) * cell_width
        if table_width <= width * 0.96 and row_count * 9 * scale <= height * 0.45:
            break
        scale -= 1
    cell_width = 6 * 5 * scale
    label_width = 8 * scale
    row_height = 9 * scale
    table_width = label_width + len(columns) * cell_width
    table_height = row_count * row_height
    left = round((width - table_width) / 2)
    top = _event_metric_table_top(
        height,
        table_height,
        options.event_metric_center_offset_percent,
    )
    for row_index in range(row_count):
        _draw_bitmap_text(
            image,
            width,
            height,
            left,
            top + row_index * row_height,
            f"P{row_index + 1}",
            scale,
            (235, 245, 255, 255),
        )
        for column_index, column in enumerate(columns[:visible_count]):
            text = _event_metric_text(column, row_index, selected_index)
            _draw_bitmap_text(
                image,
                width,
                height,
                left + label_width + column_index * cell_width,
                top + row_index * row_height,
                text,
                scale,
                _paddle_event_angle_color(column.event),
            )


def _draw_bitmap_text(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    text: str,
    scale: int,
    color: Color,
) -> None:
    gap = max(1, scale // 2)
    x = left
    for character in text:
        glyph = _glyph(character)
        _draw_bitmap(image, width, height, x + scale, top + scale, glyph, scale, (2, 5, 9, 255))
        _draw_bitmap(image, width, height, x, top, glyph, scale, color)
        x += len(glyph[0]) * scale + gap


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
    if geometry == "waterline":
        line = _extend_waterline_to_boat_span(record, line, width, height)
    return line  # type: ignore[return-value]


def _record_geometry_line(
    record: dict[str, Any],
    width: int,
    height: int,
    geometry: str,
) -> Line | None:
    values = record.get(f"{geometry}_line_xyxy")
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
    if geometry == "waterline":
        line = _extend_waterline_to_boat_span(record, line, width, height)
    return line


def _event_reference_line(
    record: dict[str, Any],
    options: ExportOptions,
    width: int,
    height: int,
    last_reference_lines: dict[str, Line],
) -> Line | None:
    geometry = (
        "waterline" if options.reference_line_mode == "waterline" else "centerline"
    )
    reference_id = _record_track_id(record)
    line = _record_geometry_line(record, width, height, geometry)
    if line is None:
        return last_reference_lines.get(reference_id)
    last_reference_lines[reference_id] = line
    return line


def _extend_waterline_to_boat_span(
    record: dict[str, Any],
    waterline: Line,
    width: int,
    height: int,
) -> Line:
    values = record.get("centerline_line_xyxy")
    if not values or len(values) != 4:
        return waterline
    try:
        scale_x, scale_y = _line_coordinate_scale(record, width, height, "centerline")
        boat_axis: Line = (
            float(values[0]) * scale_x,
            float(values[1]) * scale_y,
            float(values[2]) * scale_x,
            float(values[3]) * scale_y,
        )
    except (TypeError, ValueError):
        return waterline
    if not all(math.isfinite(value) for value in boat_axis):
        return waterline
    if (
        _acute_line_angle_degrees(waterline, boat_axis)
        > WATERLINE_BOAT_AXIS_MAX_ANGLE_DEGREES
    ):
        return waterline

    dx = waterline[2] - waterline[0]
    dy = waterline[3] - waterline[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return waterline
    unit = (dx / length, dy / length)
    origin = (waterline[0], waterline[1])
    boat_projections = [
        _dot((point[0] - origin[0], point[1] - origin[1]), unit)
        for point in ((boat_axis[0], boat_axis[1]), (boat_axis[2], boat_axis[3]))
    ]
    start = min(0.0, length, *boat_projections)
    end = max(0.0, length, *boat_projections)
    if start > -1.0:
        start = 0.0
    if end < length + 1.0:
        end = length
    return (
        origin[0] + unit[0] * start,
        origin[1] + unit[1] * start,
        origin[0] + unit[0] * end,
        origin[1] + unit[1] * end,
    )


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


def _detect_paddle_events_for_discipline(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    discipline: str,
    body_motion_frames: dict[int, dict[str, Any]] | None = None,
    progress: Callable[[str, float, str], None] | None = None,
) -> list[PaddleEvent]:
    if discipline == "canoe":
        return _detect_canoe_paddle_events(
            frames,
            options,
            width,
            height,
            scale_x,
            scale_y,
            progress,
            body_motion_frames=body_motion_frames,
        )
    return _detect_paddle_events(
        frames, options, width, height, scale_x, scale_y, progress
    )


def _detect_canoe_paddle_events(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    progress: Callable[[str, float, str], None] | None = None,
    *,
    body_motion_frames: dict[int, dict[str, Any]] | None = None,
) -> list[PaddleEvent]:
    tracks = _track_paddle_observations(
        frames, options, width, height, scale_x, scale_y, progress
    )
    directions = _canoe_directions(tracks)
    body_direction = _canoe_body_travel_direction(body_motion_frames or {})
    if body_direction[0] in {"left", "right"}:
        directions = {
            reference_id: ("body_motion", body_direction[0], body_direction[1])
            for reference_id in {
                timed.observation.reference_id
                for observations in tracks.values()
                for timed in observations
            }
        }
    tracks = _stabilize_boat_reference_lengths(
        tracks,
        directions,
        _event_boat_reference_lines(
            frames,
            options,
            width,
            height,
            scale_x,
            scale_y,
        ),
    )
    slot_count = max(
        1, options.target_slot_count, options.event_paddle_index or 0
    )
    selected_tracks = _canoe_boat_slot_tracks(
        tracks, directions, slot_count, options.event_paddle_index
    )
    detected: list[PaddleEvent] = []
    for physical_id, observations in selected_tracks.items():
        reference_id = _dominant_reference_id(observations)
        _, travel_direction, confidence = directions.get(
            reference_id, (None, None, 0.0)
        )
        if travel_direction not in {"left", "right"}:
            continue
        detected.extend(
            _canoe_phase_events(
                physical_id, observations, travel_direction, confidence
            )
        )
    deduplicated = _dedupe_paddle_events(detected, width, height)
    return [
        event
        for event in deduplicated
        if (event.kind == "catch" and options.include_catch)
        or (event.kind == "exit" and options.include_exit)
    ]


def _canoe_boat_slot_tracks(
    tracks: dict[str, list[_TimedPaddleObservation]],
    directions: dict[str, tuple[str | None, str | None, float]],
    slot_count: int,
    selected_slot: int | None,
) -> dict[str, list[_TimedPaddleObservation]]:
    """Re-identify canoe paddles by stable positions on each boat.

    Raw SAM track IDs remain candidate evidence only. The boat-relative slot is
    the event identity, matching the kayak pipeline's front-to-back selection.
    """
    anchors_by_reference = _event_paddle_slot_anchors_by_reference(
        tracks, directions, slot_count
    )
    observations_by_reference: dict[
        str, dict[int, list[_TimedPaddleObservation]]
    ] = {}
    for observations in tracks.values():
        for timed in observations:
            observations_by_reference.setdefault(
                timed.observation.reference_id, {}
            ).setdefault(timed.timestamp_ms, []).append(timed)

    slot_tracks: dict[str, list[_TimedPaddleObservation]] = {}
    for reference_id, by_timestamp in observations_by_reference.items():
        _, travel_direction, _ = directions.get(reference_id, (None, None, 0.0))
        anchors = anchors_by_reference.get(reference_id, [])
        if travel_direction not in {"left", "right"} or not anchors:
            continue
        requested_indices = (
            (selected_slot - 1,)
            if selected_slot is not None
            else tuple(range(len(anchors)))
        )
        for slot_index in requested_indices:
            if slot_index < 0 or slot_index >= len(anchors):
                continue
            slot_id = f"canoe:slot:{reference_id}:{slot_index + 1}"
            for timestamp_ms, candidates in sorted(by_timestamp.items()):
                assigned = _assign_event_paddle_slots(
                    candidates, anchors, travel_direction
                )
                timed = assigned.get(slot_index)
                if timed is None:
                    continue
                raw_line = timed.observation.raw_line or timed.observation.line
                slot_tracks.setdefault(slot_id, []).append(
                    replace(
                        timed,
                        physical_id=slot_id,
                        observation=replace(
                            timed.observation,
                            line=raw_line,
                            raw_line=raw_line,
                            phase_length_restored=False,
                        ),
                    )
                )
    return slot_tracks


def _canoe_phase_events(
    physical_id: str,
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
    direction_confidence: float,
) -> list[PaddleEvent]:
    """Detect one catch and exit per small-to-large-to-small canoe phase.

    A phase uses the blade-oriented, travel-normalized paddle angle. The longest
    observed line is a phase-length seed; restoring only the blade endpoint
    preserves the dry endpoint while allowing water-shortened geometry to cross
    the waterline.
    """
    if not observations:
        return []
    full_length = max(
        _line_length(timed.observation.raw_line or timed.observation.line)
        for timed in observations
    )
    if full_length < 1:
        return []
    samples = _canoe_phase_samples(observations, travel_direction, full_length)
    phases = _canoe_phase_segments(samples)
    events: list[PaddleEvent] = []
    for phase in phases:
        start_index = samples.index(phase.start)
        peak_index = samples.index(phase.peak)
        end_index = samples.index(phase.end)
        phase_samples = samples[start_index : end_index + 1]
        rising_samples = phase_samples[: peak_index - start_index + 1]
        catch = _canoe_phase_catch(rising_samples) if peak_index > start_index else None
        if catch is not None:
            catch_sample, catch_line = catch
            events.append(
                _canoe_phase_event(
                    "catch", physical_id, catch_sample.timed, catch_line,
                    catch_sample.blade, catch_sample.angle, travel_direction, direction_confidence
                )
            )
        # A peak is an exit only when the restore direction has been confirmed.
        if (
            phase.peak.angle - phase.end.angle >= CANOE_PHASE_MIN_TREND_DEGREES
            and (
                peak_index > start_index
                or _canoe_exit_endpoint_close_to_waterline(phase.peak)
            )
        ):
            events.append(
                _canoe_phase_event(
                    "exit", physical_id, phase.peak.timed, phase.peak.line,
                    phase.peak.blade, phase.peak.angle, travel_direction, direction_confidence
                )
            )
    return events


def _canoe_phase_samples(
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
    full_length: float,
) -> list[_CanoePhaseSample]:
    samples: list[_CanoePhaseSample] = []
    phase_forward = _canoe_phase_reference_forward(observations, travel_direction)
    for timed in observations:
        line = timed.observation.raw_line or timed.observation.line
        phase_reference_line = _canoe_phase_reference_line(
            timed.observation.reference_line, phase_forward
        )
        depths = _endpoint_signed_depths(line, phase_reference_line)
        blade = 0 if depths[0] >= depths[1] else 1
        angle = _canoe_phase_angle(
            line, phase_reference_line, blade, travel_direction
        )
        if angle is None:
            continue
        samples.append(
            _CanoePhaseSample(
                timed=timed,
                line=line,
                restored_line=_canoe_restore_blade_length(line, full_length, blade),
                blade=blade,
                angle=angle,
            )
        )
    return samples


def _canoe_phase_reference_forward(
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
) -> tuple[float, float] | None:
    vectors: list[tuple[float, float]] = []
    for timed in observations:
        geometry = _boat_reference_head_geometry(
            timed.observation.reference_line, travel_direction
        )
        if geometry is None:
            continue
        _head, forward, _observed_length = geometry
        vectors.append(forward)
    if not vectors:
        return None
    x = _median([vector[0] for vector in vectors])
    y = _median([vector[1] for vector in vectors])
    if x is None or y is None:
        return None
    return _normalize((x, y))


def _canoe_phase_reference_line(
    reference_line: Line,
    phase_forward: tuple[float, float] | None,
) -> Line:
    if phase_forward is None:
        return reference_line
    center = _line_center(reference_line)
    half_length = _line_length(reference_line) / 2
    return (
        center[0] - phase_forward[0] * half_length,
        center[1] - phase_forward[1] * half_length,
        center[0] + phase_forward[0] * half_length,
        center[1] + phase_forward[1] * half_length,
    )


def _canoe_phase_segments(samples: list[_CanoePhaseSample]) -> list[_CanoePhase]:
    """Split a slot into minimum-to-peak-to-minimum canoe phases.

    A local minimum closes the previous phase and is reused as the start of the
    next phase. The first and last segment may be partial because a video can
    start or end during either the stroke or restore period.
    """
    if len(samples) < 3:
        return []
    phases: list[_CanoePhase] = []
    start_index = 0
    peak_index = 0
    trough_index = 0
    state = "unknown"
    falling_count = 0
    initial_restore = False
    for index in range(1, len(samples)):
        delta = samples[index].angle - samples[index - 1].angle
        if state == "unknown":
            if delta >= CANOE_PHASE_MIN_TREND_DEGREES:
                state = "rising"
                peak_index = index
            elif -delta >= CANOE_PHASE_MIN_TREND_DEGREES:
                state = "falling"
                peak_index = start_index
                trough_index = index
                initial_restore = True
            elif samples[index].angle < samples[start_index].angle:
                start_index = index
                peak_index = index
            elif samples[index].angle > samples[peak_index].angle:
                peak_index = index
            continue
        if state == "rising":
            if samples[index].angle >= samples[peak_index].angle:
                peak_index = index
                falling_count = 0
            elif samples[peak_index].angle - samples[index].angle >= CANOE_PHASE_MIN_TREND_DEGREES:
                falling_count += 1
                if falling_count >= CANOE_PHASE_EXIT_CONFIRM_SAMPLES:
                    state = "falling"
                    trough_index = index
            continue
        if samples[index].angle <= samples[trough_index].angle:
            trough_index = index
            continue
        if samples[index].angle - samples[trough_index].angle >= CANOE_PHASE_MIN_TREND_DEGREES:
            end_index = trough_index
            phases.append(
                _CanoePhase(
                    samples[start_index],
                    samples[peak_index],
                    samples[end_index],
                    not initial_restore,
                )
            )
            start_index = end_index
            peak_index = max(
                range(start_index, index + 1),
                key=lambda sample_index: samples[sample_index].angle,
            )
            state = "rising"
            falling_count = 0
            initial_restore = False
    if state == "falling" and peak_index >= start_index and trough_index > peak_index:
        phases.append(
            _CanoePhase(
                samples[start_index], samples[peak_index], samples[trough_index], False
            )
        )
    return phases


def _canoe_phase_angle(
    line: Line, reference_line: Line, blade: int, travel_direction: str
) -> float | None:
    # Start from the waterline ray pointing in canoe travel direction and use
    # the dry-to-water-facing blade vector, yielding a 0-180 degree phase.
    forward = _normalize(
        (reference_line[2] - reference_line[0], reference_line[3] - reference_line[1])
    )
    if forward is None:
        return None
    if (travel_direction == "right" and forward[0] < 0) or (
        travel_direction == "left" and forward[0] > 0
    ):
        forward = (-forward[0], -forward[1])
    down = (-forward[1], forward[0])
    if down[1] < 0:
        down = (-down[0], -down[1])
    dry = (line[2], line[3]) if blade == 0 else (line[0], line[1])
    active = (line[0], line[1]) if blade == 0 else (line[2], line[3])
    vector = _normalize((active[0] - dry[0], active[1] - dry[1]))
    if vector is None:
        return None
    angle = math.degrees(math.atan2(_dot(vector, down), _dot(vector, forward)))
    return round(angle if angle >= 0 else angle + 180.0, 3)


def _canoe_restore_blade_length(line: Line, full_length: float, blade: int) -> Line:
    if _line_length(line) >= full_length:
        return line
    dry = (line[2], line[3]) if blade == 0 else (line[0], line[1])
    active = (line[0], line[1]) if blade == 0 else (line[2], line[3])
    direction = _normalize((active[0] - dry[0], active[1] - dry[1]))
    if direction is None:
        return line
    restored_active = (
        dry[0] + direction[0] * full_length,
        dry[1] + direction[1] * full_length,
    )
    return (
        restored_active[0],
        restored_active[1],
        dry[0],
        dry[1],
    ) if blade == 0 else (
        dry[0],
        dry[1],
        restored_active[0],
        restored_active[1],
    )


def _canoe_line_crosses_waterline(
    line: Line, reference_line: Line, blade: int
) -> bool:
    depths = _endpoint_signed_depths(line, reference_line)
    return depths[blade] >= 0 and depths[1 - blade] <= 0


def _canoe_phase_catch(
    rising_samples: list[_CanoePhaseSample],
) -> tuple[_CanoePhaseSample, Line] | None:
    restored = _canoe_phase_catch_restored_lines(rising_samples)
    if len(restored) < 2:
        return None
    for previous, current in zip(restored, restored[1:]):
        previous_sample, previous_line = previous
        current_sample, current_line = current
        previous_crosses = _canoe_line_crosses_waterline(
            previous_line,
            previous_sample.timed.observation.reference_line,
            previous_sample.blade,
        )
        current_crosses = _canoe_line_crosses_waterline(
            current_line,
            current_sample.timed.observation.reference_line,
            current_sample.blade,
        )
        if previous_crosses == current_crosses:
            continue
        return min(
            (previous, current),
            key=lambda entry: _canoe_active_endpoint_waterline_distance(
                entry[0], entry[1]
            ),
        )
    return None


def _canoe_phase_catch_restored_lines(
    rising_samples: list[_CanoePhaseSample],
) -> list[tuple[_CanoePhaseSample, Line]]:
    if not rising_samples:
        return []
    inherited_length = _line_length(rising_samples[0].line)
    if inherited_length < 1:
        return []
    restored: list[tuple[_CanoePhaseSample, Line]] = []
    for sample in rising_samples:
        if sample.angle > 90:
            break
        line = sample.line
        current_length = _line_length(line)
        if current_length < 1:
            continue
        if current_length < inherited_length:
            line = _canoe_restore_length_from_active_endpoint(
                line, inherited_length, sample.blade
            )
        elif (
            current_length - inherited_length
        ) / inherited_length <= CANOE_CATCH_LENGTH_RESTORE_TOLERANCE:
            inherited_length = current_length
        else:
            line = _canoe_restore_length_from_active_endpoint(
                line, inherited_length, sample.blade
            )
        restored.append((sample, line))
    return restored


def _canoe_restore_length_from_active_endpoint(
    line: Line, target_length: float, blade: int
) -> Line:
    active = (line[0], line[1]) if blade == 0 else (line[2], line[3])
    dry = (line[2], line[3]) if blade == 0 else (line[0], line[1])
    direction = _normalize((dry[0] - active[0], dry[1] - active[1]))
    if direction is None:
        return line
    restored_dry = (
        active[0] + direction[0] * target_length,
        active[1] + direction[1] * target_length,
    )
    return (
        active[0],
        active[1],
        restored_dry[0],
        restored_dry[1],
    ) if blade == 0 else (
        restored_dry[0],
        restored_dry[1],
        active[0],
        active[1],
    )


def _canoe_active_endpoint_waterline_distance(
    sample: _CanoePhaseSample, line: Line | None = None
) -> float:
    candidate = line or sample.line
    depths = _endpoint_signed_depths(
        candidate, sample.timed.observation.reference_line
    )
    reference_length = max(_line_length(sample.timed.observation.reference_line), 1.0)
    return abs(depths[sample.blade]) / reference_length + sample.angle * 1e-6


def _canoe_exit_endpoint_close_to_waterline(sample: _CanoePhaseSample) -> bool:
    return (
        _canoe_active_endpoint_waterline_distance(sample, sample.line)
        <= PADDLE_EXIT_DEPTH_RATIO
    )


def _canoe_phase_event(
    kind: str,
    physical_id: str,
    timed: _TimedPaddleObservation,
    line: Line,
    blade: int,
    phase_angle: float,
    travel_direction: str,
    confidence: float,
) -> PaddleEvent:
    return PaddleEvent(
        kind=kind,
        timestamp_ms=timed.timestamp_ms,
        instance_id=physical_id,
        line=line,
        reference_line=timed.observation.reference_line,
        confidence=confidence,
        degree=_acute_line_angle_degrees(line, timed.observation.reference_line),
        phase_angle=phase_angle,
        active_blade=blade,
        travel_direction=travel_direction,
        discipline="canoe",
    )


def _canoe_samples(
    observations: list[_TimedPaddleObservation],
) -> list[_CanoeSample]:
    if not observations:
        return []
    full_length = max(
        (
            _line_length(timed.observation.raw_line or timed.observation.line)
            for timed in observations
        ),
        default=0.0,
    )
    if full_length < 1:
        return []
    samples: list[_CanoeSample] = []
    for timed in observations:
        observation = timed.observation
        line = observation.raw_line or observation.line
        depths = _endpoint_signed_depths(line, observation.reference_line)
        blade = 0 if depths[0] >= depths[1] else 1
        air_length = _canoe_visible_air_length(line, observation.reference_line, blade)
        if air_length is None:
            continue
        ratio = _clamp(air_length / full_length, 0.0, 1.25)
        samples.append(
            _CanoeSample(
                timed=replace(
                    timed,
                    observation=replace(observation, line=line),
                ),
                blade=blade,
                blade_position=_canoe_canonical_position(
                    (line[0], line[1]) if blade == 0 else (line[2], line[3]),
                    observation.reference_line,
                ),
                air_length_ratio=ratio,
                contact=ratio <= CANOE_CONTACT_LENGTH_RATIO,
            )
        )
    return samples


def _canoe_visible_air_length(
    line: Line, reference_line: Line, blade: int
) -> float | None:
    depths = _endpoint_signed_depths(line, reference_line)
    blade_depth = depths[blade]
    dry_depth = depths[1 - blade]
    length = _line_length(line)
    if length < 1:
        return None
    if max(depths) < 0:
        return length
    denominator = blade_depth - dry_depth
    if dry_depth < 0 <= blade_depth and abs(denominator) > 1e-9:
        dry_to_intersection = -dry_depth / denominator
        return length * _clamp(dry_to_intersection, 0.0, 1.0)
    # SAM commonly truncates the submerged blade at the waterline. In that
    # case the detected line itself is the visible dry-side length.
    if abs(blade_depth) <= max(2.0, length * 0.05) and dry_depth < 0:
        return length
    return None


def _canoe_canonical_position(point: tuple[float, float], reference_line: Line) -> float:
    forward = _normalize(
        (reference_line[2] - reference_line[0], reference_line[3] - reference_line[1])
    )
    if forward is None:
        return 0.0
    if forward[0] < 0:
        forward = (-forward[0], -forward[1])
    origin = _line_center(reference_line)
    length = max(_line_length(reference_line), 1.0)
    return _dot((point[0] - origin[0], point[1] - origin[1]), forward) / length


def _canoe_pull_intervals(
    physical_id: str, samples: list[_CanoeSample]
) -> list[_CanoePullInterval]:
    intervals: list[_CanoePullInterval] = []
    entry: _CanoeSample | None = None
    minimum: _CanoeSample | None = None
    exit_candidate: _CanoeSample | None = None
    contact_samples = 0
    for sample in samples:
        if entry is not None and (
            sample.timed.timestamp_ms - entry.timed.timestamp_ms
            > PADDLE_EVENT_TRACK_GAP_MS
            or sample.timed.observation.reference_id
            != entry.timed.observation.reference_id
        ):
            entry = minimum = exit_candidate = None
            contact_samples = 0
        if sample.contact:
            if entry is None:
                entry = minimum = sample
                exit_candidate = None
                contact_samples = 1
                continue
            contact_samples += 1
            assert minimum is not None
            if sample.air_length_ratio < minimum.air_length_ratio:
                minimum = sample
                exit_candidate = None
            elif (
                sample.air_length_ratio - minimum.air_length_ratio
                >= CANOE_MIN_LENGTH_CHANGE_RATIO
                and exit_candidate is None
            ):
                exit_candidate = sample
            continue
        if entry is None:
            continue
        assert minimum is not None
        if (
            contact_samples >= CANOE_CONTACT_MIN_SAMPLES
            and exit_candidate is not None
            and sample.air_length_ratio >= CANOE_RELEASE_LENGTH_RATIO
        ):
            intervals.append(
                _CanoePullInterval(
                    physical_id=physical_id,
                    reference_id=entry.timed.observation.reference_id,
                    entry=entry,
                    exit=exit_candidate,
                    release=sample,
                    minimum_air_length_ratio=minimum.air_length_ratio,
                )
            )
        entry = minimum = exit_candidate = None
        contact_samples = 0
    return intervals


def _canoe_directions(
    tracks: dict[str, list[_TimedPaddleObservation]],
) -> dict[str, tuple[str | None, str | None, float]]:
    votes: dict[str, list[float]] = {}
    for observations in tracks.values():
        angle_samples = [
            (
                timed,
                _canoe_canonical_axis_angle(
                    timed.observation.raw_line or timed.observation.line,
                    timed.observation.reference_line,
                ),
            )
            for timed in observations
        ]
        valid_angles = [angle for _, angle in angle_samples if angle is not None]
        if (
            not valid_angles
            or max(valid_angles) - min(valid_angles)
            < CANOE_DIRECTION_MIN_AXIS_SPAN_DEGREES
        ):
            continue
        for timed, angle in angle_samples:
            observation = timed.observation
            if angle is None:
                continue
            if angle <= CANOE_DIRECTION_RIGHT_MAX_DEGREES:
                direction = 1.0
            elif angle >= CANOE_DIRECTION_LEFT_MIN_DEGREES:
                direction = -1.0
            else:
                continue
            # Votes farther from the vertical dead band and from longer lines
            # are more robust to waterline and short-fragment noise.
            boundary_distance = min(abs(angle - 90.0), 90.0) / 90.0
            weight = max(0.05, boundary_distance) * max(
                _line_length(observation.raw_line or observation.line), 1.0
            )
            votes.setdefault(observation.reference_id, []).append(direction * weight)
    results: dict[str, tuple[str | None, str | None, float]] = {}
    for reference_id, values in votes.items():
        if len(values) < CANOE_DIRECTION_MIN_VOTES:
            results[reference_id] = (None, None, 0.0)
            continue
        right_weight = sum(abs(value) for value in values if value > 0)
        left_weight = sum(abs(value) for value in values if value < 0)
        consensus = max(right_weight, left_weight) / max(
            right_weight + left_weight, 1e-9
        )
        if consensus < CANOE_DIRECTION_MIN_CONSENSUS:
            results[reference_id] = (None, None, round(consensus, 3))
        elif right_weight > left_weight:
            results[reference_id] = ("canoe_axis", "right", round(consensus, 3))
        else:
            results[reference_id] = ("canoe_axis", "left", round(consensus, 3))
    return results


def _canoe_body_travel_direction(
    body_motion_frames: dict[int, dict[str, Any]],
) -> tuple[str | None, float]:
    right_votes = 0
    left_votes = 0
    for record in body_motion_frames.values():
        landmarks = record.get("landmarks")
        if not isinstance(landmarks, dict):
            continue
        points: dict[str, float] = {}
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            landmark = landmarks.get(name)
            if not isinstance(landmark, dict):
                break
            try:
                x = float(landmark["x"])
                visibility = float(landmark.get("visibility", 0))
                presence = float(landmark.get("presence", 0))
            except (KeyError, TypeError, ValueError):
                break
            if (
                not math.isfinite(x)
                or min(visibility, presence) < MIN_LANDMARK_CONFIDENCE
            ):
                break
            points[name] = x
        if len(points) != 4:
            continue
        left_offset = points["left_shoulder"] - points["left_hip"]
        right_offset = points["right_shoulder"] - points["right_hip"]
        if left_offset > 0 and right_offset > 0:
            right_votes += 1
        elif left_offset < 0 and right_offset < 0:
            left_votes += 1
    vote_count = right_votes + left_votes
    if vote_count < CANOE_DIRECTION_MIN_VOTES:
        return None, 0.0
    consensus = max(right_votes, left_votes) / vote_count
    if right_votes == left_votes:
        return None, round(consensus, 3)
    return (
        "right" if right_votes > left_votes else "left",
        round(consensus, 3),
    )


def _canoe_canonical_axis_angle(line: Line, reference_line: Line) -> float | None:
    reference = _normalize(
        (reference_line[2] - reference_line[0], reference_line[3] - reference_line[1])
    )
    paddle = _normalize((line[2] - line[0], line[3] - line[1]))
    if reference is None or paddle is None:
        return None
    if reference[0] < 0:
        reference = (-reference[0], -reference[1])
    angle = math.degrees(math.atan2(paddle[1], paddle[0]) - math.atan2(reference[1], reference[0]))
    return angle % 180.0


def _canoe_signed_angle(
    line: Line, reference_line: Line, blade: int, travel_direction: str
) -> float | None:
    forward = _normalize(
        (reference_line[2] - reference_line[0], reference_line[3] - reference_line[1])
    )
    if forward is None:
        return None
    if (travel_direction == "right" and forward[0] < 0) or (
        travel_direction == "left" and forward[0] > 0
    ):
        forward = (-forward[0], -forward[1])
    down = (-forward[1], forward[0])
    if down[1] < 0:
        down = (-down[0], -down[1])
    dry = (line[2], line[3]) if blade == 0 else (line[0], line[1])
    active = (line[0], line[1]) if blade == 0 else (line[2], line[3])
    blade_vector = _normalize((active[0] - dry[0], active[1] - dry[1]))
    if blade_vector is None:
        return None
    angle = math.degrees(
        math.atan2(_dot(blade_vector, down), _dot(blade_vector, forward))
    )
    return round(angle, 3)


def _detect_paddle_events(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    progress: Callable[[str, float, str], None] | None = None,
) -> list[PaddleEvent]:
    tracks = _track_paddle_observations(
        frames,
        options,
        width,
        height,
        scale_x,
        scale_y,
        progress,
    )
    directions = _event_paddle_directions(tracks)
    slot_count = max(options.event_paddle_index or 0, options.target_slot_count)
    if options.event_paddle_index is not None and slot_count > 1:
        slot_anchors = _event_paddle_slot_anchors_by_reference(
            tracks,
            directions,
            slot_count,
        )
        if slot_anchors:
            # The initial pass exists only to estimate travel direction and
            # stable boat-relative slot anchors. Rebuild observations from raw
            # masks so fragment merging cannot cross those paddle slots.
            tracks = _track_paddle_observations(
                frames,
                options,
                width,
                height,
                scale_x,
                scale_y,
                progress,
                slot_anchors_by_reference=slot_anchors,
                travel_directions={
                    reference_id: travel_direction
                    for reference_id, (_, travel_direction, _) in directions.items()
                    if travel_direction in {"left", "right"}
                },
            )
            refined_directions = _event_paddle_directions(tracks)
            directions.update(
                {
                    reference_id: direction
                    for reference_id, direction in refined_directions.items()
                    if direction[0] is not None
                    and direction[1] in {"left", "right"}
                }
            )
    tracks = _stabilize_boat_reference_lengths(
        tracks,
        directions,
        _event_boat_reference_lines(
            frames,
            options,
            width,
            height,
            scale_x,
            scale_y,
        ),
    )

    event_tracks = _select_event_paddle_tracks(
        tracks,
        directions,
        options.event_paddle_index,
        options.target_slot_count,
    )
    completeness_by_source = _predict_event_paddle_completeness(
        frames,
        event_tracks,
    )
    candidate_validator = _paddle_candidate_validator(
        frames,
        completeness_by_source,
        width,
        height,
    )
    detected: list[PaddleEvent] = []
    band_upward_width = _event_band_upward_width(scale_x, scale_y)
    ordered_tracks = sorted(event_tracks.items())
    for index, (physical_id, observations) in enumerate(ordered_tracks):
        reference_id = _dominant_reference_id(observations)
        rotation_direction, travel_direction, confidence = directions.get(
            reference_id, (None, None, 0.0)
        )
        if rotation_direction is None or travel_direction is None:
            continue
        observations = _restore_track_stroke_lengths(
            observations,
            travel_direction,
            band_upward_width,
            candidate_validator=candidate_validator,
        )
        state = _PaddleEventState(
            physical_id=physical_id,
            reference_id=reference_id,
            rotation_direction=rotation_direction,
            travel_direction=travel_direction,
            direction_confidence=confidence,
        )
        for timed in observations:
            event = _update_directed_paddle_state(
                state,
                timed.observation,
                timed.timestamp_ms,
                band_upward_width,
                event_eligible=timed.observation.phase_length_verified,
            )
            if event is not None:
                detected.append(event)
        _report_progress(
            progress,
            "analyzing_events",
            9 + 4 * (index + 1) / max(len(ordered_tracks), 1),
            f"Analyzing paddle track {index + 1} of {len(ordered_tracks)}",
        )
    deduplicated = _dedupe_paddle_events(detected, width, height)
    return [
        event
        for event in deduplicated
        if (event.kind == "catch" and options.include_catch)
        or (event.kind == "exit" and options.include_exit)
    ]


def _predict_event_paddle_completeness(
    frames: dict[int, list[dict[str, Any]]],
    event_tracks: dict[str, list[_TimedPaddleObservation]],
) -> dict[tuple[int, str], bool]:
    """Batch CNN mask classification for source records used by event tracks."""
    predictor = build_paddle_completeness_predictor()
    if predictor is None:
        return {}
    required = {
        (timed.timestamp_ms, source_id)
        for observations in event_tracks.values()
        for timed in observations
        for source_id in timed.observation.source_ids
    }
    keys: list[tuple[int, str]] = []
    records: list[dict[str, Any]] = []
    for timestamp_ms, source_id in sorted(required):
        record = next(
            (
                candidate
                for candidate in frames.get(timestamp_ms, [])
                if _record_track_id(candidate) == source_id
            ),
            None,
        )
        if record is None:
            continue
        keys.append((timestamp_ms, source_id))
        records.append(record)
    try:
        probabilities = predictor.predict_records(records)
    except (OSError, RuntimeError, TypeError, ValueError):
        return {}
    return {
        key: probability < predictor.threshold
        for key, probability in zip(keys, probabilities, strict=True)
        if probability is not None
    }


def _paddle_candidate_validator(
    frames: dict[int, list[dict[str, Any]]],
    completeness_by_source: dict[tuple[int, str], bool],
    width: int,
    height: int,
) -> Callable[[_TimedPaddleObservation, Line, int], bool | None]:
    """Use the mask contributing the active endpoint for candidate eligibility."""

    def validate(
        timed: _TimedPaddleObservation,
        line: Line,
        active_blade: int,
    ) -> bool | None:
        if active_blade not in (0, 1):
            return None
        source_ids = set(timed.observation.source_ids)
        records = [
            record
            for record in frames.get(timed.timestamp_ms, [])
            if _record_track_id(record) in source_ids
        ]
        active = (line[0], line[1]) if active_blade == 0 else (line[2], line[3])
        ranked: list[tuple[float, str]] = []
        for record in records:
            source_id = _record_track_id(record)
            if (timed.timestamp_ms, source_id) not in completeness_by_source:
                continue
            source_line = _record_line(record, width, height)
            if source_line is None:
                continue
            endpoint_distance = min(
                math.dist(active, (source_line[0], source_line[1])),
                math.dist(active, (source_line[2], source_line[3])),
            )
            ranked.append((endpoint_distance, source_id))
        if not ranked:
            return None
        source_id = min(ranked)[1]
        return completeness_by_source.get((timed.timestamp_ms, source_id))

    return validate


def _track_paddle_observations(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    progress: Callable[[str, float, str], None] | None = None,
    *,
    slot_anchors_by_reference: dict[str, list[float]] | None = None,
    travel_directions: dict[str, str] | None = None,
) -> dict[str, list[_TimedPaddleObservation]]:
    states: list[_PaddleEventState] = []
    tracks: dict[str, list[_TimedPaddleObservation]] = {}
    timestamps = sorted(frames)
    target_prompt_ids = set(options.target_prompt_ids)
    last_reference_lines: dict[str, Line] = {}
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
            if is_reference:
                line = _event_reference_line(
                    record, options, width, height, last_reference_lines
                )
            else:
                line = _record_line(record, width, height)
            if line is None:
                continue
            centerline = Centerline(record=record, line=line, color=(255, 255, 255, 255))
            if is_reference:
                references.append(centerline)
            elif record.get("prompt_id") in target_prompt_ids:
                targets.append(centerline)
        observations = _consolidate_paddle_observations(
            targets,
            references,
            width,
            height,
            slot_anchors_by_reference=slot_anchors_by_reference,
            travel_directions=travel_directions,
        )
        assignments = _assign_paddle_observations(
            observations, states, timestamp_ms, width, height
        )
        for observation, state in assignments:
            raw_line = _orient_line_like(observation.line, state.last_line)
            line = _stabilize_paddle_line(state, observation.line)
            state.last_seen_ms = timestamp_ms
            state.last_line = line
            stabilized = replace(observation, line=line, raw_line=raw_line)
            tracks.setdefault(state.physical_id, []).append(
                _TimedPaddleObservation(
                    timestamp_ms=timestamp_ms,
                    physical_id=state.physical_id,
                    observation=stabilized,
                )
            )
        if index == len(timestamps) - 1 or index % max(1, len(timestamps) // 20) == 0:
            _report_progress(
                progress,
                "analyzing_events",
                5 + 4 * (index + 1) / max(len(timestamps), 1),
                f"Tracking paddle observations {index + 1} of {len(timestamps)}",
            )
    return tracks


def _dominant_reference_id(observations: list[_TimedPaddleObservation]) -> str:
    counts: dict[str, int] = {}
    for timed in observations:
        reference_id = timed.observation.reference_id
        counts[reference_id] = counts.get(reference_id, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _event_paddle_directions(
    tracks: dict[str, list[_TimedPaddleObservation]],
) -> dict[str, tuple[str | None, str | None, float]]:
    evidence_by_reference: dict[str, list[float]] = {}
    for observations in tracks.values():
        if not observations:
            continue
        reference_id = _dominant_reference_id(observations)
        evidence_by_reference.setdefault(reference_id, []).extend(
            _paddle_rotation_deltas(observations)
        )
    return {
        reference_id: _estimate_paddle_direction(deltas)
        for reference_id, deltas in evidence_by_reference.items()
    }


def _event_paddle_slot_anchors_by_reference(
    tracks: dict[str, list[_TimedPaddleObservation]],
    directions: dict[str, tuple[str | None, str | None, float]],
    slot_count: int,
) -> dict[str, list[float]]:
    observations_by_reference: dict[
        str, dict[int, list[_TimedPaddleObservation]]
    ] = {}
    for observations in tracks.values():
        for timed in observations:
            reference_id = timed.observation.reference_id
            observations_by_reference.setdefault(reference_id, {}).setdefault(
                timed.timestamp_ms,
                [],
            ).append(timed)

    anchors_by_reference: dict[str, list[float]] = {}
    for reference_id, observations_by_timestamp in observations_by_reference.items():
        _, travel_direction, _ = directions.get(reference_id, (None, None, 0.0))
        if travel_direction not in {"left", "right"}:
            continue
        anchors = _event_paddle_slot_anchors(
            observations_by_timestamp,
            travel_direction,
            slot_count,
        )
        if anchors:
            anchors_by_reference[reference_id] = anchors
    return anchors_by_reference


def _select_event_paddle_tracks(
    tracks: dict[str, list[_TimedPaddleObservation]],
    directions: dict[str, tuple[str | None, str | None, float]],
    event_paddle_index: int | None,
    target_slot_count: int = 0,
) -> dict[str, list[_TimedPaddleObservation]]:
    if event_paddle_index is None:
        return tracks

    observations_by_reference: dict[
        str, dict[int, list[_TimedPaddleObservation]]
    ] = {}
    for observations in tracks.values():
        for timed in observations:
            reference_id = timed.observation.reference_id
            observations_by_reference.setdefault(reference_id, {}).setdefault(
                timed.timestamp_ms, []
            ).append(timed)

    slot_count = max(event_paddle_index, target_slot_count)
    selected: dict[str, list[_TimedPaddleObservation]] = {}
    for reference_id, observations_by_timestamp in observations_by_reference.items():
        _, travel_direction, _ = directions.get(reference_id, (None, None, 0.0))
        if travel_direction not in {"left", "right"}:
            continue
        anchors = _event_paddle_slot_anchors(
            observations_by_timestamp,
            travel_direction,
            slot_count,
        )
        selected_index = event_paddle_index - 1
        if selected_index >= len(anchors):
            continue
        slot_id = f"paddle:slot:{reference_id}:{event_paddle_index}"
        slot_observations: list[_TimedPaddleObservation] = []
        for timestamp_ms, candidates in sorted(observations_by_timestamp.items()):
            assigned = _assign_event_paddle_slots(
                candidates,
                anchors,
                travel_direction,
            )
            timed = assigned.get(selected_index)
            if timed is None:
                continue
            raw_line = timed.observation.raw_line or timed.observation.line
            slot_observations.append(
                replace(
                    timed,
                    physical_id=slot_id,
                    observation=replace(
                        timed.observation,
                        line=raw_line,
                        raw_line=raw_line,
                        phase_length_restored=False,
                    ),
                )
            )
        if slot_observations:
            selected[slot_id] = slot_observations
    return selected


def _stabilize_boat_reference_lengths(
    tracks: dict[str, list[_TimedPaddleObservation]],
    directions: dict[str, tuple[str | None, str | None, float]],
    lines_by_reference: dict[str, dict[int, Line]] | None = None,
) -> dict[str, list[_TimedPaddleObservation]]:
    """Normalize each boat length while preserving its observed forward endpoint."""
    reference_lines = {
        reference_id: dict(lines_by_timestamp)
        for reference_id, lines_by_timestamp in (lines_by_reference or {}).items()
    }
    for observations in tracks.values():
        for timed in observations:
            reference_id = timed.observation.reference_id
            _, travel_direction, _ = directions.get(
                reference_id, (None, None, 0.0)
            )
            if travel_direction not in {"left", "right"}:
                continue
            reference_lines.setdefault(reference_id, {}).setdefault(
                timed.timestamp_ms,
                timed.observation.reference_line,
            )

    corrected = _stabilized_event_reference_lines(reference_lines, directions)

    if not corrected:
        return tracks
    stabilized: dict[str, list[_TimedPaddleObservation]] = {}
    for physical_id, observations in tracks.items():
        stabilized[physical_id] = [
            replace(
                timed,
                observation=replace(
                    timed.observation,
                    reference_line=corrected.get(
                        (
                            timed.observation.reference_id,
                            timed.timestamp_ms,
                        ),
                        timed.observation.reference_line,
                    ),
                ),
            )
            for timed in observations
        ]
    return stabilized


def _stabilized_event_reference_lines(
    lines_by_reference: dict[str, dict[int, Line]],
    directions: dict[str, tuple[str | None, str | None, float]],
) -> dict[tuple[str, int], Line]:
    corrected: dict[tuple[str, int], Line] = {}
    for reference_id, lines_by_timestamp in lines_by_reference.items():
        if reference_id not in directions:
            continue
        _, travel_direction, _ = directions[reference_id]
        if travel_direction not in {"left", "right"}:
            continue
        corrected.update(
            {
                (reference_id, timestamp_ms): line
                for timestamp_ms, line in _kalman_stabilized_boat_reference_lines(
                    lines_by_timestamp,
                    travel_direction,
                ).items()
            }
        )
    return corrected


def _event_boat_reference_lines(
    frames: dict[int, list[dict[str, Any]]],
    options: ExportOptions,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> dict[str, dict[int, Line]]:
    """Collect every selected boat observation, including frames without a paddle."""
    lines: dict[str, dict[int, Line]] = {}
    last_reference_lines: dict[str, Line] = {}
    for timestamp_ms, frame_records in frames.items():
        for raw_record in frame_records:
            record = _scale_record(raw_record, scale_x, scale_y)
            if record.get("prompt_id") != options.reference_prompt_id:
                continue
            if not _record_selected_for_export(
                record,
                options,
                width,
                height,
                timestamp_ms,
            ):
                continue
            line = _event_reference_line(
                record, options, width, height, last_reference_lines
            )
            if line is None:
                continue
            lines.setdefault(_record_track_id(record), {})[timestamp_ms] = line
    return lines


def _kalman_stabilized_boat_reference_lines(
    lines_by_timestamp: dict[int, Line],
    travel_direction: str,
) -> dict[int, Line]:
    samples: list[tuple[int, tuple[float, float], tuple[float, float], float]] = []
    for timestamp_ms, line in sorted(lines_by_timestamp.items()):
        geometry = _boat_reference_head_geometry(line, travel_direction)
        if geometry is None:
            continue
        head, forward, observed_length = geometry
        samples.append((timestamp_ms, head, forward, observed_length))
    if not samples:
        return {}

    seed_length = _median([sample[3] for sample in samples])
    if seed_length is None or seed_length < 1:
        return {}
    measurement_noise = max(
        BOAT_REFERENCE_LENGTH_MIN_NOISE_PIXELS,
        seed_length * BOAT_REFERENCE_LENGTH_MEASUREMENT_NOISE_RATIO,
    )
    process_noise = max(
        BOAT_REFERENCE_LENGTH_MIN_NOISE_PIXELS / 2,
        seed_length * BOAT_REFERENCE_LENGTH_PROCESS_NOISE_RATIO,
    )
    state = _BoatReferenceLengthFilter(
        length=seed_length,
        variance=(measurement_noise * measurement_noise) / 4,
    )
    measurement_variance = measurement_noise * measurement_noise
    process_variance_per_second = process_noise * process_noise
    stabilized: dict[int, Line] = {}
    for timestamp_ms, head, forward, observed_length in samples:
        if state.last_timestamp_ms is not None:
            elapsed_seconds = min(
                5.0,
                max(1 / 120, (timestamp_ms - state.last_timestamp_ms) / 1000),
            )
            state.variance += process_variance_per_second * elapsed_seconds
        relative_innovation = abs(observed_length - state.length) / max(
            state.length, 1.0
        )
        if relative_innovation <= BOAT_REFERENCE_LENGTH_MAX_INNOVATION_RATIO:
            gain = state.variance / (state.variance + measurement_variance)
            state.length += gain * (observed_length - state.length)
            state.variance = max(1e-6, (1 - gain) * state.variance)
        state.last_timestamp_ms = timestamp_ms
        tail = (
            head[0] - forward[0] * state.length,
            head[1] - forward[1] * state.length,
        )
        stabilized[timestamp_ms] = (tail[0], tail[1], head[0], head[1])
    return stabilized


def _boat_reference_head_geometry(
    line: Line,
    travel_direction: str,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    first = (line[0], line[1])
    second = (line[2], line[3])
    observed_length = math.dist(first, second)
    if observed_length < 1:
        return None
    if travel_direction == "right":
        head, tail = (second, first) if second[0] >= first[0] else (first, second)
    elif travel_direction == "left":
        head, tail = (second, first) if second[0] <= first[0] else (first, second)
    else:
        return None
    forward = _normalize((head[0] - tail[0], head[1] - tail[1]))
    if forward is None:
        return None
    return head, forward, observed_length


def _event_paddle_slot_anchors(
    observations_by_timestamp: dict[int, list[_TimedPaddleObservation]],
    travel_direction: str,
    slot_count: int,
) -> list[float]:
    ordered_positions = [
        sorted(
            (
                _observation_forward_position(timed.observation, travel_direction)
                for timed in candidates
            ),
            reverse=True,
        )
        for candidates in observations_by_timestamp.values()
    ]
    maximum_count = min(
        slot_count,
        max((len(positions) for positions in ordered_positions), default=0),
    )
    if maximum_count <= 0:
        return []
    complete_frames = [
        positions for positions in ordered_positions if len(positions) >= maximum_count
    ]
    return [
        _median([positions[index] for positions in complete_frames])
        for index in range(maximum_count)
    ]


def _assign_event_paddle_slots(
    candidates: list[_TimedPaddleObservation],
    anchors: list[float],
    travel_direction: str,
) -> dict[int, _TimedPaddleObservation]:
    assigned: dict[int, tuple[float, _TimedPaddleObservation]] = {}
    for timed in candidates:
        position = _observation_forward_position(
            timed.observation,
            travel_direction,
        )
        slot_index = _gated_event_paddle_slot_index(position, anchors)
        if slot_index is None:
            continue
        distance = abs(position - anchors[slot_index])
        previous = assigned.get(slot_index)
        if previous is None or distance < previous[0]:
            assigned[slot_index] = (distance, timed)
    return {slot_index: item[1] for slot_index, item in assigned.items()}


def _gated_event_paddle_slot_index(
    position: float,
    anchors: list[float],
) -> int | None:
    if not anchors or not math.isfinite(position):
        return None
    slot_index = min(
        range(len(anchors)),
        key=lambda index: abs(position - anchors[index]),
    )
    distance = abs(position - anchors[slot_index])
    neighbor_distances = [
        abs(anchors[slot_index] - anchors[neighbor_index])
        for neighbor_index in (slot_index - 1, slot_index + 1)
        if 0 <= neighbor_index < len(anchors)
    ]
    if neighbor_distances and distance > max(
        PADDLE_EVENT_SLOT_MIN_GATE_PIXELS,
        min(neighbor_distances) * PADDLE_EVENT_SLOT_MAX_SPACING_RATIO,
    ):
        return None
    return slot_index


def _observation_forward_position(
    observation: _PaddleObservation,
    travel_direction: str,
) -> float:
    reference_line = observation.reference_line
    forward = _normalize(
        (
            reference_line[2] - reference_line[0],
            reference_line[3] - reference_line[1],
        )
    )
    if forward is None:
        return -math.inf
    if (travel_direction == "right" and forward[0] < 0) or (
        travel_direction == "left" and forward[0] > 0
    ):
        forward = (-forward[0], -forward[1])
    line = observation.raw_line or observation.line
    paddle_center = _line_center(line)
    reference_center = _line_center(reference_line)
    return _dot(
        (
            paddle_center[0] - reference_center[0],
            paddle_center[1] - reference_center[1],
        ),
        forward,
    )


def _track_forward_position(
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
) -> float:
    positions = [
        _observation_forward_position(timed.observation, travel_direction)
        for timed in observations
    ]
    positions = [position for position in positions if math.isfinite(position)]
    return _median(positions) if positions else -math.inf


def _paddle_rotation_deltas(
    observations: list[_TimedPaddleObservation],
) -> list[float]:
    deltas: list[float] = []
    previous: _TimedPaddleObservation | None = None
    previous_angle: float | None = None
    for timed in observations:
        line = timed.observation.line
        angle = math.degrees(math.atan2(line[3] - line[1], line[2] - line[0])) % 180
        if (
            previous is not None
            and previous_angle is not None
            and 0 < timed.timestamp_ms - previous.timestamp_ms
            <= PADDLE_DIRECTION_MAX_SAMPLE_GAP_MS
        ):
            delta = (angle - previous_angle + 90) % 180 - 90
            if abs(delta) >= 0.5:
                deltas.append(delta)
        previous = timed
        previous_angle = angle
    return deltas


def _estimate_paddle_direction(
    deltas: list[float],
) -> tuple[str | None, str | None, float]:
    if len(deltas) < PADDLE_DIRECTION_MIN_DELTAS:
        return None, None, 0.0
    displacement = sum(abs(delta) for delta in deltas)
    if displacement < PADDLE_DIRECTION_MIN_DISPLACEMENT_DEGREES:
        return None, None, 0.0
    clockwise_weight = sum(abs(delta) for delta in deltas if delta > 0)
    anticlockwise_weight = sum(abs(delta) for delta in deltas if delta < 0)
    dominant_weight = max(clockwise_weight, anticlockwise_weight)
    consensus = dominant_weight / max(clockwise_weight + anticlockwise_weight, 1e-9)
    if consensus < PADDLE_DIRECTION_MIN_CONSENSUS:
        return None, None, round(consensus, 3)
    if clockwise_weight > anticlockwise_weight:
        return "clockwise", "right", round(consensus, 3)
    return "anticlockwise", "left", round(consensus, 3)


def _event_band_upward_width(scale_x: float, scale_y: float) -> float:
    raw = os.getenv(
        "SAM3_CENTERLINE_THICKNESS_PIXELS",
        os.getenv("SAM3_SHAFT_THICKNESS_PIXELS", "8"),
    )
    try:
        thickness = float(raw)
    except ValueError:
        thickness = 8.0
    return max(1.0, min(64.0, thickness)) * max(scale_x, scale_y)


def _consolidate_paddle_observations(
    targets: list[Centerline],
    references: list[Centerline],
    width: int,
    height: int,
    *,
    slot_anchors_by_reference: dict[str, list[float]] | None = None,
    travel_directions: dict[str, str] | None = None,
) -> list[_PaddleObservation]:
    if not references:
        return []
    grouped: dict[
        tuple[str, int | str | None],
        tuple[Centerline, list[list[Centerline]]],
    ] = {}
    for target in sorted(targets, key=lambda item: _line_length(item.line), reverse=True):
        reference = _nearest_centerline(target, references)
        reference_id = _record_track_id(reference.record)
        group_slot: int | str | None = None
        anchors = (slot_anchors_by_reference or {}).get(reference_id)
        travel_direction = (travel_directions or {}).get(reference_id)
        if anchors and travel_direction in {"left", "right"}:
            raw_observation = _PaddleObservation(
                source_ids=(_record_track_id(target.record),),
                reference_id=reference_id,
                line=target.line,
                reference_line=reference.line,
            )
            group_slot = _gated_event_paddle_slot_index(
                _observation_forward_position(raw_observation, travel_direction),
                anchors,
            )
            if group_slot is None:
                # Keep every unassigned raw detection isolated so neighboring
                # paddles cannot merge outside the slot gates.
                group_slot = f"unassigned:{_record_track_id(target.record)}"
        entry = grouped.setdefault((reference_id, group_slot), (reference, []))
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
    observations_by_reference: dict[str, list[_PaddleObservation]] = {}
    for (reference_id, _group_slot), (reference, clusters) in grouped.items():
        for cluster in clusters:
            observations_by_reference.setdefault(reference_id, []).append(
                _PaddleObservation(
                    source_ids=tuple(
                        sorted({_record_track_id(item.record) for item in cluster})
                    ),
                    reference_id=reference_id,
                    line=_merge_paddle_lines([item.line for item in cluster]),
                    reference_line=reference.line,
                )
            )
    observations: list[_PaddleObservation] = []
    for reference_observations in observations_by_reference.values():
        observations.extend(
            _filter_reflected_paddle_observations(
                reference_observations,
                width,
                height,
            )
        )
    return observations


def _filter_reflected_paddle_observations(
    observations: list[_PaddleObservation],
    width: int,
    height: int,
) -> list[_PaddleObservation]:
    if len(observations) < 2:
        return observations
    close_limit = max(
        PADDLE_REFLECTION_MIN_CLOSE_PIXELS,
        min(width, height) * PADDLE_REFLECTION_CLOSE_FRAME_RATIO,
    )
    depths = {
        observation: sum(
            _endpoint_signed_depths(observation.line, observation.reference_line)
        )
        / 2
        for observation in observations
    }
    close = [
        observation
        for observation in observations
        if abs(depths[observation]) <= close_limit
    ]
    if len(close) < 2 or not any(depths[observation] < 0 for observation in close):
        return observations
    close_set = set(close)
    return [
        observation
        for observation in observations
        if observation not in close_set or depths[observation] <= 0
    ]


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
    perpendicular_limit = max(
        PADDLE_FRAGMENT_MIN_PERPENDICULAR_PIXELS,
        min(width, height) * PADDLE_FRAGMENT_PERPENDICULAR_FRAME_RATIO,
    )
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


def _restore_track_stroke_lengths(
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
    band_upward_width: float,
    candidate_validator: Callable[
        [_TimedPaddleObservation, Line, int], bool | None
    ]
    | None = None,
) -> list[_TimedPaddleObservation]:
    """Restore each 0-180 stroke before chronological event detection.

    The 0-90 half inherits length forward in time. The 90-180 half inherits
    length backward from the observation nearest 180 degrees. A tracking gap
    starts a new independent segment, and each 360-degree cycle is isolated.
    """
    if not observations:
        return observations

    restored = list(observations)
    segment_start = 0
    for index in range(1, len(observations) + 1):
        at_end = index == len(observations)
        has_gap = not at_end and (
            observations[index].timestamp_ms - observations[index - 1].timestamp_ms
            > PADDLE_EVENT_MAX_CONFIRM_GAP_MS
        )
        if not at_end and not has_gap:
            continue
        segment = restored[segment_start:index]
        restored[segment_start:index] = _restore_track_segment_stroke_lengths(
            segment,
            travel_direction,
            band_upward_width,
            candidate_validator,
        )
        segment_start = index
    return restored


def _restore_track_segment_stroke_lengths(
    observations: list[_TimedPaddleObservation],
    travel_direction: str,
    band_upward_width: float,
    candidate_validator: Callable[
        [_TimedPaddleObservation, Line, int], bool | None
    ]
    | None = None,
) -> list[_TimedPaddleObservation]:
    oriented_lines: list[Line] = []
    previous_line: Line | None = None
    for timed in observations:
        raw_line = timed.observation.raw_line or timed.observation.line
        oriented_line = _orient_line_like(raw_line, previous_line)
        oriented_lines.append(oriented_line)
        previous_line = oriented_line

    active_blade = _infer_segment_active_blade(
        observations,
        oriented_lines,
        travel_direction,
        band_upward_width,
    )
    if active_blade not in (0, 1):
        if candidate_validator is None:
            return observations
        return [
            replace(
                timed,
                observation=replace(
                    timed.observation,
                    phase_length_verified=False,
                ),
            )
            for timed in observations
        ]

    phase_state = _PaddleEventState(
        active_blade=active_blade,
        travel_direction=travel_direction,
    )
    phase_groups: dict[int, list[tuple[int, float, Line]]] = {}
    for index, timed in enumerate(observations):
        raw_line = oriented_lines[index]
        raw_angle = _directed_blade_angle(
            raw_line,
            timed.observation.reference_line,
            active_blade,
            travel_direction,
        )
        if not _advance_directed_paddle_phase(phase_state, raw_angle):
            continue
        if phase_state.unwrapped_angle is None:
            continue
        phase_angle = phase_state.unwrapped_angle - phase_state.cycle_index * 360
        if 0 <= phase_angle <= 180:
            phase_groups.setdefault(phase_state.cycle_index, []).append(
                (index, phase_angle, raw_line)
            )

    restored_lines: dict[int, Line] = {}
    indexed_validator: Callable[[int, Line, int], bool | None] | None = None
    if candidate_validator is not None:
        def indexed_validator(
            sample_index: int,
            sample_line: Line,
            blade: int,
        ) -> bool | None:
            return candidate_validator(
                observations[sample_index],
                sample_line,
                blade,
            )

    verified_indices: set[int] = set()
    for samples in phase_groups.values():
        restored_lines.update(
            _restore_bidirectional_phase_lines(
                samples,
                active_blade,
                candidate_validator=indexed_validator,
                verified_indices=verified_indices,
            )
        )

    result = list(observations)
    for index, timed in enumerate(observations):
        line = restored_lines.get(index)
        phase_length_verified = (
            index in verified_indices if candidate_validator is not None else True
        )
        if line is None and timed.observation.phase_length_verified == phase_length_verified:
            continue
        timed = observations[index]
        result[index] = replace(
            timed,
            observation=replace(
                timed.observation,
                line=line if line is not None else timed.observation.line,
                raw_line=line if line is not None else timed.observation.raw_line,
                phase_length_restored=line is not None,
                phase_length_verified=phase_length_verified,
            ),
        )
    return result


def _infer_segment_active_blade(
    observations: list[_TimedPaddleObservation],
    oriented_lines: list[Line],
    travel_direction: str,
    band_upward_width: float,
) -> int | None:
    previous_depths: tuple[float, float] | None = None
    for timed, line in zip(observations, oriented_lines, strict=True):
        depths = _endpoint_signed_depths(line, timed.observation.reference_line)
        if previous_depths is not None:
            transitions: list[tuple[float, int]] = []
            for blade in range(2):
                delta = depths[blade] - previous_depths[blade]
                if _waterline_transition_kind(
                    previous_depths[blade],
                    depths[blade],
                    delta,
                    band_upward_width,
                ) is not None:
                    transitions.append((abs(delta), blade))
            if transitions:
                return max(transitions)[1]
        previous_depths = depths

    # A fully cropped transition may never reach the waterline. In a partial
    # stroke, the active endpoint is still identifiable as the lower endpoint
    # during its directed 0-180 half. Avoid guessing when both endpoints have
    # equal evidence (for example, a complete symmetric revolution).
    scores: list[tuple[int, float]] = []
    for blade in range(2):
        count = 0
        depth_advantage = 0.0
        for timed, line in zip(observations, oriented_lines, strict=True):
            phase_angle = _directed_blade_angle(
                line,
                timed.observation.reference_line,
                blade,
                travel_direction,
            )
            if not 0 <= phase_angle <= 180:
                continue
            depths = _endpoint_signed_depths(line, timed.observation.reference_line)
            advantage = depths[blade] - depths[1 - blade]
            if advantage <= 0:
                continue
            count += 1
            depth_advantage += advantage
        scores.append((count, depth_advantage))
    if scores[0][0] == scores[1][0] and math.isclose(
        scores[0][1],
        scores[1][1],
        abs_tol=1e-6,
    ):
        return None
    return 0 if scores[0] > scores[1] else 1


def _restore_bidirectional_phase_lines(
    samples: list[tuple[int, float, Line]],
    active_blade: int,
    candidate_validator: Callable[[int, Line, int], bool | None] | None = None,
    verified_indices: set[int] | None = None,
) -> dict[int, Line]:
    """Restore both stroke halves against fixed CNN-confirmed phase anchors."""
    restored: dict[int, Line] = {}
    restored_verified: dict[int, bool] = {}

    def retain(index: int, line: Line, verified: bool) -> None:
        existing = restored.get(index)
        existing_verified = restored_verified.get(index, False)
        if (
            existing is None
            or (verified and not existing_verified)
            or (
                verified == existing_verified
                and _line_length(line) > _line_length(existing)
            )
        ):
            restored[index] = line
            restored_verified[index] = verified

    def is_complete(index: int, line: Line) -> bool:
        return (
            candidate_validator is None
            or candidate_validator(index, line, active_blade) is True
        )

    def relative_difference(first: float, second: float) -> float:
        return abs(first - second) / max(first, 1e-6)

    def restore_half(ordered: list[tuple[int, float, Line]]) -> None:
        complete_samples = [
            sample for sample in ordered if is_complete(sample[0], sample[2])
        ]
        seed: tuple[
            tuple[int, float, Line],
            tuple[int, float, Line],
        ] | None = None
        for candidates in zip(
            complete_samples,
            complete_samples[1:],
        ):
            first_length = _line_length(candidates[0][2])
            second_length = _line_length(candidates[1][2])
            if (
                relative_difference(first_length, second_length)
                <= PADDLE_PHASE_ANCHOR_AGREEMENT_RELATIVE_TOLERANCE
            ):
                seed = candidates
                break

        if seed is None:
            for index, _phase_angle, line in ordered:
                retain(index, line, False)
            return

        phase_anchor = (
            _line_length(seed[0][2]) + _line_length(seed[1][2])
        ) / len(seed)
        previous_genuine_length = phase_anchor
        for index, _phase_angle, line in ordered:
            current_length = _line_length(line)
            candidate_is_genuine = (
                is_complete(index, line)
                and current_length >= phase_anchor
                and relative_difference(phase_anchor, current_length)
                <= PADDLE_PHASE_CANDIDATE_RELATIVE_TOLERANCE
            )
            if candidate_is_genuine:
                previous_genuine_length = current_length
                retain(index, line, True)
                continue
            retain(
                index,
                _set_paddle_active_endpoint_length(
                    line,
                    active_blade,
                    previous_genuine_length,
                ),
                True,
            )

    restore_half(
        sorted(
            (sample for sample in samples if sample[1] <= 90),
            key=lambda item: item[1],
        )
    )
    restore_half(
        sorted(
            (sample for sample in samples if sample[1] >= 90),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    if verified_indices is not None:
        verified_indices.update(
            index for index, verified in restored_verified.items() if verified
        )
    return restored


def _reset_directed_paddle_tracking_state(state: _PaddleEventState) -> None:
    """Discard state that cannot safely cross a real observation gap."""
    state.candidates.clear()
    state.phase_confident = False
    state.active_blade = None
    state.last_directed_angle = None
    state.unwrapped_angle = None
    state.emitted_events.clear()
    state.last_observation_ms = -1
    state.last_orientation_line = None
    state.last_seen_ms = -1
    state.last_line = None
    state.last_reference_line = None
    state.endpoint_depths = None
    _clear_stroke_length(state)


def _invalidate_directed_event_evidence(state: _PaddleEventState) -> None:
    """Break event confirmation without discarding physical orientation."""
    state.candidates.clear()
    state.last_seen_ms = -1
    state.last_line = None
    state.last_reference_line = None
    state.endpoint_depths = None


def _update_directed_paddle_state(
    state: _PaddleEventState,
    observation: _PaddleObservation,
    timestamp_ms: int,
    band_upward_width: float,
    *,
    event_eligible: bool = True,
) -> PaddleEvent | None:
    if (
        state.last_observation_ms >= 0
        and timestamp_ms - state.last_observation_ms
        > PADDLE_EVENT_MAX_CONFIRM_GAP_MS
    ):
        _reset_directed_paddle_tracking_state(state)

    previous_depths = state.endpoint_depths
    previous_seen_ms = state.last_seen_ms
    previous_line = state.last_line
    previous_reference_line = state.last_reference_line
    previous_cycle_index = state.cycle_index
    previous_phase_angle = (
        state.unwrapped_angle - state.cycle_index * 360
        if state.unwrapped_angle is not None
        else None
    )

    observed_line = _orient_line_like(
        observation.line,
        state.last_orientation_line or state.last_line,
    )
    raw_line = (
        _orient_line_like(observation.raw_line, observed_line)
        if observation.raw_line
        else observed_line
    )
    state.last_observation_ms = timestamp_ms
    state.last_orientation_line = observed_line

    phase_advanced = True
    if state.active_blade is not None and state.travel_direction is not None:
        raw_angle = _directed_blade_angle(
            observed_line,
            observation.reference_line,
            state.active_blade,
            state.travel_direction,
        )
        phase_advanced = _advance_directed_paddle_phase(state, raw_angle)

    if not event_eligible:
        _invalidate_directed_event_evidence(state)
        return None

    line = observed_line

    if state.active_blade is not None and state.travel_direction is not None:
        if not phase_advanced:
            depths = _endpoint_signed_depths(line, observation.reference_line)
            state.last_seen_ms = timestamp_ms
            state.last_line = line
            state.last_reference_line = observation.reference_line
            state.endpoint_depths = depths
            return None
        line = _inherit_stroke_phase_length(
            state,
            raw_line,
            fallback=observed_line,
            phase_preprocessed=observation.phase_length_restored,
        )
    depths = _endpoint_signed_depths(line, observation.reference_line)
    state.last_seen_ms = timestamp_ms
    state.last_line = line
    state.last_reference_line = observation.reference_line

    confirmed = _confirm_pending_directed_event(
        state,
        timestamp_ms,
        depths,
        band_upward_width,
    )
    if confirmed is not None:
        if confirmed.kind == "catch" and state.stroke_length is not None:
            line = _restore_immersed_paddle_length(state, line)
            state.last_line = line
            depths = _endpoint_signed_depths(line, observation.reference_line)
        state.endpoint_depths = depths
        return confirmed

    if previous_depths is None:
        state.endpoint_depths = depths
        return None

    transitions: list[tuple[float, str, int, bool]] = []
    blades = range(2) if state.active_blade is None else (state.active_blade,)
    for blade in blades:
        depth_delta = depths[blade] - previous_depths[blade]
        kind = _waterline_transition_kind(
            previous_depths[blade],
            depths[blade],
            depth_delta,
            band_upward_width,
        )
        if kind is None:
            continue
        skipped_exit_band = (
            kind == "exit"
            and previous_depths[blade] > 0
            and depths[blade] < -band_upward_width
        )
        if not skipped_exit_band and not _blade_transition_on_waterline(
            line,
            observation.reference_line,
            blade,
            band_upward_width,
        ):
            continue
        transitions.append((abs(depth_delta), kind, blade, skipped_exit_band))

    if transitions:
        _, kind, blade, skipped_exit_band = max(transitions)
        if state.active_blade is None:
            state.active_blade = blade
            if state.travel_direction is None:
                state.endpoint_depths = depths
                return None
            raw_angle = _directed_blade_angle(
                line,
                observation.reference_line,
                blade,
                state.travel_direction,
            )
            _advance_directed_paddle_phase(state, raw_angle)
            line = _inherit_stroke_phase_length(
                state,
                raw_line,
                initial_length=_line_length(previous_line) if previous_line else None,
                fallback=line,
                phase_preprocessed=observation.phase_length_restored,
            )
            depths = _endpoint_signed_depths(line, observation.reference_line)
            state.last_line = line
        if _event_phase_allowed(state, kind):
            depth_delta = depths[blade] - previous_depths[blade]
            confidence = min(
                1.0,
                state.direction_confidence
                * (
                    0.7
                    + min(0.3, abs(depth_delta) / max(band_upward_width * 2, 1))
                ),
            )
            select_crossing_geometry = previous_line is not None and (
                skipped_exit_band or kind == "catch"
            )
            if select_crossing_geometry:
                use_previous = abs(previous_depths[blade]) <= abs(depths[blade])
                selected_line = previous_line if use_previous else line
                selected_reference = (
                    previous_reference_line
                    if use_previous and previous_reference_line is not None
                    else observation.reference_line
                )
                selected_timestamp_ms = previous_seen_ms if use_previous else timestamp_ms
                selected_phase_angle = (
                    _directed_blade_angle(
                        selected_line,
                        selected_reference,
                        blade,
                        state.travel_direction,
                    )
                    if state.travel_direction is not None
                    else previous_phase_angle if use_previous else None
                )
                selected_cycle_index = previous_cycle_index if use_previous else None
                _start_directed_event_candidate(
                    state,
                    kind,
                    blade,
                    selected_timestamp_ms,
                    selected_line,
                    selected_reference,
                    confidence,
                    phase_angle=selected_phase_angle,
                    cycle_index=selected_cycle_index,
                )
                if skipped_exit_band:
                    confirmed_skip = _confirm_pending_directed_event(
                        state,
                        timestamp_ms,
                        depths,
                        band_upward_width,
                    )
                    state.endpoint_depths = depths
                    return confirmed_skip
                state.endpoint_depths = depths
                return None
            _start_directed_event_candidate(
                state,
                kind,
                blade,
                timestamp_ms,
                line,
                observation.reference_line,
                confidence,
            )
    state.endpoint_depths = depths
    return None


def _update_phase_aware_paddle_state(
    state: _PaddleEventState,
    observation: _PaddleObservation,
    timestamp_ms: int,
    band_upward_width: float,
) -> PaddleEvent | None:
    """Compatibility name for the directed phase-aware implementation."""
    line = _stabilize_paddle_line(state, observation.line)
    return _update_directed_paddle_state(
        state,
        replace(observation, line=line),
        timestamp_ms,
        band_upward_width,
    )


def _waterline_transition_kind(
    previous_depth: float,
    depth: float,
    depth_delta: float,
    band_upward_width: float,
) -> str | None:
    if (
        previous_depth < -band_upward_width
        and depth >= -band_upward_width
        and depth_delta > PADDLE_DEPTH_MOTION_EPSILON_PIXELS
    ):
        return "catch"
    if (
        previous_depth > 0
        and depth <= 0
        and depth_delta < -PADDLE_DEPTH_MOTION_EPSILON_PIXELS
    ):
        return "exit"
    return None


def _blade_transition_on_waterline(
    line: Line,
    reference_line: Line,
    blade: int,
    band_upward_width: float,
) -> bool:
    # Event analysis treats the fitted waterline as an infinite line. The finite
    # boat-span segment remains attached to the event solely for export drawing.
    depths = _endpoint_signed_depths(line, reference_line)
    active_depth = depths[blade]
    dry_depth = depths[1 - blade]
    inner_depth = active_depth + (dry_depth - active_depth) * PADDLE_BLADE_ZONE_RATIO
    blade_zone_overlaps = (
        min(active_depth, inner_depth) <= 0
        and max(active_depth, inner_depth) >= -band_upward_width
    )
    paddle_crosses = min(depths) <= 0 <= max(depths)
    return blade_zone_overlaps or paddle_crosses


def _directed_blade_angle(
    line: Line,
    reference_line: Line,
    blade: int,
    travel_direction: str,
) -> float:
    reference_axis = _normalize(
        (reference_line[2] - reference_line[0], reference_line[3] - reference_line[1])
    )
    if reference_axis is None:
        return 0.0
    forward = reference_axis
    if (travel_direction == "right" and forward[0] < 0) or (
        travel_direction == "left" and forward[0] > 0
    ):
        forward = (-forward[0], -forward[1])
    down = (-forward[1], forward[0])
    if down[1] < 0:
        down = (-down[0], -down[1])
    center = _line_center(line)
    endpoint = (line[0], line[1]) if blade == 0 else (line[2], line[3])
    blade_vector = _normalize((endpoint[0] - center[0], endpoint[1] - center[1]))
    if blade_vector is None:
        return 0.0
    return math.degrees(
        math.atan2(_dot(blade_vector, down), _dot(blade_vector, forward))
    ) % 360


def _advance_directed_paddle_phase(
    state: _PaddleEventState,
    raw_angle: float,
) -> bool:
    raw_angle %= 360
    if state.last_directed_angle is None or state.unwrapped_angle is None:
        state.last_directed_angle = raw_angle
        state.unwrapped_angle = raw_angle
        state.cycle_index = math.floor(raw_angle / 360)
        state.phase_confident = True
        return True
    delta = (raw_angle - state.last_directed_angle + 180) % 360 - 180
    if delta < -PADDLE_PHASE_BACKTRACK_TOLERANCE_DEGREES:
        state.phase_confident = False
        return False
    if delta < 0:
        state.phase_confident = True
        return True
    previous_raw_angle = state.last_directed_angle
    state.last_directed_angle = raw_angle
    state.unwrapped_angle += delta
    previous_cycle = state.cycle_index
    state.cycle_index = math.floor(state.unwrapped_angle / 360)
    if previous_raw_angle > 270 and raw_angle < 90:
        state.cycle_index = previous_cycle + 1
        state.unwrapped_angle = state.cycle_index * 360 + raw_angle
    if state.cycle_index != previous_cycle:
        _clear_stroke_length(state)
    state.phase_confident = True
    return True


def _start_directed_event_candidate(
    state: _PaddleEventState,
    kind: str,
    blade: int,
    timestamp_ms: int,
    line: Line,
    reference_line: Line,
    confidence: float,
    *,
    phase_angle: float | None = None,
    cycle_index: int | None = None,
) -> None:
    candidate_phase_angle = (
        phase_angle
        if phase_angle is not None
        else (
            _directed_blade_angle(
                line,
                reference_line,
                blade,
                state.travel_direction,
            )
            if state.travel_direction is not None
            else 0.0
        )
    )
    state.candidates[kind] = _PaddleEventCandidate(
        kind=kind,
        blade=blade,
        cycle_index=state.cycle_index if cycle_index is None else cycle_index,
        timestamp_ms=timestamp_ms,
        line=line,
        reference_line=reference_line,
        phase_angle=candidate_phase_angle,
        confidence=confidence,
    )


def _confirm_pending_directed_event(
    state: _PaddleEventState,
    timestamp_ms: int,
    depths: tuple[float, float],
    band_upward_width: float,
) -> PaddleEvent | None:
    for kind, candidate in list(state.candidates.items()):
        age_ms = timestamp_ms - candidate.timestamp_ms
        compatible = (
            depths[candidate.blade] >= -band_upward_width
            if kind == "catch"
            else depths[candidate.blade] <= 0
        )
        if age_ms > PADDLE_EVENT_MAX_CONFIRM_GAP_MS or not compatible:
            del state.candidates[kind]
            continue
        candidate.count += 1
        if candidate.count < PADDLE_EVENT_CONFIRM_SAMPLES:
            continue
        del state.candidates[kind]
        key = (candidate.cycle_index, kind)
        if key in state.emitted_events:
            continue
        state.emitted_events.add(key)
        event = PaddleEvent(
            kind=candidate.kind,
            timestamp_ms=candidate.timestamp_ms,
            instance_id=state.physical_id,
            line=candidate.line,
            confidence=round(candidate.confidence, 3),
            reference_line=candidate.reference_line,
            degree=_line_angle_degrees(candidate.line, candidate.reference_line),
            phase_angle=candidate.phase_angle,
            cycle_index=candidate.cycle_index,
            active_blade=candidate.blade,
            rotation_direction=state.rotation_direction,
            travel_direction=state.travel_direction,
        )
        return event
    return None


def _event_phase_allowed(
    state: _PaddleEventState,
    kind: str,
    cycle_index: int | None = None,
) -> bool:
    cycle = state.cycle_index if cycle_index is None else cycle_index
    return (cycle, kind) not in state.emitted_events and kind not in state.candidates


def _catch_phase_allowed(state: _PaddleEventState, blade: int = 0) -> bool:
    return (
        state.active_blade is None or state.active_blade == blade
    ) and _event_phase_allowed(
        state,
        "catch",
    )


def _restore_immersed_paddle_length(
    state: _PaddleEventState,
    observed: Line,
) -> Line:
    target_length = state.stroke_length
    blade = state.stroke_blade
    observed_length = _line_length(observed)
    if (
        target_length is None
        or blade not in (0, 1)
        or observed_length < 2
        or observed_length >= target_length
    ):
        return observed

    return _extend_paddle_active_endpoint(observed, blade, target_length)


def _extend_paddle_active_endpoint(
    observed: Line,
    active_blade: int,
    target_length: float,
) -> Line:
    observed_length = _line_length(observed)
    if (
        active_blade not in (0, 1)
        or observed_length < 2
        or observed_length >= target_length
    ):
        return observed

    first = (observed[0], observed[1])
    second = (observed[2], observed[3])
    active = first if active_blade == 0 else second
    dry = second if active_blade == 0 else first
    direction = _normalize((active[0] - dry[0], active[1] - dry[1]))
    if direction is None:
        return observed
    restored_active = (
        dry[0] + direction[0] * target_length,
        dry[1] + direction[1] * target_length,
    )
    if active_blade == 0:
        return (restored_active[0], restored_active[1], dry[0], dry[1])
    return (dry[0], dry[1], restored_active[0], restored_active[1])


def _set_paddle_active_endpoint_length(
    observed: Line,
    active_blade: int,
    target_length: float,
) -> Line:
    """Set length from the inactive endpoint, allowing an outlier to be shortened."""
    observed_length = _line_length(observed)
    if active_blade not in (0, 1) or observed_length < 2 or target_length < 2:
        return observed
    first = (observed[0], observed[1])
    second = (observed[2], observed[3])
    active = first if active_blade == 0 else second
    inactive = second if active_blade == 0 else first
    direction = _normalize((active[0] - inactive[0], active[1] - inactive[1]))
    if direction is None:
        return observed
    restored_active = (
        inactive[0] + direction[0] * target_length,
        inactive[1] + direction[1] * target_length,
    )
    if active_blade == 0:
        return (*restored_active, *inactive)
    return (*inactive, *restored_active)


def _inherit_stroke_phase_length(
    state: _PaddleEventState,
    observed: Line,
    *,
    initial_length: float | None = None,
    fallback: Line | None = None,
    phase_preprocessed: bool = False,
) -> Line:
    """Keep paddle length non-decreasing within one directed 0-180 degree phase."""
    fallback_line = observed if fallback is None else fallback
    if (
        state.active_blade not in (0, 1)
        or state.unwrapped_angle is None
        or state.cycle_index < 0
    ):
        return fallback_line

    phase_angle = state.unwrapped_angle - state.cycle_index * 360
    if phase_angle < 0 or phase_angle > 180:
        if state.stroke_cycle_index == state.cycle_index:
            _clear_stroke_length(state)
        return fallback_line

    observed_length = _line_length(observed)
    if observed_length < 2:
        return fallback_line

    phase_half = 0 if phase_angle <= 90 else 1
    if phase_preprocessed:
        state.stroke_length = observed_length
        state.stroke_blade = state.active_blade
        state.stroke_cycle_index = state.cycle_index
        state.stroke_phase_half = phase_half
        return observed

    if (
        state.stroke_cycle_index != state.cycle_index
        or state.stroke_blade != state.active_blade
        or state.stroke_length is None
    ):
        seed_length = observed_length
        if initial_length is not None and math.isfinite(initial_length):
            seed_length = max(seed_length, initial_length)
        state.stroke_length = seed_length
        state.stroke_blade = state.active_blade
        state.stroke_cycle_index = state.cycle_index
        state.stroke_phase_half = phase_half
    elif state.stroke_phase_half is None:
        state.stroke_phase_half = phase_half
    elif observed_length > state.stroke_length:
        state.stroke_length = observed_length

    return _restore_immersed_paddle_length(state, observed)


def _clear_stroke_length(state: _PaddleEventState) -> None:
    state.stroke_length = None
    state.stroke_blade = None
    state.stroke_cycle_index = None
    state.stroke_phase_half = None


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
    if event.reference_line is not None and _event_display_angle(event) is not None:
        vertex = _line_intersection(event.line, event.reference_line)
        if vertex is not None and (
            -width * 0.1 <= vertex[0] <= width * 1.1
            and -height * 0.1 <= vertex[1] <= height * 1.1
        ):
            _draw_event_angle_marker(image, width, height, event, vertex, options)


def _draw_event_angle_marker(
    image: bytearray,
    width: int,
    height: int,
    event: PaddleEvent,
    vertex: tuple[float, float],
    options: ExportOptions,
) -> None:
    assert event.reference_line is not None
    angle_color = _paddle_event_angle_color(event)
    reference_vector = _normalize(
        (
            event.reference_line[2] - event.reference_line[0],
            event.reference_line[3] - event.reference_line[1],
        )
    )
    center = _line_center(event.line)
    if event.active_blade == 0:
        paddle_vector = _normalize((event.line[0] - center[0], event.line[1] - center[1]))
    elif event.active_blade == 1:
        paddle_vector = _normalize((event.line[2] - center[0], event.line[3] - center[1]))
    else:
        paddle_vector = _normalize(
            (event.line[2] - event.line[0], event.line[3] - event.line[1])
        )
    if reference_vector is None or paddle_vector is None:
        return
    if event.travel_direction == "right" and reference_vector[0] < 0:
        reference_vector = (-reference_vector[0], -reference_vector[1])
    elif event.travel_direction == "left" and reference_vector[0] > 0:
        reference_vector = (-reference_vector[0], -reference_vector[1])
    elif event.phase_angle is None and _dot(reference_vector, paddle_vector) < 0:
        paddle_vector = (-paddle_vector[0], -paddle_vector[1])
    if event.phase_angle is not None and event.travel_direction in {"left", "right"}:
        display_angle = _event_display_angle(event)
        assert display_angle is not None
        if (
            event.discipline != "canoe"
            and event.kind == "exit"
            and 0 <= event.phase_angle <= 180
        ):
            reference_vector = (-reference_vector[0], -reference_vector[1])
            display_radians = math.radians(display_angle)
            delta = -display_radians if event.travel_direction == "right" else display_radians
        else:
            phase_radians = math.radians(event.phase_angle % 360)
            delta = phase_radians if event.travel_direction == "right" else -phase_radians
    else:
        end_angle = math.atan2(paddle_vector[1], paddle_vector[0])
        start_angle = math.atan2(reference_vector[1], reference_vector[0])
        delta = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
    start_angle = math.atan2(reference_vector[1], reference_vector[0])
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
            angle_color,
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
        angle_color,
        options,
    ):
        _draw_small_degree_label(
            image,
            width,
            height,
            label_x,
            label_y,
            text,
            angle_color,
        )


def _event_display_angle(event: PaddleEvent) -> float | None:
    if event.phase_angle is None:
        return event.degree
    if event.discipline == "canoe" and event.kind == "exit" and 0 <= event.phase_angle <= 180:
        return 180 - event.phase_angle
    if event.discipline == "canoe":
        return event.phase_angle
    if event.kind == "exit" and 0 <= event.phase_angle <= 180:
        return 180 - event.phase_angle
    return event.phase_angle


def _event_label_text(event: PaddleEvent) -> str:
    angle = _event_display_angle(event)
    if angle is None:
        return ""
    if event.discipline == "canoe":
        return f"{round(angle):+d}°"
    return f"{round(angle) % 360}°"


def _paddle_event_angle_color(event: PaddleEvent) -> Color:
    if event.kind == "catch":
        return PADDLE_EVENT_CATCH_ANGLE_COLOR
    return PADDLE_EVENT_EXIT_ANGLE_COLOR


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
    *,
    text_color_overrides: dict[int, Color] | None = None,
) -> None:
    if _draw_degree_label_block_with_pillow(
        image,
        width,
        height,
        labels,
        options,
        text_color_overrides=text_color_overrides,
    ):
        return
    _draw_degree_label_block_bitmap(
        image,
        width,
        height,
        labels,
        options,
        text_color_overrides=text_color_overrides,
    )


def _degree_label_entries(
    labels: list[DegreeLabel],
    text_color_overrides: dict[int, Color] | None = None,
) -> list[DegreeLabelEntry]:
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
                else PADDLE_ANGLE_TEXT_COLOR
            )
        if text_color_overrides is not None and index - 1 in text_color_overrides:
            text_color = text_color_overrides[index - 1]
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
    *,
    text_color_overrides: dict[int, Color] | None = None,
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

    entries = _degree_label_entries(labels, text_color_overrides)
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
    *,
    text_color_overrides: dict[int, Color] | None = None,
) -> None:
    entries = _degree_label_entries(labels, text_color_overrides)
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


def _event_metric_table_top(
    height: int,
    table_height: int,
    signed_offset_percent: float,
) -> int:
    """Place event metrics from the top for positive and bottom for negative offsets."""
    offset = max(-45.0, min(45.0, signed_offset_percent))
    inset = round(height * abs(offset) / 100)
    top = inset if offset >= 0 else height - inset - table_height
    return max(0, min(max(0, height - table_height), top))


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


def _draw_body_metric_row(
    image: bytearray,
    width: int,
    height: int,
    entries: list[tuple[str, str, Color]],
    signed_offset_percent: float,
    *,
    angle_label_font_size: int | None = None,
    draw_background: bool = True,
) -> None:
    if not entries:
        return
    if _draw_body_metric_row_with_pillow(
        image,
        width,
        height,
        entries,
        signed_offset_percent,
        angle_label_font_size=angle_label_font_size,
        draw_background=draw_background,
    ):
        return

    scale = (
        max(2, round(angle_label_font_size / 7))
        if angle_label_font_size is not None
        else max(2, round(width / 700))
    )
    glyph_gap = max(1, round(scale * 0.4))
    item_gap = max(8, scale * 5)
    padding_x = scale * 4 if draw_background else 0
    padding_y = scale * 2 if draw_background else 0
    abbreviations = {
        "L Elbow": "LE",
        "R Elbow": "RE",
        "Torso": "T",
        "L Shoulder": "LS",
        "R Shoulder": "RS",
    }
    abbreviated = [
        f"{abbreviations.get(label, label[0])} {value}"
        for label, value, _color in entries
    ]
    item_widths = [
        sum(len(_glyph(character)[0]) * scale for character in text)
        + glyph_gap * max(0, len(text) - 1)
        + padding_x * 2
        for text in abbreviated
    ]
    row_height = 7 * scale + padding_y * 2
    total_width = sum(item_widths) + item_gap * max(0, len(entries) - 1)
    left = round((width - total_width) / 2)
    top = _event_metric_table_top(height, row_height, signed_offset_percent)
    for text, item_width, (_label, _value, color) in zip(
        abbreviated, item_widths, entries
    ):
        if draw_background:
            _fill_rect(
                image,
                width,
                height,
                left,
                top,
                item_width,
                row_height,
                (2, 5, 9, 205),
            )
        x = left + padding_x
        for character in text:
            glyph = _glyph(character)
            _draw_bitmap(image, width, height, x, top + padding_y, glyph, scale, color)
            x += len(glyph[0]) * scale + glyph_gap
        left += item_width + item_gap


def _draw_body_metric_row_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    entries: list[tuple[str, str, Color]],
    signed_offset_percent: float,
    *,
    angle_label_font_size: int | None = None,
    draw_background: bool = True,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    font_path = _find_export_font()
    if font_path is None:
        return False

    font_size = (
        int(angle_label_font_size)
        if angle_label_font_size is not None
        else max(14, round(width / 58))
    )
    measuring_surface = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(measuring_surface)
    texts = [f"{label} {value}" for label, value, _color in entries]
    gap = max(10, round(font_size * 0.7))
    padding_x = max(8, round(font_size * 0.55)) if draw_background else 0
    padding_y = max(4, round(font_size * 0.3)) if draw_background else 0
    while True:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except OSError:
            return False
        boxes = [draw.textbbox((0, 0), text, font=font) for text in texts]
        item_widths = [box[2] - box[0] + padding_x * 2 for box in boxes]
        total_width = sum(item_widths) + gap * max(0, len(entries) - 1)
        if total_width <= width * 0.94 or font_size <= 10:
            break
        font_size -= 1
        gap = max(6, round(font_size * 0.7))
        padding_x = max(6, round(font_size * 0.55)) if draw_background else 0
        padding_y = max(3, round(font_size * 0.3)) if draw_background else 0

    text_height = max(box[3] - box[1] for box in boxes)
    row_height = text_height + padding_y * 2
    top = _event_metric_table_top(height, row_height, signed_offset_percent)
    row_width = round(total_width)
    row_left = round((width - row_width) / 2)
    overlay = Image.new("RGBA", (row_width, row_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left = 0
    corner_radius = max(3, round(font_size * 0.25))
    for text, box, item_width, (_label, _value, color) in zip(
        texts, boxes, item_widths, entries
    ):
        if draw_background:
            draw.rounded_rectangle(
                (left, 0, left + item_width - 1, row_height - 1),
                radius=corner_radius,
                fill=(2, 5, 9, 205),
            )
        text_left = left + (item_width - (box[2] - box[0])) / 2 - box[0]
        text_top = (row_height - (box[3] - box[1])) / 2 - box[1]
        draw.text((text_left, text_top), text, font=font, fill=color)
        left += item_width + gap
    _blend_overlay_region(
        image,
        width,
        height,
        row_left,
        top,
        row_width,
        row_height,
        overlay.tobytes(),
    )
    return True


def _body_metric_label(label: str, discipline: str) -> str:
    if discipline != "canoe":
        return label
    return {
        "L Elbow": "左肘",
        "R Elbow": "右肘",
        "Torso": "躯干",
        "L Shoulder": "左肩",
        "R Shoulder": "右肩",
        "L Knee": "左膝",
        "R Knee": "右膝",
    }.get(label, label)


def _draw_body_motion_overlay(
    image: bytearray,
    width: int,
    height: int,
    record: dict[str, Any],
    *,
    discipline: str = "kayak",
    canoe_travel_direction: str | None = None,
    angle_label_font_size: int | None = None,
    metric_offset_percent: float = 10.0,
) -> None:
    landmarks = record.get("landmarks")
    if not isinstance(landmarks, dict):
        return
    thickness = max(2, round(min(width, height) / 300))
    radius = max(3, round(min(width, height) / 170))
    upper_body_connections = (
        ("wrist", "elbow"),
        ("elbow", "shoulder"),
        ("shoulder", "hip"),
    )
    connections = (
        upper_body_connections
        + (("hip", "knee"), ("knee", "ankle"))
        if discipline == "canoe"
        else upper_body_connections
    )
    hidden_body_side: str | None = None
    if discipline == "canoe":
        if canoe_travel_direction == "right":
            hidden_body_side = "left"
        elif canoe_travel_direction == "left":
            hidden_body_side = "right"
    joint_colors = {
        ("left", "elbow"): BODY_LEFT_ELBOW_COLOR,
        ("right", "elbow"): BODY_RIGHT_ELBOW_COLOR,
        ("left", "shoulder"): BODY_LEFT_SHOULDER_COLOR,
        ("right", "shoulder"): BODY_RIGHT_SHOULDER_COLOR,
        ("left", "knee"): BODY_LEFT_KNEE_COLOR,
        ("right", "knee"): BODY_RIGHT_KNEE_COLOR,
    }
    for side, color in (("left", BODY_LEFT_COLOR), ("right", BODY_RIGHT_COLOR)):
        if side == hidden_body_side:
            continue
        for first_name, second_name in connections:
            first = _body_point(landmarks.get(f"{side}_{first_name}"), width, height)
            second = _body_point(landmarks.get(f"{side}_{second_name}"), width, height)
            if first is None or second is None:
                continue
            _draw_line(
                image,
                width,
                height,
                (first[0], first[1], second[0], second[1]),
                color,
                thickness,
            )
        joints = ("wrist", "elbow", "shoulder", "hip")
        if discipline == "canoe":
            joints += ("knee", "ankle")
        for joint in joints:
            point = _body_point(landmarks.get(f"{side}_{joint}"), width, height)
            if point is not None:
                _fill_circle(
                    image,
                    width,
                    height,
                    round(point[0]),
                    round(point[1]),
                    radius,
                    joint_colors.get((side, joint), color),
                )

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    definitions = {
        "elbow": ("shoulder", "elbow", "wrist"),
        "shoulder": ("elbow", "shoulder", "hip"),
        "knee": ("hip", "knee", "ankle"),
    }
    joint_metrics = (
        ("L Elbow", "left", "elbow", BODY_LEFT_ELBOW_COLOR),
        ("R Elbow", "right", "elbow", BODY_RIGHT_ELBOW_COLOR),
    )
    shoulder_metrics = (
        ("L Shoulder", "left", "shoulder", BODY_LEFT_SHOULDER_COLOR),
        ("R Shoulder", "right", "shoulder", BODY_RIGHT_SHOULDER_COLOR),
    )
    metric_entries: list[tuple[str, str, Color]] = []
    for label, side, joint, color in joint_metrics:
        if side == hidden_body_side:
            continue
        value = _finite_body_metric(metrics, f"{side}_{joint}_deg")
        if value is not None:
            _draw_body_joint_arc(
                image,
                width,
                height,
                landmarks,
                side,
                definitions[joint],
                color,
                radius,
                thickness,
            )
        metric_entries.append(
            (label, f"{round(value)}°" if value is not None else "--", color)
        )
    lean = _finite_body_metric(metrics, "lean_deg")
    metric_entries.append(
        (
            "Torso",
            f"{lean:+.1f}°" if lean is not None else "--",
            BODY_TORSO_COLOR,
        )
    )
    for label, side, joint, color in shoulder_metrics:
        if side == hidden_body_side:
            continue
        value = _finite_body_metric(metrics, f"{side}_{joint}_deg")
        if value is not None:
            _draw_body_joint_arc(
                image,
                width,
                height,
                landmarks,
                side,
                definitions[joint],
                color,
                radius,
                thickness,
            )
        metric_entries.append(
            (label, f"{round(value)}°" if value is not None else "--", color)
        )
    if discipline == "canoe":
        for label, side, color in (
            ("L Knee", "left", BODY_LEFT_KNEE_COLOR),
            ("R Knee", "right", BODY_RIGHT_KNEE_COLOR),
        ):
            if side == hidden_body_side:
                continue
            value = _finite_body_metric(metrics, f"{side}_knee_deg")
            if value is not None:
                _draw_body_joint_arc(
                    image,
                    width,
                    height,
                    landmarks,
                    side,
                    definitions["knee"],
                    color,
                    radius,
                    thickness,
                )
            metric_entries.append(
                (label, f"{round(value)}°" if value is not None else "--", color)
            )
    _draw_body_metric_row(
        image,
        width,
        height,
        [
            (_body_metric_label(label, discipline), value, PADDLE_ANGLE_TEXT_COLOR)
            for label, value, _color in metric_entries
        ],
        metric_offset_percent,
        angle_label_font_size=None,
        draw_background=discipline != "canoe",
    )


def _finite_body_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _draw_body_joint_arc(
    image: bytearray,
    width: int,
    height: int,
    landmarks: dict[str, Any],
    side: str,
    definition: tuple[str, str, str],
    color: Color,
    joint_radius: int,
    line_thickness: int,
) -> None:
    first_name, vertex_name, third_name = definition
    first = _body_point(landmarks.get(f"{side}_{first_name}"), width, height)
    vertex = _body_point(landmarks.get(f"{side}_{vertex_name}"), width, height)
    third = _body_point(landmarks.get(f"{side}_{third_name}"), width, height)
    if first is None or vertex is None or third is None:
        return
    first_length = math.hypot(first[0] - vertex[0], first[1] - vertex[1])
    third_length = math.hypot(third[0] - vertex[0], third[1] - vertex[1])
    arc_radius = _clamp(
        min(first_length, third_length) * 0.3,
        joint_radius * 2.5,
        joint_radius * 7.0,
    )
    start_angle = math.atan2(first[1] - vertex[1], first[0] - vertex[0])
    end_angle = math.atan2(third[1] - vertex[1], third[0] - vertex[0])
    delta = end_angle - start_angle
    while delta <= -math.pi:
        delta += math.tau
    while delta > math.pi:
        delta -= math.tau
    segments = max(8, round(abs(delta) * arc_radius / 4))
    previous = (
        vertex[0] + math.cos(start_angle) * arc_radius,
        vertex[1] + math.sin(start_angle) * arc_radius,
    )
    for index in range(1, segments + 1):
        angle = start_angle + delta * index / segments
        current = (
            vertex[0] + math.cos(angle) * arc_radius,
            vertex[1] + math.sin(angle) * arc_radius,
        )
        _draw_line(
            image,
            width,
            height,
            (previous[0], previous[1], current[0], current[1]),
            color,
            max(2, round(line_thickness * 0.7)),
        )
        previous = current


def _body_point(
    landmark: Any, width: int, height: int
) -> tuple[float, float] | None:
    if not isinstance(landmark, dict):
        return None
    try:
        visibility = float(landmark.get("visibility", 0))
        presence = float(landmark.get("presence", 0))
        x = float(landmark["x"])
        y = float(landmark["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(visibility, presence) < MIN_LANDMARK_CONFIDENCE:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x * width, y * height)


def _body_landmark_midpoint(
    landmarks: dict[str, Any],
    first_name: str,
    second_name: str,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    points = [
        point
        for name in (first_name, second_name)
        if (point := _body_point(landmarks.get(name), width, height)) is not None
    ]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _fill_circle(
    image: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: Color,
) -> None:
    radius_sq = radius * radius
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_sq:
                _blend_pixel(image, width, x, y, color)


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
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "d": ("00001", "00001", "00001", "01111", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "n": ("00000", "10110", "11001", "10001", "10001", "10001", "10001"),
    "p": ("00000", "11110", "10001", "10001", "11110", "10000", "10000"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "11110", "00000"),
    "t": ("00100", "00100", "11111", "00100", "00100", "00101", "00010"),
    "v": ("00000", "10001", "10001", "10001", "01010", "01010", "00100"),
    "x": ("00000", "10001", "01010", "00100", "01010", "10001", "00000"),
    ":": ("000", "010", "010", "000", "010", "010", "000"),
    "+": ("000", "010", "010", "111", "010", "010", "000"),
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


def _blend_overlay_region(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    overlay_width: int,
    overlay_height: int,
    overlay: bytes,
) -> None:
    for overlay_y in range(overlay_height):
        image_y = top + overlay_y
        if image_y < 0 or image_y >= height:
            continue
        for overlay_x in range(overlay_width):
            image_x = left + overlay_x
            if image_x < 0 or image_x >= width:
                continue
            index = (overlay_y * overlay_width + overlay_x) * 4
            alpha = overlay[index + 3]
            if alpha <= 0:
                continue
            _blend_pixel(
                image,
                width,
                image_x,
                image_y,
                (overlay[index], overlay[index + 1], overlay[index + 2], alpha),
            )




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
