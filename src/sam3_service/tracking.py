from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

Box = tuple[float, float, float, float]
Point = tuple[float, float]
MATCH_THRESHOLD = 0.78
MAX_TRACK_GAP_MS = 1500
INVALID_COST = 1_000_000.0


@dataclass
class _Track:
    id: str
    prompt_id: str
    index: int
    box: Box
    center: Point
    velocity: Point
    angle: float | None
    last_timestamp_ms: int
    start_ms: int
    end_ms: int
    instance_ids: set[str] = field(default_factory=set)


class StableTrackAssigner:
    def __init__(self) -> None:
        self._active: dict[str, list[_Track]] = {}
        self._tracks: list[_Track] = []
        self._next_index: dict[str, int] = {}

    def assign(self, records: list[dict[str, Any]]) -> None:
        ordered = sorted(
            records,
            key=lambda record: (
                int(record.get("timestamp_ms", 0)),
                str(record.get("prompt_id", "")),
            ),
        )
        for (timestamp_ms, prompt_id), grouped in groupby(
            ordered,
            key=lambda record: (
                int(record.get("timestamp_ms", 0)),
                str(record.get("prompt_id", "")),
            ),
        ):
            self._assign_frame(prompt_id, timestamp_ms, list(grouped))

    def entries(self) -> list[dict[str, Any]]:
        return [
            {
                "id": track.id,
                "prompt_id": track.prompt_id,
                "start_ms": track.start_ms,
                "end_ms": track.end_ms,
                "instance_ids": sorted(track.instance_ids),
            }
            for track in sorted(self._tracks, key=lambda track: (track.prompt_id, track.index))
        ]

    def _assign_frame(
        self,
        prompt_id: str,
        timestamp_ms: int,
        records: list[dict[str, Any]],
    ) -> None:
        active = [
            track
            for track in self._active.get(prompt_id, [])
            if timestamp_ms - track.last_timestamp_ms <= MAX_TRACK_GAP_MS
        ]
        self._active[prompt_id] = active
        detections = [_detection_state(record) for record in records]
        costs = [
            [_match_cost(track, detection, timestamp_ms) for detection in detections]
            for track in active
        ]
        assignments = _minimum_cost_assignment(costs, MATCH_THRESHOLD)
        matched_detection_indexes: set[int] = set()
        for track_index, detection_index in assignments:
            if detection_index >= len(records):
                continue
            cost = costs[track_index][detection_index]
            if cost > MATCH_THRESHOLD:
                continue
            track = active[track_index]
            record = records[detection_index]
            self._update_track(track, detections[detection_index], record, timestamp_ms)
            matched_detection_indexes.add(detection_index)

        for detection_index, (record, detection) in enumerate(zip(records, detections)):
            if detection_index in matched_detection_indexes:
                continue
            track = self._new_track(prompt_id, timestamp_ms, detection, record)
            active.append(track)

    def _new_track(
        self,
        prompt_id: str,
        timestamp_ms: int,
        detection: tuple[Box, Point, float | None, str],
        record: dict[str, Any],
    ) -> _Track:
        index = self._next_index.get(prompt_id, 1)
        self._next_index[prompt_id] = index + 1
        box, center, angle, instance_id = detection
        track = _Track(
            id=f"{prompt_id}:track:{index}",
            prompt_id=prompt_id,
            index=index,
            box=box,
            center=center,
            velocity=(0.0, 0.0),
            angle=angle,
            last_timestamp_ms=timestamp_ms,
            start_ms=timestamp_ms,
            end_ms=timestamp_ms,
            instance_ids={instance_id} if instance_id else set(),
        )
        record["track_id"] = track.id
        self._tracks.append(track)
        return track

    @staticmethod
    def _update_track(
        track: _Track,
        detection: tuple[Box, Point, float | None, str],
        record: dict[str, Any],
        timestamp_ms: int,
    ) -> None:
        box, center, angle, instance_id = detection
        elapsed_ms = max(timestamp_ms - track.last_timestamp_ms, 1)
        measured_velocity = (
            (center[0] - track.center[0]) / elapsed_ms,
            (center[1] - track.center[1]) / elapsed_ms,
        )
        track.velocity = (
            track.velocity[0] * 0.6 + measured_velocity[0] * 0.4,
            track.velocity[1] * 0.6 + measured_velocity[1] * 0.4,
        )
        track.box = box
        track.center = center
        track.angle = angle
        track.last_timestamp_ms = timestamp_ms
        track.end_ms = timestamp_ms
        if instance_id:
            track.instance_ids.add(instance_id)
        record["track_id"] = track.id


def assign_stable_tracks_to_chunks(chunk_paths: list[Path]) -> list[dict[str, Any]]:
    assigner = StableTrackAssigner()
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        records = payload.get("frames", [])
        assigner.assign(records)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(path)
    return assigner.entries()


def _detection_state(
    record: dict[str, Any],
) -> tuple[Box, Point, float | None, str]:
    box = _record_box(record)
    center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
    return box, center, _record_angle(record), str(record.get("instance_id", ""))


def _record_box(record: dict[str, Any]) -> Box:
    values = record.get("box_xywh") or record.get("centerline_box_xywh") or (0, 0, 1, 1)
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError):
        return (0.0, 0.0, 1.0, 1.0)
    return (x, y, max(width, 1.0), max(height, 1.0))


def _record_angle(record: dict[str, Any]) -> float | None:
    values = record.get("centerline_line_xyxy")
    if not values or len(values) != 4:
        return None
    try:
        angle = math.degrees(
            math.atan2(float(values[3]) - float(values[1]), float(values[2]) - float(values[0]))
        ) % 180
    except (TypeError, ValueError):
        return None
    return angle if math.isfinite(angle) else None


def _match_cost(
    track: _Track,
    detection: tuple[Box, Point, float | None, str],
    timestamp_ms: int,
) -> float:
    box, center, angle, instance_id = detection
    elapsed_ms = max(timestamp_ms - track.last_timestamp_ms, 0)
    predicted_center = (
        track.center[0] + track.velocity[0] * elapsed_ms,
        track.center[1] + track.velocity[1] * elapsed_ms,
    )
    scale = max(
        (math.sqrt(track.box[2] * track.box[3]) + math.sqrt(box[2] * box[3])) / 2,
        1.0,
    )
    center_distance = math.hypot(
        center[0] - predicted_center[0], center[1] - predicted_center[1]
    ) / scale
    overlap = _box_iou(track.box, box)
    size_difference = abs(
        math.log(max(box[2] * box[3], 1.0) / max(track.box[2] * track.box[3], 1.0))
    )
    angle_difference = _angle_difference(track.angle, angle)
    if center_distance > 3.5 or size_difference > math.log(4):
        return INVALID_COST
    if overlap < 0.01 and center_distance > 1.75:
        return INVALID_COST
    if angle_difference > 65 and overlap < 0.1:
        return INVALID_COST
    cost = (
        0.45 * min(center_distance / 2, 1)
        + 0.30 * (1 - overlap)
        + 0.15 * min(size_difference / math.log(4), 1)
        + 0.10 * (angle_difference / 90)
    )
    if instance_id and instance_id in track.instance_ids:
        cost -= 0.05
    return max(cost, 0.0)


def _box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def _angle_difference(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 45.0
    difference = abs(first - second) % 180
    return min(difference, 180 - difference)


def _minimum_cost_assignment(
    costs: list[list[float]], unmatched_cost: float
) -> list[tuple[int, int]]:
    if not costs:
        return []
    row_count = len(costs)
    detection_count = len(costs[0]) if costs[0] else 0
    matrix = [row + [unmatched_cost] * row_count for row in costs]
    column_count = detection_count + row_count
    potentials_rows = [0.0] * (row_count + 1)
    potentials_columns = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    path = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimums = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    matrix[current_row - 1][candidate - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[candidate]
                )
                if reduced < minimums[candidate]:
                    minimums[candidate] = reduced
                    path[candidate] = column
                if minimums[candidate] < delta:
                    delta = minimums[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    potentials_rows[matched_row[candidate]] += delta
                    potentials_columns[candidate] -= delta
                else:
                    minimums[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = path[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = []
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment.append((matched_row[column] - 1, column - 1))
    return assignment
