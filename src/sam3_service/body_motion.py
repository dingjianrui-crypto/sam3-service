from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import JobCancelled, ServiceError

BODY_LANDMARK_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    # Ankles are retained for knee-angle calculation but are not rendered.
    "left_ankle": 27,
    "right_ankle": 28,
}
MIN_LANDMARK_CONFIDENCE = 0.5
SMOOTHING_ALPHA = 0.35
SMOOTHING_MAX_GAP_MS = 200


@dataclass(frozen=True)
class BodyMotionFrame:
    frame_index: int
    timestamp_ms: int
    landmarks: dict[str, dict[str, float]]


class BodyMotionAnalyzer(Protocol):
    @property
    def model_name(self) -> str: ...

    def analyze(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[BodyMotionFrame]: ...


class MediaPipeBodyMotionAnalyzer:
    def __init__(self, model_path: Path | None) -> None:
        self.model_path = model_path

    @property
    def model_name(self) -> str:
        model = self.model_path.name if self.model_path is not None else "unconfigured"
        return f"mediapipe-pose-landmarker:{model}"

    def analyze(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[BodyMotionFrame]:
        if self.model_path is None or not self.model_path.is_file():
            raise ServiceError(
                "POSE_MODEL_UNAVAILABLE",
                "Body motion was requested, but SAM3_POSE_MODEL_PATH is not available.",
                retryable=False,
            )
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                "Body motion requires the optional MediaPipe pose dependencies.",
                retryable=False,
            ) from exc

        fps = max(float(metadata.get("fps") or 30), 1.0)
        expected_frames = max(int(metadata.get("frame_count") or 0), 1)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ServiceError(
                "POSE_INFERENCE_FAILED",
                "The normalized video could not be opened for body-motion analysis.",
                retryable=True,
            )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=MIN_LANDMARK_CONFIDENCE,
            min_pose_presence_confidence=MIN_LANDMARK_CONFIDENCE,
            min_tracking_confidence=MIN_LANDMARK_CONFIDENCE,
            output_segmentation_masks=False,
        )
        frame_index = 0
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    if cancelled():
                        raise JobCancelled()
                    available, bgr_frame = capture.read()
                    if not available:
                        break
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = round(frame_index * 1000 / fps)
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    pose = result.pose_landmarks[0] if result.pose_landmarks else None
                    landmarks = _selected_landmarks(pose) if pose is not None else {}
                    yield BodyMotionFrame(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        landmarks=landmarks,
                    )
                    frame_index += 1
                    progress(frame_index, expected_frames)
        except JobCancelled:
            raise
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "POSE_INFERENCE_FAILED",
                f"MediaPipe Pose Landmarker failed: {exc}",
                retryable=True,
            ) from exc
        finally:
            capture.release()


class MockBodyMotionAnalyzer:
    @property
    def model_name(self) -> str:
        return "mock-body-motion-v1"

    def analyze(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[BodyMotionFrame]:
        del video_path
        frame_count = max(int(metadata.get("frame_count") or 1), 1)
        fps = max(float(metadata.get("fps") or 30), 1.0)
        for frame_index in range(frame_count):
            if cancelled():
                raise JobCancelled()
            phase = math.sin(frame_index / max(frame_count - 1, 1) * math.pi * 2)
            yield BodyMotionFrame(
                frame_index=frame_index,
                timestamp_ms=round(frame_index * 1000 / fps),
                landmarks=_mock_landmarks(phase),
            )
            progress(frame_index + 1, frame_count)


def create_body_motion_analyzer(kind: str, model_path: Path | None) -> BodyMotionAnalyzer:
    if kind == "mock":
        return MockBodyMotionAnalyzer()
    if kind == "mediapipe":
        return MediaPipeBodyMotionAnalyzer(model_path)
    raise ServiceError(
        "INVALID_CONFIGURATION",
        f"Unknown body-motion analyzer: {kind}",
        status_code=500,
    )


def build_body_motion_record(
    frame: BodyMotionFrame,
    reference_line: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    confidence: dict[str, float] = {}
    landmarks = frame.landmarks
    for side in ("left", "right"):
        definitions = {
            "elbow": (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"),
            "shoulder": (f"{side}_elbow", f"{side}_shoulder", f"{side}_hip"),
            "hip": (f"{side}_shoulder", f"{side}_hip", f"{side}_knee"),
            "knee": (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
        }
        for name, points in definitions.items():
            value, score = _joint_angle(landmarks, *points)
            if value is not None:
                metrics[f"{side}_{name}_deg"] = round(value, 2)
                confidence[f"{side}_{name}"] = round(score, 3)

    primary_side = _primary_side(confidence)
    lean, lean_confidence = _signed_lean(landmarks, reference_line)
    if lean is not None:
        metrics["lean_deg"] = round(lean, 2)
        confidence["lean"] = round(lean_confidence, 3)
    return {
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "athlete_id": "primary",
        "primary_side": primary_side,
        "landmarks": landmarks,
        "metrics": metrics,
        "confidence": confidence,
    }


def smooth_body_motion_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[str, tuple[int, float]] = {}
    for record in records:
        timestamp_ms = int(record["timestamp_ms"])
        metrics = record.get("metrics", {})
        for name, raw_value in list(metrics.items()):
            value = float(raw_value)
            prior = previous.get(name)
            if prior is not None and timestamp_ms - prior[0] <= SMOOTHING_MAX_GAP_MS:
                value = prior[1] + SMOOTHING_ALPHA * (value - prior[1])
                metrics[name] = round(value, 2)
            previous[name] = (timestamp_ms, value)
    return records


def body_motion_chunk_payload(
    records: list[dict[str, Any]], sequence: int, chunk_seconds: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "start_ms": sequence * chunk_seconds * 1000,
        "end_ms": (sequence + 1) * chunk_seconds * 1000,
        "frames": records,
    }


def load_body_motion_frames_by_index(
    chunk_paths: list[Path],
) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("frames", []):
            try:
                frames[int(record["frame_index"])] = record
            except (KeyError, TypeError, ValueError):
                continue
    return frames


def _selected_landmarks(pose: Any) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for name, index in BODY_LANDMARK_INDICES.items():
        landmark = pose[index]
        visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
        presence = float(getattr(landmark, "presence", visibility) or 0.0)
        selected[name] = {
            "x": round(float(landmark.x), 6),
            "y": round(float(landmark.y), 6),
            "z": round(float(landmark.z), 6),
            "visibility": round(visibility, 4),
            "presence": round(presence, 4),
        }
    return selected


def _landmark_confidence(landmark: dict[str, float]) -> float:
    return min(float(landmark.get("visibility", 0)), float(landmark.get("presence", 0)))


def _joint_angle(
    landmarks: dict[str, dict[str, float]],
    first_name: str,
    vertex_name: str,
    third_name: str,
) -> tuple[float | None, float]:
    points = [landmarks.get(name) for name in (first_name, vertex_name, third_name)]
    if any(point is None for point in points):
        return None, 0.0
    first, vertex, third = points
    assert first is not None and vertex is not None and third is not None
    confidence = min(_landmark_confidence(point) for point in (first, vertex, third))
    if confidence < MIN_LANDMARK_CONFIDENCE:
        return None, confidence
    first_vector = (first["x"] - vertex["x"], first["y"] - vertex["y"])
    third_vector = (third["x"] - vertex["x"], third["y"] - vertex["y"])
    first_length = math.hypot(*first_vector)
    third_length = math.hypot(*third_vector)
    if first_length < 1e-6 or third_length < 1e-6:
        return None, confidence
    cosine = (
        first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1]
    ) / (first_length * third_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), confidence


def _primary_side(confidence: dict[str, float]) -> str | None:
    totals = {
        side: sum(
            confidence.get(f"{side}_{joint}", 0.0)
            for joint in ("elbow", "shoulder", "hip", "knee")
        )
        for side in ("left", "right")
    }
    if max(totals.values(), default=0.0) <= 0:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def _signed_lean(
    landmarks: dict[str, dict[str, float]],
    reference_line: tuple[float, float, float, float] | None,
) -> tuple[float | None, float]:
    shoulders = _confident_pair(landmarks, "left_shoulder", "right_shoulder")
    hips = _confident_pair(landmarks, "left_hip", "right_hip")
    if shoulders is None or hips is None or reference_line is None:
        return None, 0.0
    shoulder, shoulder_confidence = shoulders
    hip, hip_confidence = hips
    trunk = (shoulder[0] - hip[0], shoulder[1] - hip[1])
    trunk_length = math.hypot(*trunk)
    if trunk_length < 1e-6:
        return None, 0.0
    vertical_angle = math.degrees(math.atan2(abs(trunk[0]), max(-trunk[1], 1e-9)))
    x1, y1, x2, y2 = reference_line
    if (x2, y2) < (x1, y1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    reference = (x2 - x1, y2 - y1)
    reference_length = math.hypot(*reference)
    if reference_length < 1e-6:
        return None, 0.0
    projection = trunk[0] * reference[0] + trunk[1] * reference[1]
    sign = 1.0 if projection >= 0 else -1.0
    return sign * min(vertical_angle, 180.0), min(shoulder_confidence, hip_confidence)


def _confident_pair(
    landmarks: dict[str, dict[str, float]], first_name: str, second_name: str
) -> tuple[tuple[float, float], float] | None:
    available = [
        landmark
        for name in (first_name, second_name)
        if (landmark := landmarks.get(name)) is not None
        and _landmark_confidence(landmark) >= MIN_LANDMARK_CONFIDENCE
    ]
    if not available:
        return None
    return (
        (
            sum(point["x"] for point in available) / len(available),
            sum(point["y"] for point in available) / len(available),
        ),
        min(_landmark_confidence(point) for point in available),
    )


def _mock_landmarks(phase: float) -> dict[str, dict[str, float]]:
    points = {
        "left_shoulder": (0.47 + phase * 0.03, 0.30),
        "right_shoulder": (0.51 + phase * 0.03, 0.30),
        "left_elbow": (0.42 + phase * 0.04, 0.43),
        "right_elbow": (0.56 + phase * 0.04, 0.43),
        "left_wrist": (0.48 + phase * 0.05, 0.53),
        "right_wrist": (0.62 + phase * 0.05, 0.53),
        "left_hip": (0.48, 0.55),
        "right_hip": (0.52, 0.55),
        "left_knee": (0.42, 0.70),
        "right_knee": (0.58, 0.70),
        "left_ankle": (0.38, 0.86),
        "right_ankle": (0.62, 0.86),
    }
    return {
        name: {
            "x": x,
            "y": y,
            "z": 0.0,
            "visibility": 0.99,
            "presence": 0.99,
        }
        for name, (x, y) in points.items()
    }
