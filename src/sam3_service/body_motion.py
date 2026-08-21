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
SAPIENS2_BODY_LANDMARK_NAMES = tuple(BODY_LANDMARK_INDICES)
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


class Sapiens2BodyMotionAnalyzer:
    def __init__(
        self,
        checkpoint_path: Path | None,
        detector_path: Path | None,
        config_path: Path | None = None,
        *,
        device: str = "cuda:0",
        keypoint_threshold: float = MIN_LANDMARK_CONFIDENCE,
        bbox_threshold: float = 0.3,
        nms_threshold: float = 0.3,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.detector_path = detector_path
        self.config_path = config_path
        self.device = device
        self.keypoint_threshold = keypoint_threshold
        self.bbox_threshold = bbox_threshold
        self.nms_threshold = nms_threshold
        self._runtime: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        model = self.checkpoint_path.name if self.checkpoint_path else "unconfigured"
        return f"sapiens2-pose-1b:{model}"

    def analyze(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        progress: Callable[[int, int], None],
        cancelled: Callable[[], bool],
    ) -> Iterable[BodyMotionFrame]:
        self._validate_assets()
        runtime = self._ensure_runtime()
        cv2 = runtime["cv2"]
        fps = max(float(metadata.get("fps") or 30), 1.0)
        expected_frames = max(int(metadata.get("frame_count") or 0), 1)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ServiceError(
                "POSE_INFERENCE_FAILED",
                "The normalized video could not be opened for body-motion analysis.",
                retryable=True,
            )

        frame_index = 0
        previous_bbox: Any | None = None
        try:
            while True:
                if cancelled():
                    raise JobCancelled()
                available, bgr_frame = capture.read()
                if not available:
                    break
                boxes = self._detect_people(bgr_frame, runtime)
                bbox = _select_primary_bbox(boxes, previous_bbox, bgr_frame.shape[:2])
                landmarks: dict[str, dict[str, float]] = {}
                if bbox is not None:
                    previous_bbox = bbox
                    keypoints, scores = self._infer_pose(bgr_frame, bbox, runtime)
                    landmarks = _sapiens2_selected_landmarks(
                        keypoints,
                        scores,
                        runtime["keypoint_name_to_id"],
                        bgr_frame.shape[1],
                        bgr_frame.shape[0],
                        self.keypoint_threshold,
                    )
                yield BodyMotionFrame(
                    frame_index=frame_index,
                    timestamp_ms=round(frame_index * 1000 / fps),
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
                f"Sapiens2 pose estimation failed: {exc}",
                retryable=True,
            ) from exc
        finally:
            capture.release()

    def _validate_assets(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise ServiceError(
                "POSE_MODEL_UNAVAILABLE",
                "Sapiens2 body motion requires SAM3_SAPIENS2_CHECKPOINT_PATH.",
                retryable=False,
            )
        if self.detector_path is None or not self.detector_path.is_dir():
            raise ServiceError(
                "POSE_MODEL_UNAVAILABLE",
                "Sapiens2 body motion requires SAM3_SAPIENS2_DETECTOR_PATH.",
                retryable=False,
            )
        if self.config_path is not None and not self.config_path.is_file():
            raise ServiceError(
                "POSE_MODEL_UNAVAILABLE",
                "SAM3_SAPIENS2_CONFIG_PATH does not point to a file.",
                retryable=False,
            )

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import cv2
            import numpy as np
            import torch
            from PIL import Image
            from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
            from sapiens.pose.models import init_model
            from transformers import DetrForObjectDetection, DetrImageProcessor
            import sapiens
        except ImportError as exc:
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                "Sapiens2 body motion requires the optional sapiens2 dependencies.",
                retryable=False,
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                f"Sapiens2 device {self.device} is unavailable.",
                retryable=False,
            )
        config_path = self.config_path or (
            Path(sapiens.__file__).resolve().parent
            / "pose"
            / "configs"
            / "keypoints308"
            / "shutterstock_goliath_3po"
            / "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"
        )
        if not config_path.is_file():
            raise ServiceError(
                "POSE_MODEL_UNAVAILABLE",
                "The Sapiens2 1B pose config is unavailable; set SAM3_SAPIENS2_CONFIG_PATH.",
                retryable=False,
            )
        try:
            model = init_model(config_path, self.checkpoint_path, device=self.device)
            metadata_path = (
                Path(sapiens.__file__).resolve().parent
                / "pose"
                / "configs"
                / "_base_"
                / "keypoints308.py"
            )
            pose_metainfo = parse_pose_metainfo(dict(from_file=str(metadata_path)))
            model.pose_metainfo = pose_metainfo
            codec_config = dict(model.cfg.codec)
            codec_type = codec_config.pop("type")
            if codec_type != "UDPHeatmap":
                raise ValueError(f"Unsupported Sapiens2 pose codec: {codec_type}")
            model.codec = UDPHeatmap(**codec_config)
            detector_processor = DetrImageProcessor.from_pretrained(
                self.detector_path, local_files_only=True
            )
            detector = (
                DetrForObjectDetection.from_pretrained(
                    self.detector_path, local_files_only=True
                )
                .eval()
                .to(self.device)
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                f"Sapiens2 models could not be initialized: {exc}",
                retryable=False,
            ) from exc
        self._runtime = {
            "cv2": cv2,
            "np": np,
            "torch": torch,
            "Image": Image,
            "model": model,
            "detector_processor": detector_processor,
            "detector": detector,
            "keypoint_name_to_id": pose_metainfo["keypoint_name2id"],
        }
        return self._runtime

    def _detect_people(self, bgr_frame: Any, runtime: dict[str, Any]) -> Any:
        cv2 = runtime["cv2"]
        torch = runtime["torch"]
        image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = runtime["Image"].fromarray(image_rgb)
        processor = runtime["detector_processor"]
        detector = runtime["detector"]
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = detector(**inputs)
        target_sizes = torch.tensor([image_rgb.shape[:2]], device=self.device)
        result = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.bbox_threshold
        )[0]
        person_mask = result["labels"] == 1
        boxes = result["boxes"][person_mask].detach().cpu().numpy()
        scores = result["scores"][person_mask].detach().cpu().numpy()
        return _nms_boxes(boxes, scores, self.nms_threshold)

    def _infer_pose(
        self, bgr_frame: Any, bbox: Any, runtime: dict[str, Any]
    ) -> tuple[Any, Any]:
        np = runtime["np"]
        torch = runtime["torch"]
        model = runtime["model"]
        data = model.pipeline(
            {
                "img": bgr_frame,
                "bbox": np.asarray(bbox, dtype=np.float32)[None],
                "bbox_score": np.ones(1, dtype=np.float32),
            }
        )
        data = model.data_preprocessor(data)
        inputs = data["inputs"]
        with torch.no_grad():
            prediction = model(inputs)
            if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
                flipped = model(inputs.flip(-1)).flip(-1)
                flipped = flipped[:, model.pose_metainfo["flip_indices"]]
                prediction = (prediction + flipped) / 2.0
        decoded, scores = model.codec.decode(prediction[0].detach().cpu().numpy())
        sample = data["data_samples"]
        input_size = sample["meta"]["input_size"]
        bbox_center = sample["meta"]["bbox_center"]
        bbox_scale = sample["meta"]["bbox_scale"]
        decoded = decoded / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        return decoded[0], scores[0]


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


def create_body_motion_analyzer(
    kind: str,
    model_path: Path | None,
    *,
    sapiens2_checkpoint_path: Path | None = None,
    sapiens2_detector_path: Path | None = None,
    sapiens2_config_path: Path | None = None,
    sapiens2_device: str = "cuda:0",
    sapiens2_keypoint_threshold: float = MIN_LANDMARK_CONFIDENCE,
    sapiens2_bbox_threshold: float = 0.3,
    sapiens2_nms_threshold: float = 0.3,
) -> BodyMotionAnalyzer:
    if kind == "mock":
        return MockBodyMotionAnalyzer()
    if kind == "mediapipe":
        return MediaPipeBodyMotionAnalyzer(model_path)
    if kind == "sapiens2":
        return Sapiens2BodyMotionAnalyzer(
            sapiens2_checkpoint_path,
            sapiens2_detector_path,
            sapiens2_config_path,
            device=sapiens2_device,
            keypoint_threshold=sapiens2_keypoint_threshold,
            bbox_threshold=sapiens2_bbox_threshold,
            nms_threshold=sapiens2_nms_threshold,
        )
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


def _sapiens2_selected_landmarks(
    keypoints: Any,
    scores: Any,
    keypoint_name_to_id: dict[str, int],
    width: int,
    height: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for name in SAPIENS2_BODY_LANDMARK_NAMES:
        index = keypoint_name_to_id.get(name)
        if index is None or index >= len(keypoints) or index >= len(scores):
            continue
        confidence = float(scores[index])
        if confidence < threshold:
            continue
        selected[name] = {
            "x": round(float(keypoints[index][0]) / max(width, 1), 6),
            "y": round(float(keypoints[index][1]) / max(height, 1), 6),
            "z": 0.0,
            "visibility": round(confidence, 4),
            "presence": round(confidence, 4),
        }
    return selected


def _select_primary_bbox(
    boxes: Any, previous_bbox: Any | None, frame_shape: tuple[int, int]
) -> Any | None:
    if len(boxes) == 0:
        return None
    height, width = frame_shape
    frame_center = (width / 2.0, height / 2.0)

    def rank(box: Any) -> tuple[float, float, float]:
        area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
        center_distance = math.hypot(center_x - frame_center[0], center_y - frame_center[1])
        overlap = _bbox_iou(box, previous_bbox) if previous_bbox is not None else 0.0
        return overlap, area, -center_distance

    return max(boxes, key=rank)


def _bbox_iou(first: Any, second: Any) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    first_area = max(float(first[2] - first[0]), 0.0) * max(
        float(first[3] - first[1]), 0.0
    )
    second_area = max(float(second[2] - second[0]), 0.0) * max(
        float(second[3] - second[1]), 0.0
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _nms_boxes(boxes: Any, scores: Any, threshold: float) -> Any:
    if len(boxes) == 0:
        return boxes
    order = sorted(range(len(boxes)), key=lambda index: float(scores[index]), reverse=True)
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [index for index in order if _bbox_iou(boxes[current], boxes[index]) <= threshold]
    return boxes[kept]


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
