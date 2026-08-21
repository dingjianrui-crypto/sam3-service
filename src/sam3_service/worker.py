from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .body_motion import (
    BodyMotionAnalyzer,
    body_motion_chunk_payload,
    build_body_motion_record,
    create_body_motion_analyzer,
    smooth_body_motion_records,
)
from .config import Settings
from .db import Database, utc_now
from .errors import JobCancelled, ServiceError
from .media import normalize_video, probe_video
from .segmenter import FrameResult, VideoSegmenter, create_segmenter
from .storage import LocalStorage
from .tracking import assign_stable_tracks_to_chunks

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        storage: LocalStorage,
        segmenter: VideoSegmenter,
        body_motion_analyzer: BodyMotionAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.segmenter = segmenter
        self.body_motion_analyzer = body_motion_analyzer
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def process_next(self) -> bool:
        job = self.database.claim_job(self.worker_id)
        if not job:
            return False
        self.process(job)
        return True

    def process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        try:
            video = self.database.fetch_one("SELECT * FROM videos WHERE id = ?", (job["video_id"],))
            if not video or not video["source_path"]:
                raise ServiceError("INVALID_VIDEO", "The source video is unavailable.")
            source = Path(video["source_path"])
            normalized = self.storage.normalized_path(video["id"])
            self._state(job_id, "preprocessing")
            if not normalized.exists():
                metadata = normalize_video(source, normalized, video)
                self.database.execute(
                    """
                    UPDATE videos SET normalized_path = ?, width = ?, height = ?, fps = ?,
                        duration_ms = ?, frame_count = ?, codec = ? WHERE id = ?
                    """,
                    (
                        str(normalized),
                        metadata["width"],
                        metadata["height"],
                        metadata["fps"],
                        metadata["duration_ms"],
                        metadata["frame_count"],
                        metadata["codec"],
                        video["id"],
                    ),
                )
            metadata = probe_video(normalized)
            self.database.execute(
                "UPDATE jobs SET total_frames = ?, model_name = ? WHERE id = ?",
                (metadata["frame_count"], self.segmenter.model_name, job_id),
            )
            prompts = self.database.fetch_all(
                "SELECT * FROM job_prompts WHERE job_id = ? ORDER BY position", (job_id,)
            )
            settings = json.loads(job["settings_json"])
            body_motion_requested = bool(settings.get("body_motion", False))
            paddling_discipline = str(settings.get("paddling_discipline", "kayak"))
            total_passes = len(prompts) + (1 if body_motion_requested else 0)
            prompt_entries = []
            instances: dict[str, dict[str, Any]] = {}
            for prompt_index, prompt in enumerate(prompts):
                self._check_cancelled(job_id)
                self.database.execute(
                    "UPDATE job_prompts SET state = 'segmenting' WHERE id = ?", (prompt["id"],)
                )
                self._state(job_id, "segmenting")

                def progress(done: int, total: int) -> None:
                    overall = prompt_index * total + done
                    combined_total = max(1, total_passes * total)
                    self.database.execute(
                        """
                        UPDATE jobs SET processed_frames = ?, total_frames = ?,
                            worker_heartbeat_at = ? WHERE id = ?
                        """,
                        (overall, combined_total, utc_now(), job_id),
                    )

                frames = self.segmenter.segment(
                    normalized,
                    metadata,
                    prompt["id"],
                    prompt["text"],
                    float(settings.get("score_threshold", 0.3)),
                    settings,
                    progress,
                    lambda: self._cancel_requested(job_id),
                )
                self._merge_prompt_results(job_id, frames, instances)
                self.database.execute(
                    "UPDATE job_prompts SET state = 'completed' WHERE id = ?", (prompt["id"],)
                )
                prompt_entries.append(
                    {"id": prompt["id"], "text": prompt["text"], "color": prompt["color"]}
                )
            self._state(job_id, "postprocessing")
            tracks = self._assign_stable_tracks(job_id)
            body_motion_manifest: dict[str, Any] | None = None
            warnings: list[dict[str, str]] = []
            if body_motion_requested:
                self._progress_stage(job_id, "body_motion")
                try:
                    body_motion_manifest = self._analyze_body_motion(
                        job_id,
                        normalized,
                        metadata,
                        prompt_entries,
                        settings,
                        total_passes,
                    )
                except JobCancelled:
                    raise
                except Exception as exc:
                    logger.exception("body-motion analysis failed for job %s", job_id)
                    error = exc if isinstance(exc, ServiceError) else ServiceError(
                        "POSE_INFERENCE_FAILED", str(exc), retryable=True
                    )
                    body_motion_manifest = {
                        "schema_version": 1,
                        "status": "failed",
                        "discipline": paddling_discipline,
                        "error": {
                            "code": error.code,
                            "message": error.message,
                            "retryable": error.retryable,
                        },
                        "chunks": [],
                    }
                    warnings.append(
                        {"code": error.code, "message": error.message}
                    )
                self._progress_stage(job_id, "postprocessing")
            manifest = self._build_manifest(
                job_id,
                video["id"],
                metadata,
                prompt_entries,
                instances,
                tracks,
                settings,
                body_motion_manifest,
                warnings,
            )
            self._atomic_json(self.storage.manifest_path(job_id), manifest)
            self.database.execute(
                """
                UPDATE jobs SET state = 'completed', progress_stage = 'completed',
                    processed_frames = total_frames, completed_at = ?, worker_heartbeat_at = ?
                WHERE id = ?
                """,
                (utc_now(), utc_now(), job_id),
            )
        except JobCancelled:
            self.storage.delete_job(job_id)
            self.database.execute(
                "UPDATE jobs SET state = 'cancelled', progress_stage = 'cancelled', "
                "completed_at = ? WHERE id = ?",
                (utc_now(), job_id),
            )
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            error = exc if isinstance(exc, ServiceError) else ServiceError(
                "INFERENCE_FAILED", str(exc), retryable=True
            )
            self.database.execute(
                """
                UPDATE jobs SET state = 'failed', progress_stage = 'failed',
                    error_code = ?, error_detail = ?, completed_at = ?
                WHERE id = ?
                """,
                (error.code, error.message, utc_now(), job_id),
            )

    def _merge_prompt_results(
        self,
        job_id: str,
        frames: Any,
        instances: dict[str, dict[str, Any]],
    ) -> None:
        current_sequence: int | None = None
        records: list[dict[str, Any]] = []
        for frame in frames:
            sequence = frame.timestamp_ms // (self.settings.result_chunk_seconds * 1000)
            if current_sequence is not None and sequence != current_sequence:
                self._append_chunk(job_id, current_sequence, records)
                records = []
            current_sequence = sequence
            instances.setdefault(
                frame.instance_id,
                {"id": frame.instance_id, "prompt_id": frame.prompt_id},
            )
            records.append(_frame_to_dict(frame))
        if current_sequence is not None:
            self._append_chunk(job_id, current_sequence, records)

    def _append_chunk(
        self, job_id: str, sequence: int, new_records: list[dict[str, Any]]
    ) -> None:
        path = self.storage.chunk_path(job_id, sequence)
        records: list[dict[str, Any]] = []
        if path.exists():
            records = json.loads(path.read_text())["frames"]
        records.extend(new_records)
        records.sort(key=lambda item: (item["timestamp_ms"], item["prompt_id"], item["instance_id"]))
        payload = {
            "schema_version": 1,
            "start_ms": sequence * self.settings.result_chunk_seconds * 1000,
            "end_ms": (sequence + 1) * self.settings.result_chunk_seconds * 1000,
            "frames": records,
        }
        self._atomic_json(path, payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        chunk_id = f"{job_id}:{sequence}"
        self.database.execute(
            """
            INSERT INTO result_chunks
                (id, job_id, sequence, start_ms, end_ms, path, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, sequence) DO UPDATE SET
                path = excluded.path, size_bytes = excluded.size_bytes, sha256 = excluded.sha256
            """,
            (
                chunk_id,
                job_id,
                sequence,
                payload["start_ms"],
                payload["end_ms"],
                str(path),
                path.stat().st_size,
                digest,
                utc_now(),
            ),
        )

    def _build_manifest(
        self,
        job_id: str,
        video_id: str,
        metadata: dict[str, Any],
        prompts: list[dict[str, Any]],
        instances: dict[str, dict[str, Any]],
        tracks: list[dict[str, Any]],
        settings: dict[str, Any],
        body_motion: dict[str, Any] | None = None,
        warnings: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        colors = {prompt["id"]: prompt["color"] for prompt in prompts}
        chunks = self.database.fetch_all(
            "SELECT sequence, start_ms, end_ms, size_bytes FROM result_chunks "
            "WHERE job_id = ? ORDER BY sequence",
            (job_id,),
        )
        manifest = {
            "schema_version": 2,
            "job_id": job_id,
            "video": {
                "url": f"/api/v1/videos/{video_id}/content?normalized=true",
                **metadata,
            },
            "prompts": prompts,
            "settings": {
                "boat_reference_line": settings.get("boat_reference_line", "centerline"),
                "body_motion": bool(settings.get("body_motion", False)),
                "paddling_discipline": settings.get("paddling_discipline", "kayak"),
            },
            "instances": [
                {**entry, "color": colors.get(entry["prompt_id"], "#35C2FF")}
                for entry in instances.values()
            ],
            "tracks": [
                {**entry, "color": colors.get(entry["prompt_id"], "#35C2FF")}
                for entry in tracks
            ],
            "chunks": [
                {
                    **chunk,
                    "url": f"/api/v1/jobs/{job_id}/results/chunks/{chunk['sequence']}",
                }
                for chunk in chunks
            ],
        }
        if body_motion is not None:
            manifest["body_motion"] = body_motion
        if warnings:
            manifest["warnings"] = warnings
        return manifest

    def _analyze_body_motion(
        self,
        job_id: str,
        video_path: Path,
        metadata: dict[str, Any],
        prompts: list[dict[str, Any]],
        settings: dict[str, Any],
        total_passes: int,
    ) -> dict[str, Any]:
        if self.body_motion_analyzer is None:
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                "The worker has no body-motion analyzer configured.",
                retryable=False,
            )
        reference_mode = str(settings.get("boat_reference_line", "centerline"))
        reference_lines = self._body_reference_lines(
            job_id, prompts, reference_mode, metadata
        )
        prompt_passes = max(0, total_passes - 1)

        def progress(done: int, total: int) -> None:
            self.database.execute(
                "UPDATE jobs SET processed_frames = ?, total_frames = ?, "
                "worker_heartbeat_at = ? WHERE id = ?",
                (
                    prompt_passes * total + done,
                    max(1, total_passes * total),
                    utc_now(),
                    job_id,
                ),
            )

        records = [
            build_body_motion_record(frame, reference_lines.get(frame.frame_index))
            for frame in self.body_motion_analyzer.analyze(
                video_path,
                metadata,
                progress,
                lambda: self._cancel_requested(job_id),
            )
        ]
        if not records:
            raise ServiceError(
                "POSE_INFERENCE_FAILED",
                "Body-motion analysis decoded no video frames.",
                retryable=True,
            )
        if not any(record.get("landmarks") for record in records):
            raise ServiceError(
                "POSE_NOT_DETECTED",
                "No sufficiently visible primary athlete was detected.",
                retryable=False,
            )
        smooth_body_motion_records(records)
        by_sequence: dict[int, list[dict[str, Any]]] = {}
        chunk_duration_ms = self.settings.result_chunk_seconds * 1000
        for record in records:
            sequence = int(record["timestamp_ms"]) // chunk_duration_ms
            by_sequence.setdefault(sequence, []).append(record)

        descriptors: list[dict[str, Any]] = []
        for sequence, chunk_records in sorted(by_sequence.items()):
            path = self.storage.body_motion_chunk_path(job_id, sequence)
            payload = body_motion_chunk_payload(
                chunk_records, sequence, self.settings.result_chunk_seconds
            )
            self._atomic_json(path, payload)
            descriptors.append(
                {
                    "sequence": sequence,
                    "start_ms": payload["start_ms"],
                    "end_ms": payload["end_ms"],
                    "size_bytes": path.stat().st_size,
                    "url": (
                        f"/api/v1/jobs/{job_id}/results/body-motion/chunks/{sequence}"
                    ),
                }
            )
        return {
            "schema_version": 1,
            "status": "completed",
            "discipline": settings.get("paddling_discipline", "kayak"),
            "model_name": self.body_motion_analyzer.model_name,
            "reference_axis": "video_vertical",
            "direction_reference": reference_mode,
            "athlete_count": 1,
            "chunks": descriptors,
        }

    def _body_reference_lines(
        self,
        job_id: str,
        prompts: list[dict[str, Any]],
        reference_mode: str,
        metadata: dict[str, Any],
    ) -> dict[int, tuple[float, float, float, float]]:
        reference_prompt_ids = {
            str(prompt["id"])
            for prompt in prompts
            if _is_boat_prompt(str(prompt.get("text", "")))
        }
        if not reference_prompt_ids:
            return {}
        rows = self.database.fetch_all(
            "SELECT path FROM result_chunks WHERE job_id = ? ORDER BY sequence",
            (job_id,),
        )
        candidates: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            for record in payload.get("frames", []):
                if str(record.get("prompt_id")) not in reference_prompt_ids:
                    continue
                try:
                    frame_index = int(record["frame_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                candidates.setdefault(frame_index, []).append(record)
        lines: dict[int, tuple[float, float, float, float]] = {}
        width = max(float(metadata.get("width") or 1), 1.0)
        height = max(float(metadata.get("height") or 1), 1.0)
        for frame_index, records in candidates.items():
            record = max(records, key=_record_area)
            line = _normalized_reference_line(record, reference_mode, width, height)
            if line is not None:
                lines[frame_index] = line
        return lines

    def _assign_stable_tracks(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            "SELECT sequence, path FROM result_chunks WHERE job_id = ? ORDER BY sequence",
            (job_id,),
        )
        paths = [Path(row["path"]) for row in rows if Path(row["path"]).is_file()]
        tracks = assign_stable_tracks_to_chunks(paths)
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                continue
            self.database.execute(
                "UPDATE result_chunks SET size_bytes = ?, sha256 = ? "
                "WHERE job_id = ? AND sequence = ?",
                (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    job_id,
                    row["sequence"],
                ),
            )
        return tracks

    def _state(self, job_id: str, state: str) -> None:
        self.database.execute(
            "UPDATE jobs SET state = ?, progress_stage = ?, worker_heartbeat_at = ? WHERE id = ?",
            (state, state, utc_now(), job_id),
        )

    def _progress_stage(self, job_id: str, stage: str) -> None:
        self.database.execute(
            "UPDATE jobs SET progress_stage = ?, worker_heartbeat_at = ? WHERE id = ?",
            (stage, utc_now(), job_id),
        )

    def _cancel_requested(self, job_id: str) -> bool:
        row = self.database.fetch_one(
            "SELECT cancel_requested_at FROM jobs WHERE id = ?", (job_id,)
        )
        return bool(row and row["cancel_requested_at"])

    def _check_cancelled(self, job_id: str) -> None:
        if self._cancel_requested(job_id):
            raise JobCancelled()

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(path)


def _frame_to_dict(frame: FrameResult) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "prompt_id": frame.prompt_id,
        "instance_id": frame.instance_id,
        "box_xywh": frame.box_xywh,
        "score": frame.score,
        "segmentation": frame.segmentation,
        "centerline_segmentation": frame.centerline_segmentation,
        "centerline_box_xywh": frame.centerline_box_xywh,
        "centerline_line_xyxy": frame.centerline_line_xyxy,
        "waterline_segmentation": frame.waterline_segmentation,
        "waterline_box_xywh": frame.waterline_box_xywh,
        "waterline_line_xyxy": frame.waterline_line_xyxy,
        "waterline_confidence": frame.waterline_confidence,
    }


def _is_boat_prompt(value: str) -> bool:
    words = set(value.lower().replace("-", " ").split())
    return bool(words.intersection({"boat", "kayak", "canoe", "shell"}))


def _record_area(record: dict[str, Any]) -> float:
    box = record.get("box_xywh") or (0, 0, 0, 0)
    try:
        return max(0.0, float(box[2])) * max(0.0, float(box[3]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def _normalized_reference_line(
    record: dict[str, Any],
    reference_mode: str,
    video_width: float,
    video_height: float,
) -> tuple[float, float, float, float] | None:
    use_waterline = reference_mode == "waterline"
    values = (
        record.get("waterline_line_xyxy") if use_waterline else None
    ) or record.get("centerline_line_xyxy")
    if not values or len(values) != 4:
        return None
    geometry = "waterline" if use_waterline and record.get("waterline_line_xyxy") else "centerline"
    segmentation = record.get(f"{geometry}_segmentation")
    width = video_width
    height = video_height
    if isinstance(segmentation, dict) and segmentation.get("type") == "rle":
        size = segmentation.get("size")
        if isinstance(size, list) and len(size) == 2:
            try:
                height = max(float(size[0]), 1.0)
                width = max(float(size[1]), 1.0)
            except (TypeError, ValueError):
                width = video_width
                height = video_height
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def run() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    settings = Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    storage = LocalStorage(settings.data_dir)
    segmenter = create_segmenter(
        settings.segmenter,
        settings.checkpoint_path,
        offline=settings.offline,
    )
    body_motion_analyzer = create_body_motion_analyzer(
        settings.body_motion_analyzer,
        settings.pose_model_path,
        sapiens2_checkpoint_path=settings.sapiens2_checkpoint_path,
        sapiens2_detector_path=settings.sapiens2_detector_path,
        sapiens2_config_path=settings.sapiens2_config_path,
        sapiens2_device=settings.sapiens2_device,
        sapiens2_keypoint_threshold=settings.sapiens2_keypoint_threshold,
        sapiens2_bbox_threshold=settings.sapiens2_bbox_threshold,
        sapiens2_nms_threshold=settings.sapiens2_nms_threshold,
    )
    worker = Worker(settings, database, storage, segmenter, body_motion_analyzer)
    logger.info("worker %s started with %s", worker.worker_id, segmenter.model_name)
    while True:
        if not worker.process_next():
            time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    run()
