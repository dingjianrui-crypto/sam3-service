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

from .config import Settings
from .db import Database, utc_now
from .errors import JobCancelled, ServiceError
from .media import normalize_video, probe_video
from .segmenter import FrameResult, VideoSegmenter, create_segmenter
from .storage import LocalStorage

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        storage: LocalStorage,
        segmenter: VideoSegmenter,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.segmenter = segmenter
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
                    combined_total = max(1, len(prompts) * total)
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
                    float(settings.get("score_threshold", 0.5)),
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
            manifest = self._build_manifest(job_id, video["id"], metadata, prompt_entries, instances)
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
    ) -> dict[str, Any]:
        colors = {prompt["id"]: prompt["color"] for prompt in prompts}
        chunks = self.database.fetch_all(
            "SELECT sequence, start_ms, end_ms, size_bytes FROM result_chunks "
            "WHERE job_id = ? ORDER BY sequence",
            (job_id,),
        )
        return {
            "schema_version": 1,
            "job_id": job_id,
            "video": {
                "url": f"/api/v1/videos/{video_id}/content?normalized=true",
                **metadata,
            },
            "prompts": prompts,
            "instances": [
                {**entry, "color": colors.get(entry["prompt_id"], "#35C2FF")}
                for entry in instances.values()
            ],
            "chunks": [
                {
                    **chunk,
                    "url": f"/api/v1/jobs/{job_id}/results/chunks/{chunk['sequence']}",
                }
                for chunk in chunks
            ],
        }

    def _state(self, job_id: str, state: str) -> None:
        self.database.execute(
            "UPDATE jobs SET state = ?, progress_stage = ?, worker_heartbeat_at = ? WHERE id = ?",
            (state, state, utc_now(), job_id),
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
    }


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
    worker = Worker(settings, database, storage, segmenter)
    logger.info("worker %s started with %s", worker.worker_id, segmenter.model_name)
    while True:
        if not worker.process_next():
            time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    run()
