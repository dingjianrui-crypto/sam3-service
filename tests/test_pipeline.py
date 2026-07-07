from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from sam3_service.config import Settings
from sam3_service.db import Database, expires_at, utc_now
from sam3_service.segmenter import MockSegmenter
from sam3_service.storage import LocalStorage
from sam3_service.worker import Worker


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(
            data_dir=self.root,
            database_path=self.root / "sam3.sqlite3",
            static_dir=self.root / "static",
            segmenter="mock",
            checkpoint_path=None,
            offline=False,
            poll_interval_seconds=0.01,
            max_upload_bytes=20 * 1024 * 1024,
            max_video_duration_seconds=10,
            max_prompts=3,
            chunk_size_bytes=1024 * 1024,
            result_chunk_seconds=1,
        )
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()
        self.storage = LocalStorage(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_claim_and_process_mock_job(self) -> None:
        video_id = uuid.uuid4().hex
        source = self.storage.source_path(video_id)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=10:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "mp4",
                str(source),
            ],
            check=True,
        )
        now = utc_now()
        self.database.execute(
            """
            INSERT INTO videos
                (id, original_filename, state, mime_type, expected_size_bytes,
                 size_bytes, source_path, width, height, fps, duration_ms,
                 frame_count, codec, created_at, validated_at, expires_at)
            VALUES (?, 'fixture.mp4', 'ready', 'video/mp4', ?, ?, ?, 320, 240,
                    10, 1000, 10, 'h264', ?, ?, ?)
            """,
            (video_id, source.stat().st_size, source.stat().st_size, str(source), now, now, expires_at()),
        )
        job_id = uuid.uuid4().hex
        prompt_id = uuid.uuid4().hex
        self.database.execute(
            """
            INSERT INTO jobs
                (id, video_id, state, progress_stage, settings_json, created_at, expires_at)
            VALUES (?, ?, 'queued', 'queued', ?, ?, ?)
            """,
            (
                job_id,
                video_id,
                json.dumps({"score_threshold": 0.5}),
                now,
                expires_at(),
            ),
        )
        self.database.execute(
            """
            INSERT INTO job_prompts (id, job_id, position, text, color)
            VALUES (?, ?, 0, 'paddle', '#35C2FF')
            """,
            (prompt_id, job_id),
        )

        worker = Worker(
            self.settings,
            self.database,
            self.storage,
            MockSegmenter(),
        )
        self.assertTrue(worker.process_next())
        job = self.database.job_detail(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["state"], "completed")
        self.assertEqual(job["model_name"], "mock-moving-paddle-v1")

        manifest = json.loads(self.storage.manifest_path(job_id).read_text())
        self.assertEqual(manifest["job_id"], job_id)
        self.assertEqual(manifest["video"]["frame_count"], 10)
        self.assertEqual(len(manifest["chunks"]), 1)
        chunk = json.loads(self.storage.chunk_path(job_id, 0).read_text())
        self.assertEqual(len(chunk["frames"]), 10)
        self.assertEqual(chunk["frames"][0]["segmentation"]["type"], "polygon")
        self.assertEqual(chunk["frames"][0]["centerline_segmentation"]["type"], "polygon")
        self.assertIsNotNone(chunk["frames"][0]["centerline_box_xywh"])
        self.assertEqual(chunk["frames"][0]["shaft_segmentation"]["type"], "polygon")
        self.assertIsNotNone(chunk["frames"][0]["shaft_box_xywh"])

    def test_only_one_worker_claims_a_job(self) -> None:
        now = utc_now()
        self.database.execute(
            """
            INSERT INTO videos
                (id, original_filename, state, mime_type, expected_size_bytes,
                 created_at, expires_at)
            VALUES ('video', 'fixture.mp4', 'ready', 'video/mp4', 1, ?, ?)
            """,
            (now, expires_at()),
        )
        self.database.execute(
            """
            INSERT INTO jobs
                (id, video_id, state, progress_stage, settings_json, created_at, expires_at)
            VALUES ('job', 'video', 'queued', 'queued', '{}', ?, ?)
            """,
            (now, expires_at()),
        )
        first = self.database.claim_job("worker-1")
        second = self.database.claim_job("worker-2")
        self.assertEqual(first["id"], "job")
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
