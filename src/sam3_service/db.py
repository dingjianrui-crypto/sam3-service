from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def expires_at(days: int = 7) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    state TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    expected_size_bytes INTEGER NOT NULL,
    size_bytes INTEGER,
    sha256 TEXT,
    source_path TEXT,
    normalized_path TEXT,
    width INTEGER,
    height INTEGER,
    fps REAL,
    duration_ms INTEGER,
    frame_count INTEGER,
    codec TEXT,
    error_code TEXT,
    error_detail TEXT,
    created_at TEXT NOT NULL,
    validated_at TEXT,
    expires_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS upload_parts (
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    part_number INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (video_id, part_number)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id),
    state TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    progress_stage TEXT,
    processed_frames INTEGER NOT NULL DEFAULT 0,
    total_frames INTEGER,
    settings_json TEXT NOT NULL,
    model_name TEXT,
    worker_id TEXT,
    worker_heartbeat_at TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    cancel_requested_at TEXT,
    error_code TEXT,
    error_detail TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    expires_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS job_prompts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    color TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    error_code TEXT,
    UNIQUE(job_id, position)
);

CREATE TABLE IF NOT EXISTS result_chunks (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_jobs_queue
ON jobs(state, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        connection = self.connect()
        try:
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(
        self, sql: str, parameters: tuple[Any, ...] = (), *, retries: int = 3
    ) -> sqlite3.Cursor:
        for attempt in range(retries):
            connection: sqlite3.Connection | None = None
            try:
                connection = self.connect()
                cursor = connection.execute(sql, parameters)
                connection.commit()
                rows = cursor.fetchall() if cursor.description else []
                return _DetachedCursor(rows, cursor.rowcount)  # type: ignore[return-value]
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt + 1 == retries:
                    raise
                time.sleep(0.04 * (2**attempt))
            finally:
                if connection is not None:
                    connection.close()
        raise AssertionError("unreachable")

    def fetch_one(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        connection = self.connect()
        try:
            row = connection.execute(sql, parameters).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()

    def fetch_all(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        connection = self.connect()
        try:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]
        finally:
            connection.close()

    def claim_job(self, worker_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                UPDATE jobs
                SET state = 'preprocessing',
                    progress_stage = 'preprocessing',
                    worker_id = ?,
                    worker_heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE state = 'queued' AND cancel_requested_at IS NULL
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                )
                RETURNING *
                """,
                (worker_id, now, now),
            ).fetchone()
            return dict(row) if row else None

    def job_detail(self, job_id: str) -> dict[str, Any] | None:
        job = self.fetch_one("SELECT * FROM jobs WHERE id = ? AND deleted_at IS NULL", (job_id,))
        if not job:
            return None
        job["settings"] = json.loads(job.pop("settings_json"))
        job["prompts"] = self.fetch_all(
            "SELECT id, position, text, color, state, error_code "
            "FROM job_prompts WHERE job_id = ? ORDER BY position",
            (job_id,),
        )
        return job


class _DetachedCursor:
    def __init__(self, rows: list[sqlite3.Row], rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self) -> sqlite3.Row | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[sqlite3.Row]:
        return self._rows
