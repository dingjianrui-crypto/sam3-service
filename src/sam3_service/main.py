from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .db import Database, expires_at, utc_now
from .errors import ServiceError
from .exporter import ExportOptions, export_centerline_video
from .media import probe_video
from .schemas import JobCreate, VideoCreate
from .storage import LocalStorage, sha256_file

COLORS = ["#35C2FF", "#FFB547", "#A78BFA", "#4ADE80", "#FB7185"]


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured.ensure_directories()
        database = Database(configured.database_path)
        database.initialize()
        app.state.settings = configured
        app.state.database = database
        app.state.storage = LocalStorage(configured.data_dir)
        yield

    app = FastAPI(
        title="SAM 3 Paddle Review API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if configured.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(configured.cors_allow_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "The request was invalid.",
                    "details": exc.errors(),
                    "retryable": False,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.get("/api/v1/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready")
    def ready(request: Request) -> dict[str, Any]:
        database: Database = request.app.state.database
        database.fetch_one("SELECT 1 AS ok")
        stat = shutil.disk_usage(configured.data_dir)
        return {
            "status": "ready",
            "segmenter": configured.segmenter,
            "free_bytes": stat.free,
        }

    @app.post("/api/v1/videos", status_code=201)
    def create_video(payload: VideoCreate, request: Request) -> dict[str, Any]:
        if payload.size_bytes > configured.max_upload_bytes:
            raise ServiceError(
                "VIDEO_LIMIT_EXCEEDED",
                f"Video exceeds the {configured.max_upload_bytes} byte limit.",
                status_code=413,
            )
        suffix = Path(payload.filename).suffix.lower()
        if suffix not in {".mp4", ".mov"}:
            raise ServiceError("INVALID_VIDEO", "Only MP4 and MOV files are supported.")
        video_id = uuid.uuid4().hex
        now = utc_now()
        database: Database = request.app.state.database
        database.execute(
            """
            INSERT INTO videos
                (id, original_filename, state, mime_type, expected_size_bytes,
                 sha256, created_at, expires_at)
            VALUES (?, ?, 'uploading', ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                payload.filename,
                payload.mime_type,
                payload.size_bytes,
                payload.sha256,
                now,
                expires_at(),
            ),
        )
        return {
            "video_id": video_id,
            "upload_id": video_id,
            "chunk_size_bytes": configured.chunk_size_bytes,
            "expires_at": expires_at(),
        }

    @app.put("/api/v1/videos/{video_id}/parts/{part_number}", status_code=201)
    async def upload_part(
        video_id: str,
        part_number: int,
        request: Request,
        x_part_sha256: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if part_number < 0:
            raise ServiceError("VALIDATION_ERROR", "Part number must be non-negative.")
        database: Database = request.app.state.database
        video = _video_or_404(database, video_id)
        if video["state"] != "uploading":
            raise ServiceError("INVALID_STATE", "This upload is no longer open.", status_code=409)
        storage: LocalStorage = request.app.state.storage
        destination = storage.part_path(video_id, part_number)
        temporary = destination.with_suffix(".tmp")
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as output:
            async for block in request.stream():
                size += len(block)
                if size > configured.chunk_size_bytes:
                    temporary.unlink(missing_ok=True)
                    raise ServiceError(
                        "VIDEO_LIMIT_EXCEEDED", "Upload part exceeds the chunk limit.", status_code=413
                    )
                output.write(block)
                digest.update(block)
        checksum = digest.hexdigest()
        if x_part_sha256 and checksum.lower() != x_part_sha256.lower():
            temporary.unlink(missing_ok=True)
            raise ServiceError("CHECKSUM_MISMATCH", "Upload part checksum did not match.")
        temporary.replace(destination)
        database.execute(
            """
            INSERT INTO upload_parts
                (video_id, part_number, size_bytes, sha256, path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, part_number) DO UPDATE SET
                size_bytes = excluded.size_bytes, sha256 = excluded.sha256,
                path = excluded.path, created_at = excluded.created_at
            """,
            (video_id, part_number, size, checksum, str(destination), utc_now()),
        )
        return {"part_number": part_number, "size_bytes": size, "sha256": checksum}

    @app.post("/api/v1/videos/{video_id}/complete")
    def complete_video(video_id: str, request: Request) -> dict[str, Any]:
        database: Database = request.app.state.database
        video = _video_or_404(database, video_id)
        if video["state"] == "ready":
            return _public_video(video)
        if video["state"] != "uploading":
            raise ServiceError("INVALID_STATE", "Video cannot be completed.", status_code=409)
        parts = database.fetch_all(
            "SELECT * FROM upload_parts WHERE video_id = ? ORDER BY part_number", (video_id,)
        )
        if not parts or [item["part_number"] for item in parts] != list(range(len(parts))):
            raise ServiceError("UPLOAD_INCOMPLETE", "Upload parts are missing or non-contiguous.")
        if sum(item["size_bytes"] for item in parts) != video["expected_size_bytes"]:
            raise ServiceError("UPLOAD_INCOMPLETE", "Uploaded byte count does not match the file.")
        storage: LocalStorage = request.app.state.storage
        source, size, checksum = storage.assemble(
            video_id, [Path(item["path"]) for item in parts]
        )
        if video["sha256"] and checksum.lower() != video["sha256"].lower():
            source.unlink(missing_ok=True)
            raise ServiceError("CHECKSUM_MISMATCH", "Whole-file checksum did not match.")
        metadata = probe_video(source)
        if metadata["duration_ms"] > configured.max_video_duration_seconds * 1000:
            source.unlink(missing_ok=True)
            raise ServiceError(
                "VIDEO_LIMIT_EXCEEDED",
                f"Video exceeds the {configured.max_video_duration_seconds:g} second limit.",
            )
        database.execute(
            """
            UPDATE videos SET state = 'ready', source_path = ?, size_bytes = ?, sha256 = ?,
                width = ?, height = ?, fps = ?, duration_ms = ?, frame_count = ?,
                codec = ?, validated_at = ?
            WHERE id = ?
            """,
            (
                str(source),
                size,
                checksum,
                metadata["width"],
                metadata["height"],
                metadata["fps"],
                metadata["duration_ms"],
                metadata["frame_count"],
                metadata["codec"],
                utc_now(),
                video_id,
            ),
        )
        return _public_video(_video_or_404(database, video_id))

    @app.get("/api/v1/videos/{video_id}")
    def get_video(video_id: str, request: Request) -> dict[str, Any]:
        return _public_video(_video_or_404(request.app.state.database, video_id))

    @app.get("/api/v1/videos/{video_id}/content")
    def video_content(
        video_id: str,
        request: Request,
        normalized: bool = Query(default=True),
    ) -> FileResponse:
        video = _video_or_404(request.app.state.database, video_id)
        selected = video["normalized_path"] if normalized else video["source_path"]
        if not selected and normalized:
            selected = video["source_path"]
        if not selected or not Path(selected).is_file():
            raise ServiceError("NOT_FOUND", "Video content is unavailable.", status_code=404)
        media_type = "video/mp4" if normalized else (
            mimetypes.guess_type(video["original_filename"])[0] or "application/octet-stream"
        )
        return FileResponse(selected, media_type=media_type, filename=None)

    @app.post("/api/v1/jobs", status_code=202)
    def create_job(
        payload: JobCreate,
        request: Request,
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        database: Database = request.app.state.database
        video = _video_or_404(database, payload.video_id)
        if video["state"] != "ready":
            raise ServiceError("INVALID_STATE", "Video is not ready for processing.", status_code=409)
        if len(payload.prompts) > configured.max_prompts:
            raise ServiceError(
                "INVALID_PROMPT", f"At most {configured.max_prompts} prompts are allowed."
            )
        if idempotency_key:
            existing = database.fetch_one(
                "SELECT id, state FROM jobs WHERE json_extract(settings_json, '$.idempotency_key') = ?",
                (idempotency_key,),
            )
            if existing:
                return {
                    "job_id": existing["id"],
                    "state": existing["state"],
                    "status_url": f"/api/v1/jobs/{existing['id']}",
                }
        job_id = uuid.uuid4().hex
        settings_payload = _job_settings_payload(payload.settings.model_dump(), configured)
        settings_payload["idempotency_key"] = idempotency_key
        now = utc_now()
        with database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO jobs
                    (id, video_id, state, progress_stage, settings_json, created_at, expires_at)
                VALUES (?, ?, 'queued', 'queued', ?, ?, ?)
                """,
                (job_id, payload.video_id, json.dumps(settings_payload), now, expires_at()),
            )
            for position, prompt in enumerate(payload.prompts):
                connection.execute(
                    """
                    INSERT INTO job_prompts (id, job_id, position, text, color)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        job_id,
                        position,
                        prompt.text,
                        COLORS[position % len(COLORS)],
                    ),
                )
        return {
            "job_id": job_id,
            "state": "queued",
            "status_url": f"/api/v1/jobs/{job_id}",
        }

    @app.get("/api/v1/jobs")
    def list_jobs(request: Request, limit: int = Query(default=50, ge=1, le=100)):
        database: Database = request.app.state.database
        jobs = database.fetch_all(
            "SELECT id FROM jobs WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return {"items": [_public_job(database.job_detail(item["id"])) for item in jobs]}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        job = request.app.state.database.job_detail(job_id)
        if not job:
            raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
        return _public_job(job)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        database: Database = request.app.state.database
        job = database.job_detail(job_id)
        if not job:
            raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
        if job["state"] in {"completed", "failed", "cancelled"}:
            return _public_job(job)
        now = utc_now()
        if job["state"] == "queued":
            database.execute(
                "UPDATE jobs SET state = 'cancelled', progress_stage = 'cancelled', "
                "cancel_requested_at = ?, completed_at = ? WHERE id = ?",
                (now, now, job_id),
            )
        else:
            database.execute(
                "UPDATE jobs SET cancel_requested_at = ? WHERE id = ?", (now, job_id)
            )
        return _public_job(database.job_detail(job_id))

    @app.post("/api/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str, request: Request) -> dict[str, Any]:
        database: Database = request.app.state.database
        job = database.job_detail(job_id)
        if not job:
            raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
        if job["state"] not in {"failed", "cancelled"}:
            raise ServiceError("INVALID_STATE", "Only failed or cancelled jobs can retry.", status_code=409)
        request.app.state.storage.delete_job(job_id)
        database.execute("DELETE FROM result_chunks WHERE job_id = ?", (job_id,))
        database.execute(
            """
            UPDATE jobs SET state = 'queued', progress_stage = 'queued',
                processed_frames = 0, cancel_requested_at = NULL, error_code = NULL,
                error_detail = NULL, completed_at = NULL, worker_id = NULL,
                worker_heartbeat_at = NULL, attempt = attempt + 1
            WHERE id = ?
            """,
            (job_id,),
        )
        database.execute(
            "UPDATE job_prompts SET state = 'queued', error_code = NULL WHERE job_id = ?",
            (job_id,),
        )
        return _public_job(database.job_detail(job_id))

    @app.get("/api/v1/jobs/{job_id}/results")
    def get_results(job_id: str, request: Request) -> JSONResponse:
        job = request.app.state.database.job_detail(job_id)
        if not job:
            raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
        if job["state"] != "completed":
            raise ServiceError("INVALID_STATE", "Results are not ready.", status_code=409)
        path = request.app.state.storage.manifest_path(job_id)
        if not path.is_file():
            raise ServiceError("NOT_FOUND", "Result manifest is unavailable.", status_code=404)
        return JSONResponse(json.loads(path.read_text()))

    @app.get("/api/v1/jobs/{job_id}/results/chunks/{sequence}")
    def get_result_chunk(job_id: str, sequence: int, request: Request) -> FileResponse:
        row = request.app.state.database.fetch_one(
            "SELECT path FROM result_chunks WHERE job_id = ? AND sequence = ?",
            (job_id, sequence),
        )
        if not row or not Path(row["path"]).is_file():
            raise ServiceError("NOT_FOUND", "Result chunk was not found.", status_code=404)
        return FileResponse(row["path"], media_type="application/json")

    @app.get("/api/v1/jobs/{job_id}/export")
    def export_job(
        job_id: str,
        request: Request,
        angle_label_position: str = Query(default="top", pattern="^(top|bottom)$"),
        angle_label_font_size: int = Query(default=32, ge=12, le=96),
        include_spm: bool = Query(default=False),
        metric_center_offset_percent: float | None = Query(default=None, ge=0, le=45),
        reference_prompt_id: str | None = Query(default=None),
        target_prompt_ids: str | None = Query(default=None),
    ) -> FileResponse:
        database: Database = request.app.state.database
        storage: LocalStorage = request.app.state.storage
        job = database.job_detail(job_id)
        if not job:
            raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
        if job["state"] != "completed":
            raise ServiceError("INVALID_STATE", "Results are not ready.", status_code=409)
        video = _video_or_404(database, job["video_id"])
        video_path = Path(video["source_path"] or video["normalized_path"])
        manifest_path = storage.manifest_path(job_id)
        if not manifest_path.is_file():
            raise ServiceError("NOT_FOUND", "Result manifest is unavailable.", status_code=404)
        rows = database.fetch_all(
            "SELECT path FROM result_chunks WHERE job_id = ? ORDER BY sequence", (job_id,)
        )
        chunk_paths = [Path(row["path"]) for row in rows if Path(row["path"]).is_file()]
        output_path = storage.export_path(job_id)
        export_centerline_video(
            video_path=video_path,
            output_path=output_path,
            temporary_dir=storage.export_tmp_dir(job_id),
            manifest=json.loads(manifest_path.read_text()),
            chunk_paths=chunk_paths,
            options=ExportOptions(
                angle_label_position=angle_label_position,
                angle_label_font_size=angle_label_font_size,
                include_spm=include_spm,
                metric_center_offset_percent=metric_center_offset_percent,
                reference_prompt_id=reference_prompt_id,
                target_prompt_ids=tuple(
                    item.strip() for item in (target_prompt_ids or "").split(",") if item.strip()
                ),
            ),
        )
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"sam3-{job_id}-centerlines.mp4",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request) -> None:
        database: Database = request.app.state.database
        job = database.job_detail(job_id)
        if not job:
            return None
        if job["state"] not in {"completed", "failed", "cancelled"}:
            raise ServiceError("INVALID_STATE", "Cancel the active job before deleting it.", status_code=409)
        request.app.state.storage.delete_job(job_id)
        database.execute(
            "UPDATE jobs SET deleted_at = ? WHERE id = ?", (utc_now(), job_id)
        )
        return None

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        static = configured.static_dir
        requested = (static / path).resolve()
        if static.is_dir() and requested.is_relative_to(static) and requested.is_file():
            return FileResponse(requested)
        index = static / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise ServiceError(
            "NOT_FOUND",
            "Web client is not built. Run the API under /docs or build apps/web.",
            status_code=404,
        )

    return app


def _video_or_404(database: Database, video_id: str) -> dict[str, Any]:
    video = database.fetch_one(
        "SELECT * FROM videos WHERE id = ? AND deleted_at IS NULL", (video_id,)
    )
    if not video:
        raise ServiceError("NOT_FOUND", "Video was not found.", status_code=404)
    return video


def _public_video(video: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "original_filename",
        "state",
        "mime_type",
        "size_bytes",
        "width",
        "height",
        "fps",
        "duration_ms",
        "frame_count",
        "codec",
        "error_code",
        "error_detail",
        "created_at",
        "validated_at",
        "expires_at",
    )
    return {key: video.get(key) for key in fields}


def _job_settings_payload(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    requested_max = payload.get("max_detections_per_frame")
    if requested_max is None:
        requested_max = min(
            settings.default_max_detections_per_frame,
            settings.max_detections_per_frame_limit,
        )
    elif int(requested_max) > settings.max_detections_per_frame_limit:
        raise ServiceError(
            "INVALID_SETTINGS",
            "max_detections_per_frame exceeds the service limit of "
            f"{settings.max_detections_per_frame_limit}.",
        )
    requested_max = int(requested_max)

    redetect_interval = payload.get("redetect_interval_frames")
    if redetect_interval is None:
        redetect_interval = settings.default_redetect_interval_frames

    dedupe_iou = payload.get("dedupe_iou_threshold")
    if dedupe_iou is None:
        dedupe_iou = settings.default_dedupe_iou_threshold

    return {
        **payload,
        "redetect_interval_frames": max(0, int(redetect_interval)),
        "max_detections_per_frame": requested_max,
        "dedupe_iou_threshold": max(0.0, min(1.0, float(dedupe_iou))),
    }


def _public_job(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        raise ServiceError("NOT_FOUND", "Job was not found.", status_code=404)
    total = job.get("total_frames") or 0
    processed = job.get("processed_frames") or 0
    percent = min(100.0, processed * 100 / total) if total else 0.0
    return {
        "job_id": job["id"],
        "video_id": job["video_id"],
        "state": job["state"],
        "progress": {
            "stage": job.get("progress_stage"),
            "processed_frames": processed,
            "total_frames": total,
            "percent": round(percent, 1),
        },
        "prompts": job["prompts"],
        "settings": job["settings"],
        "model_name": job.get("model_name"),
        "attempt": job["attempt"],
        "error": (
            {
                "code": job["error_code"],
                "message": job["error_detail"],
                "retryable": job["error_code"]
                in {"GPU_OUT_OF_MEMORY", "INFERENCE_FAILED", "NORMALIZATION_FAILED"},
            }
            if job.get("error_code")
            else None
        ),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }


app = create_app()


def run() -> None:
    uvicorn.run(
        "sam3_service.main:app",
        host=os.getenv("SAM3_HOST", "127.0.0.1"),
        port=int(os.getenv("SAM3_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
