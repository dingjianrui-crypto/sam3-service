from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .errors import ServiceError


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _safe(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ServiceError("INVALID_PATH", "Resolved path is outside the data root.")
        return resolved

    def upload_dir(self, video_id: str) -> Path:
        path = self._safe(self.root / "uploads" / video_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def part_path(self, video_id: str, part_number: int) -> Path:
        directory = self.upload_dir(video_id) / "parts"
        directory.mkdir(parents=True, exist_ok=True)
        return self._safe(directory / f"{part_number:06d}.part")

    def source_path(self, video_id: str) -> Path:
        return self._safe(self.upload_dir(video_id) / "source")

    def video_dir(self, video_id: str) -> Path:
        path = self._safe(self.root / "videos" / video_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def normalized_path(self, video_id: str) -> Path:
        return self._safe(self.video_dir(video_id) / "normalized.mp4")

    def job_dir(self, job_id: str) -> Path:
        path = self._safe(self.root / "jobs" / job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def chunk_path(self, job_id: str, sequence: int) -> Path:
        directory = self.job_dir(job_id) / "chunks"
        directory.mkdir(parents=True, exist_ok=True)
        return self._safe(directory / f"{sequence:06d}.json")

    def manifest_path(self, job_id: str) -> Path:
        return self._safe(self.job_dir(job_id) / "manifest.json")

    def export_path(self, job_id: str) -> Path:
        return self._safe(self.job_dir(job_id) / "exports" / "centerlines.mp4")

    def export_tmp_dir(self, job_id: str) -> Path:
        path = self._safe(self.job_dir(job_id) / "exports" / "tmp")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def assemble(self, video_id: str, part_paths: list[Path]) -> tuple[Path, int, str]:
        destination = self.source_path(video_id)
        temporary = destination.with_suffix(".assembling")
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as output:
            for part in part_paths:
                with self._safe(part).open("rb") as source:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
        temporary.replace(destination)
        return destination, size, digest.hexdigest()

    def delete_video(self, video_id: str) -> None:
        for path in (self.root / "uploads" / video_id, self.root / "videos" / video_id):
            safe = self._safe(path)
            if safe.exists():
                shutil.rmtree(safe)

    def delete_job(self, job_id: str) -> None:
        path = self._safe(self.root / "jobs" / job_id)
        if path.exists():
            shutil.rmtree(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
