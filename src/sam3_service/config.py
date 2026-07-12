from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    static_dir: Path
    segmenter: str
    checkpoint_path: Path | None
    offline: bool
    poll_interval_seconds: float
    max_upload_bytes: int
    max_video_duration_seconds: float
    max_prompts: int
    chunk_size_bytes: int
    result_chunk_seconds: int
    cors_allow_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("SAM3_DATA_DIR", root / "data")).expanduser().resolve()
        return cls(
            data_dir=data_dir,
            database_path=Path(
                os.getenv("SAM3_DATABASE_PATH", data_dir / "sam3.sqlite3")
            ).expanduser().resolve(),
            static_dir=Path(
                os.getenv("SAM3_STATIC_DIR", root / "apps" / "web" / "dist")
            ).expanduser().resolve(),
            segmenter=os.getenv("SAM3_SEGMENTER", "mock").lower(),
            checkpoint_path=(
                Path(os.environ["SAM3_CHECKPOINT_PATH"]).expanduser().resolve()
                if os.getenv("SAM3_CHECKPOINT_PATH")
                else None
            ),
            offline=os.getenv("SAM3_OFFLINE", "0").lower() in {"1", "true", "yes"},
            poll_interval_seconds=float(os.getenv("SAM3_POLL_INTERVAL_SECONDS", "1")),
            max_upload_bytes=int(os.getenv("SAM3_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))),
            max_video_duration_seconds=float(
                os.getenv("SAM3_MAX_VIDEO_DURATION_SECONDS", "300")
            ),
            max_prompts=int(os.getenv("SAM3_MAX_PROMPTS", "3")),
            chunk_size_bytes=int(os.getenv("SAM3_UPLOAD_CHUNK_BYTES", str(8 * 1024 * 1024))),
            result_chunk_seconds=int(os.getenv("SAM3_RESULT_CHUNK_SECONDS", "2")),
            cors_allow_origins=_csv_env("SAM3_CORS_ALLOW_ORIGINS"),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        for name in ("uploads", "videos", "jobs", "tmp"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())
