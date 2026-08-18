from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .errors import ServiceError


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
        duration = float(
            stream.get("duration")
            or payload.get("format", {}).get("duration")
            or 0
        )
        try:
            reported_frame_count = int(stream.get("nb_frames") or 0)
        except (TypeError, ValueError):
            reported_frame_count = 0
        duration_frame_count = round(duration * fps) if fps > 0 else 0
        # Container duration can include a slightly longer audio stream. When
        # ffprobe provides the video stream's exact frame count, it is the
        # authoritative upper bound for zero-based inference frame indices.
        frame_count = reported_frame_count or duration_frame_count
        if not stream.get("width") or not stream.get("height") or duration <= 0:
            raise ValueError("missing video dimensions or duration")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "duration_ms": round(duration * 1000),
            "frame_count": frame_count,
            "codec": stream.get("codec_name", "unknown"),
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyError, ValueError) as exc:
        raise ServiceError("INVALID_VIDEO", "The uploaded file is not a readable video.") from exc


def normalize_video(source: Path, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    target_fps = max(1.0, min(float(metadata["fps"] or 30), 30.0))
    temporary = destination.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale='min(1280,iw)':-2,fps={target_fps:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
        temporary.replace(destination)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.stderr[-1000:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ServiceError(
            "NORMALIZATION_FAILED",
            f"Video normalization failed: {detail}",
            retryable=True,
        ) from exc
    return probe_video(destination)
