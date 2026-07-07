from __future__ import annotations

import json
import math
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any

from .errors import ServiceError

Color = tuple[int, int, int, int]
Line = tuple[float, float, float, float]


def export_centerline_video(
    *,
    video_path: Path,
    output_path: Path,
    temporary_dir: Path,
    manifest: dict[str, Any],
    chunk_paths: list[Path],
) -> Path:
    if not video_path.is_file():
        raise ServiceError("NOT_FOUND", "Video content is unavailable.", status_code=404)
    if not chunk_paths:
        raise ServiceError("NOT_FOUND", "Result chunks are unavailable.", status_code=404)

    width = int(manifest["video"]["width"])
    height = int(manifest["video"]["height"])
    fps = float(manifest["video"]["fps"] or 30)
    frame_count = int(manifest["video"]["frame_count"])
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ServiceError("EXPORT_FAILED", "Result manifest has invalid video metadata.")

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    frames_dir = temporary_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".tmp.mp4")

    colors = {
        prompt["id"]: _parse_hex(prompt.get("color", "#35C2FF"))
        for prompt in manifest.get("prompts", [])
    }
    frames = _load_frames_by_index(chunk_paths)
    for frame_index in range(frame_count):
        image = _transparent_image(width, height)
        records = frames.get(frame_index, [])
        _draw_frame_overlay(image, width, height, records, colors)
        _write_png_rgba(frames_dir / f"{frame_index:06d}.png", width, height, image)

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "%06d.png"),
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        temporary_output.replace(output_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        temporary_output.unlink(missing_ok=True)
        detail = exc.stderr[-1000:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ServiceError(
            "EXPORT_FAILED",
            f"Video export failed: {detail}",
            retryable=True,
            status_code=500,
        ) from exc
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return output_path


def _load_frames_by_index(chunk_paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("frames", []):
            frames.setdefault(int(record["frame_index"]), []).append(record)
    return frames


def _draw_frame_overlay(
    image: bytearray,
    width: int,
    height: int,
    records: list[dict[str, Any]],
    colors: dict[str, Color],
) -> None:
    centerlines: list[tuple[dict[str, Any], Line, Color]] = []
    for record in records:
        line = _record_line(record)
        if line is None:
            continue
        color = colors.get(record["prompt_id"], (53, 194, 255, 255))
        centerlines.append((record, line, color))
        _draw_line(image, width, height, line, color, max(3, round(min(width, height) * 0.006)))

    for first_index in range(len(centerlines)):
        for second_index in range(first_index + 1, len(centerlines)):
            first_record, first_line, _first_color = centerlines[first_index]
            second_record, second_line, _second_color = centerlines[second_index]
            if first_record["prompt_id"] == second_record["prompt_id"]:
                continue
            annotation = _angle_annotation(first_line, second_line)
            if annotation is not None:
                _draw_angle_annotation(image, width, height, annotation)


def _record_line(record: dict[str, Any]) -> Line | None:
    values = record.get("centerline_line_xyxy") or record.get("shaft_line_xyxy")
    if not values or len(values) != 4:
        return None
    try:
        line = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in line):
        return None
    return line  # type: ignore[return-value]


def _angle_annotation(first: Line, second: Line) -> dict[str, float] | None:
    intersection = _segment_intersection(first, second)
    if intersection is None:
        return None
    first_vector = _normalize((first[2] - first[0], first[3] - first[1]))
    second_vector = _normalize((second[2] - second[0], second[3] - second[1]))
    if first_vector is None or second_vector is None:
        return None
    if _dot(first_vector, second_vector) < 0:
        second_vector = (-second_vector[0], -second_vector[1])
    radians = math.acos(_clamp(_dot(first_vector, second_vector), -1, 1))
    degrees = math.degrees(radians)
    if not math.isfinite(degrees) or degrees < 2:
        return None

    first_angle = math.atan2(first_vector[1], first_vector[0])
    second_angle = math.atan2(second_vector[1], second_vector[0])
    delta = _normalize_angle(second_angle - first_angle)
    if abs(delta) > math.pi:
        delta -= math.copysign(math.tau, delta)
    return {
        "x": intersection[0],
        "y": intersection[1],
        "start_angle": first_angle,
        "end_angle": first_angle + delta,
        "degrees": min(degrees, 180 - degrees),
    }


def _segment_intersection(first: Line, second: Line) -> tuple[float, float] | None:
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    first_det = x1 * y2 - y1 * x2
    second_det = x3 * y4 - y3 * x4
    x = (first_det * (x3 - x4) - (x1 - x2) * second_det) / denominator
    y = (first_det * (y3 - y4) - (y1 - y2) * second_det) / denominator
    if not _point_on_segment(x, y, first) or not _point_on_segment(x, y, second):
        return None
    return x, y


def _point_on_segment(x: float, y: float, line: Line) -> bool:
    x1, y1, x2, y2 = line
    tolerance = 1.5
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _draw_angle_annotation(
    image: bytearray,
    width: int,
    height: int,
    annotation: dict[str, float],
) -> None:
    radius = max(24, min(width, height) * 0.07)
    start = annotation["start_angle"]
    end = annotation["end_angle"]
    delta = _normalize_angle(end - start)
    steps = max(12, int(abs(delta) * radius / 2))
    yellow = (255, 242, 168, 255)
    for step in range(steps + 1):
        angle = start + delta * step / max(steps, 1)
        x = annotation["x"] + math.cos(angle) * radius
        y = annotation["y"] + math.sin(angle) * radius
        _draw_circle(image, width, height, x, y, 2.2, yellow)

    mid = start + delta / 2
    label = f"{round(annotation['degrees'])}°"
    label_x = annotation["x"] + math.cos(mid) * (radius + 22)
    label_y = annotation["y"] + math.sin(mid) * (radius + 22)
    _draw_label(image, width, height, label_x, label_y, label)


def _draw_line(
    image: bytearray,
    width: int,
    height: int,
    line: Line,
    color: Color,
    thickness: int,
) -> None:
    x1, y1, x2, y2 = line
    radius = thickness / 2
    min_x = max(0, math.floor(min(x1, x2) - radius - 1))
    max_x = min(width - 1, math.ceil(max(x1, x2) + radius + 1))
    min_y = max(0, math.floor(min(y1, y2) - radius - 1))
    max_y = min(height - 1, math.ceil(max(y1, y2) + radius + 1))
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        return
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = _clamp(((x - x1) * dx + (y - y1) * dy) / length_sq, 0, 1)
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            if (x - closest_x) ** 2 + (y - closest_y) ** 2 <= radius * radius:
                _blend_pixel(image, width, x, y, color)


def _draw_circle(
    image: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    color: Color,
) -> None:
    min_x = max(0, math.floor(center_x - radius))
    max_x = min(width - 1, math.ceil(center_x + radius))
    min_y = max(0, math.floor(center_y - radius))
    max_y = min(height - 1, math.ceil(center_y + radius))
    radius_sq = radius * radius
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_sq:
                _blend_pixel(image, width, x, y, color)


_DIGITS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "°": ("110", "110", "000", "000", "000"),
}


def _draw_label(
    image: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    text: str,
) -> None:
    scale = max(2, round(min(width, height) / 220))
    char_width = 3 * scale
    char_height = 5 * scale
    gap = scale
    text_width = len(text) * char_width + max(0, len(text) - 1) * gap
    padding = 3 * scale
    left = round(center_x - text_width / 2)
    top = round(center_y - char_height / 2)
    _fill_rect(
        image,
        width,
        height,
        left - padding,
        top - padding,
        text_width + padding * 2,
        char_height + padding * 2,
        (2, 5, 9, 210),
    )
    x = left
    for character in text:
        _draw_bitmap_character(image, width, height, x, top, character, scale, (255, 242, 168, 255))
        x += char_width + gap


def _draw_bitmap_character(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    character: str,
    scale: int,
    color: Color,
) -> None:
    bitmap = _DIGITS.get(character)
    if not bitmap:
        return
    for row_index, row in enumerate(bitmap):
        for col_index, value in enumerate(row):
            if value == "1":
                _fill_rect(
                    image,
                    width,
                    height,
                    left + col_index * scale,
                    top + row_index * scale,
                    scale,
                    scale,
                    color,
                )


def _fill_rect(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    rect_width: int,
    rect_height: int,
    color: Color,
) -> None:
    for y in range(max(0, top), min(height, top + rect_height)):
        for x in range(max(0, left), min(width, left + rect_width)):
            _blend_pixel(image, width, x, y, color)


def _transparent_image(width: int, height: int) -> bytearray:
    return bytearray(width * height * 4)


def _blend_pixel(image: bytearray, width: int, x: int, y: int, color: Color) -> None:
    if color[3] <= 0:
        return
    index = (y * width + x) * 4
    source_alpha = color[3] / 255
    inverse = 1 - source_alpha
    image[index] = round(color[0] * source_alpha + image[index] * inverse)
    image[index + 1] = round(color[1] * source_alpha + image[index + 1] * inverse)
    image[index + 2] = round(color[2] * source_alpha + image[index + 2] * inverse)
    image[index + 3] = min(255, round(color[3] + image[index + 3] * inverse))


def _write_png_rgba(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=1)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return len(data).to_bytes(4, "big") + kind + data + checksum.to_bytes(4, "big")


def _parse_hex(value: str) -> Color:
    stripped = value.lstrip("#")
    if len(stripped) != 6:
        return (53, 194, 255, 255)
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
        255,
    )


def _normalize(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector[0], vector[1])
    if length < 1e-6:
        return None
    return vector[0] / length, vector[1] / length


def _dot(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += math.tau
    while angle > math.pi:
        angle -= math.tau
    return angle
