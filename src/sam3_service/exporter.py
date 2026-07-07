from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import zlib
from bisect import bisect_left
from pathlib import Path
from typing import Any

from .errors import ServiceError
from .media import probe_video

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

    manifest_video = manifest["video"]
    manifest_width = int(manifest_video["width"])
    manifest_height = int(manifest_video["height"])
    manifest_fps = float(manifest_video["fps"] or 30)
    video_metadata = probe_video(video_path)
    width = int(video_metadata["width"] or manifest_width)
    height = int(video_metadata["height"] or manifest_height)
    fps = float(video_metadata["fps"] or manifest_fps or 30)
    frame_count = max(
        int(video_metadata["frame_count"] or 0),
        math.ceil(float(video_metadata["duration_ms"] or 0) * fps / 1000),
    )
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
    frames = _load_frames_by_timestamp(chunk_paths)
    frame_timestamps = sorted(frames)
    scale_x = width / manifest_width if manifest_width > 0 else 1.0
    scale_y = height / manifest_height if manifest_height > 0 else 1.0
    result_tolerance_ms = max(1000 / max(fps, 1), 500 / max(manifest_fps, 1), 40)
    use_subtitle_top_label = _ffmpeg_supports_filter("subtitles")
    top_degrees: list[int | None] = []
    carried_degree: int | None = None
    for frame_index in range(frame_count):
        image = _transparent_image(width, height)
        timestamp_ms = round(frame_index * 1000 / fps)
        records = _records_for_timestamp(
            frames, frame_timestamps, timestamp_ms, result_tolerance_ms
        )
        scaled_records = [_scale_record(record, scale_x, scale_y) for record in records]
        carried_degree = _draw_frame_overlay(
            image,
            width,
            height,
            scaled_records,
            colors,
            fallback_degree=carried_degree,
            draw_top_label=not use_subtitle_top_label,
        )
        top_degrees.append(carried_degree)
        _write_png_rgba(frames_dir / f"{frame_index:06d}.png", width, height, image)

    filter_complex = "[0:v][1:v]overlay=0:0:format=auto[ov]"
    subtitle_path = temporary_dir / "paddle-degree-labels.ass"
    if use_subtitle_top_label and _write_degree_subtitles(
        subtitle_path, top_degrees, width, height, fps
    ):
        filter_complex += f";[ov]subtitles={_ffmpeg_filter_path(subtitle_path)}[v]"
    else:
        filter_complex += ";[ov]null[v]"

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
        filter_complex,
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


def _load_frames_by_timestamp(chunk_paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    for path in chunk_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("frames", []):
            frames.setdefault(int(record["timestamp_ms"]), []).append(record)
    return frames


def _ffmpeg_supports_filter(name: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return any(line.split()[1:2] == [name] for line in result.stdout.splitlines())


def _write_degree_subtitles(
    path: Path,
    degrees: list[int | None],
    width: int,
    height: int,
    fps: float,
) -> bool:
    intervals: list[tuple[int, int, int]] = []
    start_index: int | None = None
    active_degree: int | None = None
    for index, degree in enumerate(degrees + [None]):
        if degree == active_degree:
            continue
        if active_degree is not None and start_index is not None:
            intervals.append((start_index, index, active_degree))
        start_index = index if degree is not None else None
        active_degree = degree
    if not intervals:
        return False

    font_size = max(28, round(min(height * 0.085, width * 0.08)))
    margin_v = max(round(height * 0.14), font_size * 2)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TopDegree, Arial,{font_size},&H00A8F2FF,&H00FFFFFF,&H00090502,&HA0090502,1,0,0,0,100,100,0,0,4,3,0,8,24,24,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for start_index, end_index, degree in intervals:
        start = _ass_time(start_index / fps)
        end = _ass_time(end_index / fps)
        text = _ass_escape(f"Paddle Degree: {degree}")
        lines.append(f"Dialogue: 0,{start},{end},TopDegree,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return True


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ffmpeg_filter_path(path: Path) -> str:
    value = str(path)
    value = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{value}'"


def _records_for_timestamp(
    frames: dict[int, list[dict[str, Any]]],
    timestamps: list[int],
    timestamp_ms: int,
    tolerance_ms: float,
) -> list[dict[str, Any]]:
    if not timestamps:
        return []
    insertion_index = bisect_left(timestamps, timestamp_ms)
    candidates = []
    if insertion_index < len(timestamps):
        candidates.append(timestamps[insertion_index])
    if insertion_index > 0:
        candidates.append(timestamps[insertion_index - 1])
    nearest_timestamp = min(candidates, key=lambda value: abs(value - timestamp_ms))
    if abs(nearest_timestamp - timestamp_ms) > tolerance_ms:
        return []
    return frames[nearest_timestamp]


def _scale_record(record: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
    if abs(scale_x - 1) < 1e-6 and abs(scale_y - 1) < 1e-6:
        return record
    scaled = dict(record)
    for key in ("centerline_line_xyxy", "shaft_line_xyxy"):
        values = record.get(key)
        if values and len(values) == 4:
            scaled[key] = [
                float(values[0]) * scale_x,
                float(values[1]) * scale_y,
                float(values[2]) * scale_x,
                float(values[3]) * scale_y,
            ]
    return scaled


def _draw_frame_overlay(
    image: bytearray,
    width: int,
    height: int,
    records: list[dict[str, Any]],
    colors: dict[str, Color],
    *,
    fallback_degree: int | None = None,
    draw_top_label: bool = True,
) -> int | None:
    centerlines: list[tuple[dict[str, Any], Line, Color]] = []
    displayed_degree: int | None = fallback_degree
    for record in records:
        line = _record_line(record)
        if line is None:
            continue
        color = colors.get(record["prompt_id"], (53, 194, 255, 255))
        centerlines.append((record, line, color))
        _draw_line(image, width, height, line, color, max(3, round(min(width, height) * 0.006)))

    current_annotation: dict[str, float] | None = None
    for first_index in range(len(centerlines)):
        for second_index in range(first_index + 1, len(centerlines)):
            first_record, first_line, _first_color = centerlines[first_index]
            second_record, second_line, _second_color = centerlines[second_index]
            if first_record["prompt_id"] == second_record["prompt_id"]:
                continue
            annotation = _angle_annotation(first_line, second_line)
            if annotation is not None:
                current_annotation = annotation
                break
        if current_annotation is not None:
            break
    if current_annotation is not None:
        updated_degree = round(current_annotation["degrees"])
        displayed_degree = updated_degree
        _draw_angle_annotation(image, width, height, current_annotation, updated_degree)
    if displayed_degree is not None and draw_top_label:
        _draw_top_degree_label(image, width, height, displayed_degree)
    return displayed_degree


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
    displayed_degree: int | None = None,
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
    label = f"{round(annotation['degrees'] if displayed_degree is None else displayed_degree)}°"
    label_x = annotation["x"] + math.cos(mid) * (radius + 18)
    label_y = annotation["y"] + math.sin(mid) * (radius + 18)
    _draw_small_degree_label(image, width, height, label_x, label_y, label)


def _draw_top_degree_label(
    image: bytearray,
    width: int,
    height: int,
    degree: int,
) -> None:
    if _draw_top_degree_label_with_pillow(image, width, height, degree):
        return
    _draw_top_degree_label_bitmap(image, width, height, degree)


def _draw_top_degree_label_with_pillow(
    image: bytearray,
    width: int,
    height: int,
    degree: int,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    font_path = _find_export_font()
    if font_path is None:
        return False

    text = f"Paddle Degree: {degree}"
    font_size = max(24, round(min(height * 0.07, width * 0.075)))
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        return False

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _attempt in range(8):
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, round(font_size * 0.05)))
        text_width = bbox[2] - bbox[0]
        if text_width <= width * 0.86 or font_size <= 18:
            break
        font_size = round(font_size * 0.9)
        font = ImageFont.truetype(str(font_path), font_size)

    stroke_width = max(1, round(font_size * 0.05))
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x = round(font_size * 0.42)
    padding_y = round(font_size * 0.25)
    left = round(width / 2 - text_width / 2)
    top = round(max(height * 0.15, font_size * 1.8))
    box = (
        left - padding_x,
        top - padding_y,
        left + text_width + padding_x,
        top + text_height + padding_y,
    )
    radius = max(4, round(font_size * 0.16))
    draw.rounded_rectangle(box, radius=radius, fill=(2, 5, 9, 190))
    draw.text(
        (left - bbox[0], top - bbox[1]),
        text,
        font=font,
        fill=(255, 242, 168, 255),
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 9, 255),
    )
    _blend_overlay(image, width, overlay.tobytes())
    return True


def _find_export_font() -> Path | None:
    configured = os.getenv("SAM3_EXPORT_FONT_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def _draw_top_degree_label_bitmap(
    image: bytearray,
    width: int,
    height: int,
    degree: int,
) -> None:
    text = f"Paddle Degree: {degree}"
    glyphs = [_glyph(character) for character in text]
    unit_width = sum(len(glyph[0]) for glyph in glyphs) + max(0, len(glyphs) - 1) * 0.35
    glyph_height = max(len(glyph) for glyph in glyphs)
    desired_scale = max(2, round(min(height * 0.08, width * 0.12) / glyph_height))
    scale = max(2, min(desired_scale, math.floor(width * 0.84 / max(unit_width, 1))))
    gap = max(1, round(scale * 0.75))
    text_width = sum(len(glyph[0]) * scale for glyph in glyphs) + gap * (len(glyphs) - 1)
    text_height = max(len(glyph) for glyph in glyphs) * scale
    padding_x = round(scale * 1.4)
    padding_y = round(scale * 0.9)
    left = round(width / 2 - text_width / 2)
    top = round(max(height * 0.16, scale * 2.0))
    _fill_rect(
        image,
        width,
        height,
        left - padding_x,
        top - padding_y,
        text_width + padding_x * 2,
        text_height + padding_y * 2,
        (2, 5, 9, 178),
    )
    x = left
    for glyph in glyphs:
        glyph_top = top + (text_height - len(glyph) * scale) // 2
        _draw_bitmap(image, width, height, x, glyph_top, glyph, scale, (255, 242, 168, 255))
        x += len(glyph[0]) * scale + gap


def _draw_small_degree_label(
    image: bytearray,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    text: str,
) -> None:
    scale = max(2, round(width / 700))
    glyphs = [_glyph(character) for character in text]
    gap = max(1, round(scale * 0.4))
    text_width = sum(len(glyph[0]) * scale for glyph in glyphs) + gap * (len(glyphs) - 1)
    text_height = max(len(glyph) for glyph in glyphs) * scale
    padding_x = 3 * scale
    padding_y = 2 * scale
    left = round(center_x - text_width / 2)
    top = round(center_y - text_height / 2)
    _fill_rect(
        image,
        width,
        height,
        left - padding_x,
        top - padding_y,
        text_width + padding_x * 2,
        text_height + padding_y * 2,
        (2, 5, 9, 200),
    )
    x = left
    for glyph in glyphs:
        _draw_bitmap(image, width, height, x, top, glyph, scale, (255, 242, 168, 255))
        x += len(glyph[0]) * scale + gap


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


_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "d": ("00001", "00001", "00001", "01111", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    ":": ("000", "010", "010", "000", "010", "010", "000"),
    "°": ("01100", "10010", "10010", "01100", "00000", "00000", "00000"),
    " ": ("000", "000", "000", "000", "000", "000", "000"),
}


def _glyph(character: str) -> tuple[str, ...]:
    return _GLYPHS.get(character, _GLYPHS[" "])


def _draw_bitmap(
    image: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    bitmap: tuple[str, ...],
    scale: int,
    color: Color,
) -> None:
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


def _blend_overlay(image: bytearray, width: int, overlay: bytes) -> None:
    for index in range(0, len(overlay), 4):
        alpha = overlay[index + 3]
        if alpha <= 0:
            continue
        pixel = index // 4
        _blend_pixel(
            image,
            width,
            pixel % width,
            pixel // width,
            (overlay[index], overlay[index + 1], overlay[index + 2], alpha),
        )


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
