from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from PIL import Image


PADDLE_COMPLETENESS_INPUT_SIZE = 256
PADDLE_COMPLETENESS_DEFAULT_PADDING_PERCENT = 15.0
PADDLE_COMPLETENESS_DEFAULT_BATCH_SIZE = 8


class PaddleCompletenessPredictor:
    """Lazy PyTorch adapter for the mask-only paddle completeness CNN."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        padding_percent: float = PADDLE_COMPLETENESS_DEFAULT_PADDING_PERCENT,
        batch_size: int = PADDLE_COMPLETENESS_DEFAULT_BATCH_SIZE,
    ) -> None:
        import torch
        from torch import nn

        class PaddleCompletenessCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                )

            def forward(self, inputs: Any) -> Any:
                return self.classifier(self.features(inputs))

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state"), dict
        ):
            raise ValueError("Paddle completeness checkpoint has no model_state.")
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Paddle completeness checkpoint has no config.")
        architecture = config.get("architecture")
        if architecture != "paddle_cnn":
            raise ValueError(
                f"Unsupported paddle completeness architecture: {architecture!r}."
            )
        threshold = float(config.get("decision_threshold", 0.5))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Paddle completeness threshold is invalid.")

        model = PaddleCompletenessCNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.torch = torch
        self.model = model
        self.threshold = threshold
        self.padding_percent = max(0.0, min(100.0, padding_percent))
        self.batch_size = max(1, min(32, batch_size))

    def predict_records(self, records: list[dict[str, Any]]) -> list[float | None]:
        results: list[float | None] = [None] * len(records)
        normalized: list[tuple[int, bytes]] = []
        for index, record in enumerate(records):
            mask = normalize_paddle_mask_record(
                record,
                padding_percent=self.padding_percent,
            )
            if mask is not None:
                normalized.append((index, mask))

        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            tensors = [
                self.torch.tensor(list(mask), dtype=self.torch.float32).reshape(
                    1,
                    PADDLE_COMPLETENESS_INPUT_SIZE,
                    PADDLE_COMPLETENESS_INPUT_SIZE,
                )
                for _, mask in batch
            ]
            inputs = self.torch.stack(tensors)
            with self.torch.inference_mode():
                probabilities = self.torch.sigmoid(self.model(inputs).squeeze(1)).tolist()
            for (record_index, _), probability in zip(
                batch,
                probabilities,
                strict=True,
            ):
                results[record_index] = float(probability)
        return results


def build_paddle_completeness_predictor() -> PaddleCompletenessPredictor | None:
    checkpoint_path = _paddle_completeness_checkpoint_path()
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        padding_percent = float(
            os.getenv(
                "SAM3_PADDLE_COMPLETENESS_PADDING_PERCENT",
                str(PADDLE_COMPLETENESS_DEFAULT_PADDING_PERCENT),
            )
        )
        batch_size = int(
            os.getenv(
                "SAM3_PADDLE_COMPLETENESS_BATCH_SIZE",
                str(PADDLE_COMPLETENESS_DEFAULT_BATCH_SIZE),
            )
        )
        return PaddleCompletenessPredictor(
            checkpoint_path,
            padding_percent=padding_percent,
            batch_size=batch_size,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # Export remains available when the optional model/runtime is absent or
        # incompatible. Existing geometry/appearance/temporal restoration is
        # the intentional fallback.
        return None


def normalize_paddle_mask_record(
    record: dict[str, Any],
    *,
    padding_percent: float = PADDLE_COMPLETENESS_DEFAULT_PADDING_PERCENT,
    output_size: int = PADDLE_COMPLETENESS_INPUT_SIZE,
) -> bytes | None:
    """Reproduce the training repository's square mask normalization."""
    segmentation = record.get("segmentation")
    if not isinstance(segmentation, dict) or segmentation.get("type") != "rle":
        return None
    decoded = _decode_rle_points(segmentation)
    if decoded is None:
        return None
    width, height, points = decoded
    if not points or output_size <= 0:
        return None

    box = record.get("box_xywh")
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (x, y, box_width, box_height))
        or box_width <= 0
        or box_height <= 0
    ):
        return None

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center_x = (min_x + max_x + 1) / 2
    center_y = (min_y + max_y + 1) / 2
    padding = max(0.0, min(100.0, padding_percent))
    padded_left = x - box_width * padding / 100
    padded_top = y - box_height * padding / 100
    padded_right = x + box_width * (1 + padding / 100)
    padded_bottom = y + box_height * (1 + padding / 100)
    half_side = max(
        center_x - padded_left,
        padded_right - center_x,
        center_y - padded_top,
        padded_bottom - center_y,
    )
    side = max(1, math.ceil(half_side * 2))
    if side > max(width, height) * 4:
        return None
    left = math.floor(center_x - side / 2)
    top = math.floor(center_y - side / 2)
    if left + side <= 0 or top + side <= 0 or left >= width or top >= height:
        return None

    crop = bytearray(side * side)
    for point_x, point_y in points:
        crop_x = point_x - left
        crop_y = point_y - top
        if 0 <= crop_x < side and 0 <= crop_y < side:
            crop[crop_y * side + crop_x] = 255
    image = Image.frombytes("L", (side, side), bytes(crop))
    resized = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    return bytes(1 if pixel else 0 for pixel in resized.get_flattened_data())


def _decode_rle_points(
    segmentation: dict[str, Any],
) -> tuple[int, int, set[tuple[int, int]]] | None:
    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if (
        not isinstance(size, list | tuple)
        or len(size) != 2
        or not isinstance(counts, list | tuple)
    ):
        return None
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    points: set[tuple[int, int]] = set()
    position = 0
    foreground = False
    area = width * height
    for raw_count in counts:
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            return None
        if count < 0 or position + count > area:
            return None
        if foreground:
            for flat_index in range(position, position + count):
                points.add((flat_index // height, flat_index % height))
        position += count
        foreground = not foreground
    if position != area:
        return None
    return width, height, points


def _paddle_completeness_checkpoint_path() -> Path | None:
    configured = os.getenv("SAM3_PADDLE_COMPLETENESS_CHECKPOINT_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    default = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"
    return default if default.is_file() else None
