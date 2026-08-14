from __future__ import annotations

import unittest

from sam3_service.paddle_completeness import normalize_paddle_mask_record


def _rle(width: int, height: int, points: set[tuple[int, int]]) -> dict:
    counts: list[int] = []
    current = 0
    run = 0
    for x in range(width):
        for y in range(height):
            value = int((x, y) in points)
            if value == current:
                run += 1
            else:
                counts.append(run)
                current = value
                run = 1
    counts.append(run)
    return {"type": "rle", "size": [height, width], "counts": counts}


class PaddleCompletenessTest(unittest.TestCase):
    def test_normalization_returns_binary_256_square_without_stretching(self) -> None:
        points = {
            (x, y)
            for x in range(20, 80)
            for y in range(42, 58)
        }
        record = {
            "box_xywh": [20, 42, 60, 16],
            "segmentation": _rle(100, 100, points),
        }

        normalized = normalize_paddle_mask_record(record)

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(len(normalized), 256 * 256)
        self.assertEqual(set(normalized), {0, 1})
        foreground = [
            (index % 256, index // 256)
            for index, value in enumerate(normalized)
            if value
        ]
        width = max(x for x, _ in foreground) - min(x for x, _ in foreground) + 1
        height = max(y for _, y in foreground) - min(y for _, y in foreground) + 1
        self.assertGreater(width / height, 3.5)

    def test_normalization_zero_pads_crop_outside_source_frame(self) -> None:
        points = {(x, y) for x in range(0, 12) for y in range(0, 4)}
        record = {
            "box_xywh": [0, 0, 12, 4],
            "segmentation": _rle(20, 20, points),
        }

        normalized = normalize_paddle_mask_record(record)

        self.assertIsNotNone(normalized)
        assert normalized is not None
        foreground_x = [index % 256 for index, value in enumerate(normalized) if value]
        self.assertGreater(min(foreground_x), 0)

    def test_invalid_or_incomplete_rle_is_not_classified(self) -> None:
        record = {
            "box_xywh": [0, 0, 10, 10],
            "segmentation": {
                "type": "rle",
                "size": [10, 10],
                "counts": [50],
            },
        }

        self.assertIsNone(normalize_paddle_mask_record(record))


if __name__ == "__main__":
    unittest.main()
