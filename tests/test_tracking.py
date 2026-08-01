from __future__ import annotations

import unittest

from sam3_service.tracking import StableTrackAssigner


def _record(timestamp_ms: int, instance_id: str, x: float) -> dict[str, object]:
    return {
        "timestamp_ms": timestamp_ms,
        "prompt_id": "boat",
        "instance_id": instance_id,
        "box_xywh": [x, 20, 100, 30],
        "centerline_line_xyxy": [x, 35, x + 100, 35],
    }


class StableTrackAssignerTest(unittest.TestCase):
    def test_keeps_track_when_raw_instance_id_changes(self) -> None:
        records = [
            _record(0, "boat:1", 100),
            _record(100, "boat:7", 104),
            _record(200, "boat:9", 109),
        ]

        assigner = StableTrackAssigner()
        assigner.assign(records)

        self.assertEqual(
            {record["track_id"] for record in records},
            {"boat:track:1"},
        )
        self.assertEqual(
            assigner.entries()[0]["instance_ids"],
            ["boat:1", "boat:7", "boat:9"],
        )

    def test_new_boat_does_not_rename_existing_middle_boat(self) -> None:
        existing_first = _record(0, "boat:1", 200)
        entering = _record(100, "boat:1", 20)
        existing_second = _record(100, "boat:2", 204)
        records = [existing_first, entering, existing_second]

        assigner = StableTrackAssigner()
        assigner.assign(records)

        self.assertEqual(existing_first["track_id"], "boat:track:1")
        self.assertEqual(existing_second["track_id"], "boat:track:1")
        self.assertEqual(entering["track_id"], "boat:track:2")

    def test_reconnects_after_short_detection_gap(self) -> None:
        first = _record(0, "boat:1", 100)
        after_gap = _record(900, "boat:8", 112)

        assigner = StableTrackAssigner()
        assigner.assign([first, after_gap])

        self.assertEqual(first["track_id"], after_gap["track_id"])

    def test_creates_new_track_after_long_gap(self) -> None:
        first = _record(0, "boat:1", 100)
        after_gap = _record(2000, "boat:8", 100)

        assigner = StableTrackAssigner()
        assigner.assign([first, after_gap])

        self.assertNotEqual(first["track_id"], after_gap["track_id"])


if __name__ == "__main__":
    unittest.main()
