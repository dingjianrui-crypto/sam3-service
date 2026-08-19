from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sam3_service.body_motion import (
    BodyMotionFrame,
    build_body_motion_record,
    load_body_motion_frames_by_index,
    smooth_body_motion_records,
)
from sam3_service.exporter import _draw_body_motion_overlay
from sam3_service.schemas import JobSettings


def _landmark(x: float, y: float, confidence: float = 0.99) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "z": 0.0,
        "visibility": confidence,
        "presence": confidence,
    }


def _frame(shoulder_x: float = 0.6) -> BodyMotionFrame:
    landmarks = {
        "left_shoulder": _landmark(shoulder_x, 0.3),
        "right_shoulder": _landmark(shoulder_x + 0.04, 0.3),
        "left_elbow": _landmark(0.5, 0.4),
        "right_elbow": _landmark(0.54, 0.4),
        "left_wrist": _landmark(0.5, 0.5),
        "right_wrist": _landmark(0.54, 0.5),
        "left_hip": _landmark(0.5, 0.6),
        "right_hip": _landmark(0.54, 0.6),
        "left_knee": _landmark(0.5, 0.75),
        "right_knee": _landmark(0.54, 0.75),
        "left_ankle": _landmark(0.5, 0.9),
        "right_ankle": _landmark(0.54, 0.9),
    }
    return BodyMotionFrame(frame_index=0, timestamp_ms=0, landmarks=landmarks)


class BodyMotionTest(unittest.TestCase):
    def test_job_setting_is_backward_compatible_and_disabled_by_default(self) -> None:
        settings = JobSettings()

        self.assertFalse(settings.body_motion)
        self.assertEqual(settings.paddling_discipline, "kayak")
        self.assertEqual(
            JobSettings(paddling_discipline="canoe").paddling_discipline,
            "canoe",
        )

    def test_derives_joint_angles_and_signed_vertical_lean(self) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))

        self.assertAlmostEqual(record["metrics"]["left_knee_deg"], 180.0)
        self.assertAlmostEqual(record["metrics"]["right_knee_deg"], 180.0)
        expected_lean = math.degrees(math.atan2(0.1, 0.3))
        self.assertAlmostEqual(record["metrics"]["lean_deg"], expected_lean, places=1)
        self.assertIn(record["primary_side"], {"left", "right"})

    def test_reference_line_direction_is_canonicalized(self) -> None:
        forward = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        reversed_line = build_body_motion_record(_frame(), (0.9, 0.7, 0.1, 0.7))

        self.assertEqual(
            forward["metrics"]["lean_deg"],
            reversed_line["metrics"]["lean_deg"],
        )

    def test_low_confidence_joint_is_omitted(self) -> None:
        frame = _frame()
        frame.landmarks["left_wrist"]["visibility"] = 0.1

        record = build_body_motion_record(frame, (0.1, 0.7, 0.9, 0.7))

        self.assertNotIn("left_elbow_deg", record["metrics"])
        self.assertIn("right_elbow_deg", record["metrics"])

    def test_smoothing_does_not_bridge_long_gaps(self) -> None:
        records = [
            {"timestamp_ms": 0, "metrics": {"lean_deg": 0.0}},
            {"timestamp_ms": 100, "metrics": {"lean_deg": 10.0}},
            {"timestamp_ms": 500, "metrics": {"lean_deg": 20.0}},
        ]

        smooth_body_motion_records(records)

        self.assertEqual(records[1]["metrics"]["lean_deg"], 3.5)
        self.assertEqual(records[2]["metrics"]["lean_deg"], 20.0)

    def test_body_chunks_load_by_exact_source_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.json"
            path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"frame_index": 4, "timestamp_ms": 133},
                            {"frame_index": 5, "timestamp_ms": 167},
                        ]
                    }
                )
            )

            frames = load_body_motion_frames_by_index([path])

        self.assertEqual(frames[4]["timestamp_ms"], 133)
        self.assertEqual(frames[5]["timestamp_ms"], 167)

    def test_export_overlay_draws_visible_skeleton(self) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(image, 200, 120, record)

        self.assertTrue(any(image[index] for index in range(0, len(image), 4)))

    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_shows_only_upper_body_metrics(self, draw_row: Mock) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(image, 200, 120, record)

        draw_row.assert_called_once()
        entries = draw_row.call_args.args[3]
        self.assertEqual(
            [entry[0] for entry in entries],
            ["L Elbow", "R Elbow", "Torso", "L Shoulder", "R Shoulder"],
        )
        self.assertEqual(len({entry[2] for entry in entries}), 5)
        self.assertEqual(draw_row.call_args.args[4], 10.0)
        self.assertNotIn("Hip", [entry[0] for entry in entries])
        self.assertNotIn("Knee", [entry[0] for entry in entries])

    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_keeps_missing_body_metrics_in_fixed_slots(
        self, draw_row: Mock
    ) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        record["metrics"].pop("left_elbow_deg")
        record["metrics"].pop("lean_deg")
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(image, 200, 120, record)

        values = {entry[0]: entry[1] for entry in draw_row.call_args.args[3]}
        self.assertEqual(values["L Elbow"], "--")
        self.assertEqual(values["Torso"], "--")
        self.assertNotEqual(values["R Elbow"], "--")

    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_passes_body_metric_offset(self, draw_row: Mock) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(
            image, 200, 120, record, metric_offset_percent=-12.5
        )

        self.assertEqual(draw_row.call_args.args[4], -12.5)


if __name__ == "__main__":
    unittest.main()
