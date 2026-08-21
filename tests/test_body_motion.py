from __future__ import annotations

import json
import math
import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import Mock, patch

from sam3_service.body_motion import (
    BodyMotionFrame,
    Sapiens2BodyMotionAnalyzer,
    _select_primary_bbox,
    _sapiens2_selected_landmarks,
    build_body_motion_record,
    create_body_motion_analyzer,
    load_body_motion_frames_by_index,
    smooth_body_motion_records,
)
from sam3_service.config import Settings
from sam3_service.errors import ServiceError
from sam3_service.exporter import (
    PADDLE_ANGLE_TEXT_COLOR,
    _draw_body_metric_row,
    _draw_body_motion_overlay,
)
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

    def test_sapiens2_maps_named_body_joints_and_discards_low_scores(self) -> None:
        names = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_hip": 9,
            "right_hip": 10,
            "left_knee": 11,
            "right_knee": 12,
            "left_ankle": 13,
            "right_ankle": 14,
            "right_wrist": 41,
            "left_wrist": 62,
        }
        keypoints = [[float(index), float(index * 2)] for index in range(308)]
        scores = [0.9] * 308
        scores[41] = 0.49

        landmarks = _sapiens2_selected_landmarks(
            keypoints, scores, names, width=100, height=200, threshold=0.5
        )

        self.assertEqual(set(landmarks), set(names) - {"right_wrist"})
        self.assertEqual(landmarks["left_wrist"]["x"], 0.62)
        self.assertEqual(landmarks["left_wrist"]["y"], 0.62)
        self.assertEqual(landmarks["left_wrist"]["z"], 0.0)
        self.assertEqual(landmarks["left_wrist"]["visibility"], 0.9)

    def test_sapiens2_primary_bbox_prefers_area_then_previous_overlap(self) -> None:
        boxes = [[5, 5, 25, 25], [40, 20, 90, 90]]
        first = _select_primary_bbox(boxes, None, (100, 100))
        continued = _select_primary_bbox(boxes, [0, 0, 30, 30], (100, 100))

        self.assertEqual(first, boxes[1])
        self.assertEqual(continued, boxes[0])
        self.assertIsNone(_select_primary_bbox([], None, (100, 100)))

    def test_sapiens2_analyzer_requires_local_assets(self) -> None:
        analyzer = Sapiens2BodyMotionAnalyzer(None, None)

        with self.assertRaises(ServiceError) as raised:
            list(analyzer.analyze(Path("missing.mp4"), {}, Mock(), lambda: False))

        self.assertEqual(raised.exception.code, "POSE_MODEL_UNAVAILABLE")

    def test_factory_selects_sapiens2_without_loading_the_model(self) -> None:
        analyzer = create_body_motion_analyzer(
            "sapiens2",
            None,
            sapiens2_checkpoint_path=Path("pose.safetensors"),
            sapiens2_detector_path=Path("detector"),
        )

        self.assertIsInstance(analyzer, Sapiens2BodyMotionAnalyzer)
        self.assertEqual(analyzer.model_name, "sapiens2-pose-1b:pose.safetensors")

    def test_sapiens2_settings_are_read_from_environment(self) -> None:
        values = {
            "SAM3_SAPIENS2_CHECKPOINT_PATH": "models/pose.safetensors",
            "SAM3_SAPIENS2_DETECTOR_PATH": "models/detector",
            "SAM3_SAPIENS2_CONFIG_PATH": "models/config.py",
            "SAM3_SAPIENS2_DEVICE": "cuda:1",
            "SAM3_SAPIENS2_KEYPOINT_THRESHOLD": "0.6",
            "SAM3_SAPIENS2_BBOX_THRESHOLD": "0.4",
            "SAM3_SAPIENS2_NMS_THRESHOLD": "0.2",
        }
        with patch.dict(environ, values, clear=False):
            settings = Settings.from_env()

        self.assertEqual(
            settings.sapiens2_checkpoint_path, Path("models/pose.safetensors").resolve()
        )
        self.assertEqual(settings.sapiens2_detector_path, Path("models/detector").resolve())
        self.assertEqual(settings.sapiens2_config_path, Path("models/config.py").resolve())
        self.assertEqual(settings.sapiens2_device, "cuda:1")
        self.assertEqual(settings.sapiens2_keypoint_threshold, 0.6)
        self.assertEqual(settings.sapiens2_bbox_threshold, 0.4)
        self.assertEqual(settings.sapiens2_nms_threshold, 0.2)

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
        self.assertTrue(
            all(entry[2] == PADDLE_ANGLE_TEXT_COLOR for entry in entries)
        )
        self.assertEqual(draw_row.call_args.args[4], 10.0)
        self.assertNotIn("Hip", [entry[0] for entry in entries])
        self.assertNotIn("Knee", [entry[0] for entry in entries])

    @patch("sam3_service.exporter._draw_body_joint_arc")
    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_adds_left_and_right_knee_metrics_for_canoe(
        self, draw_row: Mock, draw_arc: Mock
    ) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(
            image, 200, 120, record, discipline="canoe"
        )

        entries = draw_row.call_args.args[3]
        self.assertEqual(
            [entry[0] for entry in entries],
            [
                "左肘",
                "右肘",
                "躯干",
                "左肩",
                "右肩",
                "左膝",
                "右膝",
            ],
        )
        values = {entry[0]: entry[1] for entry in entries}
        self.assertEqual(values["左膝"], "180°")
        self.assertEqual(values["右膝"], "180°")
        knee_calls = [
            call
            for call in draw_arc.call_args_list
            if call.args[5] == ("hip", "knee", "ankle")
        ]
        self.assertEqual(len(knee_calls), 2)

    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_keeps_missing_canoe_knee_metric_in_fixed_slot(
        self, draw_row: Mock
    ) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        record["metrics"].pop("left_knee_deg")
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(
            image, 200, 120, record, discipline="canoe"
        )

        values = {entry[0]: entry[1] for entry in draw_row.call_args.args[3]}
        self.assertEqual(values["左膝"], "--")
        self.assertEqual(values["右膝"], "180°")

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

    @patch("sam3_service.exporter._draw_body_metric_row")
    def test_export_passes_angle_font_size_to_body_metrics(
        self, draw_row: Mock
    ) -> None:
        record = build_body_motion_record(_frame(), (0.1, 0.7, 0.9, 0.7))
        image = bytearray([0, 0, 0, 255] * 200 * 120)

        _draw_body_motion_overlay(
            image,
            200,
            120,
            record,
            discipline="canoe",
            angle_label_font_size=32,
        )

        self.assertEqual(draw_row.call_args.kwargs["angle_label_font_size"], 32)

    def test_body_metric_row_uses_angle_title_and_value_font_sizes(self) -> None:
        try:
            from PIL import ImageFont
        except ImportError:
            self.skipTest("Pillow is not installed")
        image = bytearray([0, 0, 0, 255] * 400 * 120)
        default_font = ImageFont.load_default()
        requested_sizes: list[int] = []

        def fake_truetype(_path: str, size: int):
            requested_sizes.append(size)
            return default_font

        with (
            patch("sam3_service.exporter._find_export_font", return_value=Path("font.ttf")),
            patch("PIL.ImageFont.truetype", side_effect=fake_truetype),
        ):
            _draw_body_metric_row(
                image,
                400,
                120,
                [("左肘", "80°", PADDLE_ANGLE_TEXT_COLOR)],
                10.0,
                angle_label_font_size=32,
                draw_background=False,
            )

        self.assertEqual(requested_sizes[:2], [32, 23])


if __name__ == "__main__":
    unittest.main()
