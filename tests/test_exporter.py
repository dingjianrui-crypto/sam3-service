from __future__ import annotations

import unittest

from sam3_service.exporter import (
    Centerline,
    DegreeLabel,
    ExportOptions,
    SpmEstimator,
    _degree_label_entries,
    _degree_labels,
    _metric_label_top,
    _record_line,
    _spm_label_top,
)


class ExporterTest(unittest.TestCase):
    def test_computes_degree_label_for_each_target_paddle(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "instance_id": "boat:1"},
            line=(0, 0, 100, 0),
            color=(255, 255, 255, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "instance_id": f"paddle:{index}"},
                line=(10.0 * index, 0, 10.0 * index, 50),
                color=(53, 194, 255, 255),
            )
            for index in range(1, 5)
        ]

        labels = _degree_labels(
            [reference, *targets],
            ExportOptions(
                reference_prompt_id="boat",
                target_prompt_ids=("paddle",),
            ),
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual([label.degree for label in labels], [90, 90, 90, 90])

    def test_degree_label_entries_use_short_text_and_highlight_outlier(self) -> None:
        labels = [
            DegreeLabel(
                instance_id=f"paddle:{index}",
                degree=degree,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            )
            for index, degree in enumerate([40, 42, 41, 75], start=1)
        ]

        entries = _degree_label_entries(labels)

        self.assertEqual(
            [entry.text for entry in entries],
            ["1: 40°", "2: 42°", "3: 41°", "4: 75°"],
        )
        self.assertEqual(entries[3].text_color, (255, 82, 96, 255))
        self.assertTrue(all(entry.text_color != (255, 82, 96, 255) for entry in entries[:3]))

    def test_single_degree_label_omits_index(self) -> None:
        entries = _degree_label_entries(
            [
                DegreeLabel(
                    instance_id="paddle:1",
                    degree=42,
                    line=(0, 0, 1, 1),
                    color=(53, 194, 255, 255),
                )
            ]
        )

        self.assertEqual([entry.text for entry in entries], ["42°"])

    def test_record_line_scales_rle_centerline_coordinates_to_output_size(self) -> None:
        line = _record_line(
            {
                "centerline_line_xyxy": [10, 20, 30, 40],
                "centerline_segmentation": {
                    "type": "rle",
                    "size": [100, 200],
                    "counts": [],
                },
            },
            400,
            300,
        )

        self.assertEqual(line, (20, 60, 60, 120))

    def test_spm_estimator_reports_instant_and_average(self) -> None:
        estimator = SpmEstimator(window_ms=5000)
        estimate = None
        for timestamp_ms, degree in [
            (0, 40),
            (1000, 60),
            (2000, 40),
            (3000, 60),
            (4000, 40),
        ]:
            estimate = estimator.update(
                timestamp_ms,
                [
                    DegreeLabel(
                        instance_id="paddle:1",
                        degree=degree,
                        line=(0, 0, 1, 1),
                        color=(53, 194, 255, 255),
                    )
                ],
            )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(round(estimate.instantaneous or 0), 60)
        self.assertEqual(round(estimate.average or 0), 60)

    def test_spm_label_uses_opposite_side_from_angle_label(self) -> None:
        self.assertGreater(
            _spm_label_top(100, 100, 10, 12, ExportOptions(angle_label_position="top")),
            70,
        )
        self.assertLess(
            _spm_label_top(100, 100, 10, 12, ExportOptions(angle_label_position="bottom")),
            30,
        )

    def test_portrait_metric_labels_move_toward_center(self) -> None:
        landscape_top = _metric_label_top(1920, 1080, 30, 32, "top")
        portrait_top = _metric_label_top(1080, 1920, 30, 32, "top")
        portrait_bottom = _metric_label_top(1080, 1920, 30, 32, "bottom")

        self.assertGreater(portrait_top, landscape_top)
        self.assertGreater(portrait_bottom, 1500)
        self.assertLess(portrait_bottom, 1700)

    def test_metric_center_offset_percent_controls_both_positions(self) -> None:
        self.assertEqual(_metric_label_top(1000, 1000, 10, 12, "top", 25), 250)
        self.assertEqual(_metric_label_top(1000, 1000, 10, 12, "bottom", 25), 740)
        self.assertEqual(
            _spm_label_top(
                1000,
                1000,
                10,
                12,
                ExportOptions(
                    angle_label_position="top",
                    metric_center_offset_percent=25,
                ),
            ),
            740,
        )


if __name__ == "__main__":
    unittest.main()
