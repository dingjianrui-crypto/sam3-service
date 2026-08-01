from __future__ import annotations

import unittest

from sam3_service.exporter import (
    Centerline,
    DegreeLabel,
    ExportOptions,
    SpmEstimator,
    _degree_label_entries,
    _degree_labels,
    _degree_slots,
    _metric_label_top,
    _maximum_target_count_in_selection,
    _record_line,
    _record_selected_for_export,
    _resolve_requested_track_ids,
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

    def test_export_instance_selection_filters_paddles_and_boats(self) -> None:
        options = ExportOptions(
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
            reference_track_ids=("boat:track:2",),
            target_track_ids=("paddle:track:1",),
        )

        self.assertFalse(
            _record_selected_for_export(
                {"prompt_id": "boat", "instance_id": "boat:1", "track_id": "boat:track:1"}, options
            )
        )
        self.assertTrue(
            _record_selected_for_export(
                {"prompt_id": "boat", "instance_id": "boat:7", "track_id": "boat:track:2"}, options
            )
        )
        self.assertTrue(
            _record_selected_for_export(
                {"prompt_id": "paddle", "instance_id": "paddle:4", "track_id": "paddle:track:1"}, options
            )
        )
        self.assertFalse(
            _record_selected_for_export(
                {"prompt_id": "paddle", "instance_id": "paddle:2", "track_id": "paddle:track:2"}, options
            )
        )

    def test_degree_labels_use_only_selected_instances(self) -> None:
        centerlines = [
            Centerline(
                record={"prompt_id": "boat", "instance_id": "boat:1"},
                line=(0, 0, 100, 0),
                color=(255, 255, 255, 255),
            ),
            Centerline(
                record={"prompt_id": "boat", "instance_id": "boat:2"},
                line=(0, 0, 0, 100),
                color=(255, 255, 255, 255),
            ),
            Centerline(
                record={"prompt_id": "paddle", "instance_id": "paddle:1"},
                line=(10, 0, 10, 50),
                color=(53, 194, 255, 255),
            ),
            Centerline(
                record={"prompt_id": "paddle", "instance_id": "paddle:2"},
                line=(90, 0, 90, 50),
                color=(53, 194, 255, 255),
            ),
        ]

        labels = _degree_labels(
            centerlines,
            ExportOptions(
                reference_prompt_id="boat",
                target_prompt_ids=("paddle",),
                reference_track_ids=("boat:1",),
                target_track_ids=("paddle:2",),
            ),
        )

        self.assertEqual([label.instance_id for label in labels], ["paddle:2"])
        self.assertEqual([label.degree for label in labels], [90])

    def test_export_rectangle_filters_by_centerline_center(self) -> None:
        options = ExportOptions(selection_rect=(0.1, 0.1, 0.3, 0.3))

        self.assertTrue(
            _record_selected_for_export(
                {"centerline_line_xyxy": [10, 20, 30, 20]}, options, 100, 100
            )
        )
        self.assertFalse(
            _record_selected_for_export(
                {"centerline_line_xyxy": [70, 20, 90, 20]}, options, 100, 100
            )
        )

    def test_selection_uses_maximum_paddle_count_for_fixed_slots(self) -> None:
        options = ExportOptions(
            target_prompt_ids=("paddle",),
            selection_rect=(0, 0, 0.5, 1),
        )
        frames = {
            0: [
                {"prompt_id": "paddle", "centerline_line_xyxy": [10, 10, 20, 20]},
                {"prompt_id": "paddle", "centerline_line_xyxy": [30, 10, 40, 20]},
            ],
            100: [
                {"prompt_id": "paddle", "centerline_line_xyxy": [10, 10, 20, 20]},
            ],
        }

        count = _maximum_target_count_in_selection(
            frames, options, 100, 100, 1, 1
        )

        self.assertEqual(count, 2)

    def test_raw_instance_selection_resolves_to_stable_track(self) -> None:
        resolved = _resolve_requested_track_ids(
            ("boat:7",),
            [
                {
                    "id": "boat:track:1",
                    "prompt_id": "boat",
                    "instance_ids": ["boat:1", "boat:7"],
                }
            ],
        )

        self.assertEqual(resolved, ("boat:track:1",))

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

    def test_degree_slots_keep_missing_tracks_visible(self) -> None:
        labels = [
            DegreeLabel(
                instance_id="paddle:track:1",
                degree=42,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            ),
            DegreeLabel(
                instance_id="paddle:track:3",
                degree=51,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            ),
        ]

        slots = _degree_slots(
            labels,
            ExportOptions(
                target_track_ids=tuple(f"paddle:track:{index}" for index in range(1, 5))
            ),
        )
        entries = _degree_label_entries(slots)

        self.assertEqual([slot.degree for slot in slots], [42, None, 51, None])
        self.assertEqual(
            [entry.text for entry in entries],
            ["1: 42°", "2: Missing", "3: 51°", "4: Missing"],
        )
        self.assertEqual(entries[1].text_color, (148, 163, 184, 255))

    def test_missing_degrees_are_excluded_from_outlier_highlight(self) -> None:
        labels = [
            DegreeLabel(
                instance_id=f"paddle:track:{index}",
                degree=degree,
                line=None if degree is None else (0, 0, 1, 1),
                color=(53, 194, 255, 255),
            )
            for index, degree in enumerate([40, 42, 41, None], start=1)
        ]

        entries = _degree_label_entries(labels)

        self.assertTrue(all(entry.text_color != (255, 82, 96, 255) for entry in entries))

    def test_rectangle_degree_slots_use_stable_positions(self) -> None:
        labels = [
            DegreeLabel(
                instance_id="raw:7",
                degree=42,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            )
        ]

        slots = _degree_slots(labels, ExportOptions(target_slot_count=3))

        self.assertEqual([slot.instance_id for slot in slots], ["slot:1", "slot:2", "slot:3"])
        self.assertEqual([slot.degree for slot in slots], [42, None, None])

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
            (5000, 60),
            (6000, 40),
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

    def test_spm_estimator_suppresses_startup_wiggles(self) -> None:
        estimator = SpmEstimator(window_ms=5000)
        estimate = None
        for timestamp_ms, degree in [
            (0, 40),
            (250, 55),
            (500, 40),
            (750, 55),
            (1000, 40),
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
        self.assertIsNone(estimate.instantaneous)
        self.assertIsNone(estimate.average)

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
