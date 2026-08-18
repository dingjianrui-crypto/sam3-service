from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sam3_service.exporter import (
    Centerline,
    DegreeLabel,
    EventMetricColumn,
    ExportOptions,
    PaddleEvent,
    SpmEstimator,
    _PaddleObservation,
    _PaddleEventState,
    _TimedPaddleObservation,
    _advance_directed_paddle_phase,
    _aligned_companion_degree,
    _blade_transition_on_waterline,
    _blade_waterline_overlaps,
    _catch_phase_allowed,
    _consolidate_paddle_observations,
    _dedupe_paddle_events,
    _detect_paddle_events,
    _draw_frame_overlay,
    _draw_paddle_event_label,
    _degree_label_entries,
    _degree_labels,
    _degree_slots,
    _directed_blade_angle,
    _estimate_paddle_direction,
    _event_phase_allowed,
    _event_label_text,
    _event_companion_degree_slots,
    _event_frame_degree_slots,
    _event_freeze_frame_count,
    _event_metric_table_top,
    _event_metric_text,
    _event_metric_values,
    _event_boat_reference_lines,
    _freeze_audio_filter,
    _freeze_moments,
    _freeze_segments,
    _inherit_stroke_phase_length,
    _line_intersection,
    _line_length,
    _load_frames_by_index,
    _kalman_stabilized_boat_reference_lines,
    _metric_label_top,
    _maximum_target_count_in_selection,
    _normalize_export_options,
    _observation_forward_position,
    _paddle_water_depth_ratio,
    _paddle_rotation_deltas,
    _paddle_event_angle_color,
    _paddle_candidate_validator,
    _predict_event_paddle_completeness,
    _record_line,
    _record_selected_for_export,
    _restore_immersed_paddle_length,
    _resolve_requested_track_ids,
    _resolved_export_frame_count,
    _restore_bidirectional_phase_lines,
    _restore_track_stroke_lengths,
    _selection_rect_at,
    _select_event_paddle_tracks,
    _spm_label_top,
    _stabilize_boat_reference_lengths,
    _update_directed_paddle_state,
    _update_paddle_event_state,
    _waterline_transition_kind,
)


def _rotating_observation(
    angle: float,
    *,
    travel_direction: str = "right",
) -> _PaddleObservation:
    radians = math.radians(angle)
    forward_x = 1.0 if travel_direction == "right" else -1.0
    vector = (30 * math.cos(radians) * forward_x, 30 * math.sin(radians))
    center = (50.0, 30.0)
    return _PaddleObservation(
        source_ids=("paddle:1",),
        reference_id="boat:1",
        line=(
            center[0] - vector[0],
            center[1] - vector[1],
            center[0] + vector[0],
            center[1] + vector[1],
        ),
        reference_line=(0.0, 50.0, 100.0, 50.0),
    )


def _axis_track(angles: list[float]) -> list[_TimedPaddleObservation]:
    observations: list[_TimedPaddleObservation] = []
    for index, angle in enumerate(angles):
        radians = math.radians(angle)
        vector = (30 * math.cos(radians), 30 * math.sin(radians))
        observation = _PaddleObservation(
            source_ids=("paddle:1",),
            reference_id="boat:1",
            line=(50 - vector[0], 30 - vector[1], 50 + vector[0], 30 + vector[1]),
            reference_line=(0.0, 50.0, 100.0, 50.0),
        )
        observations.append(
            _TimedPaddleObservation(index * 100, "paddle:physical:1", observation)
        )
    return observations


class ExporterTest(unittest.TestCase):
    def test_export_frame_count_prefers_decoded_count_over_rounded_duration(self) -> None:
        metadata = {"frame_count": 563, "duration_ms": 18767}

        self.assertEqual(_resolved_export_frame_count(metadata, 30.0), 563)

    def test_export_frame_count_uses_rounded_duration_only_as_fallback(self) -> None:
        metadata = {"frame_count": 0, "duration_ms": 18767}

        self.assertEqual(_resolved_export_frame_count(metadata, 30.0), 563)

    def test_loads_export_records_by_exact_source_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "chunk.json"
            chunk_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"frame_index": 561, "timestamp_ms": 18700, "id": "previous"},
                            {"frame_index": 562, "timestamp_ms": 18733, "id": "final"},
                        ]
                    }
                )
            )

            frames = _load_frames_by_index([chunk_path])

        self.assertIsNotNone(frames)
        assert frames is not None
        self.assertEqual([record["id"] for record in frames[561]], ["previous"])
        self.assertEqual([record["id"] for record in frames[562]], ["final"])
        self.assertEqual(frames.get(560, []), [])

    def test_legacy_chunks_without_frame_indices_use_timestamp_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "chunk.json"
            chunk_path.write_text(
                json.dumps({"frames": [{"timestamp_ms": 18700, "id": "legacy"}]})
            )

            frames = _load_frames_by_index([chunk_path])

        self.assertIsNone(frames)

    def test_rotation_direction_maps_to_kayak_travel_direction(self) -> None:
        clockwise = _estimate_paddle_direction(
            _paddle_rotation_deltas(_axis_track([0, 15, 30, 45, 60, 75, 90]))
        )
        anticlockwise = _estimate_paddle_direction(
            _paddle_rotation_deltas(_axis_track([170, 155, 140, 125, 110, 95, 80]))
        )

        self.assertEqual(clockwise[:2], ("clockwise", "right"))
        self.assertEqual(anticlockwise[:2], ("anticlockwise", "left"))
        self.assertGreaterEqual(clockwise[2], 0.75)
        self.assertGreaterEqual(anticlockwise[2], 0.75)

    def test_mirrored_travel_uses_the_same_normalized_phase_angle(self) -> None:
        for angle in [0, 30, 90, 180, 270, 350]:
            right = _rotating_observation(angle, travel_direction="right")
            left = _rotating_observation(angle, travel_direction="left")

            self.assertAlmostEqual(
                _directed_blade_angle(right.line, right.reference_line, 1, "right"),
                angle,
            )
            self.assertAlmostEqual(
                _directed_blade_angle(left.line, left.reference_line, 1, "left"),
                angle,
            )

    def test_ignored_backtracking_does_not_accumulate_phase_drift(self) -> None:
        state = _PaddleEventState()

        for angle in [340, 345, 342, 343, 346, 355, 2, 10]:
            self.assertTrue(_advance_directed_paddle_phase(state, angle))

        self.assertEqual(state.cycle_index, 1)
        self.assertAlmostEqual(state.unwrapped_angle or 0, 370)
        self.assertAlmostEqual(state.last_directed_angle or 0, 10)

    def test_event_angle_uses_geometry_after_ignored_backtracking(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = []
        for index, angle in enumerate([0, 10, 7, 8, 12, 30, 45]):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                8.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual([event.kind for event in events], ["catch"])
        self.assertAlmostEqual(events[0].phase_angle or 0, 30)
        self.assertAlmostEqual(events[0].degree or 0, 30)

    def test_directed_detector_backdates_catch_and_exit(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = []
        for index, angle in enumerate([0, 30, 45, 60, 90, 120, 135, 150, 165]):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                8.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual([(event.kind, event.timestamp_ms) for event in events], [
            ("catch", 100),
            ("exit", 700),
        ])
        self.assertEqual([event.cycle_index for event in events], [0, 0])
        self.assertEqual([event.active_blade for event in events], [1, 1])
        self.assertEqual(events[0].travel_direction, "right")
        self.assertAlmostEqual(events[0].phase_angle or 0, 30)
        self.assertEqual(_event_label_text(events[0]), "30°")
        self.assertEqual(_event_label_text(events[1]), "30°")
        self.assertEqual(_paddle_event_angle_color(events[0]), (255, 82, 96, 255))
        self.assertEqual(_paddle_event_angle_color(events[1]), (46, 204, 113, 255))

    def test_stroke_phase_length_restores_cropped_immersed_blade_through_exit(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events: list[PaddleEvent] = []
        lengths_at_event: list[float] = []
        for index, angle in enumerate([0, 30, 45, 60, 90, 120, 135, 150, 165]):
            observation = _rotating_observation(angle)
            if angle >= 90:
                line = observation.line
                observation = _PaddleObservation(
                    source_ids=observation.source_ids,
                    reference_id=observation.reference_id,
                    line=(
                        line[0],
                        line[1],
                        line[0] + (line[2] - line[0]) * 0.55,
                        line[1] + (line[3] - line[1]) * 0.55,
                    ),
                    reference_line=observation.reference_line,
                )
            event = _update_directed_paddle_state(
                state,
                observation,
                index * 100,
                8.0,
            )
            if event is not None:
                events.append(event)
                lengths_at_event.append(math.dist(event.line[:2], event.line[2:]))

        self.assertEqual([event.kind for event in events], ["catch", "exit"])
        self.assertAlmostEqual(lengths_at_event[0], 60.0)
        self.assertAlmostEqual(lengths_at_event[1], 60.0)
        self.assertAlmostEqual(state.stroke_length or 0, 60.0)

    def test_immersed_length_restoration_extends_only_the_active_blade(self) -> None:
        state = _PaddleEventState(
            stroke_length=100.0,
            stroke_blade=0,
            stroke_cycle_index=0,
        )

        restored = _restore_immersed_paddle_length(
            state,
            (40.0, 0.0, 100.0, 0.0),
        )

        self.assertEqual(restored, (0.0, 0.0, 100.0, 0.0))

    def test_stroke_length_restores_forward_to_90_and_backward_from_180(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 100.0, 0.0)),
            (1, 45.0, (0.0, 0.0, 98.0, 0.0)),
            (2, 90.0, (0.0, 0.0, 60.0, 0.0)),
            (3, 135.0, (0.0, 0.0, 108.0, 0.0)),
            (4, 180.0, (0.0, 0.0, 110.0, 0.0)),
        ]

        restored = _restore_bidirectional_phase_lines(samples, active_blade=1)

        self.assertEqual(
            [_line_length(restored[index]) for index in range(5)],
            [100.0, 100.0, 110.0, 110.0, 110.0],
        )
        self.assertTrue(all(restored[index][:2] == (0.0, 0.0) for index in range(5)))

    def test_fixed_reverse_anchor_rejects_cnn_complete_length_inflation(self) -> None:
        samples = [
            (0, 90.0, (0.0, 0.0, 145.0, 0.0)),
            (1, 110.0, (0.0, 0.0, 146.0, 0.0)),
            (2, 130.0, (0.0, 0.0, 185.0, 0.0)),
            (3, 150.0, (0.0, 0.0, 144.0, 0.0)),
            (4, 165.0, (0.0, 0.0, 148.0, 0.0)),
            (5, 180.0, (0.0, 0.0, 147.0, 0.0)),
        ]

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: True,
        )

        self.assertEqual(
            [round(_line_length(restored[index]), 1) for index in range(6)],
            [148.0, 148.0, 148.0, 148.0, 148.0, 147.5],
        )

    def test_reverse_anchor_keeps_only_lengths_within_relative_tolerance(self) -> None:
        samples = [
            (0, 140.0, (0.0, 0.0, 142.2, 0.0)),
            (1, 150.0, (0.0, 0.0, 183.5, 0.0)),
            (2, 161.0, (0.0, 0.0, 210.9, 0.0)),
            (3, 175.0, (0.0, 0.0, 208.8, 0.0)),
        ]
        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: True,
        )

        self.assertEqual(
            [round(_line_length(restored[index]), 1) for index in range(4)],
            [210.9, 210.9, 210.9, 209.9],
        )

    def test_fixed_reverse_anchor_rejects_gradual_length_drift(self) -> None:
        samples = [
            (0, 130.0, (0.0, 0.0, 105.0, 0.0)),
            (1, 140.0, (0.0, 0.0, 115.0, 0.0)),
            (2, 150.0, (0.0, 0.0, 125.0, 0.0)),
            (3, 160.0, (0.0, 0.0, 135.0, 0.0)),
            (4, 170.0, (0.0, 0.0, 145.0, 0.0)),
            (5, 180.0, (0.0, 0.0, 147.0, 0.0)),
        ]

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: True,
        )

        self.assertEqual(
            [_line_length(restored[index]) for index in range(6)],
            [147.0, 147.0, 147.0, 147.0, 147.0, 147.0],
        )

    def test_reverse_anchor_is_a_minimum_for_cnn_complete_candidates(self) -> None:
        samples = [
            (0, 121.0, (0.0, 0.0, 112.71, 0.0)),
            (1, 128.0, (0.0, 0.0, 119.20, 0.0)),
            (2, 173.0, (0.0, 0.0, 137.10, 0.0)),
            (3, 175.0, (0.0, 0.0, 136.18, 0.0)),
        ]

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda index, _line, _blade: index != 0,
        )

        self.assertEqual(
            [round(_line_length(restored[index]), 2) for index in range(4)],
            [137.10, 137.10, 137.10, 136.64],
        )

    def test_cnn_cropped_masks_cannot_seed_forward_or_backward_length(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 100.0, 0.0)),
            (1, 45.0, (0.0, 0.0, 180.0, 0.0)),
            (2, 90.0, (0.0, 0.0, 105.0, 0.0)),
            (3, 135.0, (0.0, 0.0, 170.0, 0.0)),
            (4, 180.0, (0.0, 0.0, 120.0, 0.0)),
        ]

        verified_indices: set[int] = set()
        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda index, _line, _blade: index not in {1, 3},
            verified_indices=verified_indices,
        )

        self.assertEqual(
            [round(_line_length(restored[index]), 1) for index in range(5)],
            [102.5, 102.5, 120.0, 120.0, 120.0],
        )
        self.assertEqual(verified_indices, set(range(5)))

    def test_unknown_cnn_status_cannot_seed_restoration(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 100.0, 0.0)),
            (1, 45.0, (0.0, 0.0, 130.0, 0.0)),
        ]
        verified_indices: set[int] = set()

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: None,
            verified_indices=verified_indices,
        )

        self.assertEqual(_line_length(restored[1]), 130.0)
        self.assertEqual(verified_indices, set())

    def test_disagreeing_complete_candidates_cannot_seed_restoration(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 100.0, 0.0)),
            (1, 45.0, (0.0, 0.0, 130.0, 0.0)),
        ]
        verified_indices: set[int] = set()

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: True,
            verified_indices=verified_indices,
        )

        self.assertEqual(_line_length(restored[0]), 100.0)
        self.assertEqual(_line_length(restored[1]), 130.0)
        self.assertEqual(verified_indices, set())

    def test_cropped_phase_edges_require_a_complete_seed(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 80.0, 0.0)),
            (1, 45.0, (0.0, 0.0, 90.0, 0.0)),
            (2, 90.0, (0.0, 0.0, 120.0, 0.0)),
            (3, 135.0, (0.0, 0.0, 90.0, 0.0)),
            (4, 180.0, (0.0, 0.0, 80.0, 0.0)),
        ]
        verified_indices: set[int] = set()

        _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda index, _line, _blade: index == 2,
            verified_indices=verified_indices,
        )

        self.assertEqual(verified_indices, set())

    def test_track_without_complete_cnn_seed_is_ineligible_for_events(self) -> None:
        restored = _restore_track_stroke_lengths(
            _axis_track([0, 15, 30, 45, 60, 90, 120, 150, 165]),
            "right",
            8.0,
            candidate_validator=lambda _timed, _line, _blade: None,
        )

        self.assertTrue(restored)
        self.assertFalse(
            any(timed.observation.phase_length_verified for timed in restored)
        )

    def test_fixed_forward_anchor_rejects_cnn_complete_length_inflation(self) -> None:
        samples = [
            (0, 0.0, (0.0, 0.0, 100.0, 0.0)),
            (1, 20.0, (0.0, 0.0, 105.0, 0.0)),
            (2, 45.0, (0.0, 0.0, 180.0, 0.0)),
        ]

        restored = _restore_bidirectional_phase_lines(
            samples,
            active_blade=1,
            candidate_validator=lambda _index, _line, _blade: True,
        )

        self.assertEqual(_line_length(restored[2]), 105.0)

    def test_completeness_predictions_use_checkpoint_threshold(self) -> None:
        class Predictor:
            threshold = 0.5

            def predict_records(self, records: list[dict]) -> list[float]:
                return [record["probability"] for record in records]

        frames = {
            100: [
                {"track_id": "paddle:1", "probability": 0.2},
                {"track_id": "paddle:2", "probability": 0.8},
            ]
        }
        observation = _PaddleObservation(
            source_ids=("paddle:1", "paddle:2"),
            reference_id="boat:1",
            line=(0.0, 0.0, 10.0, 10.0),
            reference_line=(0.0, 20.0, 100.0, 20.0),
        )
        tracks = {
            "slot:1": [_TimedPaddleObservation(100, "slot:1", observation)]
        }

        with patch(
            "sam3_service.exporter.build_paddle_completeness_predictor",
            return_value=Predictor(),
        ):
            statuses = _predict_event_paddle_completeness(frames, tracks)

        self.assertEqual(statuses, {(100, "paddle:1"): True, (100, "paddle:2"): False})

    def test_active_endpoint_source_mask_controls_candidate_status(self) -> None:
        frames = {
            100: [
                {
                    "track_id": "paddle:left",
                    "centerline_line_xyxy": [0, 0, 40, 0],
                },
                {
                    "track_id": "paddle:right",
                    "centerline_line_xyxy": [40, 0, 100, 0],
                },
            ]
        }
        observation = _PaddleObservation(
            source_ids=("paddle:left", "paddle:right"),
            reference_id="boat:1",
            line=(0.0, 0.0, 100.0, 0.0),
            reference_line=(0.0, 20.0, 100.0, 20.0),
        )
        timed = _TimedPaddleObservation(100, "slot:1", observation)
        validator = _paddle_candidate_validator(
            frames,
            {(100, "paddle:left"): True, (100, "paddle:right"): False},
            100,
            100,
        )

        self.assertTrue(validator(timed, observation.line, 0))
        self.assertFalse(validator(timed, observation.line, 1))

    def test_track_is_bidirectionally_restored_before_event_detection(self) -> None:
        angles = [0, 30, 45, 60, 90, 120, 135, 150, 165, 180]
        observed_lengths = [60, 58, 45, 40, 30, 40, 45, 50, 58, 60]
        track: list[_TimedPaddleObservation] = []
        for index, (angle, observed_length) in enumerate(
            zip(angles, observed_lengths, strict=True)
        ):
            full = _rotating_observation(angle)
            inactive = full.line[:2]
            active = full.line[2:]
            unit = (
                (active[0] - inactive[0]) / 60.0,
                (active[1] - inactive[1]) / 60.0,
            )
            cropped = (
                inactive[0],
                inactive[1],
                inactive[0] + unit[0] * observed_length,
                inactive[1] + unit[1] * observed_length,
            )
            track.append(
                _TimedPaddleObservation(
                    index * 100,
                    "paddle:physical:1",
                    replace(full, line=cropped, raw_line=cropped),
                )
            )

        restored = _restore_track_stroke_lengths(
            track,
            "right",
            8.0,
            candidate_validator=lambda _timed, line, _blade: _line_length(line) >= 55,
        )

        self.assertTrue(
            all(timed.observation.phase_length_restored for timed in restored)
        )
        self.assertEqual(
            [round(_line_length(timed.observation.line)) for timed in restored],
            [60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
        )
        for timed in restored:
            self.assertEqual(timed.observation.line[:2], timed.observation.raw_line[:2])

        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = [
            event
            for timed in restored
            if (
                event := _update_directed_paddle_state(
                    state,
                    timed.observation,
                    timed.timestamp_ms,
                    8.0,
                )
            )
            is not None
        ]
        self.assertEqual([event.kind for event in events], ["catch", "exit"])

    def test_reversed_cropped_fragment_inherits_pre_restored_stroke_length(self) -> None:
        angles = [0, 36, 46, 58, 104, 116, 135, 150, 165]
        timestamps = [333, 400, 433, 467, 600, 633, 667, 700, 733]
        track: list[_TimedPaddleObservation] = []
        for angle, timestamp_ms in zip(angles, timestamps, strict=True):
            observation = _rotating_observation(angle)
            line = observation.line
            if angle == 104:
                inactive = line[:2]
                active = line[2:]
                unit = (
                    (active[0] - inactive[0]) / 60.0,
                    (active[1] - inactive[1]) / 60.0,
                )
                cropped_active = (
                    inactive[0] + unit[0] * 30.0,
                    inactive[1] + unit[1] * 30.0,
                )
                line = (*cropped_active, *inactive)
            track.append(
                _TimedPaddleObservation(
                    timestamp_ms,
                    "paddle:slot:boat:1:1",
                    replace(observation, line=line, raw_line=line),
                )
            )

        restored = _restore_track_stroke_lengths(track, "right", 8.0)

        self.assertTrue(restored[4].observation.phase_length_restored)
        state = _PaddleEventState(
            physical_id="paddle:slot:boat:1:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = [
            event
            for timed in restored
            if (
                event := _update_directed_paddle_state(
                    state,
                    timed.observation,
                    timed.timestamp_ms,
                    8.0,
                )
            )
            is not None
        ]

        self.assertEqual([event.kind for event in events], ["catch", "exit"])
        self.assertAlmostEqual(events[0].phase_angle or 0, 36)
        self.assertGreater(events[1].phase_angle or 0, 120)
        self.assertAlmostEqual(state.stroke_length or 0, 60.0)

    def test_runtime_length_envelope_resets_for_backward_restored_half(self) -> None:
        state = _PaddleEventState(
            active_blade=0,
            cycle_index=0,
            unwrapped_angle=80.0,
        )

        first_half = _inherit_stroke_phase_length(
            state,
            (0.0, 0.0, 100.0, 0.0),
            phase_preprocessed=True,
        )
        state.unwrapped_angle = 110.0
        second_half = _inherit_stroke_phase_length(
            state,
            (30.0, 0.0, 100.0, 0.0),
            phase_preprocessed=True,
        )
        state.unwrapped_angle = 130.0
        cropped_second_half = _inherit_stroke_phase_length(
            state,
            (40.0, 0.0, 100.0, 0.0),
        )

        self.assertAlmostEqual(_line_length(first_half), 100.0)
        self.assertAlmostEqual(_line_length(second_half), 70.0)
        self.assertAlmostEqual(_line_length(cropped_second_half), 70.0)
        self.assertEqual(state.stroke_phase_half, 1)
        self.assertEqual(state.stroke_length, 70.0)

    def test_stroke_length_is_inherited_only_within_one_zero_to_180_phase(self) -> None:
        state = _PaddleEventState(
            active_blade=0,
            cycle_index=2,
            unwrapped_angle=750.0,
        )

        first = _inherit_stroke_phase_length(state, (0.0, 0.0, 100.0, 0.0))
        cropped = _inherit_stroke_phase_length(state, (25.0, 0.0, 100.0, 0.0))

        self.assertEqual(first, (0.0, 0.0, 100.0, 0.0))
        self.assertEqual(cropped, (0.0, 0.0, 100.0, 0.0))
        self.assertEqual(state.stroke_length, 100.0)

        state.unwrapped_angle = 901.0
        recovery = _inherit_stroke_phase_length(state, (25.0, 0.0, 100.0, 0.0))

        self.assertEqual(recovery, (25.0, 0.0, 100.0, 0.0))
        self.assertIsNone(state.stroke_length)

        state.cycle_index = 3
        state.unwrapped_angle = 1110.0
        next_stroke = _inherit_stroke_phase_length(
            state,
            (25.0, 0.0, 100.0, 0.0),
        )

        self.assertEqual(next_stroke, (25.0, 0.0, 100.0, 0.0))
        self.assertEqual(state.stroke_length, 75.0)

    def test_phase_inheritance_uses_raw_line_before_generic_stabilization(self) -> None:
        full = _rotating_observation(30)
        dry = full.line[:2]
        active = full.line[2:]
        unit = (
            (active[0] - dry[0]) / 60.0,
            (active[1] - dry[1]) / 60.0,
        )
        cropped_active = (
            dry[0] + unit[0] * 42.0,
            dry[1] + unit[1] * 42.0,
        )
        wrongly_stabilized = (
            cropped_active[0] - unit[0] * 60.0,
            cropped_active[1] - unit[1] * 60.0,
            cropped_active[0],
            cropped_active[1],
        )
        observation = _PaddleObservation(
            source_ids=full.source_ids,
            reference_id=full.reference_id,
            line=wrongly_stabilized,
            reference_line=full.reference_line,
            raw_line=(dry[0], dry[1], cropped_active[0], cropped_active[1]),
        )
        state = _PaddleEventState(
            active_blade=1,
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
            last_directed_angle=30.0,
            unwrapped_angle=30.0,
            cycle_index=0,
            stroke_length=60.0,
            stroke_blade=1,
            stroke_cycle_index=0,
        )

        self.assertIsNone(_update_directed_paddle_state(state, observation, 100, 8.0))

        assert state.last_line is not None
        for actual, expected in zip(state.last_line, full.line, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_event_label_prefers_directed_phase_angle_over_acute_angle(self) -> None:
        event = PaddleEvent(
            kind="catch",
            timestamp_ms=100,
            instance_id="paddle:physical:1",
            line=(20, 20, 80, 80),
            confidence=1.0,
            degree=45,
            phase_angle=315,
            active_blade=1,
            rotation_direction="clockwise",
            travel_direction="right",
        )

        self.assertEqual(_event_label_text(event), "315°")

    def test_exit_event_label_uses_complementary_acute_angle(self) -> None:
        event = PaddleEvent(
            kind="exit",
            timestamp_ms=100,
            instance_id="paddle:physical:1",
            line=(0, 0, 30, 40),
            confidence=1.0,
            phase_angle=151,
            degree=29,
        )

        self.assertEqual(_event_label_text(event), "29°")

    def test_opposite_blade_is_ignored_and_next_360_cycle_resets_catch(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = []
        angles = [
            0, 30, 45, 60, 90, 120, 135, 150,
            180, 210, 225, 240, 270, 300, 315, 330,
            360, 390, 405, 420,
        ]
        for index, angle in enumerate(angles):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                8.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual(
            [(event.kind, event.cycle_index) for event in events],
            [("catch", 0), ("exit", 0), ("catch", 1)],
        )

    def test_catch_and_exit_gates_reset_independently_each_cycle(self) -> None:
        state = _PaddleEventState(
            active_blade=1,
            stroke_length=60.0,
            stroke_blade=1,
            stroke_cycle_index=0,
        )
        state.emitted_events.update({(0, "catch"), (0, "exit")})
        state.last_directed_angle = 350
        state.unwrapped_angle = 350

        self.assertFalse(_catch_phase_allowed(state, 1))
        self.assertFalse(_event_phase_allowed(state, "exit"))
        self.assertTrue(_advance_directed_paddle_phase(state, 10))
        self.assertEqual(state.cycle_index, 1)
        self.assertIsNone(state.stroke_length)
        self.assertTrue(_catch_phase_allowed(state, 1))
        self.assertTrue(_event_phase_allowed(state, "exit"))

    def test_exit_can_bootstrap_without_a_detected_catch(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = []
        for index, angle in enumerate([90, 120, 135, 150, 165]):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                8.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual([(event.kind, event.timestamp_ms) for event in events], [
            ("exit", 300),
        ])
        self.assertNotIn((0, "catch"), state.emitted_events)

    def test_event_analysis_pipeline_handles_both_travel_directions(self) -> None:
        for travel_direction in ("right", "left"):
            with self.subTest(travel_direction=travel_direction):
                frames: dict[int, list[dict[str, object]]] = {}
                for index, angle in enumerate(
                    [0, 15, 30, 45, 60, 90, 120, 135, 150, 165]
                ):
                    observation = _rotating_observation(
                        angle,
                        travel_direction=travel_direction,
                    )
                    frames[index * 100] = [
                        {
                            "prompt_id": "boat",
                            "instance_id": "boat:1",
                            "track_id": "boat:track:1",
                            "centerline_line_xyxy": list(observation.reference_line),
                            "waterline_line_xyxy": list(observation.reference_line),
                        },
                        {
                            "prompt_id": "paddle",
                            "instance_id": "paddle:1",
                            "track_id": "paddle:track:1",
                            "centerline_line_xyxy": list(observation.line),
                        },
                    ]

                completeness = {
                    (timestamp_ms, "paddle:track:1"): True
                    for timestamp_ms in frames
                }
                with patch(
                    "sam3_service.exporter._predict_event_paddle_completeness",
                    return_value=completeness,
                ):
                    events = _detect_paddle_events(
                        frames,
                        ExportOptions(
                            include_catch=True,
                            include_exit=True,
                            reference_prompt_id="boat",
                            reference_line_mode="waterline",
                            target_prompt_ids=("paddle",),
                        ),
                        100,
                        100,
                        1.0,
                        1.0,
                    )

                self.assertEqual(
                    [
                        (event.kind, event.timestamp_ms, event.travel_direction)
                        for event in events
                    ],
                    [
                        ("catch", 200, travel_direction),
                        ("exit", 800, travel_direction),
                    ],
                )

    def test_event_paddle_index_ranks_tracks_from_front_in_travel_direction(self) -> None:
        def track(physical_id: str, center_x: float) -> list[_TimedPaddleObservation]:
            observation = _PaddleObservation(
                source_ids=(physical_id,),
                reference_id="boat:1",
                line=(center_x - 5, 20, center_x + 5, 40),
                reference_line=(0, 50, 100, 50),
            )
            return [_TimedPaddleObservation(0, physical_id, observation)]

        tracks = {
            "paddle:left": track("paddle:left", 20),
            "paddle:middle": track("paddle:middle", 50),
            "paddle:right": track("paddle:right", 80),
        }

        rightward = _select_event_paddle_tracks(
            tracks,
            {"boat:1": ("clockwise", "right", 1.0)},
            1,
        )
        leftward = _select_event_paddle_tracks(
            tracks,
            {"boat:1": ("anticlockwise", "left", 1.0)},
            1,
        )
        second_rightward = _select_event_paddle_tracks(
            tracks,
            {"boat:1": ("clockwise", "right", 1.0)},
            2,
        )

        self.assertEqual(
            next(iter(rightward.values()))[0].observation.source_ids,
            ("paddle:right",),
        )
        self.assertEqual(
            next(iter(leftward.values()))[0].observation.source_ids,
            ("paddle:left",),
        )
        self.assertEqual(
            next(iter(second_rightward.values()))[0].observation.source_ids,
            ("paddle:middle",),
        )

    def test_boat_reference_length_filter_anchors_head_and_rejects_tail_outliers(self) -> None:
        stabilized = _kalman_stabilized_boat_reference_lines(
            {
                0: (100.0, 100.0, 500.0, 100.0),
                100: (-100.0, 100.0, 500.0, 100.0),
                200: (250.0, 100.0, 500.0, 100.0),
                300: (90.0, 100.0, 500.0, 100.0),
            },
            "right",
        )

        self.assertEqual(set(stabilized), {0, 100, 200, 300})
        for line in stabilized.values():
            self.assertEqual(line[2:], (500.0, 100.0))
            self.assertAlmostEqual(_line_length(line), 405.0, delta=5.0)
        self.assertLess(_line_length(stabilized[100]), 430.0)
        self.assertGreater(_line_length(stabilized[200]), 380.0)

    def test_boat_reference_length_filter_anchors_leftward_head(self) -> None:
        stabilized = _kalman_stabilized_boat_reference_lines(
            {
                0: (100.0, 100.0, 500.0, 100.0),
                100: (100.0, 100.0, 620.0, 100.0),
                200: (100.0, 100.0, 500.0, 100.0),
            },
            "left",
        )

        for line in stabilized.values():
            self.assertEqual(line[2:], (100.0, 100.0))
            self.assertAlmostEqual(line[0], 500.0, delta=5.0)

    def test_boat_reference_length_stabilization_prevents_projection_shift(self) -> None:
        def timed(timestamp_ms: int, reference_line: tuple[float, float, float, float]):
            observation = _PaddleObservation(
                source_ids=("paddle:1",),
                reference_id="boat:1",
                line=(440.0, 20.0, 460.0, 40.0),
                reference_line=reference_line,
            )
            return _TimedPaddleObservation(timestamp_ms, "paddle:1", observation)

        tracks = {
            "paddle:1": [
                timed(0, (100.0, 50.0, 500.0, 50.0)),
                timed(100, (-100.0, 50.0, 500.0, 50.0)),
                timed(200, (100.0, 50.0, 500.0, 50.0)),
            ]
        }

        raw_inflated_position = _observation_forward_position(
            tracks["paddle:1"][1].observation,
            "right",
        )
        stabilized = _stabilize_boat_reference_lengths(
            tracks,
            {"boat:1": ("clockwise", "right", 1.0)},
        )
        positions = [
            _observation_forward_position(timed.observation, "right")
            for timed in stabilized["paddle:1"]
        ]

        self.assertEqual(raw_inflated_position, 250.0)
        self.assertEqual(positions, [150.0, 150.0, 150.0])

    def test_boat_reference_filter_uses_frames_without_a_paddle(self) -> None:
        tracks = {
            "paddle:1": [
                _TimedPaddleObservation(
                    0,
                    "paddle:1",
                    _PaddleObservation(
                        source_ids=("paddle:1",),
                        reference_id="boat:1",
                        line=(440.0, 20.0, 460.0, 40.0),
                        reference_line=(-100.0, 50.0, 500.0, 50.0),
                    ),
                )
            ]
        }

        stabilized = _stabilize_boat_reference_lengths(
            tracks,
            {"boat:1": ("clockwise", "right", 1.0)},
            {
                "boat:1": {
                    0: (-100.0, 50.0, 500.0, 50.0),
                    100: (100.0, 50.0, 500.0, 50.0),
                    200: (100.0, 50.0, 500.0, 50.0),
                }
            },
        )

        reference_line = stabilized["paddle:1"][0].observation.reference_line
        self.assertEqual(reference_line[2:], (500.0, 50.0))
        self.assertEqual(_line_length(reference_line), 400.0)

    def test_event_boat_reference_lines_collects_boat_only_frames(self) -> None:
        def boat_record(timestamp_ms: int) -> dict[str, object]:
            return {
                "timestamp_ms": timestamp_ms,
                "prompt_id": "boat",
                "instance_id": "boat:1",
                "centerline_line_xyxy": [100.0, 50.0, 500.0, 50.0],
                "box_xywh": [100.0, 40.0, 400.0, 20.0],
                "segmentation": {"type": "polygon", "points": []},
            }

        lines = _event_boat_reference_lines(
            {0: [boat_record(0)], 100: [boat_record(100)]},
            ExportOptions(reference_prompt_id="boat", reference_line_mode="centerline"),
            640,
            360,
            1.0,
            1.0,
        )

        self.assertEqual(set(lines["boat:1"]), {0, 100})

    def test_event_paddle_slot_stitches_fragments_without_collapsing_a_gap(self) -> None:
        def timed(
            timestamp_ms: int,
            physical_id: str,
            center_x: float,
        ) -> _TimedPaddleObservation:
            observation = _PaddleObservation(
                source_ids=(physical_id,),
                reference_id="boat:1",
                line=(center_x - 5, 20, center_x + 5, 40),
                reference_line=(0, 50, 100, 50),
            )
            return _TimedPaddleObservation(timestamp_ms, physical_id, observation)

        tracks = {
            "front:fragment:1": [timed(0, "front:fragment:1", 80)],
            "front:fragment:2": [timed(200, "front:fragment:2", 82)],
            "rear": [
                timed(0, "rear", 20),
                timed(100, "rear", 20),
                timed(200, "rear", 20),
            ],
        }

        selected = _select_event_paddle_tracks(
            tracks,
            {"boat:1": ("clockwise", "right", 1.0)},
            1,
            2,
        )

        observations = next(iter(selected.values()))
        self.assertEqual([item.timestamp_ms for item in observations], [0, 200])
        self.assertEqual(
            [item.observation.source_ids for item in observations],
            [("front:fragment:1",), ("front:fragment:2",)],
        )

    def test_all_event_paddles_preserves_every_track(self) -> None:
        tracks = {
            "paddle:1": _axis_track([0, 30]),
            "paddle:2": _axis_track([0, 30]),
        }

        self.assertIs(
            _select_event_paddle_tracks(tracks, {}, None),
            tracks,
        )

    def test_event_detection_uses_only_the_selected_directional_paddle(self) -> None:
        def shifted_track(physical_id: str, offset_x: float) -> list[_TimedPaddleObservation]:
            shifted: list[_TimedPaddleObservation] = []
            for timed in _axis_track([0, 15, 30, 45, 60, 90, 120, 135, 150, 165]):
                line = timed.observation.line
                observation = replace(
                    timed.observation,
                    source_ids=(physical_id,),
                    line=(
                        line[0] + offset_x,
                        line[1],
                        line[2] + offset_x,
                        line[3],
                    ),
                )
                shifted.append(
                    replace(timed, physical_id=physical_id, observation=observation)
                )
            return shifted

        tracks = {
            "paddle:rear": shifted_track("paddle:rear", -20),
            "paddle:front": shifted_track("paddle:front", 20),
        }
        with patch(
            "sam3_service.exporter._track_paddle_observations",
            return_value=tracks,
        ), patch(
            "sam3_service.exporter._paddle_candidate_validator",
            return_value=lambda _timed, _line, _blade: True,
        ):
            events = _detect_paddle_events(
                {},
                ExportOptions(
                    include_catch=True,
                    include_exit=True,
                    reference_prompt_id="boat",
                    target_prompt_ids=("paddle",),
                    target_slot_count=2,
                    event_paddle_index=1,
                ),
                100,
                100,
                1.0,
                1.0,
            )

        self.assertEqual([event.kind for event in events], ["catch", "exit"])
        self.assertEqual(len({event.instance_id for event in events}), 1)
        self.assertTrue(events[0].instance_id.endswith(":1"))

    def test_gap_does_not_synthesize_a_waterline_transition(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )

        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(30), 0, 8.0)
        )
        state.stroke_length = 60.0
        state.stroke_blade = 1
        state.stroke_cycle_index = 0
        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(60), 500, 8.0)
        )
        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(90), 600, 8.0)
        )
        self.assertEqual(state.emitted_events, set())
        self.assertIsNone(state.stroke_length)

    def test_unverified_recovery_preserves_endpoint_and_phase_continuity(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
            active_blade=1,
        )

        self.assertIsNone(
            _update_directed_paddle_state(
                state,
                _rotating_observation(176),
                0,
                8.0,
            )
        )
        for timestamp_ms, angle in [
            (100, 196),
            (200, 236),
            (300, 296),
            (367, 356),
        ]:
            self.assertIsNone(
                _update_directed_paddle_state(
                    state,
                    _rotating_observation(angle),
                    timestamp_ms,
                    8.0,
                    event_eligible=False,
                )
            )

        self.assertIsNone(
            _update_directed_paddle_state(
                state,
                _rotating_observation(3),
                400,
                8.0,
            )
        )

        self.assertEqual(state.active_blade, 1)
        self.assertEqual(state.cycle_index, 1)
        self.assertAlmostEqual(state.last_directed_angle or 0.0, 3.0)
        self.assertAlmostEqual(state.unwrapped_angle or 0.0, 363.0)
        self.assertEqual(state.candidates, {})

    def test_unverified_recovery_cannot_emit_but_next_cycle_can(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events: list[PaddleEvent] = []
        sequence = [
            *[(angle, True) for angle in [0, 30, 45, 60, 90, 120, 135, 150, 165]],
            *[(angle, False) for angle in [180, 210, 240, 270, 300, 330, 350]],
            *[(angle, True) for angle in [0, 30, 45, 60, 90, 120, 135, 150, 165]],
        ]
        for index, (angle, event_eligible) in enumerate(sequence):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                8.0,
                event_eligible=event_eligible,
            )
            if event is not None:
                events.append(event)

        self.assertEqual(
            [(event.kind, event.cycle_index) for event in events],
            [("catch", 0), ("exit", 0), ("catch", 1), ("exit", 1)],
        )

    def test_unverified_observation_cannot_confirm_pending_event(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
            active_blade=1,
        )

        self.assertIsNone(
            _update_directed_paddle_state(
                state, _rotating_observation(0), 0, 8.0
            )
        )
        self.assertIsNone(
            _update_directed_paddle_state(
                state, _rotating_observation(30), 100, 8.0
            )
        )
        self.assertIn("catch", state.candidates)
        self.assertIsNone(
            _update_directed_paddle_state(
                state,
                _rotating_observation(45),
                200,
                8.0,
                event_eligible=False,
            )
        )
        self.assertEqual(state.candidates, {})
        self.assertIsNone(
            _update_directed_paddle_state(
                state, _rotating_observation(60), 300, 8.0
            )
        )

    def test_collinear_exit_fragments_are_one_observation_and_one_event(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "track_id": "boat:track:1"},
            line=(0.0, 70.0, 160.0, 70.0),
            color=(255, 255, 255, 255),
        )
        fragments = [
            Centerline(
                record={"prompt_id": "paddle", "track_id": f"paddle:{index}"},
                line=line,
                color=(255, 255, 255, 255),
            )
            for index, line in enumerate(
                [
                    (20.0, 20.0, 80.0, 45.0),
                    (82.0, 46.0, 145.0, 72.0),
                ],
                start=1,
            )
        ]

        observations = _consolidate_paddle_observations(
            fragments, [reference], 160, 120
        )
        events = _dedupe_paddle_events(
            [
                PaddleEvent(
                    "exit", 1000, "paddle:1", fragments[0].line, 0.8, reference.line, 22
                ),
                PaddleEvent(
                    "exit", 1100, "paddle:2", fragments[1].line, 0.9, reference.line, 20
                ),
            ],
            160,
            120,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(len(events), 1)

    def test_fragment_consolidation_does_not_merge_across_paddle_slots(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "track_id": "boat:track:1"},
            line=(0.0, 100.0, 400.0, 100.0),
            color=(255, 255, 255, 255),
        )
        rear = Centerline(
            record={"prompt_id": "paddle", "track_id": "paddle:rear"},
            line=(20.0, -2.0, 190.0, 18.0),
            color=(255, 255, 255, 255),
        )
        front = Centerline(
            record={"prompt_id": "paddle", "track_id": "paddle:front"},
            line=(210.0, 20.0, 380.0, 40.0),
            color=(255, 255, 255, 255),
        )

        legacy = _consolidate_paddle_observations(
            [rear, front],
            [reference],
            400,
            200,
        )
        slot_aware = _consolidate_paddle_observations(
            [rear, front],
            [reference],
            400,
            200,
            slot_anchors_by_reference={"boat:track:1": [95.0, -95.0]},
            travel_directions={"boat:track:1": "right"},
        )

        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(slot_aware), 2)
        self.assertEqual(
            {observation.source_ids for observation in slot_aware},
            {("paddle:front",), ("paddle:rear",)},
        )

    def test_offset_parallel_exit_lines_from_screenshot_are_consolidated(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "track_id": "boat:track:1"},
            line=(120.0, 190.0, 400.0, 170.0),
            color=(255, 255, 255, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "track_id": f"paddle:{index}"},
                line=line,
                color=(255, 255, 255, 255),
            )
            for index, line in enumerate(
                [
                    (232.0, 9.0, 279.0, 224.0),
                    (248.0, 10.0, 301.0, 214.0),
                ],
                start=1,
            )
        ]

        observations = _consolidate_paddle_observations(
            targets, [reference], 537, 243
        )
        events = _dedupe_paddle_events(
            [
                PaddleEvent("exit", 1000, "paddle:1", targets[0].line, 0.8),
                PaddleEvent("exit", 1100, "paddle:2", targets[1].line, 0.9),
            ],
            537,
            243,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(len(events), 1)

    def test_below_boat_paddle_reflection_is_ignored_when_real_paddle_exists(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "track_id": "boat:track:1"},
            line=(0.0, 70.0, 160.0, 70.0),
            color=(255, 255, 255, 255),
        )
        real_paddle = Centerline(
            record={"prompt_id": "paddle", "track_id": "paddle:real"},
            line=(20.0, 20.0, 80.0, 60.0),
            color=(255, 255, 255, 255),
        )
        reflected_paddle = Centerline(
            record={"prompt_id": "paddle", "track_id": "paddle:reflection"},
            line=(20.0, 80.0, 80.0, 120.0),
            color=(255, 255, 255, 255),
        )

        observations = _consolidate_paddle_observations(
            [real_paddle, reflected_paddle],
            [reference],
            160,
            120,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source_ids, ("paddle:real",))

    def test_single_below_boat_paddle_is_retained_without_reflection_pair(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "track_id": "boat:track:1"},
            line=(0.0, 70.0, 160.0, 70.0),
            color=(255, 255, 255, 255),
        )
        paddle = Centerline(
            record={"prompt_id": "paddle", "track_id": "paddle:1"},
            line=(20.0, 80.0, 80.0, 120.0),
            color=(255, 255, 255, 255),
        )

        observations = _consolidate_paddle_observations(
            [paddle],
            [reference],
            160,
            120,
        )

        self.assertEqual(len(observations), 1)

    def test_paddle_water_depth_is_normalized_against_waterline(self) -> None:
        waterline = (0.0, 50.0, 100.0, 50.0)

        self.assertAlmostEqual(
            _paddle_water_depth_ratio((50.0, 10.0, 50.0, 60.0), waterline) or 0,
            0.1,
        )
        self.assertLess(
            _paddle_water_depth_ratio((50.0, 10.0, 50.0, 45.0), waterline) or 0,
            0,
        )

    def test_paddle_event_state_confirms_and_backdates_catch_and_exit(self) -> None:
        state = _PaddleEventState()
        line = (20.0, 10.0, 20.0, 40.0)

        self.assertIsNone(_update_paddle_event_state(state, "paddle:1", 0, line, 0.0))
        self.assertIsNone(_update_paddle_event_state(state, "paddle:1", 100, line, 0.02))
        catch = _update_paddle_event_state(state, "paddle:1", 200, line, 0.03)
        self.assertIsNotNone(catch)
        assert catch is not None
        self.assertEqual((catch.kind, catch.timestamp_ms), ("catch", 100))

        self.assertIsNone(_update_paddle_event_state(state, "paddle:1", 300, line, 0.02))
        self.assertIsNone(_update_paddle_event_state(state, "paddle:1", 400, line, 0.0))
        exit_event = _update_paddle_event_state(state, "paddle:1", 500, line, -0.01)
        self.assertIsNotNone(exit_event)
        assert exit_event is not None
        self.assertEqual((exit_event.kind, exit_event.timestamp_ms), ("exit", 400))

    def test_simultaneous_events_share_one_freeze_moment(self) -> None:
        first = PaddleEvent(
            kind="catch",
            timestamp_ms=1000,
            instance_id="paddle:1",
            line=(0, 0, 10, 10),
            confidence=0.9,
        )
        second = PaddleEvent(
            kind="catch",
            timestamp_ms=1150,
            instance_id="paddle:2",
            line=(20, 0, 30, 10),
            confidence=0.8,
        )
        later = PaddleEvent(
            kind="exit",
            timestamp_ms=1800,
            instance_id="paddle:1",
            line=(0, 0, 10, 10),
            confidence=0.9,
        )

        moments = _freeze_moments([later, second, first], 30, 90)

        self.assertEqual([moment.frame_index for moment in moments], [30, 54])
        self.assertEqual(moments[0].events, (first, second))

    def test_nearby_catch_and_exit_use_separate_freeze_moments(self) -> None:
        catch = PaddleEvent(
            kind="catch",
            timestamp_ms=1000,
            instance_id="paddle:1",
            line=(0, 0, 10, 10),
            confidence=0.9,
        )
        exit_event = PaddleEvent(
            kind="exit",
            timestamp_ms=1200,
            instance_id="paddle:1",
            line=(10, 0, 0, 10),
            confidence=0.9,
        )

        moments = _freeze_moments([catch, exit_event], 30, 90)

        self.assertEqual([moment.frame_index for moment in moments], [30, 36])
        self.assertEqual(moments[0].events, (catch,))
        self.assertEqual(moments[1].events, (exit_event,))

    def test_freeze_moment_respects_configured_metric_count(self) -> None:
        events = [
            PaddleEvent(
                kind="exit",
                timestamp_ms=1000 + index * 50,
                instance_id=f"paddle:{index}",
                line=line,
                confidence=confidence,
            )
            for index, (line, confidence) in enumerate(
                [
                    ((232.0, 9.0, 279.0, 224.0), 0.8),
                    ((248.0, 10.0, 301.0, 214.0), 0.9),
                ],
                start=1,
            )
        ]

        moments = _freeze_moments(
            events, 30, 90, max_events_per_moment=1
        )

        self.assertEqual(len(moments), 1)
        self.assertEqual(len(moments[0].events), 1)
        self.assertEqual(moments[0].events[0].instance_id, "paddle:2")

    def test_freeze_filters_insert_pause_before_resuming_event_frame(self) -> None:
        event = PaddleEvent(
            kind="catch",
            timestamp_ms=1000,
            instance_id="paddle:1",
            line=(0, 0, 10, 10),
            confidence=0.9,
        )
        moments = _freeze_moments([event], 30, 90)

        audio_filter = _freeze_audio_filter(moments, 45, 30, 90)

        self.assertEqual(
            _freeze_segments(moments, 90),
            (
                ("normal", 0, 30),
                ("freeze", 30, 31),
                ("normal", 30, 90),
            ),
        )
        self.assertIn("atrim=start=1.000000000:end=1.033333333", audio_filter)
        self.assertIn("volume=0", audio_filter)

    def test_event_freeze_and_metrics_are_opt_in_with_new_defaults(self) -> None:
        options = ExportOptions()

        self.assertFalse(options.include_event_freeze)
        self.assertEqual(options.event_hold_seconds, 1.2)
        self.assertFalse(options.include_event_metrics)
        self.assertFalse(options.include_body_motion)
        self.assertEqual(_event_freeze_frame_count(options, 30), 0)
        self.assertEqual(
            _event_freeze_frame_count(
                replace(options, include_event_freeze=True),
                30,
            ),
            36,
        )

    def test_event_metrics_keep_live_angles_and_use_an_independent_offset(self) -> None:
        options = _normalize_export_options(
            ExportOptions(include_angles=True, include_event_metrics=True),
            {
                "prompts": [
                    {"id": "boat", "text": "boat"},
                    {"id": "paddle", "text": "paddle"},
                ],
                "tracks": [],
                "settings": {},
            },
            1080,
            1920,
        )

        self.assertTrue(options.include_angles)
        self.assertEqual(options.event_metric_center_offset_percent, 5.5)
        self.assertEqual(options.metric_center_offset_percent, 16.0)

    def test_signed_event_metric_offset_selects_top_or_bottom_anchor(self) -> None:
        self.assertEqual(_event_metric_table_top(1000, 100, 10), 100)
        self.assertEqual(_event_metric_table_top(1000, 100, -10), 800)
        self.assertEqual(_event_metric_table_top(1000, 100, 0), 0)

    def test_event_angle_uses_first_crossing_geometry(self) -> None:
        state = _PaddleEventState(immersed=False)
        paddle = (50.0, 0.0, 50.0, 80.0)
        waterline = (0.0, 50.0, 100.0, 50.0)

        self.assertIsNone(
            _update_paddle_event_state(
                state, "paddle:1", 100, paddle, 0.02, waterline
            )
        )
        event = _update_paddle_event_state(
            state, "paddle:1", 200, (52.0, 0.0, 52.0, 80.0), 0.03, waterline
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.timestamp_ms, 100)
        self.assertEqual(event.line, paddle)
        self.assertEqual(event.reference_line, waterline)
        self.assertEqual(event.degree, 90)
        self.assertEqual(_event_label_text(event), "90°")
        self.assertNotIn("CATCH", _event_label_text(event))
        self.assertNotIn("EXIT", _event_label_text(event))
        self.assertEqual(_line_intersection(paddle, waterline), (50.0, 50.0))

    def test_export_can_hide_angles_without_removing_angle_measurements(self) -> None:
        records = [
            {
                "prompt_id": "boat",
                "instance_id": "boat:1",
                "centerline_line_xyxy": [0, 20, 100, 20],
            },
            {
                "prompt_id": "paddle",
                "instance_id": "paddle:1",
                "centerline_line_xyxy": [50, 0, 50, 80],
            },
        ]
        options = ExportOptions(
            include_angles=False,
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
        )

        with (
            patch("sam3_service.exporter._draw_degree_label_block") as draw_block,
            patch("sam3_service.exporter._draw_target_degree_marker") as draw_marker,
        ):
            _draw_frame_overlay(
                bytearray(100 * 100 * 4),
                100,
                100,
                records,
                {"boat": (255, 181, 71, 255), "paddle": (53, 194, 255, 255)},
                export_options=options,
                timestamp_ms=0,
                spm_estimator=SpmEstimator(),
            )

        draw_block.assert_not_called()
        draw_marker.assert_not_called()

    def test_export_angles_use_only_the_top_or_bottom_block(self) -> None:
        records = [
            {
                "prompt_id": "boat",
                "instance_id": "boat:1",
                "centerline_line_xyxy": [0, 20, 100, 20],
            },
            {
                "prompt_id": "paddle",
                "instance_id": "paddle:1",
                "centerline_line_xyxy": [50, 0, 50, 80],
            },
        ]
        options = ExportOptions(
            include_angles=True,
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
        )

        with (
            patch("sam3_service.exporter._draw_degree_label_block") as draw_block,
            patch("sam3_service.exporter._draw_target_degree_marker") as draw_marker,
        ):
            _draw_frame_overlay(
                bytearray(100 * 100 * 4),
                100,
                100,
                records,
                {"boat": (255, 181, 71, 255), "paddle": (53, 194, 255, 255)},
                export_options=options,
                timestamp_ms=0,
                spm_estimator=SpmEstimator(),
            )

        draw_block.assert_called_once()
        draw_marker.assert_not_called()

    def test_event_geometry_draws_degree_label_around_the_arc(self) -> None:
        event = PaddleEvent(
            kind="catch",
            timestamp_ms=0,
            instance_id="paddle:1",
            line=(50.0, 0.0, 50.0, 80.0),
            reference_line=(0.0, 50.0, 100.0, 50.0),
            confidence=1.0,
            phase_angle=90.0,
            travel_direction="right",
        )

        with (
            patch("sam3_service.exporter._draw_paddle_event_label_with_pillow") as pillow_label,
            patch("sam3_service.exporter._draw_small_degree_label") as bitmap_label,
        ):
            _draw_paddle_event_label(
                bytearray(100 * 100 * 4),
                100,
                100,
                event,
                ExportOptions(),
            )

        pillow_label.assert_called_once()
        self.assertEqual(pillow_label.call_args.args[5], "90°")
        bitmap_label.assert_not_called()

    def test_record_line_can_select_waterline_and_fall_back_to_centerline(self) -> None:
        record = {
            "centerline_line_xyxy": [0, 20, 100, 20],
            "waterline_line_xyxy": [0, 30, 100, 32],
        }

        self.assertEqual(
            _record_line(record, 100, 100, use_waterline=True),
            (0.0, 30.0, 100.0, 32.0),
        )
        self.assertEqual(
            _record_line(
                {"centerline_line_xyxy": [0, 20, 100, 20]},
                100,
                100,
                use_waterline=True,
            ),
            (0.0, 20.0, 100.0, 20.0),
        )

    def test_waterline_is_extended_to_the_boat_centerline_span(self) -> None:
        record = {
            "centerline_line_xyxy": [0, 20, 100, 20],
            "waterline_line_xyxy": [25, 30, 75, 30],
        }

        self.assertEqual(
            _record_line(record, 100, 100, use_waterline=True),
            (0.0, 30.0, 100.0, 30.0),
        )

    def test_event_analysis_treats_short_waterline_as_infinite(self) -> None:
        paddle = (200.0, 0.0, 200.0, 55.0)
        short_waterline = (0.0, 50.0, 100.0, 50.0)

        self.assertFalse(_blade_waterline_overlaps(paddle, short_waterline, 4.0)[1])
        self.assertTrue(
            _blade_transition_on_waterline(paddle, short_waterline, 1, 8.0)
        )

    def test_event_waterline_band_extends_eight_pixels_upward_only(self) -> None:
        self.assertEqual(_waterline_transition_kind(-9.0, -7.0, 2.0, 8.0), "catch")
        self.assertIsNone(_waterline_transition_kind(5.0, 1.0, -4.0, 8.0))
        self.assertEqual(_waterline_transition_kind(1.0, -1.0, -2.0, 8.0), "exit")

    def test_catch_crossing_uses_the_observation_closest_to_waterline(self) -> None:
        reference = (0.0, 0.0, 100.0, 0.0)

        def observation(active: tuple[float, float]) -> _PaddleObservation:
            return _PaddleObservation(
                source_ids=("paddle:1",),
                reference_id="boat:1",
                line=(50.0, -100.0, *active),
                reference_line=reference,
            )

        previous = observation((70.0, -9.0))
        crossing = observation((55.0, 32.0))
        confirmation = observation((52.0, 35.0))
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
            active_blade=1,
        )

        self.assertIsNone(
            _update_directed_paddle_state(state, previous, 100, 8.0)
        )
        self.assertIsNone(
            _update_directed_paddle_state(state, crossing, 200, 8.0)
        )
        event = _update_directed_paddle_state(
            state,
            confirmation,
            300,
            8.0,
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual((event.kind, event.timestamp_ms), ("catch", 100))
        self.assertEqual(event.line, previous.line)

    def test_exit_skipping_the_full_band_uses_the_closest_observation(self) -> None:
        cases = [
            (135, 165, 100, 135),
            (90, 158, 133, 158),
        ]
        for previous_angle, current_angle, expected_timestamp, expected_phase in cases:
            with self.subTest(
                previous_angle=previous_angle,
                current_angle=current_angle,
            ):
                state = _PaddleEventState(
                    physical_id="paddle:physical:1",
                    rotation_direction="clockwise",
                    travel_direction="right",
                    direction_confidence=1.0,
                    active_blade=1,
                )
                previous = _rotating_observation(previous_angle)
                current = _rotating_observation(current_angle)

                self.assertIsNone(
                    _update_directed_paddle_state(state, previous, 100, 8.0)
                )
                event = _update_directed_paddle_state(state, current, 133, 8.0)

                self.assertIsNotNone(event)
                assert event is not None
                self.assertEqual(event.kind, "exit")
                self.assertEqual(event.timestamp_ms, expected_timestamp)
                self.assertAlmostEqual(event.phase_angle or 0, expected_phase)
                expected_line = previous.line if expected_timestamp == 100 else current.line
                for actual, expected in zip(event.line, expected_line, strict=True):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(state.candidates, {})

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

    def test_selection_rectangle_holds_until_next_keyframe(self) -> None:
        options = ExportOptions(
            selection_keyframes=(
                (0, 0.0, 0.1, 0.2, 0.3),
                (1000, 0.4, 0.3, 0.4, 0.5),
            )
        )

        self.assertEqual(_selection_rect_at(options, -10), (0.0, 0.1, 0.2, 0.3))
        self.assertEqual(_selection_rect_at(options, 500), (0.0, 0.1, 0.2, 0.3))
        self.assertEqual(_selection_rect_at(options, 1000), (0.4, 0.3, 0.4, 0.5))
        self.assertEqual(_selection_rect_at(options, 2000), (0.4, 0.3, 0.4, 0.5))

    def test_keyframed_rectangle_filters_at_frame_time(self) -> None:
        options = ExportOptions(
            selection_keyframes=(
                (0, 0.0, 0.0, 0.3, 1.0),
                (1000, 0.7, 0.0, 0.3, 1.0),
            )
        )
        left_record = {"centerline_line_xyxy": [5, 20, 15, 20]}
        right_record = {"centerline_line_xyxy": [85, 20, 95, 20]}

        self.assertTrue(_record_selected_for_export(left_record, options, 100, 100, 0))
        self.assertFalse(_record_selected_for_export(right_record, options, 100, 100, 0))
        self.assertFalse(_record_selected_for_export(left_record, options, 100, 100, 1000))
        self.assertTrue(_record_selected_for_export(right_record, options, 100, 100, 1000))
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
            ["1: 42°", "", "3: 51°", ""],
        )
        self.assertEqual(
            [entry.layout_text for entry in entries],
            ["1: 000°", "2: 000°", "3: 000°", "4: 000°"],
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

    def test_configured_metric_count_truncates_extra_detections(self) -> None:
        labels = [
            DegreeLabel(
                instance_id=f"paddle:{index}",
                degree=40 + index,
                line=(index, 0, index, 10),
                color=(53, 194, 255, 255),
            )
            for index in range(3)
        ]

        slots = _degree_slots(labels, ExportOptions(target_slot_count=1))

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0].degree, 40)
        self.assertEqual(slots[0].instance_id, "slot:1")

    def test_configured_metric_count_overrides_track_slot_count(self) -> None:
        labels = [
            DegreeLabel(
                instance_id="paddle:track:2",
                degree=42,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            )
        ]

        slots = _degree_slots(
            labels,
            ExportOptions(
                target_slot_count=1,
                target_track_ids=("paddle:track:1", "paddle:track:2"),
            ),
        )

        self.assertEqual([slot.degree for slot in slots], [42])

    def test_event_companion_angles_follow_directional_slots_and_skip_selected(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "instance_id": "boat:1"},
            line=(0.0, 50.0, 100.0, 50.0),
            color=(255, 181, 71, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "instance_id": f"paddle:{index}"},
                line=(center_x - 10.0, 30.0, center_x + 10.0, 30.0 + rise),
                color=(53, 194, 255, 255),
            )
            for index, (center_x, rise) in enumerate(
                [(20.0, 12.0), (50.0, 20.0), (80.0, 35.0)],
                start=1,
            )
        ]
        options = ExportOptions(
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
            target_slot_count=3,
            event_paddle_index=2,
        )

        rightward = _event_companion_degree_slots(
            [reference, *targets],
            options,
            PaddleEvent(
                kind="catch",
                timestamp_ms=0,
                instance_id="slot:2",
                line=targets[1].line,
                confidence=1.0,
                phase_angle=45,
                travel_direction="right",
            ),
        )
        leftward = _event_companion_degree_slots(
            [reference, *targets],
            options,
            PaddleEvent(
                kind="catch",
                timestamp_ms=0,
                instance_id="slot:2",
                line=targets[1].line,
                confidence=1.0,
                phase_angle=45,
                travel_direction="left",
            ),
        )

        self.assertEqual([slot.degree for slot in rightward], [60, 45, 31])
        self.assertEqual([slot.degree for slot in leftward], [31, 45, 60])
        self.assertIsNone(rightward[1].line)
        self.assertIsNone(leftward[1].line)

        single = _event_companion_degree_slots(
            [reference, targets[0]],
            replace(options, target_slot_count=1, event_paddle_index=1),
            PaddleEvent(
                kind="catch",
                timestamp_ms=0,
                instance_id="slot:1",
                line=targets[0].line,
                confidence=1.0,
                phase_angle=45,
                travel_direction="left",
            ),
        )
        self.assertEqual([slot.degree for slot in single], [45])
        self.assertIsNone(single[0].line)

    def test_all_event_paddles_replace_nearest_top_or_bottom_slots(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "instance_id": "boat:1"},
            line=(0.0, 50.0, 100.0, 50.0),
            color=(255, 181, 71, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "instance_id": f"paddle:{index}"},
                line=(center_x - 10.0, 30.0, center_x + 10.0, 30.0 + rise),
                color=(53, 194, 255, 255),
            )
            for index, (center_x, rise) in enumerate(
                [(20.0, 12.0), (50.0, 20.0), (80.0, 35.0)],
                start=1,
            )
        ]
        options = ExportOptions(
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
            target_slot_count=3,
        )
        events = (
            PaddleEvent(
                kind="catch",
                timestamp_ms=0,
                instance_id="event:left",
                line=targets[0].line,
                confidence=1.0,
                phase_angle=44.0,
            ),
            PaddleEvent(
                kind="exit",
                timestamp_ms=0,
                instance_id="event:right",
                line=targets[2].line,
                confidence=1.0,
                phase_angle=140.0,
            ),
        )

        slots = _event_frame_degree_slots([reference, *targets], options, events)

        self.assertEqual(slots[0].degree, 44)
        self.assertEqual(slots[2].degree, 40)
        self.assertIsNone(slots[0].line)
        self.assertIsNotNone(slots[1].line)
        self.assertIsNone(slots[2].line)
        self.assertEqual(slots[0].color, (255, 82, 96, 255))
        self.assertEqual(slots[2].color, (46, 204, 113, 255))

    def test_event_metric_rows_use_selected_raw_angle_and_signed_differences(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "instance_id": "boat:1"},
            line=(0.0, 50.0, 100.0, 50.0),
            color=(255, 181, 71, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "instance_id": f"paddle:{index}"},
                line=(center_x - 10.0, 30.0, center_x + 10.0, 30.0 + rise),
                color=(53, 194, 255, 255),
            )
            for index, (center_x, rise) in enumerate(
                [(20.0, 12.0), (50.0, 20.0), (80.0, 35.0)],
                start=1,
            )
        ]
        options = ExportOptions(
            reference_prompt_id="boat",
            target_prompt_ids=("paddle",),
            target_slot_count=3,
            event_paddle_index=2,
        )
        event = PaddleEvent(
            kind="catch",
            timestamp_ms=1000,
            instance_id="slot:2",
            line=targets[1].line,
            confidence=1.0,
            phase_angle=45,
            travel_direction="right",
        )

        values = _event_metric_values([reference, *targets], options, event)
        column = EventMetricColumn(event=event, values=values)

        self.assertEqual(values, (15, 45, -14))
        self.assertEqual(
            [_event_metric_text(column, index, 1) for index in range(3)],
            ["+15°", "45°", "-14°"],
        )
        self.assertEqual(_aligned_companion_degree(31, 149), 149)

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
