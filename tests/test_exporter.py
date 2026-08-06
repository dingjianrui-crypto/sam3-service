from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sam3_service.exporter import (
    Centerline,
    DegreeLabel,
    ExportOptions,
    PaddleEvent,
    SpmEstimator,
    _PaddleObservation,
    _PaddleEventState,
    _TimedPaddleObservation,
    _advance_directed_paddle_phase,
    _blade_waterline_overlaps,
    _catch_phase_allowed,
    _consolidate_paddle_observations,
    _dedupe_paddle_events,
    _detect_paddle_events,
    _draw_frame_overlay,
    _degree_label_entries,
    _degree_labels,
    _degree_slots,
    _directed_blade_angle,
    _estimate_paddle_direction,
    _event_phase_allowed,
    _event_label_text,
    _freeze_audio_filter,
    _freeze_moments,
    _freeze_segments,
    _line_intersection,
    _load_frames_by_index,
    _metric_label_top,
    _maximum_target_count_in_selection,
    _paddle_water_depth_ratio,
    _paddle_rotation_deltas,
    _paddle_event_angle_color,
    _record_line,
    _record_selected_for_export,
    _restore_immersed_paddle_length,
    _resolve_requested_track_ids,
    _resolved_export_frame_count,
    _selection_rect_at,
    _spm_label_top,
    _update_directed_paddle_state,
    _update_paddle_event_state,
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

    def test_directed_detector_backdates_catch_and_exit(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events = []
        for index, angle in enumerate([0, 30, 45, 60, 90, 120, 135, 150]):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                4.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual([(event.kind, event.timestamp_ms) for event in events], [
            ("catch", 200),
            ("exit", 600),
        ])
        self.assertEqual([event.cycle_index for event in events], [0, 0])
        self.assertEqual([event.active_blade for event in events], [1, 1])
        self.assertEqual(events[0].travel_direction, "right")
        self.assertAlmostEqual(events[0].phase_angle or 0, 45)
        self.assertEqual(_event_label_text(events[0]), "45°")
        self.assertEqual(_event_label_text(events[1]), "135°")
        self.assertEqual(_paddle_event_angle_color(events[0]), (255, 82, 96, 255))
        self.assertEqual(_paddle_event_angle_color(events[1]), (46, 204, 113, 255))

    def test_catch_length_restores_cropped_immersed_blade_until_exit(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )
        events: list[PaddleEvent] = []
        lengths_at_event: list[float] = []
        for index, angle in enumerate([0, 30, 45, 60, 90, 120, 135, 150]):
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
                4.0,
            )
            if event is not None:
                events.append(event)
                lengths_at_event.append(math.dist(event.line[:2], event.line[2:]))

        self.assertEqual([event.kind for event in events], ["catch", "exit"])
        self.assertAlmostEqual(lengths_at_event[0], 60.0)
        self.assertAlmostEqual(lengths_at_event[1], 60.0)
        self.assertIsNone(state.stroke_length)

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
                4.0,
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
        for index, angle in enumerate([90, 120, 135, 150]):
            event = _update_directed_paddle_state(
                state,
                _rotating_observation(angle),
                index * 100,
                4.0,
            )
            if event is not None:
                events.append(event)

        self.assertEqual([(event.kind, event.timestamp_ms) for event in events], [
            ("exit", 200),
        ])
        self.assertNotIn((0, "catch"), state.emitted_events)

    def test_event_analysis_pipeline_handles_both_travel_directions(self) -> None:
        for travel_direction in ("right", "left"):
            with self.subTest(travel_direction=travel_direction):
                frames: dict[int, list[dict[str, object]]] = {}
                for index, angle in enumerate([0, 15, 30, 45, 60, 90, 120, 135, 150]):
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
                        ("catch", 300, travel_direction),
                        ("exit", 700, travel_direction),
                    ],
                )

    def test_gap_does_not_synthesize_a_waterline_transition(self) -> None:
        state = _PaddleEventState(
            physical_id="paddle:physical:1",
            rotation_direction="clockwise",
            travel_direction="right",
            direction_confidence=1.0,
        )

        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(30), 0, 4.0)
        )
        state.stroke_length = 60.0
        state.stroke_blade = 1
        state.stroke_cycle_index = 0
        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(60), 500, 4.0)
        )
        self.assertIsNone(
            _update_directed_paddle_state(state, _rotating_observation(90), 600, 4.0)
        )
        self.assertEqual(state.emitted_events, set())
        self.assertIsNone(state.stroke_length)

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
