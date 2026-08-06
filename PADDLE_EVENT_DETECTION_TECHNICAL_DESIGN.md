# Paddle Catch and Exit Event Detection — Technical Design

## 1. Status

- Status: Implemented
- Scope: Side-view sprint-kayak videos
- Source requirement: [`PADDLE_EVENT_DETECTION.md`](PADDLE_EVENT_DETECTION.md)
- Primary integration point: `src/sam3_service/exporter.py`

## 2. Summary

The event detector will model each physical paddle as a directed, continuous
rotation from `0°` through `360°`. The detected rotation direction determines
the kayak travel direction:

- anticlockwise paddle rotation means the kayak travels left;
- clockwise paddle rotation means the kayak travels right.

The detector will normalize both cases into the same local coordinate system.
Its zero ray points forward along the waterline, and increasing angle initially
points below the waterline. One phase is one complete `360°` revolution of the
same physical blade.

Within a phase, the detector may emit at most one catch and one exit for the
camera-visible blade. Catch and exit are independent. Their eligibility resets
at every new phase even if either event was missed in the preceding phase.

Event detection is based only on observed two-dimensional geometry. The system
does not synthesize an event for a stroke hidden on the other side of the
camera and does not attempt to infer unsupported camera-side depth.

## 3. Goals

1. Determine whether a tracked paddle rotates clockwise or anticlockwise.
2. Derive the kayak travel direction from the rotation direction.
3. Produce a stable directed paddle angle in `[0°, 360°)` relative to the
   detected waterline.
4. Divide the paddle motion into complete `360°` phases.
5. Detect the first visible catch and first visible exit in each phase.
6. Prevent an opposite-blade transition from generating a second event in the
   same revolution.
7. Recover automatically after missed detections, short tracking gaps, and SAM
   track-ID changes.
8. Preserve the first-transition timestamp while using temporal confirmation
   to reject single-frame noise.
9. Preserve a non-decreasing paddle length during each directed `0°-180°`
   stroke phase while its active blade is partially occluded by water.
10. Treat the fitted waterline as infinite for event analysis while drawing it
    only across the detected boat's longitudinal bow-to-stern span.
11. Suppress a below-boat mirrored paddle candidate when a distinct above-boat
    candidate is detected near the same boat.

## 4. Non-goals

- Reconstructing the three-dimensional paddle trajectory from one side-view
  camera.
- Determining camera-side depth when the visible evidence is ambiguous.
- Generating catch or exit events for fully occluded strokes.
- Replacing SAM segmentation, centerline fitting, or waterline fitting.
- Changing the existing export API or freeze-frame presentation in the first
  implementation.

## 5. Terminology

### 5.1 Physical paddle

A paddle identity maintained across model track-ID changes and short detection
gaps. One physical paddle owns one rotation and event state.

### 5.2 Blade endpoints

The two endpoints of the stabilized paddle centerline. Endpoint identity is
maintained by temporal continuity rather than by the arbitrary endpoint order
returned by line fitting.

### 5.3 Active blade

The blade endpoint associated with the first reliable visible waterline
transition in a track segment. Once anchored, only this endpoint can generate
catch and exit events. The other endpoint is the opposite blade and is ignored
for event generation until the track segment is reinitialized.

The first reliable transition may be either catch or exit. This allows a video
that starts mid-stroke, or a cycle with a missed catch, to establish the active
blade from an exit.

### 5.4 Phase

One complete `360°` revolution of the active blade. A phase has a monotonically
increasing unwrapped angle interval:

```text
[cycle_index × 360°, (cycle_index + 1) × 360°)
```

### 5.5 Catch

The first confirmed transition of the active blade from the dry side of the
waterline into the waterline band or underwater side during a phase.

### 5.6 Exit

The first confirmed transition of the active blade from the underwater side of
the waterline into the waterline band or dry side during a phase.

Catch does not require a previously detected exit, and exit does not require a
previously detected catch.

## 6. Coordinate System and Angle Definition

Image coordinates use `x` increasing to the right and `y` increasing toward
the bottom of the image.

For each matched waterline, construct an orthonormal local basis:

- `forward`: the normalized waterline tangent pointing in the kayak's detected
  travel direction;
- `down`: the waterline normal pointing toward the bottom of the image.

The waterline tangent is oriented as follows:

- left-moving kayak: `forward.x < 0`;
- right-moving kayak: `forward.x > 0`.

Let `blade_vector` be the vector from the stabilized paddle center to the
active blade endpoint. The normalized phase angle is:

```text
angle = degrees(atan2(dot(blade_vector, down),
                      dot(blade_vector, forward))) mod 360
```

This definition produces the required mirrored behavior:

| Travel | Rotation | `0°` direction | Increasing angle initially points |
|---|---|---|---|
| Left | Anticlockwise | Left along the waterline | Below the waterline |
| Right | Clockwise | Right along the waterline | Below the waterline |

The angle used for event phase control is the directed `[0°, 360°)` angle. The
existing acute paddle-to-waterline angle may remain available separately for
the export label and angle arc.

## 7. High-level Architecture

Event analysis will use two logical passes. Export rendering remains a later,
independent operation.

### 7.1 Pass A: observation tracking and direction estimation

1. Load and scale all result records.
2. Match every paddle observation to its nearest selected waterline.
3. Consolidate nearby collinear fragments into one physical observation.
4. When multiple candidates are near one boat, discard below-water candidates
   if at least one above-water candidate exists.
5. Associate observations with persistent physical-paddle tracks.
6. Stabilize line length and endpoint ordering.
7. Extend the fitted waterline slope to the projected span of the matched boat
   centerline.
8. Estimate rotation direction from the temporal paddle-axis sequence.
9. Derive travel direction and a confidence score.

### 7.2 Pass B: phase and event analysis

1. Build the direction-normalized waterline basis.
2. Anchor or maintain active-blade endpoint identity.
3. Calculate directed and unwrapped blade angles.
4. Assign every observation to a `cycle_index`.
5. Classify the active blade as above, within, or below the waterline band.
6. Start and confirm catch/exit candidates.
7. Enforce one event of each kind per cycle.
8. Deduplicate nearby same-kind events as a final safeguard.

This separation permits direction to be estimated from the full track before
events near the beginning of the video are evaluated.

## 8. Rotation and Travel-direction Estimation

### 8.1 Undirected axis samples

Before active-blade identity is known, the paddle centerline is an undirected
axis and therefore has a `180°` ambiguity. For each stabilized observation:

```text
axis_angle = degrees(atan2(dy, dx)) mod 180
```

For adjacent samples, calculate the shortest modulo-`180°` change in
`[-90°, 90°)`. In image coordinates:

- positive change is clockwise;
- negative change is anticlockwise.

Samples are rejected when the time gap, center displacement, line-angle jump,
or line-length change makes the association unreliable.

### 8.2 Robust decision

Rotation is locked only after a configurable minimum number of valid deltas,
minimum accumulated angular displacement, and directional consensus. A robust
median or trimmed weighted mean is used rather than a single-frame slope.

Initial values:

| Parameter | Initial value |
|---|---:|
| Minimum valid direction deltas | 5 |
| Minimum accumulated displacement | `45°` |
| Minimum same-sign consensus | `75%` |
| Maximum direction sample gap | `400 ms` |

The result is mapped to travel direction:

```text
median angular velocity < 0  => anticlockwise => left
median angular velocity > 0  => clockwise     => right
```

Direction is preferably estimated by consensus across all physical paddles
matched to the same boat/reference track. A low-confidence or conflicting
result produces no event for the affected segment; it must not guess a travel
direction.

If a sustained reversal is detected, the track is split into a new direction
segment and its phase/event state is reinitialized.

## 9. Endpoint Identity and Active-blade Bootstrap

Line-fit endpoint ordering is not stable. Each new line is oriented to minimize
the sum of distances between its endpoints and the preceding stabilized
endpoints. Short gaps may use constant-angular-velocity prediction before this
matching step.

Before the active blade is anchored, the detector evaluates waterline
transitions for both temporally oriented endpoints. The first reliable catch or
exit transition:

1. anchors that endpoint as the active blade;
2. initializes its directed angle parity;
3. starts the corresponding event candidate;
4. leaves the opposite endpoint ineligible for events.

The event may be emitted after normal temporal confirmation. Bootstrap does not
discard the first visible event.

This design treats the camera-side limitation as an evidence constraint. If a
far-side stroke is fully hidden, it supplies no reliable transition and cannot
be selected. If both sides are equally visible in the two-dimensional input,
camera-side depth is inherently ambiguous; the chosen active endpoint and
confidence should be exposed in diagnostics.

## 10. Angle Unwrapping and Phase Progression

After endpoint identity is anchored, choose the equivalent directed angle
closest to the predicted angle from the prior sample. This creates a continuous
unwrapped angle:

```text
..., 342°, 355°, 367°, 381°, ...
```

The phase is:

```text
cycle_index = floor(unwrapped_angle / 360°)
phase_angle = unwrapped_angle - cycle_index × 360°
```

When `cycle_index` advances, the new cycle starts with independent event flags:

```text
catch_emitted = false
exit_emitted = false
```

The reset does not inspect the preceding cycle. It therefore occurs when:

- both events were detected;
- only catch was detected;
- only exit was detected; or
- neither event was detected.

Backward phase movement caused by small jitter is ignored. A sustained true
rotation reversal creates a new direction segment rather than decrementing the
cycle index.

## 11. Waterline-transition Geometry

### 11.1 Signed depth

For each endpoint, calculate signed perpendicular depth relative to the
waterline:

```text
depth > 0 => below the waterline
depth < 0 => above the waterline
```

Use a finite-width band to tolerate fitted-line jitter:

The event waterline has an asymmetric 8-pixel band that extends upward from the
fitted waterline, rather than four pixels to each side:

```text
ABOVE: depth < -band_upward_width
BAND:  -band_upward_width <= depth <= 0
BELOW: depth > 0
```

The default `band_upward_width` is `8px`, scaled with the analyzed video. Catch
enters the upper edge at `-8px`; exit enters the lower edge at the fitted
waterline (`0px`).

The robust waterline fitter continues to estimate slope from the central hull
boundary, excluding curved and noisy bow/stern pixels. The fitted direction has
two uses:

1. event geometry treats it as an infinite line for signed depth, paddle
   intersection, and angle estimation, so an incomplete boat mask cannot make a
   valid crossing miss the fitted waterline merely because it is outside the
   detected segment;
2. export rendering projects the detected boat centerline endpoints onto the
   fitted slope and draws only that finite bow-to-stern span.

### 11.2 Catch candidate

Start a catch candidate when all conditions hold:

1. the active endpoint moves from `ABOVE` toward `BAND` or `BELOW`, including a
   direct `ABOVE -> BELOW` transition at low frame rates;
2. signed depth is increasing beyond a small motion epsilon;
3. the active blade zone overlaps the infinite fitted waterline;
4. no catch has already been emitted for the candidate's cycle.

### 11.3 Exit candidate

Start an exit candidate when all conditions hold:

1. the active endpoint moves from `BELOW` toward `BAND` or `ABOVE`, including a
   direct `BELOW -> ABOVE` transition;
2. signed depth is decreasing beyond a small motion epsilon;
3. the active blade zone overlaps the infinite fitted waterline;
4. no exit has already been emitted for the candidate's cycle.

The signed-depth state represents immersion across the complete underwater
portion of the stroke. It avoids interpreting separation from a narrow
waterline band while the blade is moving deeper as an exit.

### 11.4 Stroke-phase paddle-length reconstruction

Within each directed `0°-180°` stroke phase, maintain a phase-local accepted
paddle length:

1. maintain temporal endpoint ordering;
2. retain the identified active immersed blade for the phase;
3. initialize the accepted length from the first usable observation in the
   phase, including the immediately preceding observation when the active blade
   is first identified mid-phase;
4. increase the accepted length when a longer valid line is observed, but never
   decrease it within the same phase;
5. when the observed line is shorter, anchor the inactive endpoint and extend
   only the active endpoint along the current observed axis to the accepted
   length;
6. use the reconstructed line for signed-depth classification, catch/exit
   detection, event angle, and event geometry.

Track association may retain a generically stabilized line for identity and
angle continuity, but event reconstruction also retains the pre-stabilized,
consistently oriented line. The phase-local reconstruction operates on that raw
line so generic stabilization cannot anchor the cropped active endpoint before
the active-blade-aware rule runs.

The accepted length remains inherited after exit until the directed angle moves
beyond `180°`. It is not inherited into the recovery half of the revolution or
the next `0°-180°` stroke phase. A continuity-breaking tracking gap also clears
it.

## 12. Candidate Confirmation and Timestamping

A transition is confirmed by two consecutive compatible observations. The
event is backdated to the timestamp and geometry of the first transition
observation.

Each pending candidate stores:

```text
kind
cycle_index
first_timestamp_ms
first_line
first_reference_line
first_phase_angle
active_endpoint
confirmation_count
confidence
```

A candidate expires after `400 ms` or when subsequent geometry contradicts its
motion direction. A candidate remains assigned to the cycle containing its
first observation even if confirmation arrives immediately after a phase
boundary. Event uniqueness is checked using `(physical_id, cycle_index, kind)`.

Catch and exit candidates are independent in their eligibility. Confirming or
missing one kind does not alter the other kind's phase gate.

## 13. State Model

Each physical paddle direction segment owns state equivalent to:

```text
PaddleTrackState
  physical_id
  reference_id
  source_ids
  last_seen_ms
  stabilized_line
  recent_pre_catch_lengths
  stroke_length
  stroke_blade
  stroke_cycle_index
  endpoint_tracks[2]
  rotation_direction
  travel_direction
  direction_confidence
  active_endpoint
  last_directed_angle
  unwrapped_angle
  cycle_index
  endpoint_surface_state
  emitted[(cycle_index, catch)]
  emitted[(cycle_index, exit)]
  pending_catch
  pending_exit
  phase_confident
```

The event state machine for each event kind is:

```text
ELIGIBLE
  | matching first transition
  v
PENDING
  | confirmed                         | contradicted/expired
  v                                   v
EMITTED_FOR_CYCLE                   ELIGIBLE
  | next 360° cycle
  v
ELIGIBLE
```

Catch and exit run separate instances of this state machine.

## 14. Missing Data and Recovery

### 14.1 Short observation gap

For a gap no longer than the `400 ms` event/direction continuity window:

- preserve physical identity and active endpoint;
- predict endpoint orientation for association;
- do not synthesize waterline transitions inside the gap;
- on return, update the current surface state from observed geometry;
- allow a later independent catch or exit transition in the same phase.

For a longer gap that is still within the `1500 ms` physical-track association
window, preserve the physical identity but clear active-blade, phase, pending
candidate, per-cycle eligibility, and catch-scoped length state. The first
returned observation is used only as a new baseline; a later reliable
transition re-anchors the active blade. This prevents a transition from being
synthesized across missing frames.

### 14.2 Ambiguous phase after a gap

If more than one endpoint parity or more than one `180°` continuation is
plausible, mark the phase uncertain. Start a new track segment and wait for a
new reliable transition rather than risking a false event or wrong cycle.

### 14.3 Track-ID change

Continue the physical state when spatial position, predicted motion, line
orientation, length, and reference association pass the existing physical
matching gates. Raw model track-ID agreement is supporting evidence only.

### 14.4 Missing event

No special recovery action is required. The next `360°` phase automatically
creates fresh catch and exit eligibility.

## 15. Multiple Paddles and Boats

- State is independent per physical paddle.
- Each paddle is assigned to its nearest eligible boat/reference waterline.
- If multiple distinct candidates are close to the same boat and at least one
  candidate center is above its waterline, below-water candidates are treated
  as reflections and excluded before tracking and direction estimation.
- The reflection rule does not remove a lone below-water candidate and does not
  collapse multiple genuine candidates whose centers are above the boat.
- Rotation direction is estimated per boat direction segment using consensus
  across its paddles when possible.
- Spatially separate crew paddles must not share active-blade or event state.
- Existing post-analysis temporal/spatial deduplication remains a safeguard for
  fragmented detections, not the primary uniqueness mechanism.
- Events from synchronized paddles may continue to share one export freeze
  moment under the existing `250 ms` grouping rule.

## 16. Event Output

The public export request remains unchanged. Internally, extend event metadata
to include diagnostics while retaining current rendering fields:

```text
PaddleEvent
  kind: "catch" | "exit"
  timestamp_ms
  physical_id
  line
  reference_line
  degree                 # existing acute display angle
  phase_angle            # new directed [0, 360) angle
  cycle_index
  active_endpoint
  rotation_direction
  travel_direction
  confidence
```

The freeze frame displays `phase_angle` in the full directed `[0°, 360°)` range.
Catch angle arcs and labels are red; exit angle arcs and labels are green. The
legacy acute `degree` remains available for compatibility and non-event angle
metrics.

## 17. Confidence

Event confidence should combine:

- rotation-direction confidence;
- physical-track association confidence;
- endpoint-parity confidence;
- signed-depth margin and motion consistency;
- finite-waterline intersection validity;
- number and spacing of confirmation samples;
- centerline and waterline fit confidence when available.

Low direction or endpoint confidence suppresses the event. The detector should
prefer a missed event over a confidently wrong phase or camera-side guess.

## 18. Implementation Plan

### 18.1 Module boundary

The initial implementation remains in `src/sam3_service/exporter.py` so it can
reuse the existing selection, centerline, physical-association, geometry,
deduplication, and freeze-rendering helpers without introducing a circular
interface. The event-analysis functions form a separate internal section and
run before rendering. They can move to `paddle_events.py` after the observation
and event interfaces stabilize without changing the export API.

### 18.2 Reusable current behavior

Retain or extract the existing implementations for:

- paddle-fragment consolidation;
- physical-paddle association;
- stabilized line length;
- temporal endpoint orientation;
- finite segment geometry;
- two-sample confirmation and backdating;
- nearby-event deduplication;
- export freeze grouping and rendering.

### 18.3 Behavior to replace

Remove the event gate based on the undirected acute `0° -> 90° -> 0°` reversal
sequence. Replace its four-phase counter with the directed unwrapped `360°`
cycle model described here. Do not retain a two-phase opposite-blade exception.

### 18.4 Compatibility

- Existing API query parameters retain their behavior. The optional
  `include_event_paddle_length` parameter defaults to `false` and changes only
  event-label rendering.
- `include_catch`, `include_exit`, and `event_hold_seconds` keep their existing
  meaning.
- The event analysis pass remains offline and deterministic for a fixed result
  manifest.
- Existing export selection, metric count, audio freeze, and rendering logic
  are unaffected.
- Update `SAM3_API_DEVELOPMENT_GUIDE.md` when the implementation lands because
  it currently documents the old four-acute-phase algorithm.

## 19. Configuration

Initial tunable values should be centralized rather than embedded throughout
the state machine:

| Parameter | Initial value |
|---|---:|
| Event confirmation samples | `2` |
| Candidate maximum age | `400 ms` |
| Physical-track gap | `1500 ms` |
| Direction minimum deltas | `5` |
| Direction minimum displacement | `45°` |
| Direction consensus | `75%` |
| Direction sample maximum gap | `400 ms` |
| Event deduplication window | `250 ms` |
| Waterline band half-width | Derived from fitted line thickness, minimum `1 px` |

Thresholds must be validated against representative left-moving and
right-moving videos before being exposed as user-facing configuration.

## 20. Testing Strategy

### 20.1 Geometry unit tests

- The waterline basis points forward and down for sloped lines.
- Left/anticlockwise and mirrored right/clockwise samples produce the same
  normalized phase angles.
- Endpoint signed depth is positive below the waterline.
- Crossings outside the finite drawn waterline extent are accepted against the
  infinite analysis waterline.
- Endpoint order swaps do not change active-blade identity.
- A central fitted waterline extends to the projected bow-to-stern boat span.
- A cropped immersed endpoint is reconstructed without moving the dry endpoint.
- A below-boat mirrored candidate is removed when its above-boat counterpart is
  present near the same reference.

### 20.2 Direction unit tests

- Noisy anticlockwise samples resolve to left.
- Noisy clockwise samples resolve to right.
- Oscillation without sufficient displacement remains unknown.
- Conflicting paddle tracks suppress boat direction until consensus exists.
- A sustained true reversal creates a new segment.

### 20.3 Phase unit tests

- `355° -> 4°` advances exactly one cycle.
- Small wrap-boundary jitter does not advance multiple cycles.
- An opposite endpoint transition near `180°` does not emit an event.
- Each full active-blade revolution advances exactly one cycle.
- Ambiguous continuation after a long gap starts a new uncertain segment.

### 20.4 Event-state unit tests

- Dry-to-underwater motion emits one backdated catch.
- Underwater-to-dry motion emits one backdated exit.
- Deepening below the waterline does not emit exit.
- Two catches in one cycle emit only the first.
- Two exits in one cycle emit only the first.
- A missed catch does not block exit.
- A missed exit does not block the next cycle's catch.
- Missing both events does not block either event in the next cycle.
- A candidate confirming across a phase boundary keeps its first-frame cycle.
- A single-frame waterline touch emits no event.
- Paddle length never decreases within one `0°-180°` stroke phase; cropped
  observations anchor the inactive endpoint and extend the active endpoint.
- Stroke length remains available after exit, then clears beyond `180°`, at the
  next phase, and after a continuity-breaking gap.

### 20.5 Integration tests

- Left-moving, anticlockwise fixture with visible catch and exit.
- Horizontally mirrored right-moving, clockwise fixture with equivalent output.
- Video beginning mid-underwater stroke detects exit without requiring catch.
- Short SAM track-ID break preserves physical paddle and phase.
- Long ambiguous gap reinitializes without a false event.
- Multiple paddles retain independent cycles and synchronized freeze grouping.
- Catch-only, exit-only, and catch-plus-exit export options filter output without
  changing analysis results.

## 21. Acceptance Criteria

The implementation is acceptable when all of the following hold:

1. Rotation direction determines travel direction according to the requirement.
2. Both travel directions use a directed normalized angle in `[0°, 360°)`.
3. One physical active blade defines one phase per complete revolution.
4. At most one catch and one exit are emitted per physical paddle per phase.
5. Catch is the first confirmed visible dry-to-water crossing in the phase.
6. Exit is the first confirmed visible water-to-dry separation in the phase.
7. Catch and exit do not depend on one another.
8. Phase eligibility resets after every `360°`, including cycles with missed
   events.
9. The opposite endpoint cannot create a second catch or exit in the cycle.
10. Hidden strokes do not generate synthetic events.
11. Mirrored left/right test sequences produce equivalent normalized results.
12. Existing export API and freeze rendering remain backward compatible.
13. A paddle crossing anywhere within the detected bow-to-stern span remains
    eligible for catch or exit.
14. Partial underwater segmentation does not shorten the active paddle line
    used for exit detection during a confirmed stroke.
15. A mirrored below-boat paddle does not create a second physical track or
    contribute to rotation and event estimation when the real paddle is visible.

## 22. Diagnostics and Validation

During development, optionally write a machine-readable diagnostic trace per
physical paddle containing:

```text
timestamp_ms
physical_id
source_ids
rotation_direction
direction_confidence
active_endpoint
phase_angle
unwrapped_angle
cycle_index
signed_depth
surface_state
candidate_kind
emitted_event
event_confidence
```

An optional debug overlay should show the forward waterline ray, active blade,
directed phase angle, cycle index, and signed-depth state. These diagnostics are
development aids and are not required in the normal exported video.
