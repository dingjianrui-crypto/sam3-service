# Canoe Direction, Paddle Angle, Catch, and Exit Detection — Technical Design

## 1. Status

- Status: Initial export implementation complete; real-video calibration pending
- Scope: Side-view canoe videos in which the camera commonly moves with the boat
- Inputs already available: tracked paddle centerline and tracked boat waterline
- Intended integration point: a canoe-specific analysis strategy called by the export pipeline
- Related implementation: `src/sam3_service/exporter.py`
- Related kayak design: [`PADDLE_EVENT_DETECTION_TECHNICAL_DESIGN.md`](PADDLE_EVENT_DETECTION_TECHNICAL_DESIGN.md)

This document records the aligned design decisions for implementation. Numeric
thresholds described as initial values are calibration parameters, not finalized
product requirements.

The first implementation selects this strategy from the persisted job setting
`settings.paddling_discipline`. It reuses stabilized paddle/boat tracks, estimates
full length from the longest tracked observation, detects shortened-paddle contact
intervals, requires two pull intervals for direction, calculates a signed angle,
and emits backdated catch/exit events. Completeness-assisted confidence, recovery
scoring, direction-segment reversal, and production threshold tuning remain
follow-up work.

## 2. Summary

Canoe paddle motion is a pendulum-like cycle rather than the continuous
rotation modeled by the kayak detector:

```text
forward recovery -> catch -> sternward pull -> exit -> forward recovery
```

The canoe pipeline must therefore not use the kayak rule that maps clockwise
rotation to rightward travel and anticlockwise rotation to leftward travel. For
the agreed upright, non-mirrored production camera convention, it primarily
infers travel from body pose. A vertical screen axis passes through each hip.
When both shoulders lie to the right of their corresponding hip axes the frame
votes right; when both lie to the left the frame votes left. Canonical
paddle-axis evidence remains a fallback when completed pose data is unavailable.

After direction bootstrap, the pipeline calculates a direction-normalized phase
angle. One phase is small-to-large-to-small. Catch is the first rising
waterline crossing of a blade-endpoint-restored line; exit is the peak-angle
frame after persistent decreasing angle evidence confirms the reversal.

The implementation is a buffered, two-pass analysis:

1. Build boat-relative paddle observations without assuming travel direction.
2. Infer canoe travel direction from bilateral shoulder-to-hip horizontal votes,
   with canonical paddle-axis evidence as fallback.
3. Orient the boat basis in the detected direction.
4. Reprocess buffered observations to calculate signed angles and detect catch,
   pull, exit, and recovery.

Insufficient evidence produces an unknown direction and no canoe events. The
pipeline must not guess or fall back to kayak direction logic.

## 3. Goals

1. Remove camera/boat screen translation by measuring paddle motion relative to
   the detected waterline in every frame.
2. Maintain stable paddle, boat-reference, and blade-endpoint identities.
3. Bootstrap left/right canoe travel from complete immersed pull intervals.
4. Produce a direction-normalized signed paddle angle in `[-180°, 180°)`.
5. Preserve the real angular reversal between pull and recovery rather than
   unwrapping it into a kayak-style `360°` revolution.
6. Detect catch from immersion and visible-length shortening.
7. Detect exit from sustained visible-length recovery after the immersed
   minimum, confirmed by water and motion evidence.
8. Backdate confirmed events to their first supported physical transition.
9. Suppress events when paddle, endpoint, waterline, or direction evidence is
   unreliable.
10. Share existing tracking, selection, rendering, and export infrastructure
    without changing kayak behavior.

## 4. Non-goals

- Inferring canoe direction from screen-space boat movement.
- Reusing clockwise/anticlockwise rotation as the canoe direction signal.
- Reconstructing three-dimensional paddle or athlete motion.
- Emitting events for intervals hidden by long occlusions.
- Treating a single length change or a single waterline crossing as a confirmed
  event.
- Finalizing thresholds without representative canoe footage.
- Replacing segmentation, centerline fitting, waterline fitting, or the existing
  mask-completeness classifier.

## 5. Terminology

### 5.1 Canonical boat axis

Before travel direction is known, the stabilized waterline tangent is oriented
left-to-right in image space. It provides a consistent axis for direction votes
without claiming which end is the bow.

### 5.2 Directed boat basis

After direction is known, `forward` is the stabilized waterline tangent pointing
in the detected travel direction. `down` is its perpendicular normal pointing
toward image-bottom/water.

### 5.3 Boat-relative paddle slot

A canoe paddle identity is a stable slot on its associated boat, not a raw SAM
track ID. The pipeline projects paddle centers onto the detected travel-oriented
boat axis, derives stable front-to-back anchors, and assigns one candidate per
slot in each frame. Raw track IDs are candidate evidence only and may change
without creating a new paddle identity.

### 5.4 Blade endpoint

The endpoint of the paddle centerline that enters the water. It is selected from
water-contact evidence and then maintained by temporal continuity. It must not
be inferred from arbitrary fitted-line endpoint order.

### 5.5 Dry endpoint

The endpoint opposite the blade. During immersion it normally remains above the
water and anchors the visible shaft.

### 5.6 Visible above-water length

The distance from the dry endpoint to the paddle/waterline intersection. It is
not the raw detected centerline length. It is normalized by a stable full-paddle
length estimate.

### 5.7 Provisional entry and release

Water-contact transitions detected before travel direction is known. Direction
bootstrap uses `entry -> immersed interval -> release`; only the directed second
pass names confirmed transitions catch and exit.

## 6. Boat-relative Coordinate System

Image coordinates have `x` increasing right and `y` increasing down. For every
frame, stabilize the selected boat waterline and define:

```text
origin    = center of stabilized waterline
tangent   = normalized waterline direction, canonicalized so tangent.x >= 0
down      = perpendicular(tangent), flipped so down.y >= 0
boat_size = stabilized waterline or boat-centerline projected length
```

For a paddle point `p`:

```text
longitudinal(p) = dot(p - origin, tangent) / boat_size
depth(p)        = dot(p - origin, down) / boat_size
```

These normalized measurements remove shared camera/boat translation and reduce
sensitivity to resolution or zoom. A missing or unstable waterline may use the
existing boat-centerline fallback, but the observation must record which
reference was used and lower confidence accordingly.

The waterline endpoint order must never be treated as bow direction.

## 7. Observation Construction and Tracking

The canoe strategy may reuse the existing fragment consolidation, stable
physical-paddle tracking, reflection filtering, and boat-reference association.
Each observation additionally needs:

```text
timestamp_ms
boat_slot_id
reference_boat_id
stabilized_paddle_line
stabilized_reference_line
blade_endpoint and blade_confidence
dry_endpoint
blade_longitudinal_position
blade_depth
visible_air_length
normalized_air_length
full_length_estimate and full_length_confidence
paddle_completeness, when available
observation_confidence
```

Tracking or reference switches divide the data into new analysis segments. Do
not calculate motion deltas across such boundaries or across a gap longer than
the configured continuity window.

## 8. Blade Endpoint Bootstrap

Before blade identity is known, calculate signed depth for both endpoints over a
short observation window. The blade candidate is the endpoint that repeatedly:

- approaches, intersects, or passes below the waterline;
- becomes the truncated endpoint while visible paddle length shortens; and
- maintains plausible spatial continuity between frames.

Once established, maintain endpoint identity using the stabilized line and
nearest endpoint correspondence. Preserve it through above-water recovery.

Do not publish a directed angle when blade identity is uncertain. A short gap may
preserve the prior identity; a long gap or an implausible endpoint jump starts a
new endpoint bootstrap. A silent endpoint swap would shift the signed angle by
approximately `180°` and corrupt both phase and events.

## 9. Full and Above-water Paddle Length

Estimate full paddle length from high-confidence complete detections during
above-water recovery. The existing paddle-completeness classifier may gate
eligible samples. Use a robust statistic over recent accepted samples rather
than allowing a single long detection to change the baseline.

When the paddle line intersects the waterline:

```text
visible_air_length = distance(dry_endpoint, waterline_intersection)
normalized_air_length = visible_air_length / full_length_estimate
```

When the complete paddle is confidently above water, the visible length may use
the complete stabilized line length. If no reliable intersection or full-length
estimate exists, mark the length observation unavailable instead of substituting
stale geometry.

Use the visible-length time series only after smoothing short segmentation noise.
Smoothing must not bridge long detection gaps or erase genuine minima and trend
reversals.

## 10. Provisional Water-contact Intervals

Before direction is known, classify blade contact using signed endpoint depth,
waterline intersection, visible-length behavior, and completeness evidence:

```text
ABOVE -> ENTRY_CANDIDATE -> IMMERSED -> RELEASE_CANDIDATE -> ABOVE
```

Use separate entry and release depth thresholds to provide hysteresis around the
waterline. Require persistence for multiple observations or a minimum duration.

A valid provisional contact interval contains:

```text
entry observation
continuous immersed pull observations
release observation
```

Reject it when:

- the paddle or boat reference changes;
- continuity gaps exceed the limit;
- blade identity becomes unreliable;
- immersion lasts too few observations;
- longitudinal pull displacement is too small relative to boat length; or
- entry and release evidence are both low confidence.

## 11. Canoe Travel-direction Bootstrap

Use body-motion landmarks first. A frame is eligible only when left/right
shoulders and left/right hips all pass the landmark confidence threshold. The
frame votes right when both `shoulder.x - hip.x` values are positive and left
when both are negative. Mixed-side frames cast no vote. Require at least five
eligible frames and select the simple majority; an exact tie remains unknown.
The winning share is reported as direction confidence.

If completed pose frames are unavailable or do not produce a direction, use the
following paddle-axis fallback.

Canonicalize each waterline screen-left to screen-right. For every paddle line,
calculate its undirected axis angle relative to that waterline in `[0°, 180°)`:

```text
axis_angle = (paddle_axis_angle - canonical_waterline_angle) mod 180°
```

| Axis angle | Travel vote |
|---|---|
| `0°-80°` | Right |
| `80°-100°` | No vote: vertical dead band |
| `100°-180°` | Left |

The dead band rejects near-vertical jitter. A paddle track must also span at
least `8°`; static fragments are excluded because this bootstrap relies on
observed paddle movement. Aggregate eligible observations per boat/reference
segment, weighting each by paddle length and distance from `90°`. Lock direction
only after at least five eligible observations and `70%` weighted consensus.

Direction output is:

```text
travel_direction: left | right | null
direction_method: body_motion | canoe_axis
direction_confidence: 0.0 to 1.0
supporting_strokes: integer
```

A single reverse stroke must not flip a locked direction. Sustained opposite
angle evidence across moving tracks starts a new direction segment and
reinitializes angle/event state.

## 12. Direction-normalized Signed Paddle Angle

Once direction is locked, orient the stabilized waterline tangent as `forward`:

- right travel: `forward.x > 0`;
- left travel: `forward.x < 0`.

Keep `down` pointing toward image-bottom/water. The `0°` ray is therefore the
waterline pointing in canoe travel direction. Orient the paddle vector from the
dry endpoint to the water-facing blade endpoint, then define:

```text
blade_vector = blade_endpoint - dry_endpoint
canoe_angle = degrees(atan2(dot(blade_vector, down),
                            dot(blade_vector, forward)))
```

Store the angle in `[-180°, 180°)`:

- `0°`: blade vector points forward toward the bow;
- `+90°`: blade vector points down toward the water;
- near `+180°` or `-180°`: blade vector points toward the stern;
- negative values place the blade above the forward/stern axis.

The angle is direction-normalized, so mirrored left- and right-moving canoes
share the same phase interpretation: small forward-side angle, increasing stroke
to a peak, then decreasing restore to the next small forward-side angle.

Processing order:

1. Stabilize the boat reference.
2. Maintain blade endpoint identity.
3. Calculate raw signed angle.
4. Unwrap only across the `-180°/180°` representation boundary.
5. Smooth short jitter.
6. Calculate angular velocity and a persistent trend.

Do not unwrap the pull/recovery reversal into a continuing revolution. Classify
small velocity as stationary and require a minimum displacement and persistence
before changing between increasing and decreasing trends. Initial trend concepts
to calibrate are `3°-5°` minimum displacement and `2-3` persistent frames.

Each directed observation should expose:

```text
canoe_angle_deg
canoe_angular_velocity_deg_per_second
angle_trend: increasing | decreasing | stationary
angle_confidence
```

## 13. Directed Stroke State Machine

After bootstrap, reprocess buffered observations through a canoe-specific state
machine:

```text
RECOVERY
  -> CATCH_CANDIDATE
  -> PULL
  -> EXIT_CANDIDATE
  -> RECOVERY
```

The video may begin or end in either state. The initial segment is retained as a
partial phase when it starts mid-stroke (`98° -> 120° -> 52°`) or mid-restore
(`120° -> 52°`). A phase boundary is the confirmed local minimum angle: it
closes the previous phase and is reused as the start of the next phase. A full
phase is therefore represented as `minimum frame -> peak frame -> next minimum
frame`; a trailing phase remains partial when the next minimum has not yet been
confirmed.

Phase angles use a slot-level robust boat reference direction rather than each
frame's raw waterline angle. Per-frame waterline geometry can jitter, especially
near the water surface, so it must not create artificial minima, peaks, or phase
boundaries. Catch crossing tests still use the event frame's local reference
line so the rendered event geometry remains tied to the visible waterline.

Expected signals are:

| Phase | Water state | Angle trend | Longitudinal motion | Air length |
|---|---|---|---|---|
| Recovery | Above | Toward forward/catch extreme | Toward bow | High/recovering |
| Catch candidate | Restored line crosses waterline | Increasing | Recovery stops | Shortening can support confidence |
| Pull | Immersed | Toward rear extreme | Toward stern | Low/near minimum |
| Exit candidate | Releasing | Near trend transition | Pull slows/reverses | Increasing |

Angle trend is validation, not the sole phase signal. The same inclination may
occur during pull and recovery. Water state, longitudinal motion, and visible
length distinguish them.

## 14. Catch Detection

Detect catch independently inside each broken-down phase. During the rising
portion from phase start to peak, seed paddle length from the phase-start
minimum-angle observation. Until the phase angle exceeds `90°`, a later shorter
detection inherits the previous phase length. A longer detection updates the
inherited length only when the growth is within `15%`; larger jumps inherit the
previous length as well. Restoration anchors the active endpoint and stretches
the dry endpoint backward along the detected centerline. Catch candidates are
the before/after frames around a restored
centerline-waterline crossing, and the selected catch is the candidate whose
active endpoint is closest to the local event-frame waterline.

Supporting evidence is:

- forward recovery motion slows or reverses into a sternward pull;
- the angle trend changes into the expected pull trend;
- subsequent observations remain immersed; and
- paddle completeness changes from complete toward cropped/partial.

The closest crossing-neighbor active-endpoint observation is the catch timestamp
and geometry. A phase with no rising portion or no restored crossing before
`90°` emits no catch. Visible-length shortening and immersion remain confidence
signals, not the primary timestamp selector.

Cancel the candidate if the blade returns above water, visible length recovers,
or pull motion fails to develop within the confirmation window.

## 15. Exit Detection

Detect exit independently inside each broken-down phase. The exit timestamp is
the phase peak frame. Confirm it only after the phase has a restore segment with
angle decrease by the phase trend threshold. If the video starts mid-restore and
the phase peak is also the first phase sample, require that peak's active
endpoint to be close to the local event-frame waterline; otherwise a high-angle
start frame away from water is not an exit. Later phase peaks do not use this
proximity gate. Store the peak-angle frame and geometry as the exit, not the
later confirming samples. Visible-length recovery and waterline release increase
confidence but do not replace the peak selector.

Cancel the candidate and return to `PULL` if the blade becomes deeper again,
length growth disappears, or recovery does not develop within the confirmation
window.

## 16. Hysteresis and Confidence

All thresholds should be normalized by stabilized boat or full-paddle length
where possible. Required independent signal groups prevent geometry jitter from
creating events.

Parameters to expose as named constants and calibrate include:

- maximum observation continuity gap;
- entry and release depth thresholds;
- minimum contact persistence;
- minimum normalized pull displacement;
- minimum visible-length decrease for catch;
- minimum visible-length growth and growth rate for exit;
- minimum trend displacement and persistence;
- minimum direction votes and consensus; and
- candidate confirmation timeout.

Event confidence combines endpoint, reference, length, water transition, motion,
angle trend, completeness, and temporal confirmation evidence. Missing optional
evidence may lower confidence; missing primary evidence suppresses the event.

## 17. Failure and Recovery Behavior

- No stable boat reference: no direction vote, directed angle, or event.
- Unknown travel direction: retain provisional observations but emit no canoe
  angle/event result.
- Uncertain blade endpoint: suppress directed angle and events until re-bootstrap.
- Short observation gap: preserve state without fabricating intermediate motion.
- Long gap or track/reference switch: begin a new analysis segment.
- Missing catch: an independently reliable release may be retained as provisional
  evidence, but the initial implementation should not emit an exit until a valid
  pull state has been established.
- Missing exit: time out the pull segment and resume endpoint/contact bootstrap;
  do not synthesize an exit.
- Conflicting paddles on the same boat: leave direction unknown until weighted
  consensus is restored.
- Reverse stroke: treat it as an outlier unless sustained complete-stroke evidence
  supports a new direction segment.

## 18. Architecture and Integration

Implement a discipline-specific strategy boundary rather than adding canoe
conditions throughout the kayak state machine:

```text
event analysis coordinator
  |- kayak strategy: rotation consensus and 360-degree phase
  `- canoe strategy: canonical axis-angle consensus and pendulum phase
```

Both strategies may return a shared high-level contract:

```text
discipline
travel_direction
direction_method
direction_confidence
events[]
diagnostics
```

Canoe-specific observations and state should use separate dataclasses and helper
functions. Do not overload kayak fields such as `rotation_direction`,
`cycle_index`, or continuously unwrapped phase angle with different meanings.

Export configuration should select the strategy from the persisted job setting
`paddling_discipline`. Existing kayak jobs and exports must retain current
behavior. Canoe events may reuse the existing event rendering, freeze-frame,
metric-slot, and progress-reporting infrastructure after detection.

## 19. Proposed Processing Passes

### Pass A: Geometry and provisional contact

1. Load selected paddle and boat records.
2. Stabilize waterlines and paddle lines.
3. Associate raw paddle candidates with boat references and derive stable slots.
4. Bootstrap blade endpoint identity.
5. Estimate full paddle length from complete recovery samples.
6. Calculate normalized longitudinal, depth, and visible-length observations.
7. Detect provisional entry/immersed/release intervals.

### Pass B: Direction bootstrap

1. Canonicalize waterlines screen-left to screen-right.
2. Calculate each tracked paddle's undirected `[0°, 180°)` axis angle.
3. Exclude the vertical dead band and static tracks with less than `8°` span.
4. Aggregate weighted left/right votes by boat/reference segment.
5. Lock direction only when support and consensus pass thresholds.
6. Split sustained reversal evidence into a new direction segment.

### Pass C: Directed angle and events

1. Orient the waterline basis in the locked travel direction.
2. Calculate signed paddle angle, velocity, and trend.
3. Reprocess buffered observations through the canoe state machine.
4. Confirm and backdate catch/exit candidates.
5. Deduplicate only events from the same physical paddle and stroke segment.

### Pass D: Rendering

1. Map events to source frame indices.
2. Reuse existing angle/event labels and optional freeze behavior.
3. Include canoe direction and event diagnostics in development output.

## 20. Testing Plan

Unit tests should cover:

- camera and boat translating together while boat-relative paddle motion remains
  unchanged;
- reversed waterline endpoints producing identical canonical observations;
- right- and left-moving mirrored strokes producing the same normalized phase;
- endpoint order changing without changing blade identity;
- right-half and left-half paddle axes voting expected travel directions;
- vertical dead-band and static-axis tracks producing no direction votes;
- insufficient eligible angle evidence or consensus returning unknown direction;
- pendulum reversal remaining a reversal rather than a `360°` unwrap;
- catch shortening followed by persistent immersion;
- false shortening that immediately recovers producing no catch;
- exit growth from an immersed minimum followed by upward release;
- growth caused only by angle jitter producing no exit;
- cropped-to-complete recovery confirming but not delaying an exit timestamp;
- missing/low-confidence endpoints or waterline suppressing events;
- gaps and track/reference changes splitting state;
- one reverse stroke not flipping a locked direction; and
- kayak strategy output remaining unchanged.

Integration fixtures should include representative side-view canoe clips with
camera follow/pan, both travel directions, occlusion at catch/exit, incomplete
paddle masks, and manually labeled entry/release frames. Report direction
accuracy and catch/exit frame error separately.

## 21. Decisions Recorded

The following decisions are aligned for implementation:

1. Canoe uses a dedicated pipeline; it does not reuse kayak rotation direction.
2. Direction is measured in boat-relative coordinates because the camera follows
   the boat.
3. Travel direction is primarily inferred from bilateral shoulder-to-hip
   horizontal position over valid pose frames.
4. Canonical paddle-axis half-plane voting is retained as fallback when pose
   direction is unavailable.
5. Body direction requires at least five eligible frames and a simple majority;
   paddle fallback retains its weighted-consensus threshold.
6. Early observations are buffered and reprocessed after direction bootstrap.
7. Canoe paddle angle is signed and direction-normalized in `[-180°, 180°)`.
8. Pull/recovery angular reversals are preserved; they are not kayak cycles.
9. Catch primarily uses water entry plus visible above-water shortening.
10. Exit starts from sustained visible-length growth after the immersed minimum
    and requires independent water, motion, angle, completeness, or recovery
    confirmation.
11. Confirmed events are backdated to the first supported transition.
12. Uncertain direction, endpoint identity, or primary event evidence suppresses
    output rather than invoking a fallback guess.

## 22. Calibration Work Before Production

Implementation may begin with named provisional thresholds, but production
enablement requires labeled canoe footage to determine:

- normalized entry/release depth bands;
- minimum valid pull and recovery displacement;
- full-paddle baseline stability;
- catch shortening and exit growth thresholds;
- event persistence and confirmation windows;
- direction-vote count and consensus;
- acceptable endpoint/reference confidence; and
- expected event frame tolerance.

Calibration must include both travel directions and camera-follow footage.
