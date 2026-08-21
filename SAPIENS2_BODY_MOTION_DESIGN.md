# Sapiens2 Body-Motion Backend Design

## Status

Accepted for personal testing. This document describes an optional Sapiens2 1B pose backend;
MediaPipe remains the default body-motion analyzer. Model weights and detector assets are deployed
separately and are not stored in this repository.

## Goals

- Allow `sam3-worker` to use Sapiens2 1B instead of MediaPipe for optional body-motion jobs.
- Preserve the existing job request, result chunk, player, metric, and export contracts.
- Retain only the body joints required by the current canoe/kayak calculations.
- Keep Sapiens-specific model loading and inference behind `BodyMotionAnalyzer`.
- Keep one primary athlete stable enough for single-athlete review clips.

## Non-goals

- Storing or exposing all 308 Sapiens2 keypoints.
- Adding face, head, finger, toe, or heel analysis.
- Adding 3D pose; Sapiens2 pose output is 2D.
- Selecting the pose backend per job or loading MediaPipe and Sapiens2 simultaneously.
- Downloading checkpoints automatically.
- Supporting multiple athletes in body-motion results.

## Upstream model and runtime

The initial backend uses the `sapiens2_1b` 308-keypoint pose checkpoint at 1024 by 768 input
resolution. Sapiens2 pose is top-down: a person detector first produces bounding boxes, then pose
inference runs on the selected crop. The upstream reference implementation uses the Hugging Face
DETR ResNet-101 DC5 detector. Both the pose checkpoint and detector directory are required local
deployment assets.

Sapiens2 requires Python 3.12 or newer and PyTorch 2.7 or newer. The backend is CUDA-oriented and
shares the worker GPU with SAM. The model is initialized lazily on the first requested body-motion
job and retained by the long-lived worker for later jobs. Deployment-like testing must establish
whether SAM and Sapiens2 1B fit concurrently on the target GPU.

## Configuration

The backend is selected for the whole worker:

```bash
SAM3_BODY_MOTION_ANALYZER=sapiens2
SAM3_SAPIENS2_CHECKPOINT_PATH=/opt/models/sapiens2_1b_pose.safetensors
SAM3_SAPIENS2_DETECTOR_PATH=/opt/models/detr-resnet-101-dc5
SAM3_SAPIENS2_DEVICE=cuda:0
```

Optional tuning settings are:

- `SAM3_SAPIENS2_CONFIG_PATH`: override the upstream 1B 308-keypoint config. By default the
  installed Sapiens2 package's matching config is used.
- `SAM3_SAPIENS2_KEYPOINT_THRESHOLD`: minimum retained keypoint confidence; default `0.5`.
- `SAM3_SAPIENS2_BBOX_THRESHOLD`: DETR person threshold; default `0.3`.
- `SAM3_SAPIENS2_NMS_THRESHOLD`: person-box non-maximum suppression threshold; default `0.3`.

Invalid paths, an unavailable runtime, and inference failures use the existing non-fatal body-motion
error behavior: segmentation completes, while the manifest contains a failed body-motion entry and
a warning.

## Canonical keypoint mapping

The adapter resolves Sapiens2 keypoints by name from its model metadata instead of assuming COCO
ordering. This is important because the wrists occur after detailed hand joints in the 308-point
format. The following canonical points are retained:

| Canonical point | Sapiens2 keypoint | Reviewed index | Purpose |
| --- | --- | ---: | --- |
| left/right shoulder | `left_shoulder` / `right_shoulder` | 5 / 6 | shoulder and lean |
| left/right elbow | `left_elbow` / `right_elbow` | 7 / 8 | arm angles |
| left/right wrist | `left_wrist` / `right_wrist` | 62 / 41 | elbow endpoint |
| left/right hip | `left_hip` / `right_hip` | 9 / 10 | torso, hip, direction |
| left/right knee | `left_knee` / `right_knee` | 11 / 12 | hip and knee angles |
| left/right ankle | `left_ankle` / `right_ankle` | 13 / 14 | knee endpoint |

Wrists and ankles are retained as body-joint endpoints; detailed hand and foot points are discarded.
Sapiens pixel coordinates are normalized by source-frame width and height. The Sapiens keypoint
score populates both `visibility` and `presence`; `z` is `0.0`. Keypoints below the configured
threshold are omitted. This produces the same `BodyMotionFrame` consumed by current metric code.

## Primary-athlete selection

The result contract remains single-athlete. For the first detected frame, the adapter chooses the
largest person box, with proximity to the frame center as a deterministic tie-breaker. On later
frames it prefers the detection with the greatest overlap with the previous box and then center
proximity. A frame with no person detection produces an empty landmark map rather than a full-frame
pose guess. The previous tracked box is retained across short misses so detection can reacquire the
athlete.

This is intentionally lightweight continuity, not a second persistent tracking system. If testing
shows identity switches in multi-person footage, the next iteration should associate detections with
SAM person/boat regions or introduce a dedicated tracker.

## Processing and lifecycle

1. Open the normalized video and preserve its exact frame index/timestamp timeline.
2. Lazily initialize DETR, the Sapiens2 1B pose model, pose metadata, and heatmap decoder.
3. Detect people on each frame and select the primary athlete.
4. Run top-down pose inference on the selected crop.
5. Decode 308 heatmaps, map the 12 required joints, and emit one `BodyMotionFrame`.
6. Reuse existing angle, confidence, smoothing, chunking, UI, direction, and export behavior.

The initial implementation processes every frame for correctness and exact-frame overlays. Detection
intervals, crop tracking, frame sampling, batching, and interpolation are deferred until benchmark
data justifies them. Cancellation and progress callbacks continue to run once per decoded frame.

## Compatibility

- `settings.body_motion` and `settings.paddling_discipline` are unchanged.
- Body-motion chunks remain schema version 1.
- The manifest records a provider-specific `model_name`, for example
  `sapiens2-pose-1b:sapiens2_1b_pose.safetensors`.
- Existing clients and old result chunks remain compatible.
- MediaPipe and the deterministic mock backend remain available.

## Dependencies and deployment

Sapiens2 is installed through a dedicated optional dependency extra pinned to a reviewed upstream
commit. A production-style worker needs the SAM and Sapiens2 extras, but does not need the MediaPipe
extra unless that backend is also used in the same environment. Sapiens2 declares the GUI OpenCV
package while this project uses headless OpenCV for SAM, so the final environment's `cv2` import and
version must be verified on the deployment host.

The worker must have local access to:

- the Sapiens2 1B pose `.safetensors` checkpoint;
- a complete local DETR ResNet-101 DC5 model directory;
- the Sapiens2 package and its default 1B pose config, unless overridden; and
- the configured CUDA device.

## Validation

Automated tests do not download weights or require a GPU. They cover configuration parsing, analyzer
selection, primary-box continuity, named keypoint mapping, normalization, confidence filtering, and
runtime/path failure envelopes. Existing body-motion and pipeline tests verify that canonical records
remain compatible downstream.

Manual acceptance on the target host must cover:

1. cold model initialization and a short 1B inference job;
2. peak GPU memory with the configured SAM model resident;
3. processing throughput and total job latency;
4. joint stability and left/right correctness in kayak and canoe clips;
5. primary-athlete continuity with background people; and
6. comparison against MediaPipe for occlusion, spray, paddle overlap, and seated/kneeling posture.

## Known risks

- The 1B model and high-resolution per-frame inference may dominate job time and GPU memory.
- DETR is image-based; without temporal tracking, crowded scenes can cause identity switches.
- Mapping one confidence score to both canonical confidence fields is an approximation.
- Sapiens2 provides no depth coordinate, so all current measurements remain 2D projection angles.
- Upstream API/config structure is not a stable external API; the dependency revision must remain
  pinned and upgrades require adapter tests plus a real-model smoke test.
