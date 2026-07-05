# Product Requirements Document: SAM 3 Paddle Video Segmentation

**Status:** Draft for alignment  
**Version:** 0.1  
**Date:** 2026-07-05  
**Owner:** TBD

## 1. Summary

Build a browser-based application and web API that let a user upload a local video, describe an object with a short text prompt such as `paddle`, `paddle shaft`, or `paddle blade`, run SAM 3.1 segmentation and tracking on a remote NVIDIA T4 server, and review the resulting object masks overlaid on the video.

The first release is an internal, single-GPU tool optimized for correctness, clear progress reporting, and easy visual review rather than real-time inference or high concurrency.

## 2. Problem

Reviewing paddle-sport videos currently requires a person to find and trace the paddle or shaft frame by frame. The user needs a simple workflow that:

1. accepts an ordinary video from their computer;
2. identifies an object from a natural-language prompt;
3. tracks each matching object through the video;
4. displays the result in a browser for visual validation; and
5. exposes the same workflow through an API for future integrations.

Thin shafts, motion blur, occlusion, multiple paddles, scene cuts, and similarly shaped objects are expected failure cases and must be visible to the reviewer rather than silently hidden.

## 3. Goals

### MVP goals

- Upload a video from a desktop browser to a remote server.
- Submit one or more short text prompts, initially focused on `paddle`, `paddle shaft`, and `paddle blade`.
- Process videos asynchronously on one NVIDIA T4 GPU.
- Detect, segment, and temporally track every matching object instance.
- Preserve stable object IDs where the model supports them.
- Review masks, labels, confidence, and object IDs over synchronized video playback.
- Show upload and processing status, progress, errors, and retry actions.
- Make jobs and results available through a documented HTTP API.
- Automatically expire uploaded videos and generated results after a configurable retention period.

### Success criteria

- A first-time user can upload, process, and review a supported video without command-line access.
- The service completes the agreed validation set without GPU out-of-memory failures.
- A failed job returns a useful error and can be retried without uploading the video again.
- Paddle or shaft masks are visible and correctly synchronized during playback.
- On the validation set, the target remains identified on at least 80% of clearly visible frames, with no unreported mask/video synchronization errors.
- Performance on the T4 is measured and documented before production limits or turnaround promises are finalized.

## 4. Non-goals for MVP

- Real-time or live-stream segmentation.
- Model training or fine-tuning.
- Guaranteed production-scale availability or multiple concurrent GPU jobs.
- Native mobile applications.
- Team workspaces, comments, or complex role-based access.
- Frame-by-frame mask editing.
- Manual refinement with point, box, or mask prompts.
- Permanent media storage.
- Automated sports analytics derived from the masks.

## 5. Target user and primary journey

The initial user is a researcher, coach, or developer reviewing paddle-sport footage.

1. The user opens the web client.
2. The user selects a local video.
3. The client validates the file and uploads it with visible progress.
4. The user enters a prompt or selects a preset such as `paddle shaft`.
5. The user starts processing.
6. The job enters a queue and reports its current stage.
7. When processing is complete, the user opens the review player.
8. The player shows the source video with colored masks, labels, confidence, and stable instance IDs.
9. The user can pause, seek, step frame by frame, toggle prompts or instances, and adjust overlay opacity.
10. The user may delete the job and its media.

## 6. Product requirements

### 6.1 Upload

- Support browser upload from local storage.
- MVP input formats: MP4 and MOV.
- Validate extension, MIME type, file signature, duration, resolution, and size server-side.
- Display upload percentage and allow cancellation.
- Use resumable or multipart upload when files exceed the configured threshold.
- Transcode unsupported browser codecs to a review-compatible MP4 while preserving timing.
- Store the original frame rate and timing metadata needed to align masks.

Initial limits, subject to the T4 benchmark:

- maximum file size: 500 MB;
- maximum duration: 5 minutes;
- maximum input resolution: 1920 × 1080;
- inference resolution: configurable, defaulting to a 720p-class working resolution;
- frame rate: preserve up to 30 fps and normalize higher-rate inputs.

### 6.2 Prompting

- Accept a short, non-empty text phrase in English.
- Provide presets for `paddle`, `paddle shaft`, and `paddle blade`.
- Allow multiple prompts in one job, with each prompt displayed as a distinct class/color.
- Store the exact prompt with the result for reproducibility.
- Warn that a broad prompt such as `paddle` may cover both blade and shaft, while a narrow prompt may be less reliably detected.
- Reject excessive prompt count or length using configurable limits.

### 6.3 Processing

- Run segmentation as an asynchronous job; HTTP requests must not remain open for the full inference duration.
- Keep the SAM 3.1 model loaded in the GPU worker between jobs where practical.
- Process one GPU inference job at a time by default on the T4.
- Expose these job states: `created`, `uploading`, `queued`, `preprocessing`, `segmenting`, `postprocessing`, `completed`, `failed`, `cancelled`, and `expired`.
- Report coarse progress by stage and processed frame count when available.
- Support cancellation and retry.
- Use automatic mixed precision when validated not to cause unacceptable quality loss.
- Persist model/checkpoint version, prompt, inference settings, input metadata, and timestamps with every result.
- Fail safely on GPU out-of-memory, corrupt video, unsupported codec, missing checkpoint access, or worker restart.

### 6.4 Segmentation result

For each processed frame and detected instance, retain:

- frame index and presentation timestamp;
- prompt/class;
- stable object ID when available;
- segmentation mask in a compact representation such as COCO RLE;
- bounding box;
- confidence score, when provided by the model.

The API may deliver mask metadata in time-based chunks so the browser does not download the complete result before playback begins.

### 6.5 Review player

- Play, pause, seek, change playback speed, and step one frame backward or forward.
- Draw masks in sync with the displayed video frame.
- Show a distinct color for each prompt or instance.
- Toggle all overlays, individual prompts, and individual object IDs.
- Adjust mask opacity.
- Optionally show bounding boxes, labels, IDs, and confidence.
- Show “no object detected” without treating it as a processing failure.
- Preserve correct mask alignment after seeking.
- Provide a clear empty, loading, failed, expired, and completed state.

### 6.6 Job management

- Show a simple list of recent jobs with filename, prompt, state, creation time, and progress.
- Open a completed job in the review player.
- Retry failed processing without re-uploading while the source video remains available.
- Delete a job and all associated media/results.

### 6.7 API

Use versioned JSON APIs under `/api/v1`. Large binary transfers may use pre-signed or dedicated upload/download URLs.

Minimum endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/videos` | Create an upload and return upload instructions |
| `POST` | `/videos/{video_id}/complete` | Confirm upload and start media validation |
| `GET` | `/videos/{video_id}` | Read video metadata and validation state |
| `POST` | `/jobs` | Create a segmentation job with video ID and prompts |
| `GET` | `/jobs` | List recent jobs |
| `GET` | `/jobs/{job_id}` | Read status, progress, settings, and errors |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a queued or running job |
| `POST` | `/jobs/{job_id}/retry` | Retry a failed job |
| `GET` | `/jobs/{job_id}/results` | Read result manifest and mask chunk references |
| `DELETE` | `/jobs/{job_id}` | Delete job, source media, and results |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | API, storage, worker, model, and GPU readiness |

API behavior:

- Return stable machine-readable error codes plus human-readable messages.
- Make job creation idempotent when an idempotency key is supplied.
- Paginate list and result endpoints.
- Publish an OpenAPI specification.
- Restrict cross-origin access to configured web-client origins.

## 7. Proposed system shape

```text
Browser client
  ├── uploads video ───────────────┐
  ├── creates and monitors job     │
  └── reviews video + mask chunks  │
                                   ▼
API service ── metadata/status ── database
  │                               
  ├── media/results ───────────── object or local storage
  │
  └── queued job ──────────────── GPU worker
                                   ├── video decode/preprocess
                                   ├── SAM 3.1 detector + tracker
                                   └── compact mask postprocess
```

The API and GPU worker may run on the same remote machine for MVP, but their responsibilities should remain separate so the worker can later move or scale independently. The queue can begin as a durable database-backed queue; adding Redis or a managed queue is an implementation choice, not a product requirement.

## 8. Technical constraints and assumptions

- Target model: latest compatible SAM 3.1 checkpoint and repository code.
- Upstream prerequisites currently specify Python 3.12+, PyTorch 2.7+, and CUDA 12.6+.
- The checkpoint is gated on Hugging Face; deployment requires approved access and a server-side token.
- SAM 3 has 848M parameters and uses a shared vision encoder with a detector/tracker architecture.
- SAM 3.1 Object Multiplex improves multi-object tracking efficiency, but published speed claims use an H100 and cannot be transferred directly to a T4.
- A T4 has limited memory and compute relative to upstream benchmark hardware. Maximum duration, working resolution, prompt count, object count, precision mode, and processing time must be established with a deployment-like benchmark.
- The remote GPU host must have a driver compatible with the selected CUDA/PyTorch stack.
- FFmpeg or an equivalent decoder/transcoder is required for media normalization.
- Use of SAM materials and any redistribution must comply with the SAM License. Product/legal review is required before external or commercial release.

## 9. Security, privacy, and retention

- MVP runs only inside a trusted secure environment and relies on network-level access control; application authentication is not required.
- Plain HTTP is acceptable inside that environment. TLS and application authentication become required before exposure to an untrusted network.
- Do not place credentials or Hugging Face tokens in the browser.
- Sanitize filenames and never execute uploaded content.
- Isolate media processing and enforce size, duration, and resource limits.
- Default retention: delete source videos, normalized media, and results 7 days after job creation; make the period configurable.
- Allow immediate user-initiated deletion.
- Avoid logging video contents, prompts beyond operational need, access tokens, or signed URLs.
- Record basic audit events for upload, processing, access, and deletion.

## 10. Observability and operations

- Log job ID, stage, duration, frame count, model version, and structured error code.
- Collect queue time, preprocessing time, inference time, postprocessing time, total turnaround, GPU memory peak, GPU utilization, failure rate, and result size.
- Readiness must fail if the model/checkpoint or GPU is unavailable.
- On worker restart, queued jobs resume and interrupted jobs become retryable rather than remaining stuck.
- Configure disk/storage alerts and automatic cleanup.

## 11. Acceptance criteria

The MVP is accepted when:

1. A user can upload a supported video from the browser and see accurate progress.
2. A user can submit at least one text prompt and receive a job ID immediately.
3. Status updates continue without keeping the create request open.
4. The T4 worker completes the agreed validation set without OOM or stuck jobs.
5. The review player displays masks aligned to their source frames during play, pause, seek, and frame stepping.
6. Prompt/class and instance controls correctly show and hide overlays.
7. Invalid media, failed inference, cancellation, retry, deletion, and expiry have tested user-visible behavior.
8. API endpoints are documented in OpenAPI and pass automated contract tests.
9. Uploaded and derived files are removed by explicit deletion and automatic retention cleanup.
10. Benchmark and quality reports document tested video properties, settings, turnaround, GPU memory peak, and known failure cases.

## 12. Delivery phases

### Phase 0: feasibility spike

- Obtain checkpoint access and reproduce upstream video inference.
- Run representative paddle videos on the target T4.
- Compare the prompts `paddle`, `paddle shaft`, and `paddle blade`.
- Measure memory, speed, object count behavior, resolution tradeoffs, and mask quality.
- Confirm browser video timing and mask synchronization strategy.
- Set final MVP limits and performance targets from evidence.

### Phase 1: API and worker

- Upload, validation, storage, job lifecycle, queue, SAM inference, compact results, cleanup, health checks, and OpenAPI.

### Phase 2: web review client

- Upload workflow, prompt entry, job progress/list, review player, overlay controls, errors, retry, and deletion.

### Phase 3: hardening

- End-to-end validation, security review, deployment automation, monitoring, recovery tests, and benchmark report.

### Candidate post-MVP work

- Point/box corrections and propagation.
- Downloadable annotated MP4, alpha-mask video, image sequence, or COCO-format export.
- User accounts and shared projects.
- Batch upload and multi-GPU scheduling.
- Paddle-specific fine-tuning or fallback tracking.
- Derived analytics such as paddle angle, stroke timing, or trajectory.

## 13. Decisions to confirm

The draft currently assumes:

1. **Deployment:** a private/internal tool, not a public multi-tenant service.
2. **Target object:** general text prompting is supported, with paddle-related presets; paddle, shaft, and blade can be separate prompts.
3. **Result:** browser review is required; downloadable annotated video and raw-mask export are post-MVP.
4. **Correction:** review is visual only in MVP; the user cannot repair a bad mask.
5. **Capacity:** one active inference job on one T4, with additional jobs queued.
6. **Limits:** 500 MB, 5 minutes, 1080p input, and up to 30 fps are provisional until the feasibility spike.
7. **Access:** the secure environment is the MVP trust boundary; no application login or TLS is required.
8. **Retention:** media and results expire after 7 days by default.

Questions requiring owner confirmation:

- Is the first deployment for one trusted user, a small internal team, or external users?
- Does “shaft” mean only the narrow handle/shaft excluding the blade, and must blade and shaft be separate masks?
- Can a video contain multiple people and paddles that must retain separate IDs?
- Is visual review sufficient, or is downloadable annotated video/raw mask data required in MVP?
- Is manual correction with clicks or boxes required when text prompting fails?
- What are the typical and maximum video duration, resolution, frame rate, codec, and file size?
- What turnaround time is acceptable for a typical video?
- May uploaded videos be stored on the GPU server, and what retention period is acceptable?
- What network ingress or access-control mechanism defines the trusted secure environment?

## 14. Source notes

- [SAM 3 README](https://github.com/facebookresearch/sam3/blob/main/README.md)
- [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
