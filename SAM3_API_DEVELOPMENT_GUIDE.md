# SAM3 Object Segmentation API Development Guide

This guide describes how another application can use the SAM3 object segmentation service, what request and response payloads look like, and how to turn the result masks into centerlines, angles, and other derived measurements.

The service is asynchronous: clients upload a video, create a segmentation job with one or more text prompts, poll the job until it completes, then fetch a result manifest and time-based result chunks.

## Service Overview

Default local base URL:

```text
http://127.0.0.1:8000
```

Versioned API prefix:

```text
/api/v1
```

The API process stores uploads and serves HTTP routes. A separate worker process normalizes videos, runs segmentation, and writes result chunks. Both processes must share the same `SAM3_DATA_DIR`.

Health checks:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health/live` | Process liveness |
| `GET` | `/api/v1/health/ready` | Database/storage readiness and configured segmenter |

## End-to-End Client Flow

1. Create an upload session with `POST /api/v1/videos`.
2. Upload binary parts with `PUT /api/v1/videos/{video_id}/parts/{part_number}`.
3. Finalize the video with `POST /api/v1/videos/{video_id}/complete`.
4. Create a segmentation job with `POST /api/v1/jobs`.
5. Poll `GET /api/v1/jobs/{job_id}` until `state` is `completed`.
6. Fetch the result manifest with `GET /api/v1/jobs/{job_id}/results`.
7. Fetch each chunk listed in `manifest.chunks[*].url`.
8. Use `video.url` from the manifest for synchronized playback, or call `GET /api/v1/jobs/{job_id}/export` for an MP4 with centerline overlays.

## Video Upload API

### Create Upload

`POST /api/v1/videos`

Request:

```json
{
  "filename": "race.mp4",
  "size_bytes": 73400320,
  "mime_type": "video/mp4",
  "sha256": "optional-64-character-whole-file-sha256"
}
```

Rules:

- `filename` must be only a filename, not a path.
- Supported extensions are `.mp4` and `.mov`.
- `size_bytes` must be greater than zero and no larger than `SAM3_MAX_UPLOAD_BYTES`.
- `sha256` is optional. If supplied, the completed upload must match it.

Response:

```json
{
  "video_id": "8d1f...",
  "upload_id": "8d1f...",
  "chunk_size_bytes": 8388608,
  "expires_at": "2026-07-09T12:00:00Z"
}
```

### Upload Parts

`PUT /api/v1/videos/{video_id}/parts/{part_number}`

Send the raw bytes for one contiguous part. Part numbers are zero-based and must be contiguous when the upload is completed.

Optional header:

```text
x-part-sha256: <64-character part sha256>
```

Response:

```json
{
  "part_number": 0,
  "size_bytes": 8388608,
  "sha256": "..."
}
```

### Complete Upload

`POST /api/v1/videos/{video_id}/complete`

The server assembles the parts, validates size/checksum, probes video metadata, and marks the video as ready.

Response:

```json
{
  "id": "8d1f...",
  "original_filename": "race.mp4",
  "state": "ready",
  "mime_type": "video/mp4",
  "size_bytes": 73400320,
  "width": 1920,
  "height": 1080,
  "fps": 29.97,
  "duration_ms": 45000,
  "frame_count": 1349,
  "codec": "h264",
  "error_code": null,
  "error_detail": null,
  "created_at": "...",
  "validated_at": "...",
  "expires_at": "..."
}
```

## Segmentation Job API

### Create Job

`POST /api/v1/jobs`

Recommended header:

```text
idempotency-key: <client-generated-id>
```

Request:

```json
{
  "video_id": "8d1f...",
  "prompts": [
    { "text": "paddle" },
    { "text": "boat" }
  ],
  "settings": {
    "working_max_dimension": 1280,
    "include_boxes": true,
    "score_threshold": 0.3,
    "redetect_interval_frames": 1,
    "max_detections_per_frame": 13,
    "dedupe_iou_threshold": 0.6,
    "boat_reference_line": "centerline"
  }
}
```

Fields:

| Field | Type | Meaning |
|---|---|---|
| `video_id` | string | A completed video ID |
| `prompts` | array | Text prompts, one to `SAM3_MAX_PROMPTS`; each text is normalized whitespace and max 80 chars |
| `settings.score_threshold` | number | Model confidence threshold, `0` to `1` |
| `settings.redetect_interval_frames` | integer | How often to re-run text grounding; `0` keeps frame-0-only tracking, `1` attempts every frame |
| `settings.max_detections_per_frame` | integer | Maximum kept detections per prompt per frame after de-duplication; defaults to 13 and cannot exceed `SAM3_MAX_DETECTIONS_PER_FRAME` |
| `settings.dedupe_iou_threshold` | number | Same-frame box IoU above which lower-scored duplicate detections are dropped |
| `settings.boat_reference_line` | `centerline` or `waterline` | Boat geometry used as the paddle-angle reference; defaults to `centerline`. The web upload card exposes this choice. |
| `settings.include_boxes` | boolean | Reserved client preference; current chunks include boxes |
| `settings.working_max_dimension` | integer | Accepted range `320` to `1920`; reserved for processing-size control |

Detection mode presets used by the web UI:

| Mode | `score_threshold` | `redetect_interval_frames` | `max_detections_per_frame` | `dedupe_iou_threshold` | Use when |
|---|---:|---:|---:|---:|---|
| `Recall first` | `0.3` | `1` | `13` | `0.6` | Paddle count, visibility, or camera zoom changes over time; prioritize finding as many paddle parts as possible |
| `Balanced` | `0.35` | `10` | `13` | `0.6` | Need recovery from drift or new objects, but every-frame grounding is too slow |
| `Fast tracking` | `0.5` | `0` | `13` | `0.6` | Objects are already visible at frame 0 and speed matters more than rediscovery |
| `Custom` | user-defined | user-defined | user-defined | user-defined | Advanced tuning for a specific video, GPU budget, or false-positive/false-negative balance |

Mode semantics:

- `redetect_interval_frames: 0` keeps the original behavior: text grounding runs on frame 0, then SAM3 propagation handles later frames.
- `redetect_interval_frames: 1` attempts text grounding on every frame, then de-duplicates and caps detections before storing results.
- Values above `1` re-ground on periodic anchor frames, for example `10` means frame `0, 10, 20, ...`.
- `max_detections_per_frame` is applied per prompt after same-frame de-duplication. The service rejects values above `SAM3_MAX_DETECTIONS_PER_FRAME`.

For paddle scenes with up to four paddlers, the recall-first default is `max_detections_per_frame: 13`: up to three visible paddle parts per paddler, plus room for a boat/reference prompt when used separately. Increase the service-side SAM3 object cap above this, for example `SAM3_MAX_TRACKED_OBJECTS=16` or `24`, so duplicate candidates do not consume all model slots before API de-duplication.

Response:

```json
{
  "job_id": "7a2b...",
  "state": "queued",
  "status_url": "/api/v1/jobs/7a2b..."
}
```

### Poll Job

`GET /api/v1/jobs/{job_id}`

Response:

```json
{
  "job_id": "7a2b...",
  "video_id": "8d1f...",
  "state": "segmenting",
  "progress": {
    "stage": "segmenting",
    "processed_frames": 320,
    "total_frames": 2698,
    "percent": 11.9
  },
  "prompts": [
    { "id": "prompt-id-1", "text": "paddle", "color": "#35C2FF", "state": "segmenting" }
  ],
  "settings": {
    "working_max_dimension": 1280,
    "include_boxes": true,
    "score_threshold": 0.3,
    "redetect_interval_frames": 1,
    "max_detections_per_frame": 13,
    "dedupe_iou_threshold": 0.6,
    "idempotency_key": "..."
  },
  "model_name": "sam3.1-object-multiplex",
  "attempt": 1,
  "error": null,
  "created_at": "...",
  "started_at": "...",
  "completed_at": null
}
```

Job states:

```text
queued -> preprocessing -> segmenting -> postprocessing -> completed
```

Terminal states:

```text
completed, failed, cancelled
```

Other useful routes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/jobs?limit=50` | List recent jobs |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Request cancellation |
| `POST` | `/api/v1/jobs/{job_id}/retry` | Retry a failed or cancelled job |
| `DELETE` | `/api/v1/jobs/{job_id}` | Soft-delete a terminal job |

## Result Manifest

`GET /api/v1/jobs/{job_id}/results`

Only available when the job state is `completed`.

```json
{
  "schema_version": 2,
  "job_id": "7a2b...",
  "video": {
    "url": "/api/v1/videos/8d1f.../content?normalized=true",
    "width": 1920,
    "height": 1080,
    "fps": 29.97,
    "duration_ms": 45000,
    "frame_count": 1349,
    "codec": "h264"
  },
  "prompts": [
    { "id": "prompt-id-1", "text": "paddle", "color": "#35C2FF" },
    { "id": "prompt-id-2", "text": "boat", "color": "#FFB547" }
  ],
  "instances": [
    { "id": "prompt-id-1:1", "prompt_id": "prompt-id-1", "color": "#35C2FF" }
  ],
  "tracks": [
    {
      "id": "prompt-id-1:track:1",
      "prompt_id": "prompt-id-1",
      "start_ms": 0,
      "end_ms": 44800,
      "instance_ids": ["prompt-id-1:1", "prompt-id-1:7"],
      "color": "#35C2FF"
    }
  ],
  "chunks": [
    {
      "sequence": 0,
      "start_ms": 0,
      "end_ms": 2000,
      "size_bytes": 12345,
      "url": "/api/v1/jobs/7a2b.../results/chunks/0"
    }
  ]
}
```

Use `video.url` for playback. The masks and geometry in chunks use the manifest video coordinate system: `(0, 0)` is the top-left pixel, `x` grows right, and `y` grows downward.

## Result Chunk Format

`GET /api/v1/jobs/{job_id}/results/chunks/{sequence}`

```json
{
  "schema_version": 1,
  "start_ms": 0,
  "end_ms": 2000,
  "frames": [
    {
      "frame_index": 12,
      "timestamp_ms": 400,
      "prompt_id": "prompt-id-1",
      "instance_id": "prompt-id-1:1",
      "track_id": "prompt-id-1:track:1",
      "box_xywh": [512.2, 301.4, 180.5, 44.1],
      "score": 0.93,
      "segmentation": {
        "type": "rle",
        "size": [1080, 1920],
        "counts": [3456, 12, 144, 17]
      },
      "centerline_segmentation": {
        "type": "rle",
        "size": [1080, 1920],
        "counts": [6789, 8, 190, 8]
      },
      "centerline_box_xywh": [510, 315, 188, 12],
      "centerline_line_xyxy": [512.0, 321.0, 696.0, 330.5],
      "waterline_segmentation": null,
      "waterline_box_xywh": null,
      "waterline_line_xyxy": null,
      "waterline_confidence": null
    }
  ]
}
```

Frame fields:

| Field | Type | Meaning |
|---|---|---|
| `frame_index` | integer | Zero-based source frame index |
| `timestamp_ms` | integer | Presentation timestamp in milliseconds |
| `prompt_id` | string | ID from `manifest.prompts` |
| `instance_id` | string | Raw object ID emitted by the model; it may change after re-detection |
| `track_id` | string | Stable post-processed identity that can span multiple raw instance IDs |
| `box_xywh` | number[4] | Object box: `[x, y, width, height]` |
| `score` | number or null | Model confidence |
| `segmentation` | object | Full object mask |
| `centerline_segmentation` | object or null | Thin mask around the fitted centerline |
| `centerline_box_xywh` | number[4] or null | Box around the centerline mask |
| `centerline_line_xyxy` | number[4] or null | Centerline segment `[x1, y1, x2, y2]` |
| `waterline_segmentation` | object or null | Thin mask around a fitted boat waterline; generated for boat-like prompts when `boat_reference_line` is `waterline` |
| `waterline_box_xywh` | number[4] or null | Box around the waterline mask |
| `waterline_line_xyxy` | number[4] or null | Boat waterline segment `[x1, y1, x2, y2]` |
| `waterline_confidence` | number or null | Waterline fit confidence from `0` to `1` |

### Stable Track Identity

SAM object IDs are retained as `instance_id`, but repeated text grounding can assign a new raw ID to the same physical object. During post-processing, the service associates detections independently within each prompt and writes a stable `track_id`.

For each timestamp, active tracks and detections are compared using predicted center movement, box overlap, relative size change, and undirected centerline-angle change. A one-to-one minimum-cost assignment prevents a newly entering object from taking an existing object's identity simply because model detection order changed. Raw-ID agreement is only a small preference. Implausible movement, size change, or angle change rejects a match. Tracks survive detection gaps up to 1.5 seconds; a later reappearance starts a new track when continuity cannot be established confidently.

Manifest `tracks` entries include their visible time range and all raw `instance_ids` associated with them. Track IDs remain available for temporal analytics and compatibility with older API clients, but the web export UI uses its video rectangle instead of exposing track identity. Existing schema-v1 completed results are upgraded to schema v2 when their manifest is next requested.

Mask formats:

- Polygon: `{ "type": "polygon", "points": [[x, y], ...] }`
- RLE: `{ "type": "rle", "size": [height, width], "counts": [...] }`

The RLE is uncompressed COCO-style run-length encoding in column-major order. It starts with the number of zero/background pixels, then alternates foreground and background run lengths.

## Minimal JavaScript Client

```js
async function requestJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function uploadVideo(file) {
  const upload = await requestJson("/api/v1/videos", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      size_bytes: file.size,
      mime_type: file.type || "application/octet-stream"
    })
  });

  const partCount = Math.ceil(file.size / upload.chunk_size_bytes);
  for (let part = 0; part < partCount; part += 1) {
    const start = part * upload.chunk_size_bytes;
    const body = file.slice(start, Math.min(file.size, start + upload.chunk_size_bytes));
    await fetch(`/api/v1/videos/${upload.video_id}/parts/${part}`, {
      method: "PUT",
      body
    });
  }

  await requestJson(`/api/v1/videos/${upload.video_id}/complete`, { method: "POST" });
  return upload.video_id;
}

async function runSegmentation(file, prompts) {
  const videoId = await uploadVideo(file);
  const job = await requestJson("/api/v1/jobs", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "idempotency-key": crypto.randomUUID()
    },
    body: JSON.stringify({
      video_id: videoId,
      prompts: prompts.map((text) => ({ text })),
      settings: {
        score_threshold: 0.3,
        redetect_interval_frames: 1,
        max_detections_per_frame: 13,
        dedupe_iou_threshold: 0.6,
        include_boxes: true
      }
    })
  });

  while (true) {
    const status = await requestJson(`/api/v1/jobs/${job.job_id}`);
    if (status.state === "completed") break;
    if (status.state === "failed" || status.state === "cancelled") {
      throw new Error(status.error?.message || `Job ${status.state}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  const manifest = await requestJson(`/api/v1/jobs/${job.job_id}/results`);
  const chunks = await Promise.all(manifest.chunks.map((chunk) => requestJson(chunk.url)));
  return { manifest, frames: chunks.flatMap((chunk) => chunk.frames) };
}
```

## Rendering and Frame Synchronization

For a video player, choose result records by timestamp instead of assuming exact frame callback alignment.

Recommended matching:

1. Group chunk records by `timestamp_ms`.
2. For the current playback time, compute `current_ms = video.currentTime * 1000`.
3. Use the nearest result timestamp within a tolerance.
4. A practical tolerance is `max(1000 / fps, 40)` milliseconds.

If masks are rendered at a different size from the manifest video:

```js
const scaleX = canvas.width / manifest.video.width;
const scaleY = canvas.height / manifest.video.height;

function scaleLine([x1, y1, x2, y2]) {
  return [x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY];
}

function scaleBox([x, y, w, h]) {
  return [x * scaleX, y * scaleY, w * scaleX, h * scaleY];
}
```

For RLE masks, decode in column-major order:

```js
function decodeRle(rle) {
  const [height, width] = rle.size;
  const mask = Array.from({ length: height }, () => new Uint8Array(width));
  let flatIndex = 0;
  let value = 0;

  for (const count of rle.counts) {
    if (value === 1) {
      for (let offset = 0; offset < count; offset += 1) {
        const columnMajor = flatIndex + offset;
        const x = Math.floor(columnMajor / height);
        const y = columnMajor % height;
        mask[y][x] = 1;
      }
    }
    flatIndex += count;
    value = 1 - value;
  }

  return mask;
}
```

## Using Centerline Output

Prefer the server-provided line when it is present:

```js
function getCenterline(record) {
  const line = record.centerline_line_xyxy;
  if (!line || line.length !== 4 || line.some((v) => !Number.isFinite(v))) {
    return null;
  }
  return line;
}
```

Common measurements:

```js
function lineCenter([x1, y1, x2, y2]) {
  return [(x1 + x2) / 2, (y1 + y2) / 2];
}

function lineLength([x1, y1, x2, y2]) {
  return Math.hypot(x2 - x1, y2 - y1);
}

function lineAngleDegrees([x1, y1, x2, y2]) {
  return (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
}
```

Angle between two centerlines:

```js
function angleBetweenLines(a, b) {
  const av = normalize([a[2] - a[0], a[3] - a[1]]);
  let bv = normalize([b[2] - b[0], b[3] - b[1]]);
  if (!av || !bv) return null;

  if (dot(av, bv) < 0) bv = [-bv[0], -bv[1]];
  const radians = Math.acos(clamp(dot(av, bv), -1, 1));
  const degrees = (radians * 180) / Math.PI;
  return Math.min(degrees, 180 - degrees);
}

function normalize([x, y]) {
  const length = Math.hypot(x, y);
  return length < 1e-6 ? null : [x / length, y / length];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
```

For multiple paddles against one boat centerline, group records by `prompt_id` for the current frame, choose the boat prompt as the reference, and calculate each paddle instance against that reference:

```js
function paddleAnglesForFrame(records, boatPromptId, paddlePromptIds) {
  const boat = records.find(
    (record) => record.prompt_id === boatPromptId && record.centerline_line_xyxy
  );
  if (!boat) return [];

  return records
    .filter(
      (record) =>
        paddlePromptIds.has(record.prompt_id) &&
        record.centerline_line_xyxy
    )
    .map((paddle) => ({
      paddle_instance_id: paddle.instance_id,
      boat_instance_id: boat.instance_id,
      degrees: angleBetweenLines(
        paddle.centerline_line_xyxy,
        boat.centerline_line_xyxy
      )
    }))
    .filter((item) => item.degrees != null);
}
```

The API intentionally does not emit angle records because different applications may choose different reference prompts, pairing rules, or smoothing behavior.

Intersection point for two centerline segments:

```js
function segmentIntersection(a, b) {
  const [x1, y1, x2, y2] = a;
  const [x3, y3, x4, y4] = b;
  const d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(d) < 1e-6) return null;

  const det1 = x1 * y2 - y1 * x2;
  const det2 = x3 * y4 - y3 * x4;
  const x = (det1 * (x3 - x4) - (x1 - x2) * det2) / d;
  const y = (det1 * (y3 - y4) - (y1 - y2) * det2) / d;

  return pointOnSegment(x, y, a) && pointOnSegment(x, y, b) ? { x, y } : null;
}

function pointOnSegment(x, y, [x1, y1, x2, y2]) {
  const tolerance = 1.5;
  return (
    x >= Math.min(x1, x2) - tolerance &&
    x <= Math.max(x1, x2) + tolerance &&
    y >= Math.min(y1, y2) - tolerance &&
    y <= Math.max(y1, y2) + tolerance
  );
}
```

## Recomputing Centerlines from Masks

The service already computes `centerline_line_xyxy`, but another application may want to recompute centerlines with custom rules. The current server-side strategy is:

1. Convert the object mask into foreground pixel points `(x, y)`.
2. Fit a robust line through the long axis using RANSAC.
3. Fall back to a principal-component line if RANSAC cannot find a good line.
4. Refit the line from inlier points near the first line.
5. Project all mask pixels onto the final line.
6. Use the minimum and maximum projections as the centerline endpoints.

When `settings.boat_reference_line` is `waterline`, boat-like prompts (`boat`, `kayak`, `canoe`, or `shell`) also receive a waterline derived from the same full mask. The service samples the image-bottom-facing mask boundary along the fitted boat axis, removes the outer 15% at both ends to avoid curved bow and stern regions, fits the remaining boundary robustly, and records the resulting line and confidence. Paddle centerlines remain unchanged. Preview and export use the waterline only for the selected boat reference and fall back to its centerline when no valid waterline is available.
7. Optionally create a fixed-thickness line-band mask for visualization.

Important constants used by the service:

```text
initial_threshold = max(3, min(16, max(sqrt(mask_area) * 0.08, min(width, height) * 0.025)))
centerline_thickness_pixels = SAM3_CENTERLINE_THICKNESS_PIXELS, default 8
```

Simplified principal-component implementation:

```js
function centerlineFromBinaryMask(mask) {
  const height = mask.length;
  const width = height ? mask[0].length : 0;
  const points = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (mask[y][x]) points.push([x, y]);
    }
  }
  if (points.length < 8) return null;

  const cx = points.reduce((sum, [x]) => sum + x, 0) / points.length;
  const cy = points.reduce((sum, [, y]) => sum + y, 0) / points.length;

  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const [x, y] of points) {
    const dx = x - cx;
    const dy = y - cy;
    xx += dx * dx;
    yy += dy * dy;
    xy += dx * dy;
  }
  if (xx === 0 && yy === 0) return null;

  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const ux = Math.cos(angle);
  const uy = Math.sin(angle);

  let start = Infinity;
  let end = -Infinity;
  for (const [x, y] of points) {
    const projection = (x - cx) * ux + (y - cy) * uy;
    start = Math.min(start, projection);
    end = Math.max(end, projection);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || end - start < 2) return null;

  return [
    cx + start * ux,
    cy + start * uy,
    cx + end * ux,
    cy + end * uy
  ];
}
```

For noisy masks, add RANSAC before the principal-component step:

- Randomly sample two foreground pixels to define a candidate line.
- Count foreground pixels within `initial_threshold` pixels of that candidate.
- Score candidates by `inlier_count * inlier_span`.
- Refit the principal-component line using the best inliers.
- Keep endpoint extent from all foreground pixels, not only inliers, so long objects keep their full length.

## Exported Centerline Video

`GET /api/v1/jobs/{job_id}/export`

Returns an MP4 with centerlines and angle annotations over the original video.

Use this endpoint when another application needs a quick visual artifact rather than raw analytical data. For analytics, prefer the manifest and chunks because they preserve per-frame masks, lines, boxes, scores, prompt IDs, and instance IDs.

Query parameters:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `angle_label_position` | `top` or `bottom` | `top` | Vertical placement for the stacked degree label block |
| `angle_label_font_size` | integer, `12` to `96` | `32` | Font size in video pixels for the burned-in degree labels |
| `include_angles` | boolean | `true` | Draw degree labels and near-paddle angle markers; angle measurements remain available internally for SPM when this is `false` |
| `include_spm` | boolean | `false` | Draw instantaneous and average SPM |
| `include_catch` | boolean | `false` | Detect catch events and draw the catch angle in red |
| `include_exit` | boolean | `false` | Detect exit events and draw the exit angle in green |
| `include_event_paddle_length` | boolean | `false` | Append the event detector's paddle centerline length in pixels to catch and exit angle labels |
| `event_hold_seconds` | number, `0.1` to `10` | `1.5` | Duration for which the exported video freezes on each enabled catch or exit event |
| `export_task_id` | string | generated by server | Client task identifier used to query live export progress |
| `metric_count` | integer, `1` to `4` | none | Fix the number of paddle metric positions; extra detections are ignored and unavailable positions remain blank |
| `event_paddle_index` | integer, `1` to `4` | none (`ALL`) | Restrict catch and exit analysis to the indexed paddle-line slot, ordered front-to-back in the detected travel direction; must not exceed `metric_count` when both are supplied |
| `metric_center_offset_percent` | number, `0` to `45` | `5.5` landscape, `16` portrait | Distance from the nearest video edge toward the centerline for both metric rows |
| `reference_prompt_id` | string | inferred boat prompt | Prompt used as the reference centerline, usually `boat` |
| `target_prompt_ids` | comma-separated string | inferred paddle prompts | Prompts whose instances receive degree labels |
| `selection_x` | number, `0` to `1` | none | Normalized left edge of the export rectangle |
| `selection_y` | number, `0` to `1` | none | Normalized top edge of the export rectangle |
| `selection_width` | number, greater than `0` to `1` | none | Normalized export rectangle width |
| `selection_height` | number, greater than `0` to `1` | none | Normalized export rectangle height |
| `selection_keyframes` | JSON string | none | Moving rectangle keyframes encoded as `[[time_ms,x,y,width,height],...]`; overrides the static rectangle |
| `reference_track_ids` | comma-separated string | none | Deprecated compatibility filter; not used by the web UI |
| `target_track_ids` | comma-separated string | none | Deprecated compatibility filter; not used by the web UI |

For each exported frame, the server finds every target centerline, matches it to the nearest reference centerline, and prints one degree label per target on the same horizontal row under the Chinese title `桨叶角度`, without a background panel. For example, if four paddle instances are detected, the exported video can show `1: 42°   2: 51°   3: 37°   4: 48°`. The same index-and-degree label is also drawn near each paddle centerline. When only one paddle is detected, the exported video omits the index and shows only the value, such as `42°`. When more than three paddle labels are present, the label farthest from the average degree is highlighted in red.

When `metric_count` is supplied, the degree row always reserves exactly that many positions, independently of the number of paddle detection records. Paddles are ordered left to right in each frame, extra detections are ignored, and an unavailable position remains blank without shifting the other metric positions or carrying forward a stale value. Empty slots are excluded from outlier highlighting and SPM estimation, and no near-paddle marker is drawn for them. The web UI always sends this value and defaults it to `1`.

`event_paddle_index` affects only catch and exit analysis. For each reference
boat, the server derives stable paddle-line slots from direction-normalized
centerline positions, ordered front-to-back in the travel direction. Index `1`
is the leading slot. On every frame, a detected centerline is assigned to its
nearest slot only when it is sufficiently close to that slot's anchor. This
stitches fragmented physical track IDs into the same event stream while leaving
a temporarily missing slot blank instead of shifting a different paddle into
it. Omitting the parameter analyzes all eligible physical tracks. If the
requested slot is unavailable, no paddle from that reference boat is
substituted and no event is emitted for it. Angle and SPM rendering continue to
use `metric_count` independently.

For backward compatibility, rectangle exports that omit `metric_count` reserve the maximum number of paddle positions observed inside the rectangle over the complete video.

The video toolbar provides Rectangle and Undo tools. Rectangle pauses playback and lets the user drag an export area directly over the video viewport. The first rectangle records the initial keyframe. Moving the timeline alone does not create a keyframe. When the user moves or redraws the rectangle, the client compares its normalized position and size with the last recorded rectangle at or before that timestamp and creates or updates a milestone only when they differ. A dedicated application-controlled timeline below the viewport maps pointer coordinates directly to video milliseconds and shows each saved rectangle position as a numbered milestone with a vertical tick. Milestones change color as playback passes them, and the time readout reports the total keyframe count. The milestone uses the timeline's exact current timestamp. The client and export renderer hold each rectangle fixed until playback reaches the next milestone, then switch immediately to that milestone's rectangle. Undo removes the most recently edited keyframe and returns export filtering to the full frame after the last keyframe is removed.

The client sends normalized coordinates, so responsive display scaling does not change the selected source area. During export, a boat or paddle is included in each frame only when the center of its centerline is inside the rectangle at that time. The rectangle is a filter and is not burned into the exported video. `selection_keyframes` takes precedence over the legacy static `selection_x`, `selection_y`, `selection_width`, and `selection_height` parameters.

The export UI does not expose track IDs. `reference_track_ids`, `target_track_ids`, `reference_instance_ids`, and `target_instance_ids` remain API compatibility parameters for older clients only.

The web viewer separates Playback controls from Export controls. Playback keeps global overlay opacity, boxes and IDs, optional angle preview, and one visibility/overlay-target row per prompt so different object types can independently show their full mask, centerline, or generated waterline. Export owns `include_angles`, reference and target prompts, degree placement, font size, metric count, center offset, SPM, catch and exit events, event-paddle selection, optional event paddle length, event freeze time, and MP4 generation. The angle switch is grouped with the angle controls, while the SPM switch is grouped with the general export controls. The export reference and target selections also drive the optional live angle preview.

When catch or exit is enabled, export performs a two-pass event analysis before rendering. The first pass merges nearby collinear paddle fragments, associates them with persistent physical-paddle tracks across raw track-ID changes, stabilizes line length and endpoint ordering, and estimates rotation direction from the undirected paddle-axis sequence. Clockwise rotation maps to rightward kayak travel; anticlockwise rotation maps to leftward travel. Direction must have at least five valid angular deltas, 45 degrees of accumulated motion, and 75 percent directional consensus. Low-confidence direction segments emit no events.

Before event detection, each directed `0°-180°` stroke restores paddle length with independent forward `0°-90°` and reverse `180°-90°` fixed anchors. Each half traverses from its terminal angle and requires the first pair of CNN-complete lengths to agree within `15%`; their mean establishes the fixed phase anchor. A later CNN-complete length remains genuine only when it is not shorter than that anchor and its relative difference from the anchor is at most `10%`. Cropped, unknown, shorter-than-anchor, and out-of-tolerance observations inherit the preceding genuine length in traversal order without moving the anchor. Inherited geometry anchors the inactive endpoint and adjusts only the active endpoint, and accepted lengths remain isolated to their phase, cycle, and continuous track segment. A half-phase without a confirmed seed remains ineligible to start or confirm catch/exit events, but its raw centerlines continue endpoint-orientation and directed-phase tracking. Only a genuine paddle-observation gap resets that physical continuity state.

The second pass creates a local coordinate basis from the matched waterline. Its forward ray points in the detected travel direction and its normal points below the waterline. The selected blade endpoint receives a directed angle from 0 through 360 degrees using this basis, so rightward/clockwise and mirrored leftward/anticlockwise motion share the same normalized phase. A complete 360-degree revolution is one cycle. The first reliable visible catch or exit anchors the active blade endpoint; the opposite endpoint is then ignored for event generation until tracking is reinitialized.

Catch and exit use the active endpoint's signed waterline depth and a finite-width tolerance band. A catch candidate begins on the first observed above-to-water transition while moving downward. The preceding and current catch observations are compared, and the one whose active endpoint is closest to the fitted waterline supplies the event timestamp, angle, and geometry. An exit candidate begins on the first observed underwater-to-air transition while moving upward. The transition must occur on the finite detected waterline. Two consecutive compatible samples confirm the transition. Candidates expire after 400 ms, and a detection gap longer than that reinitializes phase/contact observation rather than synthesizing a transition across missing frames.

Each physical paddle may emit at most one catch and one exit per 360-degree cycle. Catch eligibility does not depend on a confirmed exit, and exit eligibility does not depend on a confirmed catch. Cycle-indexed eligibility resets automatically at the next revolution even if one or both events were missed. Hidden opposite-side strokes do not produce synthetic events, and ambiguous two-dimensional camera-side depth is not guessed.

Nearby collinear same-type events within 250 ms are merged after state analysis as an additional safeguard. Fragment consolidation allows a modest perpendicular offset between nearly parallel detections so duplicate full-length lines from a split paddle remain one physical observation. Spatially separate paddles in a multi-athlete boat remain independent. When `metric_count` is configured, each freeze moment also renders no more than that number of event angles, choosing the highest-confidence and longest event lines first.

At each enabled event, the exporter inserts a pause at the event timestamp and freezes the complete source frame for `event_hold_seconds`. Audio is silent during the pause, then video and audio resume from the same source timestamp without skipping source content. Events from different paddles within 250 ms share one freeze so a synchronized crew does not create several consecutive pauses. SPM and selection calculations remain on the original source timeline. Only the inserted pause frames redraw the paddle centerline in cyan and its matched boat centerline or waterline in amber. The angle arc and numeric label use the directed phase angle from 0 through 359 degrees: catch is red and exit is green. When `include_event_paddle_length=true`, the label also prints the rounded pixel length of the exact paddle centerline used to estimate that event, such as `56° 249 px`. No catch or exit text is rendered. The temporary highlights disappear and the configured overlay colors return when normal playback resumes.

While an export request is running, query `GET /api/v1/jobs/{job_id}/export/status?task_id={export_task_id}`. The response reports `state`, `stage`, `percent`, and `message`. Stages currently include `preparing`, `analyzing_events`, `rendering`, `encoding`, `finalizing`, `completed`, and `failed`. The web client polls this endpoint and displays the current message and progress bar while retaining the existing MP4 download response from the export request.

When `include_spm=true`, the export estimates cadence from the degree time series and draws `瞬时桨频` and `平均桨频` under the Chinese title `桨频` as text only, without a background panel. SPM is always placed on the side opposite `angle_label_position`: angle labels at the top put SPM near the bottom, and angle labels at the bottom put SPM near the top. The export renderer uses the configured CJK-capable font path when available, then falls back to installed Ubuntu CJK fonts such as Noto Sans CJK.

For portrait videos, the default `metric_center_offset_percent` is higher so top and bottom metric overlays sit closer to the centerline of the frame. Passing `metric_center_offset_percent` overrides that default for both the degree row and the SPM row.

## Error Format

Errors return a JSON body:

```json
{
  "error": {
    "code": "INVALID_STATE",
    "message": "Results are not ready.",
    "retryable": false,
    "request_id": "..."
  }
}
```

Common status codes:

| Status | Meaning |
|---|---|
| `400` | Invalid request or unsupported video |
| `404` | Resource not found |
| `409` | Invalid state, such as requesting results before completion |
| `413` | Upload exceeds configured limits |
| `422` | Schema validation error |
| `500` | Internal or export failure |
| `503` | Model/runtime unavailable |

## Operational Notes

- The API currently assumes the secure deployment network is the authorization boundary. Put it behind your own authentication/TLS layer before exposing it outside a trusted network.
- Remote clients can call the API when the server binds to a reachable address such as `SAM3_HOST=0.0.0.0` and the network path is open.
- Browser applications served from a different origin need that origin listed in `SAM3_CORS_ALLOW_ORIGINS`, for example `http://localhost:5173,https://review.example.com`.
- Server-to-server clients are not subject to browser CORS checks.
- Results are chunked by time; default chunk duration is controlled by `SAM3_RESULT_CHUNK_SECONDS`.
- Upload chunk size is controlled by `SAM3_UPLOAD_CHUNK_BYTES`.
- The real worker uses `SAM3_SEGMENTER=sam3`; local development can use `SAM3_SEGMENTER=mock`.
- `SAM3_CENTERLINE_THICKNESS_PIXELS` controls the thickness of generated centerline masks, not the `centerline_line_xyxy` endpoints.
- Always treat `centerline_*` fields as optional. Some masks may be too small or too ambiguous to fit a line.
