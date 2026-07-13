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
    "dedupe_iou_threshold": 0.6
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
  "schema_version": 1,
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
      "centerline_line_xyxy": [512.0, 321.0, 696.0, 330.5]
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
| `instance_id` | string | Stable tracked object ID within this job |
| `box_xywh` | number[4] | Object box: `[x, y, width, height]` |
| `score` | number or null | Model confidence |
| `segmentation` | object | Full object mask |
| `centerline_segmentation` | object or null | Thin mask around the fitted centerline |
| `centerline_box_xywh` | number[4] or null | Box around the centerline mask |
| `centerline_line_xyxy` | number[4] or null | Centerline segment `[x1, y1, x2, y2]` |

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
| `include_spm` | boolean | `false` | Draw instantaneous and average SPM on the side opposite the angle labels |
| `reference_prompt_id` | string | inferred boat prompt | Prompt used as the reference centerline, usually `boat` |
| `target_prompt_ids` | comma-separated string | inferred paddle prompts | Prompts whose instances receive degree labels |

For each exported frame, the server finds every target centerline, matches it to the nearest reference centerline, and prints one degree label per target on the same horizontal row. For example, if four paddle instances are detected, the exported video can show `1: 42°   2: 51°   3: 37°   4: 48°`. The same index-and-degree label is also drawn near each paddle centerline. When more than three paddle labels are present, the label farthest from the average degree is highlighted in red.

When `include_spm=true`, the export estimates cadence from the degree time series and draws `Inst SPM` and `Avg SPM` as text only, without a background panel. If the angle labels are at the top, SPM is drawn near the bottom; if angle labels are at the bottom, SPM is drawn near the top.

For portrait videos, top and bottom metric overlays are automatically moved farther from the video edges so they sit closer to the centerline of the frame.

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
