export type Prompt = { id: string; text: string; color: string; state?: string };

export type Job = {
  job_id: string;
  video_id: string;
  state: string;
  progress: {
    stage: string;
    processed_frames: number;
    total_frames: number;
    percent: number;
  };
  prompts: Prompt[];
  settings: JobSettings;
  model_name?: string;
  attempt: number;
  error?: { code: string; message: string; retryable: boolean } | null;
  created_at: string;
};

export type JobSettings = {
  working_max_dimension?: number;
  include_boxes?: boolean;
  score_threshold: number;
  redetect_interval_frames: number;
  max_detections_per_frame: number;
  dedupe_iou_threshold: number;
  boat_reference_line: "centerline" | "waterline";
  body_motion?: boolean;
  paddling_discipline?: "kayak" | "canoe";
};

export type BodyLandmark = {
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence: number;
};

export type BodyMotionFrame = {
  frame_index: number;
  timestamp_ms: number;
  athlete_id: string;
  primary_side?: "left" | "right" | null;
  landmarks: Record<string, BodyLandmark>;
  metrics: Record<string, number>;
  confidence: Record<string, number>;
};

export type BodyMotionChunk = {
  schema_version: number;
  start_ms: number;
  end_ms: number;
  frames: BodyMotionFrame[];
};

export type FrameMask = {
  frame_index: number;
  timestamp_ms: number;
  prompt_id: string;
  instance_id: string;
  track_id?: string;
  box_xywh: number[];
  centerline_box_xywh?: number[] | null;
  centerline_line_xyxy?: number[] | null;
  waterline_box_xywh?: number[] | null;
  waterline_line_xyxy?: number[] | null;
  score: number | null;
  segmentation:
    | { type: "polygon"; points: number[][] }
    | { type: "rle"; size: number[]; counts: number[] };
  centerline_segmentation?:
    | { type: "polygon"; points: number[][] }
    | { type: "rle"; size: number[]; counts: number[] }
    | null;
  waterline_segmentation?:
    | { type: "polygon"; points: number[][] }
    | { type: "rle"; size: number[]; counts: number[] }
    | null;
};

export type ResultManifest = {
  schema_version: number;
  job_id: string;
  video: {
    url: string;
    width: number;
    height: number;
    fps: number;
    duration_ms: number;
    frame_count: number;
  };
  prompts: Prompt[];
  settings?: Pick<
    JobSettings,
    "boat_reference_line" | "body_motion" | "paddling_discipline"
  >;
  instances: { id: string; prompt_id: string; color: string }[];
  chunks: {
    sequence: number;
    start_ms: number;
    end_ms: number;
    size_bytes: number;
    url: string;
  }[];
  body_motion?: {
    schema_version: number;
    status: "completed" | "failed";
    model_name?: string;
    reference_axis?: "video_vertical";
    direction_reference?: "centerline" | "waterline";
    athlete_count?: number;
    discipline?: "kayak" | "canoe";
    error?: { code: string; message: string; retryable: boolean };
    chunks: {
      sequence: number;
      start_ms: number;
      end_ms: number;
      size_bytes: number;
      url: string;
    }[];
  };
  warnings?: { code: string; message: string }[];
};

export type ExportVideoOptions = {
  angle_label_position: "top" | "bottom";
  angle_label_font_size: number;
  include_angles: boolean;
  include_spm: boolean;
  include_catch: boolean;
  include_exit: boolean;
  include_event_freeze: boolean;
  event_hold_seconds: number;
  include_event_metrics: boolean;
  include_body_motion: boolean;
  body_joint_names: string[];
  metric_count: number;
  event_paddle_index?: number;
  event_metric_center_offset_percent: number;
  metric_center_offset_percent?: number;
  reference_prompt_id?: string;
  target_prompt_ids?: string[];
  selection_rect?: { x: number; y: number; width: number; height: number };
  selection_keyframes?: Array<{
    time_ms: number;
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
};

export type ExportProgress = {
  task_id: string;
  state: "running" | "completed" | "failed";
  stage: string;
  percent: number;
  message: string;
  error?: string;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

async function sha256(blob: Blob): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;

  const bytes = await blob.arrayBuffer();
  const digest = await subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function idempotencyKey(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

export async function uploadVideo(
  file: File,
  onProgress: (percent: number) => void
): Promise<string> {
  const created = await request<{
    video_id: string;
    chunk_size_bytes: number;
  }>("/api/v1/videos", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      size_bytes: file.size,
      mime_type: file.type || "application/octet-stream"
    })
  });
  const chunkSize = created.chunk_size_bytes;
  const partCount = Math.ceil(file.size / chunkSize);
  for (let part = 0; part < partCount; part += 1) {
    const start = part * chunkSize;
    const chunk = file.slice(start, Math.min(file.size, start + chunkSize));
    const checksum = await sha256(chunk);
    await request(`/api/v1/videos/${created.video_id}/parts/${part}`, {
      method: "PUT",
      headers: checksum ? { "x-part-sha256": checksum } : undefined,
      body: chunk
    });
    onProgress(((part + 1) / partCount) * 100);
  }
  await request(`/api/v1/videos/${created.video_id}/complete`, { method: "POST" });
  return created.video_id;
}

export async function createJob(
  videoId: string,
  prompts: string[],
  settings: JobSettings
): Promise<Job> {
  const created = await request<{ job_id: string }>("/api/v1/jobs", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "idempotency-key": idempotencyKey()
    },
    body: JSON.stringify({
      video_id: videoId,
      prompts: prompts.map((text) => ({ text })),
      settings: { ...settings, include_boxes: true }
    })
  });
  return getJob(created.job_id);
}

export function getJob(jobId: string): Promise<Job> {
  return request(`/api/v1/jobs/${jobId}`);
}

export async function listJobs(): Promise<Job[]> {
  const response = await request<{ items: Job[] }>("/api/v1/jobs");
  return response.items;
}

export function getManifest(jobId: string): Promise<ResultManifest> {
  return request(`/api/v1/jobs/${jobId}/results`);
}

export function getChunk(url: string): Promise<{ frames: FrameMask[] }> {
  return request(url);
}

export function getBodyMotionChunk(url: string): Promise<BodyMotionChunk> {
  return request(url);
}

export function cancelJob(jobId: string): Promise<Job> {
  return request(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

export function retryJob(jobId: string): Promise<Job> {
  return request(`/api/v1/jobs/${jobId}/retry`, { method: "POST" });
}

export function deleteJob(jobId: string): Promise<void> {
  return request(`/api/v1/jobs/${jobId}`, { method: "DELETE" });
}

export async function exportJobVideo(
  jobId: string,
  options: ExportVideoOptions,
  onProgress?: (progress: ExportProgress) => void
): Promise<Blob> {
  const taskId = idempotencyKey();
  const params = new URLSearchParams({
    t: String(Date.now()),
    export_task_id: taskId,
    angle_label_position: options.angle_label_position,
    angle_label_font_size: String(options.angle_label_font_size),
    include_angles: String(options.include_angles),
    include_spm: String(options.include_spm),
    include_catch: String(options.include_catch),
    include_exit: String(options.include_exit),
    include_event_freeze: String(options.include_event_freeze),
    event_hold_seconds: String(options.event_hold_seconds),
    include_event_metrics: String(options.include_event_metrics),
    include_body_motion: String(options.include_body_motion),
    body_joint_names: options.body_joint_names.join(","),
    metric_count: String(options.metric_count),
    event_metric_center_offset_percent: String(
      options.event_metric_center_offset_percent
    )
  });
  if (options.metric_center_offset_percent != null) {
    params.set(
      "metric_center_offset_percent",
      String(options.metric_center_offset_percent)
    );
  }
  if (options.event_paddle_index != null) {
    params.set("event_paddle_index", String(options.event_paddle_index));
  }
  if (options.reference_prompt_id) {
    params.set("reference_prompt_id", options.reference_prompt_id);
  }
  if (options.target_prompt_ids?.length) {
    params.set("target_prompt_ids", options.target_prompt_ids.join(","));
  }
  if (options.selection_rect) {
    params.set("selection_x", String(options.selection_rect.x));
    params.set("selection_y", String(options.selection_rect.y));
    params.set("selection_width", String(options.selection_rect.width));
    params.set("selection_height", String(options.selection_rect.height));
  }
  if (options.selection_keyframes?.length) {
    params.set(
      "selection_keyframes",
      JSON.stringify(
        options.selection_keyframes.map(({ time_ms, x, y, width, height }) => [
          time_ms,
          x,
          y,
          width,
          height
        ])
      )
    );
  }
  let finished = false;
  const responsePromise = fetch(`/api/v1/jobs/${jobId}/export?${params}`, {
    cache: "no-store"
  }).finally(() => {
    finished = true;
  });
  const poller = (async () => {
    while (!finished) {
      const progressResponse = await fetch(
        `/api/v1/jobs/${jobId}/export/status?task_id=${encodeURIComponent(taskId)}`,
        { cache: "no-store" }
      ).catch(() => null);
      if (progressResponse?.ok) {
        const progress = (await progressResponse.json()) as ExportProgress;
        onProgress?.(progress);
      }
      await new Promise((resolve) => window.setTimeout(resolve, 400));
    }
  })();
  const response = await responsePromise;
  await poller;
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `${response.status} ${response.statusText}`);
  }
  onProgress?.({
    task_id: taskId,
    state: "completed",
    stage: "completed",
    percent: 100,
    message: "Export complete"
  });
  return response.blob();
}
