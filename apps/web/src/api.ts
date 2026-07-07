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
  model_name?: string;
  attempt: number;
  error?: { code: string; message: string; retryable: boolean } | null;
  created_at: string;
};

export type FrameMask = {
  frame_index: number;
  timestamp_ms: number;
  prompt_id: string;
  instance_id: string;
  box_xywh: number[];
  centerline_box_xywh?: number[] | null;
  centerline_line_xyxy?: number[] | null;
  shaft_box_xywh?: number[] | null;
  shaft_line_xyxy?: number[] | null;
  score: number | null;
  segmentation:
    | { type: "polygon"; points: number[][] }
    | { type: "rle"; size: number[]; counts: number[] };
  centerline_segmentation?:
    | { type: "polygon"; points: number[][] }
    | { type: "rle"; size: number[]; counts: number[] }
    | null;
  shaft_segmentation?:
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
  instances: { id: string; prompt_id: string; color: string }[];
  chunks: {
    sequence: number;
    start_ms: number;
    end_ms: number;
    size_bytes: number;
    url: string;
  }[];
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

export async function createJob(videoId: string, prompts: string[]): Promise<Job> {
  const created = await request<{ job_id: string }>("/api/v1/jobs", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "idempotency-key": idempotencyKey()
    },
    body: JSON.stringify({
      video_id: videoId,
      prompts: prompts.map((text) => ({ text })),
      settings: { score_threshold: 0.5, include_boxes: true }
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

export function cancelJob(jobId: string): Promise<Job> {
  return request(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

export function retryJob(jobId: string): Promise<Job> {
  return request(`/api/v1/jobs/${jobId}/retry`, { method: "POST" });
}

export function deleteJob(jobId: string): Promise<void> {
  return request(`/api/v1/jobs/${jobId}`, { method: "DELETE" });
}

export async function exportJobVideo(jobId: string): Promise<Blob> {
  const response = await fetch(`/api/v1/jobs/${jobId}/export`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `${response.status} ${response.statusText}`);
  }
  return response.blob();
}
