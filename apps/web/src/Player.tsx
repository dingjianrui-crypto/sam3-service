import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { exportJobVideo, FrameMask, getChunk, ResultManifest } from "./api";

type Props = {
  manifest: ResultManifest;
};

type OverlayMode = "mask" | "centerline";

type CenterlineRecord = {
  record: FrameMask;
  line: [number, number, number, number];
  color: string;
};

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: { mediaTime: number }) => void
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

export function Player({ manifest }: Props) {
  const videoRef = useRef<VideoWithFrameCallback>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chunksRef = useRef(new Map<number, FrameMask[]>());
  const loadingRef = useRef(new Set<number>());
  const [opacity, setOpacity] = useState(0.48);
  const [showBoxes, setShowBoxes] = useState(true);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("mask");
  const [exporting, setExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState("");
  const [enabledPrompts, setEnabledPrompts] = useState(
    new Set(manifest.prompts.map((prompt) => prompt.id))
  );
  const [status, setStatus] = useState("Loading result…");
  const colorByPrompt = useMemo(
    () => new Map(manifest.prompts.map((prompt) => [prompt.id, prompt.color])),
    [manifest]
  );

  const ensureChunk = useCallback(
    async (timeMs: number) => {
      const descriptor = manifest.chunks.find(
        (chunk) => timeMs >= chunk.start_ms && timeMs < chunk.end_ms
      );
      if (!descriptor || chunksRef.current.has(descriptor.sequence)) return;
      if (loadingRef.current.has(descriptor.sequence)) return;
      loadingRef.current.add(descriptor.sequence);
      try {
        const payload = await getChunk(descriptor.url);
        chunksRef.current.set(descriptor.sequence, payload.frames);
        setStatus("");
      } finally {
        loadingRef.current.delete(descriptor.sequence);
      }
    },
    [manifest]
  );

  const draw = useCallback(
    (mediaTime: number) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || !video.videoWidth) return;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const context = canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      const timeMs = mediaTime * 1000;
      void ensureChunk(timeMs);
      const descriptor = manifest.chunks.find(
        (chunk) => timeMs >= chunk.start_ms && timeMs < chunk.end_ms
      );
      if (!descriptor) return;
      const records = chunksRef.current.get(descriptor.sequence);
      if (!records) return;
      const nearby = recordsForTime(records, timeMs, manifest.video.fps, enabledPrompts);
      drawOverlay(context, nearby, {
        colorByPrompt,
        opacity,
        overlayMode,
        showBoxes
      });
    },
    [
      colorByPrompt,
      enabledPrompts,
      ensureChunk,
      manifest.chunks,
      manifest.video.fps,
      opacity,
      overlayMode,
      showBoxes
    ]
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let handle = 0;
    if (video.requestVideoFrameCallback) {
      const callback = (_now: number, metadata: { mediaTime: number }) => {
        draw(metadata.mediaTime);
        handle = video.requestVideoFrameCallback!(callback);
      };
      handle = video.requestVideoFrameCallback(callback);
      return () => video.cancelVideoFrameCallback?.(handle);
    }
    const fallback = () => draw(video.currentTime);
    video.addEventListener("timeupdate", fallback);
    return () => video.removeEventListener("timeupdate", fallback);
  }, [draw]);

  function togglePrompt(id: string) {
    setEnabledPrompts((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const exportCenterlineVideo = useCallback(async () => {
    setExporting(true);
    setExportStatus("Rendering MP4 on server…");
    try {
      const blob = await exportJobVideo(manifest.job_id);
      downloadBlob(blob, `sam3-${manifest.job_id}-centerlines.mp4`);
      setExportStatus("Export complete.");
    } catch (reason) {
      setExportStatus(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setExporting(false);
    }
  }, [manifest.job_id]);

  return (
    <section className="viewer">
      <div className="video-shell">
        <div className="video-frame">
          <video ref={videoRef} src={manifest.video.url} controls playsInline />
          <canvas ref={canvasRef} />
        </div>
        {status && <div className="video-status">{status}</div>}
      </div>
      <div className="viewer-controls">
        <div className="prompt-toggles">
          {manifest.prompts.map((prompt) => (
            <button
              className={enabledPrompts.has(prompt.id) ? "chip active" : "chip"}
              key={prompt.id}
              onClick={() => togglePrompt(prompt.id)}
              style={{ "--chip-color": prompt.color } as React.CSSProperties}
            >
              <span />
              {prompt.text}
            </button>
          ))}
        </div>
        <label>
          Overlay
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
        </label>
        <label>
          Overlay target
          <select
            value={overlayMode}
            onChange={(event) => setOverlayMode(event.target.value as OverlayMode)}
          >
            <option value="mask">Detected mask</option>
            <option value="centerline">Centerline</option>
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={(event) => setShowBoxes(event.target.checked)}
          />
          Boxes and IDs
        </label>
        <button className="secondary export-button" disabled={exporting} onClick={exportCenterlineVideo}>
          {exporting ? "Exporting…" : "Export"}
        </button>
        {exportStatus && <span className="export-status">{exportStatus}</span>}
      </div>
    </section>
  );
}

function recordsForTime(
  records: FrameMask[],
  timeMs: number,
  fps: number,
  enabledPrompts: Set<string>
) {
  const tolerance = 500 / Math.max(fps, 1);
  return records.filter(
    (record) =>
      Math.abs(record.timestamp_ms - timeMs) <= tolerance &&
      enabledPrompts.has(record.prompt_id)
  );
}

function drawOverlay(
  context: CanvasRenderingContext2D,
  records: FrameMask[],
  options: {
    colorByPrompt: Map<string, string>;
    opacity: number;
    overlayMode: OverlayMode;
    showBoxes: boolean;
  }
) {
  const centerlines: CenterlineRecord[] = [];
  for (const record of records) {
    const color = options.colorByPrompt.get(record.prompt_id) ?? "#35C2FF";
    const line = getCenterlineLine(record);
    if (options.overlayMode === "centerline" && line) {
      centerlines.push({ record, line, color });
    }
    const segmentation =
      options.overlayMode === "centerline" &&
      (record.centerline_segmentation || record.shaft_segmentation)
        ? (record.centerline_segmentation ?? record.shaft_segmentation)!
        : record.segmentation;
    context.save();
    context.globalAlpha = options.opacity;
    context.fillStyle = color;
    if (segmentation.type === "polygon") {
      context.beginPath();
      segmentation.points.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.closePath();
      context.fill();
    } else {
      drawRle(context, segmentation, color, options.opacity);
    }
    context.restore();
    if (options.showBoxes) {
      const [x, y, width, height] =
        options.overlayMode === "centerline" &&
        (record.centerline_box_xywh || record.shaft_box_xywh)
          ? (record.centerline_box_xywh ?? record.shaft_box_xywh)!
          : record.box_xywh;
      context.strokeStyle = color;
      context.lineWidth = Math.max(2, context.canvas.width / 600);
      context.strokeRect(x, y, width, height);
      context.fillStyle = color;
      context.font = `${Math.max(13, context.canvas.width / 60)}px system-ui`;
      context.fillText(
        `${record.instance_id}${record.score == null ? "" : ` · ${record.score.toFixed(2)}`}`,
        x,
        Math.max(18, y - 6)
      );
    }
  }
  if (options.overlayMode === "centerline") {
    drawAngleAnnotations(context, centerlines);
  }
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function getCenterlineLine(record: FrameMask): [number, number, number, number] | null {
  const line = record.centerline_line_xyxy ?? record.shaft_line_xyxy;
  if (!line || line.length !== 4 || line.some((value) => !Number.isFinite(value))) {
    return null;
  }
  return [line[0], line[1], line[2], line[3]];
}

function drawAngleAnnotations(
  context: CanvasRenderingContext2D,
  centerlines: CenterlineRecord[]
) {
  let visibleDegree: number | null = null;
  for (let firstIndex = 0; firstIndex < centerlines.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < centerlines.length; secondIndex += 1) {
      const first = centerlines[firstIndex];
      const second = centerlines[secondIndex];
      if (first.record.prompt_id === second.record.prompt_id) continue;
      const annotation = angleAnnotation(first.line, second.line);
      if (!annotation) continue;
      drawAngleAnnotation(context, annotation);
      visibleDegree ??= annotation.degrees;
    }
  }
  if (visibleDegree != null) {
    drawTopDegreeLabel(context, visibleDegree);
  }
}

function angleAnnotation(
  first: [number, number, number, number],
  second: [number, number, number, number]
) {
  const intersection = segmentIntersection(first, second);
  if (!intersection) return null;

  const firstVector = normalize([first[2] - first[0], first[3] - first[1]]);
  let secondVector = normalize([second[2] - second[0], second[3] - second[1]]);
  if (!firstVector || !secondVector) return null;

  if (dot(firstVector, secondVector) < 0) {
    secondVector = [-secondVector[0], -secondVector[1]];
  }
  const radians = Math.acos(clamp(dot(firstVector, secondVector), -1, 1));
  const degrees = (radians * 180) / Math.PI;
  if (!Number.isFinite(degrees) || degrees < 2) return null;

  const firstAngle = Math.atan2(firstVector[1], firstVector[0]);
  let secondAngle = Math.atan2(secondVector[1], secondVector[0]);
  let delta = normalizeAngle(secondAngle - firstAngle);
  if (Math.abs(delta) > Math.PI) {
    delta -= Math.sign(delta) * Math.PI * 2;
  }
  secondAngle = firstAngle + delta;

  return {
    x: intersection.x,
    y: intersection.y,
    startAngle: firstAngle,
    endAngle: secondAngle,
    degrees: Math.min(degrees, 180 - degrees)
  };
}

function segmentIntersection(
  first: [number, number, number, number],
  second: [number, number, number, number]
): { x: number; y: number } | null {
  const [x1, y1, x2, y2] = first;
  const [x3, y3, x4, y4] = second;
  const denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denominator) < 1e-6) return null;

  const firstDet = x1 * y2 - y1 * x2;
  const secondDet = x3 * y4 - y3 * x4;
  const x = (firstDet * (x3 - x4) - (x1 - x2) * secondDet) / denominator;
  const y = (firstDet * (y3 - y4) - (y1 - y2) * secondDet) / denominator;
  if (!pointOnSegment(x, y, first) || !pointOnSegment(x, y, second)) return null;
  return { x, y };
}

function pointOnSegment(x: number, y: number, line: [number, number, number, number]) {
  const [x1, y1, x2, y2] = line;
  const tolerance = 1.5;
  return (
    x >= Math.min(x1, x2) - tolerance &&
    x <= Math.max(x1, x2) + tolerance &&
    y >= Math.min(y1, y2) - tolerance &&
    y <= Math.max(y1, y2) + tolerance
  );
}

function drawAngleAnnotation(
  context: CanvasRenderingContext2D,
  annotation: {
    x: number;
    y: number;
    startAngle: number;
    endAngle: number;
    degrees: number;
  }
) {
  const radius = Math.max(24, Math.min(context.canvas.width, context.canvas.height) * 0.07);
  const counterclockwise = annotation.endAngle < annotation.startAngle;
  context.save();
  context.globalAlpha = 1;
  context.strokeStyle = "#fff2a8";
  context.fillStyle = "#fff2a8";
  context.lineWidth = Math.max(2, context.canvas.width / 700);
  context.beginPath();
  context.arc(
    annotation.x,
    annotation.y,
    radius,
    annotation.startAngle,
    annotation.endAngle,
    counterclockwise
  );
  context.stroke();

  const midAngle = annotation.startAngle + normalizeAngle(annotation.endAngle - annotation.startAngle) / 2;
  const labelX = annotation.x + Math.cos(midAngle) * (radius + 18);
  const labelY = annotation.y + Math.sin(midAngle) * (radius + 18);
  const label = `${Math.round(annotation.degrees)}°`;
  context.font = `${Math.max(14, context.canvas.width / 55)}px system-ui`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  const metrics = context.measureText(label);
  context.fillStyle = "rgba(2, 5, 9, 0.78)";
  context.fillRect(labelX - metrics.width / 2 - 6, labelY - 11, metrics.width + 12, 22);
  context.fillStyle = "#fff2a8";
  context.fillText(label, labelX, labelY);
  context.restore();
}

function drawTopDegreeLabel(context: CanvasRenderingContext2D, degrees: number) {
  const label = `Degree/角度：${Math.round(degrees)}`;
  let fontSize = Math.max(84, (context.canvas.width / 55) * 6);
  context.save();
  do {
    context.font = `700 ${fontSize}px system-ui, sans-serif`;
    if (context.measureText(label).width <= context.canvas.width * 0.86 || fontSize <= 18) {
      break;
    }
    fontSize *= 0.9;
  } while (true);
  const labelX = context.canvas.width / 2;
  const labelY = Math.max(fontSize * 1.25, context.canvas.height * 0.18);
  context.globalAlpha = 1;
  context.textAlign = "center";
  context.textBaseline = "middle";
  const metrics = context.measureText(label);
  const paddingX = fontSize * 0.16;
  const paddingY = fontSize * 0.12;
  context.fillStyle = "rgba(2, 5, 9, 0.68)";
  context.fillRect(
    labelX - metrics.width / 2 - paddingX,
    labelY - fontSize / 2 - paddingY,
    metrics.width + paddingX * 2,
    fontSize + paddingY * 2
  );
  context.fillStyle = "#fff2a8";
  context.lineWidth = Math.max(2, fontSize * 0.045);
  context.strokeStyle = "rgba(2, 5, 9, 0.95)";
  context.strokeText(label, labelX, labelY);
  context.fillText(label, labelX, labelY);
  context.restore();
}

function normalize(vector: number[]): [number, number] | null {
  const length = Math.hypot(vector[0], vector[1]);
  if (length < 1e-6) return null;
  return [vector[0] / length, vector[1] / length];
}

function dot(first: [number, number], second: [number, number]) {
  return first[0] * second[0] + first[1] * second[1];
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function normalizeAngle(angle: number) {
  let result = angle;
  while (result <= -Math.PI) result += Math.PI * 2;
  while (result > Math.PI) result -= Math.PI * 2;
  return result;
}

function drawRle(
  context: CanvasRenderingContext2D,
  rle: { size: number[]; counts: number[] },
  color: string,
  opacity: number
) {
  const [height, width] = rle.size;
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const offscreenContext = offscreen.getContext("2d");
  if (!offscreenContext) return;
  const image = offscreenContext.createImageData(width, height);
  const rgb = parseHex(color);
  let flatIndex = 0;
  let value = 0;
  for (const count of rle.counts) {
    if (value === 1) {
      for (let offset = 0; offset < count; offset += 1) {
        const columnMajor = flatIndex + offset;
        const x = Math.floor(columnMajor / height);
        const y = columnMajor % height;
        const pixel = (y * width + x) * 4;
        image.data[pixel] = rgb[0];
        image.data[pixel + 1] = rgb[1];
        image.data[pixel + 2] = rgb[2];
        image.data[pixel + 3] = Math.round(255 * opacity);
      }
    }
    flatIndex += count;
    value = 1 - value;
  }
  offscreenContext.putImageData(image, 0, 0);
  context.globalAlpha = 1;
  context.drawImage(offscreen, 0, 0, context.canvas.width, context.canvas.height);
}

function parseHex(color: string): number[] {
  const value = color.replace("#", "");
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16)
  ];
}
