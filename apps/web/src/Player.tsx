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

type AngleConfig = {
  enabled: boolean;
  referencePromptId: string;
  targetPromptIds: Set<string>;
};

type ExportLabelPosition = "top" | "bottom";

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: { mediaTime: number }) => void
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

const OVERLAY_FONT_FAMILY =
  'Arial, Helvetica, system-ui, sans-serif';

function defaultMetricCenterOffsetPercent(manifest: ResultManifest) {
  return manifest.video.height > manifest.video.width ? 16 : 5.5;
}

export function Player({ manifest }: Props) {
  const videoRef = useRef<VideoWithFrameCallback>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chunksRef = useRef(new Map<number, FrameMask[]>());
  const loadingRef = useRef(new Set<number>());
  const defaultReferencePromptId = useMemo(
    () => defaultAngleReferencePromptId(manifest),
    [manifest]
  );
  const defaultTargetPromptIds = useMemo(
    () => defaultAngleTargetPromptIds(manifest, defaultReferencePromptId),
    [defaultReferencePromptId, manifest]
  );
  const [opacity, setOpacity] = useState(0.48);
  const [showBoxes, setShowBoxes] = useState(true);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("mask");
  const [angleEnabled, setAngleEnabled] = useState(false);
  const [angleReferencePromptId, setAngleReferencePromptId] = useState(defaultReferencePromptId);
  const [angleTargetPromptIds, setAngleTargetPromptIds] = useState(defaultTargetPromptIds);
  const [exportLabelPosition, setExportLabelPosition] = useState<ExportLabelPosition>("top");
  const [exportMetricCenterOffsetPercent, setExportMetricCenterOffsetPercent] = useState(
    defaultMetricCenterOffsetPercent(manifest)
  );
  const [exportFontSize, setExportFontSize] = useState(32);
  const [exportSpmEnabled, setExportSpmEnabled] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState("");
  const [enabledPrompts, setEnabledPrompts] = useState(
    new Set(manifest.prompts.map((prompt) => prompt.id))
  );
  const [status, setStatus] = useState("Loading result…");

  useEffect(() => {
    setExportMetricCenterOffsetPercent(defaultMetricCenterOffsetPercent(manifest));
  }, [manifest.job_id, manifest.video.height, manifest.video.width]);

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
        angleConfig: {
          enabled: angleEnabled,
          referencePromptId: angleReferencePromptId,
          targetPromptIds: angleTargetPromptIds
        },
        colorByPrompt,
        opacity,
        overlayMode,
        showBoxes
      });
    },
    [
      angleEnabled,
      angleReferencePromptId,
      angleTargetPromptIds,
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
    setAngleEnabled(false);
    setAngleReferencePromptId(defaultReferencePromptId);
    setAngleTargetPromptIds(defaultTargetPromptIds);
  }, [defaultReferencePromptId, defaultTargetPromptIds, manifest.job_id]);

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

  function toggleAngleTargetPrompt(id: string) {
    setAngleTargetPromptIds((current) => {
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
      const blob = await exportJobVideo(manifest.job_id, {
        angle_label_position: exportLabelPosition,
        angle_label_font_size: exportFontSize,
        include_spm: exportSpmEnabled,
        metric_center_offset_percent: exportMetricCenterOffsetPercent,
        reference_prompt_id: angleReferencePromptId,
        target_prompt_ids: [...angleTargetPromptIds]
      });
      downloadBlob(blob, `sam3-${manifest.job_id}-centerlines.mp4`);
      setExportStatus("Export complete.");
    } catch (reason) {
      setExportStatus(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setExporting(false);
    }
  }, [
    angleReferencePromptId,
    angleTargetPromptIds,
    exportFontSize,
    exportLabelPosition,
    exportMetricCenterOffsetPercent,
    exportSpmEnabled,
    manifest.job_id
  ]);

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
        <label className="checkbox">
          <input
            type="checkbox"
            checked={angleEnabled}
            onChange={(event) => {
              setAngleEnabled(event.target.checked);
              if (event.target.checked) setOverlayMode("centerline");
            }}
          />
          Angles
        </label>
        {angleEnabled && (
          <div className="angle-controls">
            <label>
              Reference
              <select
                value={angleReferencePromptId}
                onChange={(event) => {
                  const nextReference = event.target.value;
                  setAngleReferencePromptId(nextReference);
                  setAngleTargetPromptIds((current) => {
                    const next = new Set(current);
                    next.delete(nextReference);
                    if (next.size === 0) {
                      manifest.prompts
                        .filter((prompt) => prompt.id !== nextReference)
                        .forEach((prompt) => next.add(prompt.id));
                    }
                    return next;
                  });
                }}
              >
                {manifest.prompts.map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {prompt.text}
                  </option>
                ))}
              </select>
            </label>
            <div className="angle-targets">
              {manifest.prompts
                .filter((prompt) => prompt.id !== angleReferencePromptId)
                .map((prompt) => (
                  <label key={prompt.id} className="checkbox">
                    <input
                      type="checkbox"
                      checked={angleTargetPromptIds.has(prompt.id)}
                      onChange={() => toggleAngleTargetPrompt(prompt.id)}
                    />
                    {prompt.text}
                  </label>
                ))}
            </div>
          </div>
        )}
        <div className="export-controls">
          <label>
            Degree position
            <select
              value={exportLabelPosition}
              onChange={(event) => setExportLabelPosition(event.target.value as ExportLabelPosition)}
            >
              <option value="top">Top</option>
              <option value="bottom">Bottom</option>
            </select>
          </label>
          <label>
            Font size
            <input
              type="number"
              min="12"
              max="96"
              step="2"
              value={exportFontSize}
              onChange={(event) => {
                const value = Number(event.target.value);
                setExportFontSize(Number.isFinite(value) ? clamp(value, 12, 96) : 12);
              }}
            />
          </label>
          <label>
            Center offset %
            <input
              type="number"
              min="0"
              max="45"
              step="0.5"
              value={exportMetricCenterOffsetPercent}
              onChange={(event) => {
                const value = Number(event.target.value);
                setExportMetricCenterOffsetPercent(
                  Number.isFinite(value) ? clamp(value, 0, 45) : 0
                );
              }}
            />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={exportSpmEnabled}
              onChange={(event) => setExportSpmEnabled(event.target.checked)}
            />
            SPM
          </label>
          <button className="secondary export-button" disabled={exporting} onClick={exportCenterlineVideo}>
            {exporting ? "Exporting…" : "Export"}
          </button>
          {exportStatus && <span className="export-status">{exportStatus}</span>}
        </div>
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
  let nearestTimestamp: number | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const record of records) {
    const distance = Math.abs(record.timestamp_ms - timeMs);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestTimestamp = record.timestamp_ms;
    }
  }
  if (nearestTimestamp == null || nearestDistance > tolerance) return [];
  return records.filter(
    (record) =>
      record.timestamp_ms === nearestTimestamp &&
      enabledPrompts.has(record.prompt_id)
  );
}

function defaultAngleReferencePromptId(manifest: ResultManifest) {
  return (
    manifest.prompts.find((prompt) => /\bboat\b/i.test(prompt.text))?.id ??
    manifest.prompts[1]?.id ??
    manifest.prompts[0]?.id ??
    ""
  );
}

function defaultAngleTargetPromptIds(manifest: ResultManifest, referencePromptId: string) {
  const preferred = manifest.prompts.filter(
    (prompt) => prompt.id !== referencePromptId && /\bpaddle\b/i.test(prompt.text)
  );
  const targets = preferred.length
    ? preferred
    : manifest.prompts.filter((prompt) => prompt.id !== referencePromptId);
  return new Set(targets.map((prompt) => prompt.id));
}

function drawOverlay(
  context: CanvasRenderingContext2D,
  records: FrameMask[],
  options: {
    angleConfig: AngleConfig;
    colorByPrompt: Map<string, string>;
    opacity: number;
    overlayMode: OverlayMode;
    showBoxes: boolean;
  }
): void {
  const centerlines: CenterlineRecord[] = [];
  for (const record of records) {
    const color = options.colorByPrompt.get(record.prompt_id) ?? "#35C2FF";
    const line = getCenterlineLine(record, context.canvas.width, context.canvas.height);
    if (line) {
      centerlines.push({ record, line, color });
    }
    const segmentation =
      options.overlayMode === "centerline" && record.centerline_segmentation
        ? record.centerline_segmentation
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
        options.overlayMode === "centerline" && record.centerline_box_xywh
          ? scaleCenterlineBox(record, context.canvas.width, context.canvas.height)
          : record.box_xywh;
      context.strokeStyle = color;
      context.lineWidth = Math.max(2, context.canvas.width / 600);
      context.strokeRect(x, y, width, height);
      context.fillStyle = color;
      context.font = `${Math.max(13, context.canvas.width / 60)}px ${OVERLAY_FONT_FAMILY}`;
      context.fillText(
        `${record.instance_id}${record.score == null ? "" : ` · ${record.score.toFixed(2)}`}`,
        x,
        Math.max(18, y - 6)
      );
    }
  }
  if (options.angleConfig.enabled) {
    drawAngleAnnotations(context, centerlines, options.angleConfig);
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

function getCenterlineLine(
  record: FrameMask,
  canvasWidth: number,
  canvasHeight: number
): [number, number, number, number] | null {
  const line = record.centerline_line_xyxy;
  if (!line || line.length !== 4 || line.some((value) => !Number.isFinite(value))) {
    return null;
  }
  const [scaleX, scaleY] = centerlineCoordinateScale(record, canvasWidth, canvasHeight);
  return [line[0] * scaleX, line[1] * scaleY, line[2] * scaleX, line[3] * scaleY];
}

function scaleCenterlineBox(
  record: FrameMask,
  canvasWidth: number,
  canvasHeight: number
): [number, number, number, number] {
  const box = record.centerline_box_xywh ?? record.box_xywh;
  const [scaleX, scaleY] = centerlineCoordinateScale(record, canvasWidth, canvasHeight);
  return [box[0] * scaleX, box[1] * scaleY, box[2] * scaleX, box[3] * scaleY];
}

function centerlineCoordinateScale(
  record: FrameMask,
  canvasWidth: number,
  canvasHeight: number
): [number, number] {
  const segmentation = record.centerline_segmentation;
  if (!segmentation || segmentation.type !== "rle") return [1, 1];
  const [maskHeight, maskWidth] = segmentation.size;
  if (!maskWidth || !maskHeight) return [1, 1];
  return [canvasWidth / maskWidth, canvasHeight / maskHeight];
}

function drawAngleAnnotations(
  context: CanvasRenderingContext2D,
  centerlines: CenterlineRecord[],
  angleConfig: AngleConfig
) {
  const references = centerlines.filter(
    (centerline) => centerline.record.prompt_id === angleConfig.referencePromptId
  );
  if (!references.length || !angleConfig.targetPromptIds.size) return;

  const targets = centerlines.filter((centerline) =>
    angleConfig.targetPromptIds.has(centerline.record.prompt_id)
  );
  for (const target of targets) {
    const reference = nearestCenterline(target, references);
    const degrees = angleBetweenLines(target.line, reference.line);
    if (degrees == null) continue;
    drawAngleAnnotation(context, target, degrees);
  }
}

function nearestCenterline(target: CenterlineRecord, references: CenterlineRecord[]) {
  const [targetX, targetY] = lineCenter(target.line);
  return references.reduce((nearest, candidate) => {
    const [nearestX, nearestY] = lineCenter(nearest.line);
    const [candidateX, candidateY] = lineCenter(candidate.line);
    const nearestDistance = Math.hypot(targetX - nearestX, targetY - nearestY);
    const candidateDistance = Math.hypot(targetX - candidateX, targetY - candidateY);
    return candidateDistance < nearestDistance ? candidate : nearest;
  });
}

function angleBetweenLines(
  first: [number, number, number, number],
  second: [number, number, number, number]
): number | null {
  const firstVector = normalize([first[2] - first[0], first[3] - first[1]]);
  let secondVector = normalize([second[2] - second[0], second[3] - second[1]]);
  if (!firstVector || !secondVector) return null;

  if (dot(firstVector, secondVector) < 0) {
    secondVector = [-secondVector[0], -secondVector[1]];
  }
  const radians = Math.acos(clamp(dot(firstVector, secondVector), -1, 1));
  const degrees = (radians * 180) / Math.PI;
  if (!Number.isFinite(degrees) || degrees < 2) return null;
  return Math.min(degrees, 180 - degrees);
}

function lineCenter(line: [number, number, number, number]): [number, number] {
  return [(line[0] + line[2]) / 2, (line[1] + line[3]) / 2];
}

function drawAngleAnnotation(
  context: CanvasRenderingContext2D,
  target: CenterlineRecord,
  degrees: number
) {
  const [centerX, centerY] = lineCenter(target.line);
  const dx = target.line[2] - target.line[0];
  const dy = target.line[3] - target.line[1];
  const length = Math.max(Math.hypot(dx, dy), 1);
  const offset = Math.max(18, Math.min(context.canvas.width, context.canvas.height) * 0.035);
  const labelX = centerX + (-dy / length) * offset;
  const labelY = centerY + (dx / length) * offset;
  const label = `${Math.round(degrees)}°`;
  context.save();
  context.globalAlpha = 1;
  context.strokeStyle = target.color;
  context.fillStyle = target.color;
  context.lineWidth = Math.max(2, context.canvas.width / 900);
  context.beginPath();
  context.moveTo(centerX, centerY);
  context.lineTo(labelX, labelY);
  context.stroke();
  context.font = `${Math.max(14, context.canvas.width / 55)}px ${OVERLAY_FONT_FAMILY}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  const metrics = context.measureText(label);
  context.fillStyle = "rgba(2, 5, 9, 0.78)";
  context.fillRect(labelX - metrics.width / 2 - 6, labelY - 11, metrics.width + 12, 22);
  context.fillStyle = target.color;
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
