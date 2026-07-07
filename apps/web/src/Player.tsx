import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FrameMask, getChunk, ResultManifest } from "./api";

type Props = {
  manifest: ResultManifest;
};

type OverlayMode = "paddle" | "shaft";

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
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("paddle");
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
      const tolerance = 500 / Math.max(manifest.video.fps, 1);
      const nearby = records.filter(
        (record) =>
          Math.abs(record.timestamp_ms - timeMs) <= tolerance &&
          enabledPrompts.has(record.prompt_id)
      );
      for (const record of nearby) {
        const color = colorByPrompt.get(record.prompt_id) ?? "#35C2FF";
        const segmentation =
          overlayMode === "shaft" && record.shaft_segmentation
            ? record.shaft_segmentation
            : record.segmentation;
        context.save();
        context.globalAlpha = opacity;
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
          drawRle(context, segmentation, color, opacity);
        }
        context.restore();
        if (showBoxes) {
          const [x, y, width, height] =
            overlayMode === "shaft" && record.shaft_box_xywh
              ? record.shaft_box_xywh
              : record.box_xywh;
          context.strokeStyle = color;
          context.lineWidth = Math.max(2, canvas.width / 600);
          context.strokeRect(x, y, width, height);
          context.fillStyle = color;
          context.font = `${Math.max(13, canvas.width / 60)}px system-ui`;
          context.fillText(
            `${record.instance_id}${record.score == null ? "" : ` · ${record.score.toFixed(2)}`}`,
            x,
            Math.max(18, y - 6)
          );
        }
      }
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
            <option value="paddle">Paddle mask</option>
            <option value="shaft">Shaft centerline</option>
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
      </div>
    </section>
  );
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
