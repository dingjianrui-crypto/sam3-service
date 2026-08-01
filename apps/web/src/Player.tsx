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

type ExportRect = { x: number; y: number; width: number; height: number };
type ExportSelectionKeyframe = ExportRect & { time_ms: number };
type SelectionInteraction =
  | { mode: "draw"; start: { x: number; y: number }; timeMs: number }
  | {
      mode: "move";
      start: { x: number; y: number };
      rectangle: ExportRect;
      timeMs: number;
    };

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
  const selectionRectangleRef = useRef<HTMLDivElement>(null);
  const chunksRef = useRef(new Map<number, FrameMask[]>());
  const loadingRef = useRef(new Set<number>());
  const selectionInteractionRef = useRef<SelectionInteraction | null>(null);
  const lastEditedKeyframeRef = useRef<number | null>(null);
  const selectionKeyframesRef = useRef<ExportSelectionKeyframe[]>([]);
  const playbackTimeRef = useRef(0);
  const pendingSeekTimeRef = useRef<number | null>(null);
  const manualTimelineTimeRef = useRef(0);
  const manualTimelineActiveRef = useRef(false);
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
  const [rectangleToolActive, setRectangleToolActive] = useState(false);
  const [selectionKeyframes, setSelectionKeyframes] = useState<ExportSelectionKeyframe[]>([]);
  const [draftSelection, setDraftSelection] = useState<ExportRect | null>(null);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const [videoDurationMs, setVideoDurationMs] = useState(manifest.video.duration_ms);
  const [exportLabelPosition, setExportLabelPosition] = useState<ExportLabelPosition>("top");
  const [exportMetricCenterOffsetPercent, setExportMetricCenterOffsetPercent] = useState(
    defaultMetricCenterOffsetPercent(manifest)
  );
  const [exportFontSize, setExportFontSize] = useState(32);
  const [exportMetricCount, setExportMetricCount] = useState(1);
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
  const exportSelection = useMemo(
    () => selectionAtTime(selectionKeyframes, playbackTimeMs),
    [playbackTimeMs, selectionKeyframes]
  );

  useEffect(() => {
    selectionKeyframesRef.current = selectionKeyframes;
  }, [selectionKeyframes]);
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
    setRectangleToolActive(false);
    setSelectionKeyframes([]);
    selectionKeyframesRef.current = [];
    setDraftSelection(null);
    setPlaybackTimeMs(0);
    playbackTimeRef.current = 0;
    pendingSeekTimeRef.current = null;
    manualTimelineTimeRef.current = 0;
    manualTimelineActiveRef.current = false;
    setVideoDurationMs(manifest.video.duration_ms);
    lastEditedKeyframeRef.current = null;
  }, [defaultReferencePromptId, defaultTargetPromptIds, manifest.job_id]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let handle = 0;
    if (video.requestVideoFrameCallback) {
      const callback = (_now: number, metadata: { mediaTime: number }) => {
        draw(metadata.mediaTime);
        const timeMs = Math.round(metadata.mediaTime * 1000);
        const accepted = acceptVideoTime(timeMs);
        const displayTimeMs = accepted ? timeMs : playbackTimeRef.current;
        const rectangle = selectionAtTime(selectionKeyframesRef.current, displayTimeMs);
        if (selectionRectangleRef.current && rectangle && !selectionInteractionRef.current) {
          applySelectionStyle(selectionRectangleRef.current, rectangle);
        }
        handle = video.requestVideoFrameCallback!(callback);
      };
      handle = video.requestVideoFrameCallback(callback);
      return () => video.cancelVideoFrameCallback?.(handle);
    }
    const fallback = () => draw(video.currentTime);
    video.addEventListener("timeupdate", fallback);
    return () => video.removeEventListener("timeupdate", fallback);
  }, [draw]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const syncPlayback = () => {
      const timeMs = Math.round(video.currentTime * 1000);
      acceptVideoTime(timeMs);
    };
    const clearManualTimelineTime = () => {
      manualTimelineActiveRef.current = false;
      video.closest<HTMLElement>(".video-shell")?.removeAttribute("data-timeline-time-ms");
    };
    const syncDuration = () => {
      if (Number.isFinite(video.duration)) setVideoDurationMs(Math.round(video.duration * 1000));
    };
    video.addEventListener("timeupdate", syncPlayback);
    video.addEventListener("seeked", syncPlayback);
    video.addEventListener("play", clearManualTimelineTime);
    video.addEventListener("loadedmetadata", syncDuration);
    video.addEventListener("durationchange", syncDuration);
    syncDuration();
    return () => {
      video.removeEventListener("timeupdate", syncPlayback);
      video.removeEventListener("seeked", syncPlayback);
      video.removeEventListener("play", clearManualTimelineTime);
      video.removeEventListener("loadedmetadata", syncDuration);
      video.removeEventListener("durationchange", syncDuration);
    };
  }, [manifest.job_id]);

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

  function beginRectangle(event: React.PointerEvent<HTMLDivElement>) {
    const point = normalizedPointer(event);
    const interactionTimeMs = manualTimelineActiveRef.current
      ? manualTimelineTimeRef.current
      : playbackTimeMs;
    if (rectangleToolActive) {
      selectionInteractionRef.current = {
        mode: "draw",
        start: point,
        timeMs: interactionTimeMs
      };
      setDraftSelection({ x: point.x, y: point.y, width: 0, height: 0 });
    } else if (exportSelection && pointInRectangle(point, exportSelection)) {
      selectionInteractionRef.current = {
        mode: "move",
        start: point,
        rectangle: exportSelection,
        timeMs: interactionTimeMs
      };
      setDraftSelection(exportSelection);
    } else {
      return;
    }
    videoRef.current?.pause();
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function updateRectangle(event: React.PointerEvent<HTMLDivElement>) {
    const interaction = selectionInteractionRef.current;
    if (!interaction) return;
    const point = normalizedPointer(event);
    setDraftSelection(
      interaction.mode === "draw"
        ? rectangleFromPoints(interaction.start, point)
        : moveRectangle(interaction.rectangle, interaction.start, point)
    );
  }

  function finishRectangle(event: React.PointerEvent<HTMLDivElement>) {
    const interaction = selectionInteractionRef.current;
    if (!interaction) return;
    const point = normalizedPointer(event);
    const rectangle =
      interaction.mode === "draw"
        ? rectangleFromPoints(interaction.start, point)
        : moveRectangle(interaction.rectangle, interaction.start, point);
    selectionInteractionRef.current = null;
    setDraftSelection(null);
    if (rectangle.width >= 0.01 && rectangle.height >= 0.01) {
      const videoShell = event.currentTarget.ownerDocument.querySelector<HTMLElement>(
        ".video-shell[data-timeline-time-ms]"
      );
      const storedTime = videoShell?.dataset.timelineTimeMs;
      const storedTimeMs = storedTime == null ? Number.NaN : Number(storedTime);
      const timeMs = Number.isFinite(storedTimeMs)
        ? storedTimeMs
        : interaction.timeMs;
      setSelectionKeyframes((current) => {
        const previousRectangle = lastRecordedRectangle(current, timeMs);
        if (previousRectangle && rectanglesEqual(previousRectangle, rectangle)) {
          return current;
        }
        const next = upsertSelectionKeyframe(current, timeMs, rectangle);
        selectionKeyframesRef.current = next;
        lastEditedKeyframeRef.current = timeMs;
        return next;
      });
      setRectangleToolActive(false);
    }
  }

  function undoSelectionKeyframe() {
    setSelectionKeyframes((current) => {
      if (!current.length) return current;
      const editedTime = lastEditedKeyframeRef.current;
      const index = editedTime == null
        ? current.length - 1
        : current.findIndex((keyframe) => keyframe.time_ms === editedTime);
      const removeIndex = index >= 0 ? index : current.length - 1;
      const next = current.filter((_, keyframeIndex) => keyframeIndex !== removeIndex);
      lastEditedKeyframeRef.current = next.at(-1)?.time_ms ?? null;
      return next;
    });
    setDraftSelection(null);
    setRectangleToolActive(false);
  }

  function seekTo(timeMs: number) {
    const clampedTimeMs = clamp(timeMs, 0, Math.max(videoDurationMs, 0));
    pendingSeekTimeRef.current = clampedTimeMs;
    playbackTimeRef.current = clampedTimeMs;
    manualTimelineTimeRef.current = clampedTimeMs;
    manualTimelineActiveRef.current = true;
    const video = videoRef.current;
    if (video) video.currentTime = clampedTimeMs / 1000;
    setPlaybackTimeMs(clampedTimeMs);
  }

  function acceptVideoTime(timeMs: number) {
    const pendingTimeMs = pendingSeekTimeRef.current;
    const toleranceMs = Math.max(100, 2000 / Math.max(manifest.video.fps, 1));
    if (pendingTimeMs != null && Math.abs(timeMs - pendingTimeMs) > toleranceMs) {
      return false;
    }
    pendingSeekTimeRef.current = null;
    playbackTimeRef.current = timeMs;
    setPlaybackTimeMs(timeMs);
    return true;
  }

  function seekTimelinePointer(event: React.PointerEvent<HTMLDivElement>) {
    if (event.type === "pointermove" && event.buttons !== 1) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = clamp(
      (event.clientX - bounds.left) / Math.max(bounds.width, 1),
      0,
      1
    );
    const frameStepMs = Math.max(1, Math.round(1000 / Math.max(manifest.video.fps, 1)));
    const timeMs = Math.round((ratio * videoDurationMs) / frameStepMs) * frameStepMs;
    if (event.type === "pointerdown") {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    event.currentTarget.closest<HTMLElement>(".video-shell")!.dataset.timelineTimeMs = String(
      timeMs
    );
    seekTo(timeMs);
  }

  function seekTimelineKey(event: React.KeyboardEvent<HTMLDivElement>) {
    const frameStepMs = Math.max(1, Math.round(1000 / Math.max(manifest.video.fps, 1)));
    const keyTimes: Record<string, number> = {
      ArrowLeft: playbackTimeMs - frameStepMs,
      ArrowRight: playbackTimeMs + frameStepMs,
      Home: 0,
      End: videoDurationMs
    };
    const timeMs = keyTimes[event.key];
    if (timeMs == null) return;
    event.preventDefault();
    event.currentTarget.closest<HTMLElement>(".video-shell")!.dataset.timelineTimeMs = String(
      timeMs
    );
    seekTo(timeMs);
  }

  const exportCenterlineVideo = useCallback(async () => {
    setExporting(true);
    setExportStatus("Rendering MP4 on server…");
    try {
      const blob = await exportJobVideo(manifest.job_id, {
        angle_label_position: exportLabelPosition,
        angle_label_font_size: exportFontSize,
        include_spm: exportSpmEnabled,
        metric_count: exportMetricCount,
        metric_center_offset_percent: exportMetricCenterOffsetPercent,
        reference_prompt_id: angleReferencePromptId,
        target_prompt_ids: [...angleTargetPromptIds],
        selection_keyframes: selectionKeyframes.length ? selectionKeyframes : undefined
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
    exportMetricCount,
    exportMetricCenterOffsetPercent,
    exportSpmEnabled,
    manifest.job_id,
    selectionKeyframes
  ]);

  return (
    <section className="viewer">
      <div className="video-shell">
        <div className="video-toolbar" role="toolbar" aria-label="Video selection tools">
          <button
            className={rectangleToolActive ? "tool-button active" : "tool-button"}
            title="Draw export rectangle"
            aria-pressed={rectangleToolActive}
            onClick={() => {
              setRectangleToolActive((current) => !current);
              videoRef.current?.pause();
            }}
          >
            <span className="rectangle-tool-icon" aria-hidden="true" />
            Rectangle
          </button>
          <button
            className="tool-button"
            title="Remove export rectangle"
            disabled={!selectionKeyframes.length && !draftSelection}
            onClick={undoSelectionKeyframe}
          >
            Undo
          </button>
        </div>
        <div className="video-frame">
          <video ref={videoRef} src={manifest.video.url} controls playsInline />
          <canvas ref={canvasRef} />
          <div
            className={
              rectangleToolActive
                ? "selection-surface active"
                : exportSelection
                  ? "selection-surface movable"
                  : "selection-surface"
            }
            onPointerDown={beginRectangle}
            onPointerMove={updateRectangle}
            onPointerUp={finishRectangle}
            onPointerCancel={() => {
              selectionInteractionRef.current = null;
              setDraftSelection(null);
            }}
          >
            {(draftSelection ?? exportSelection) && (
              <div
                ref={selectionRectangleRef}
                className="export-selection-rectangle"
                style={selectionStyle(draftSelection ?? exportSelection!)}
              />
            )}
          </div>
        </div>
        <div className="selection-timeline">
          <div
            className="selection-timeline-track"
            role="slider"
            tabIndex={0}
            aria-label="Video timeline"
            aria-valuemin={0}
            aria-valuemax={Math.max(videoDurationMs, 1)}
            aria-valuenow={Math.min(playbackTimeMs, Math.max(videoDurationMs, 1))}
            onPointerDown={seekTimelinePointer}
            onPointerMove={seekTimelinePointer}
            onPointerUp={seekTimelinePointer}
            onKeyDown={seekTimelineKey}
          >
            <div className="selection-timeline-rail" />
            <div
              className="selection-timeline-progress"
              style={{ width: `${(playbackTimeMs / Math.max(videoDurationMs, 1)) * 100}%` }}
            />
            <div
              className="selection-timeline-playhead"
              style={{ left: `${(playbackTimeMs / Math.max(videoDurationMs, 1)) * 100}%` }}
            />
            {selectionKeyframes.map((keyframe, index) => (
              <button
                className={
                  keyframe.time_ms <= playbackTimeMs
                    ? "selection-keyframe passed"
                    : "selection-keyframe"
                }
                key={keyframe.time_ms}
                style={{ left: `${(keyframe.time_ms / Math.max(videoDurationMs, 1)) * 100}%` }}
                title={`Rectangle keyframe at ${formatTime(keyframe.time_ms)}`}
                aria-label={`Rectangle keyframe at ${formatTime(keyframe.time_ms)}`}
                onClick={() => seekTo(keyframe.time_ms)}
              >
                <span>{index + 1}</span>
              </button>
            ))}
          </div>
          <output>
            {formatTime(playbackTimeMs)} / {formatTime(videoDurationMs)} · {selectionKeyframes.length} keyframes
          </output>
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
                  const nextTargets = new Set(angleTargetPromptIds);
                  nextTargets.delete(nextReference);
                  if (nextTargets.size === 0) {
                    manifest.prompts
                      .filter((prompt) => prompt.id !== nextReference)
                      .forEach((prompt) => nextTargets.add(prompt.id));
                  }
                  setAngleReferencePromptId(nextReference);
                  setAngleTargetPromptIds(nextTargets);
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
            Metric count
            <input
              type="number"
              min="1"
              max="4"
              step="1"
              value={exportMetricCount}
              onChange={(event) => {
                const value = Number(event.target.value);
                setExportMetricCount(Number.isFinite(value) ? clamp(Math.round(value), 1, 4) : 1);
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
          <button
            className="secondary export-button"
            disabled={exporting}
            onClick={exportCenterlineVideo}
          >
            {exporting ? "Exporting…" : "Export"}
          </button>
          {exportStatus && <span className="export-status">{exportStatus}</span>}
        </div>
      </div>
    </section>
  );
}

function normalizedPointer(event: React.PointerEvent<HTMLDivElement>) {
  const bounds = event.currentTarget.getBoundingClientRect();
  return {
    x: clamp((event.clientX - bounds.left) / Math.max(bounds.width, 1), 0, 1),
    y: clamp((event.clientY - bounds.top) / Math.max(bounds.height, 1), 0, 1)
  };
}

function rectangleFromPoints(
  first: { x: number; y: number },
  second: { x: number; y: number }
): ExportRect {
  return {
    x: Math.min(first.x, second.x),
    y: Math.min(first.y, second.y),
    width: Math.abs(second.x - first.x),
    height: Math.abs(second.y - first.y)
  };
}

function moveRectangle(
  rectangle: ExportRect,
  start: { x: number; y: number },
  current: { x: number; y: number }
): ExportRect {
  return {
    ...rectangle,
    x: clamp(rectangle.x + current.x - start.x, 0, 1 - rectangle.width),
    y: clamp(rectangle.y + current.y - start.y, 0, 1 - rectangle.height)
  };
}

function pointInRectangle(point: { x: number; y: number }, rectangle: ExportRect) {
  return (
    point.x >= rectangle.x &&
    point.x <= rectangle.x + rectangle.width &&
    point.y >= rectangle.y &&
    point.y <= rectangle.y + rectangle.height
  );
}

function upsertSelectionKeyframe(
  keyframes: ExportSelectionKeyframe[],
  timeMs: number,
  rectangle: ExportRect
) {
  const frameToleranceMs = 20;
  const next = keyframes.filter(
    (keyframe) => Math.abs(keyframe.time_ms - timeMs) > frameToleranceMs
  );
  next.push({ ...rectangle, time_ms: timeMs });
  return next.sort((first, second) => first.time_ms - second.time_ms);
}

function lastRecordedRectangle(
  keyframes: ExportSelectionKeyframe[],
  timeMs: number
): ExportRect | null {
  if (!keyframes.length) return null;
  let previous = keyframes[0];
  for (const keyframe of keyframes) {
    if (keyframe.time_ms > timeMs) break;
    previous = keyframe;
  }
  return previous;
}

function rectanglesEqual(first: ExportRect, second: ExportRect) {
  const epsilon = 0.0005;
  return (
    Math.abs(first.x - second.x) <= epsilon &&
    Math.abs(first.y - second.y) <= epsilon &&
    Math.abs(first.width - second.width) <= epsilon &&
    Math.abs(first.height - second.height) <= epsilon
  );
}

function selectionAtTime(
  keyframes: ExportSelectionKeyframe[],
  timeMs: number
): ExportRect | null {
  if (!keyframes.length) return null;
  let active = keyframes[0];
  for (const keyframe of keyframes) {
    if (keyframe.time_ms > timeMs) break;
    active = keyframe;
  }
  return rectangleFromKeyframe(active);
}

function rectangleFromKeyframe(keyframe: ExportSelectionKeyframe): ExportRect {
  return {
    x: keyframe.x,
    y: keyframe.y,
    width: keyframe.width,
    height: keyframe.height
  };
}

function formatTime(timeMs: number) {
  const totalSeconds = Math.max(0, Math.round(timeMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function selectionStyle(rectangle: ExportRect): React.CSSProperties {
  return {
    left: `${rectangle.x * 100}%`,
    top: `${rectangle.y * 100}%`,
    width: `${rectangle.width * 100}%`,
    height: `${rectangle.height * 100}%`
  };
}

function applySelectionStyle(element: HTMLDivElement, rectangle: ExportRect) {
  element.style.left = `${rectangle.x * 100}%`;
  element.style.top = `${rectangle.y * 100}%`;
  element.style.width = `${rectangle.width * 100}%`;
  element.style.height = `${rectangle.height * 100}%`;
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
        `${recordTrackLabel(record)}${record.score == null ? "" : ` · ${record.score.toFixed(2)}`}`,
        x,
        Math.max(18, y - 6)
      );
    }
  }
  if (options.angleConfig.enabled) {
    drawAngleAnnotations(context, centerlines, options.angleConfig);
  }
}

function recordTrackLabel(record: FrameMask) {
  return record.instance_id;
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
