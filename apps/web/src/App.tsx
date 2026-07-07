import { FormEvent, useEffect, useState } from "react";
import {
  cancelJob,
  createJob,
  deleteJob,
  getJob,
  getManifest,
  Job,
  listJobs,
  ResultManifest,
  retryJob,
  uploadVideo
} from "./api";
import { Player } from "./Player";

const ACTIVE_STATES = new Set([
  "queued",
  "preprocessing",
  "segmenting",
  "postprocessing"
]);

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<Job | null>(null);
  const [manifest, setManifest] = useState<ResultManifest | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("paddle, boat");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refreshJobs() {
    try {
      const items = await listJobs();
      setJobs(items);
      if (selected) {
        const updated = items.find((job) => job.job_id === selected.job_id);
        if (updated) setSelected(updated);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  useEffect(() => {
    void refreshJobs();
  }, []);

  useEffect(() => {
    if (!jobs.some((job) => ACTIVE_STATES.has(job.state))) return;
    const interval = window.setInterval(() => void refreshJobs(), 1500);
    return () => window.clearInterval(interval);
  }, [jobs]);

  useEffect(() => {
    if (selected?.state !== "completed") {
      setManifest(null);
      return;
    }
    void getManifest(selected.job_id).then(setManifest).catch((reason) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, [selected?.job_id, selected?.state]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file || !prompt.trim()) return;
    setBusy(true);
    setError("");
    setUploadProgress(0);
    try {
      const videoId = await uploadVideo(file, setUploadProgress);
      const created = await createJob(
        videoId,
        prompt
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean)
      );
      setSelected(created);
      setFile(null);
      await refreshJobs();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function act(action: () => Promise<unknown>) {
    setError("");
    try {
      await action();
      await refreshJobs();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return (
    <div className="app">
      <header>
        <div className="brand-mark" aria-hidden="true">
          <i />
        </div>
        <div>
          <p className="eyebrow">SAM 3.1 VIDEO SEGMENTATION</p>
          <h1>Paddle Vision</h1>
        </div>
        <span className="environment">Secure environment · HTTP</span>
      </header>

      <main>
        <aside>
          <form className="upload-card" onSubmit={submit}>
            <div>
              <p className="step">NEW ANALYSIS</p>
              <h2>Track objects</h2>
              <p className="muted">Upload a video and describe what SAM should follow.</p>
            </div>
            <label className={file ? "dropzone selected" : "dropzone"}>
              <input
                type="file"
                accept=".mp4,.mov,video/mp4,video/quicktime"
                disabled={busy}
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
              <strong>{file ? file.name : "Choose MP4 or MOV"}</strong>
              <small>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
                  : "Up to 500 MB · 5 minutes"}
              </small>
            </label>
            <label className="field">
              <span>Text prompt</span>
              <input
                value={prompt}
                disabled={busy}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="paddle, boat"
              />
              <small>Separate up to three prompts with commas.</small>
            </label>
            {busy && (
              <div className="upload-progress">
                <span style={{ width: `${uploadProgress}%` }} />
                <small>
                  {uploadProgress < 100
                    ? `Uploading ${uploadProgress.toFixed(0)}%`
                    : "Validating and creating job…"}
                </small>
              </div>
            )}
            <button className="primary" disabled={!file || !prompt.trim() || busy}>
              {busy ? "Preparing…" : "Start segmentation"}
            </button>
          </form>

          <section className="jobs">
            <div className="section-title">
              <h2>Recent jobs</h2>
              <button className="quiet" onClick={() => void refreshJobs()}>
                Refresh
              </button>
            </div>
            {jobs.length === 0 && <p className="empty">No videos yet.</p>}
            {jobs.map((job) => (
              <button
                key={job.job_id}
                className={selected?.job_id === job.job_id ? "job active" : "job"}
                onClick={() => setSelected(job)}
              >
                <span className={`status-dot ${job.state}`} />
                <span>
                  <strong>{job.prompts.map((item) => item.text).join(", ")}</strong>
                  <small>
                    {job.state} ·{" "}
                    {new Date(job.created_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit"
                    })}
                  </small>
                </span>
                {ACTIVE_STATES.has(job.state) && <b>{job.progress.percent}%</b>}
              </button>
            ))}
          </section>
        </aside>

        <section className="workspace">
          {error && (
            <div className="alert">
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          {!selected && (
            <div className="welcome">
              <div className="orb">
                <i />
              </div>
              <p className="step">READY</p>
              <h2>Your segmentation workspace</h2>
              <p>Start a new analysis or select a recent job to review its tracked masks.</p>
            </div>
          )}
          {selected && (
            <>
              <div className="job-header">
                <div>
                  <p className="step">JOB {selected.job_id.slice(0, 8)}</p>
                  <h2>{selected.prompts.map((item) => item.text).join(" + ")}</h2>
                  <p className="muted">
                    {selected.model_name ?? "Waiting for worker"} · attempt {selected.attempt ?? 1}
                  </p>
                </div>
                <div className="job-actions">
                  {ACTIVE_STATES.has(selected.state) && (
                    <button
                      className="secondary"
                      onClick={() => void act(() => cancelJob(selected.job_id))}
                    >
                      Cancel
                    </button>
                  )}
                  {["failed", "cancelled"].includes(selected.state) && (
                    <button
                      className="secondary"
                      onClick={() => void act(() => retryJob(selected.job_id))}
                    >
                      Retry
                    </button>
                  )}
                  {["failed", "cancelled", "completed"].includes(selected.state) && (
                    <button
                      className="danger"
                      onClick={() =>
                        void act(async () => {
                          await deleteJob(selected.job_id);
                          setSelected(null);
                        })
                      }
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
              {ACTIVE_STATES.has(selected.state) && (
                <div className="processing">
                  <div className="radar">
                    <i />
                    <i />
                  </div>
                  <p className="step">{selected.progress.stage}</p>
                  <h3>Finding your objects</h3>
                  <div className="progress-track">
                    <span style={{ width: `${selected.progress.percent}%` }} />
                  </div>
                  <p className="muted">
                    {selected.progress.processed_frames.toLocaleString()} /{" "}
                    {selected.progress.total_frames.toLocaleString()} frames
                  </p>
                </div>
              )}
              {selected.state === "failed" && (
                <div className="state-card failed">
                  <h3>Processing failed</h3>
                  <p>{selected.error?.message ?? "The worker could not complete this job."}</p>
                </div>
              )}
              {selected.state === "cancelled" && (
                <div className="state-card">
                  <h3>Job cancelled</h3>
                  <p>The uploaded video is still available, so you can retry this job.</p>
                </div>
              )}
              {selected.state === "completed" &&
                (manifest ? <Player manifest={manifest} /> : <div className="loading">Loading result…</div>)}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
