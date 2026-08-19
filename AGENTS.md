# Repository Guide

## Project Overview

SAM 3 Paddle Review is a local-first video review application for segmenting paddles and boats with SAM 3.1, tracking detections over time, deriving paddle/body-motion metrics, and exporting annotated video. The repository currently supports a complete deterministic mock pipeline; real SAM inference is optional and requires a compatible CUDA host and gated checkpoint access.

The deployed MVP is intentionally a single-host system with two Python processes and one browser client:

- `sam3-api` is a FastAPI process. It validates chunked uploads and job requests, persists state, serves result/media endpoints, starts export tasks, and serves the built SPA.
- `sam3-worker` is the only SQLite queue consumer. It normalizes video, runs each text prompt through the selected segmenter, optionally runs body-motion analysis, assigns stable tracks, and writes chunked results/manifests. It keeps the real model resident between jobs.
- SQLite in WAL mode stores metadata and the durable queue. Run exactly one API process and one claiming worker; multiple Uvicorn workers or GPU workers are not supported by this architecture. Each process opens its own short-lived transactions.
- Local filesystem storage under `SAM3_DATA_DIR` holds upload parts, source/normalized videos, job chunks, exports, temporary files, and the SQLite database. API and worker must share the same data directory on local durable disk.
- `apps/web` is a React 19/TypeScript/Vite SPA. It uploads videos in checksummed chunks, creates/polls jobs, loads result chunks, renders overlays in the review player, and requests annotated exports. Vite proxies `/api` to `127.0.0.1:8000` in development.

Important implementation boundaries:

- `src/sam3_service/main.py`: application factory, HTTP contracts, errors, health, media/results, export endpoints, and SPA fallback. Register API routes before any fallback route.
- `src/sam3_service/worker.py`: queue-processing state machine and orchestration. Preserve cancellation checks, heartbeat updates, and the distinction between segmentation failure and optional body-motion warnings.
- `src/sam3_service/db.py`: schema initialization, SQLite connection policy, and atomic job claiming. Database state is the coordination mechanism between API and worker.
- `src/sam3_service/segmenter.py`: `VideoSegmenter` boundary with deterministic `MockSegmenter` and optional `Sam3Segmenter`. Keep upstream SAM-specific compatibility code inside this adapter.
- `src/sam3_service/storage.py` and `media.py`: generated storage paths, upload assembly, FFprobe validation, and FFmpeg normalization.
- `src/sam3_service/tracking.py`: stable identities across raw per-frame model instance IDs.
- `src/sam3_service/body_motion.py`: optional MediaPipe or mock pose analysis and derived joint/lean metrics.
- `src/sam3_service/exporter.py`: annotated-video rendering, geometry, paddle phase/catch/exit analysis, SPM, optional pose overlays, and FFmpeg export. This module contains tightly coupled temporal/geometry behavior; add focused regression tests for changes.
- `src/sam3_service/paddle_completeness.py`: optional mask-only CNN gate used by paddle event export; its crop/padding behavior must match training.
- `src/sam3_service/schemas.py`: Pydantic request contracts. Mirror contract changes in `apps/web/src/api.ts` and `SAM3_API_DEVELOPMENT_GUIDE.md`.

`PRD.md` and `TECHNICAL_DESIGN.md` explain product intent and the target MVP. Prefer implemented code and `README.md` when those documents describe planned layout or behavior that is not present. `SAM3_API_DEVELOPMENT_GUIDE.md` is the detailed external contract. Paddle event logic is specified in `PADDLE_EVENT_DETECTION_TECHNICAL_DESIGN.md`.

## Build & Commands

Prerequisites are Python 3.12+, `uv`, Node.js 20+ with npm, and FFmpeg/FFprobe. Real inference additionally needs compatible NVIDIA/CUDA/PyTorch support, SAM 3.1, and checkpoint access.

Install the API, mock worker, and test dependencies from the repository root:

```bash
UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --extra dev
```

Build the web client:

```bash
cd apps/web
npm install
npm run build
```

Run the mock-mode application in separate terminals from the repository root:

```bash
SAM3_SEGMENTER=mock uv run sam3-api
SAM3_SEGMENTER=mock uv run sam3-worker
```

To exercise body motion without a MediaPipe asset, use:

```bash
SAM3_SEGMENTER=mock SAM3_BODY_MOTION_ANALYZER=mock uv run sam3-worker
```

For live frontend development:

```bash
cd apps/web
npm run dev
```

The frontend production check is `npm run build`; `npm run typecheck` runs TypeScript only. There is no repository-defined frontend lint or test command. Ruff has a 100-character line-length setting in `pyproject.toml`, but no Ruff script or pinned Ruff development dependency is defined.

Install optional real-inference and pose dependencies with:

```bash
UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --extra sam3 --extra pose
SAM3_SEGMENTER=sam3 uv run sam3-worker
```

The `sam3` extra pins an upstream SAM commit and Linux CUDA PyTorch. Do not casually change the SAM commit, PyTorch/CUDA pairing, or the two OpenCV distributions; validate them together on the deployment host. The default upstream path requires Hugging Face access. Offline mode requires both `SAM3_OFFLINE=1` and `SAM3_CHECKPOINT_PATH`.

Production deployment is native `systemd`, not Docker: build the SPA, install the Python project at one pinned revision, and run `sam3-api.service` plus `sam3-worker.service` as unprivileged users with a shared environment file and data directory. The complete unit examples and backup/upgrade sequence are in `README.md`; no deployment unit files or install script are committed. Use SQLite's online backup command or `VACUUM INTO`, not a raw copy of a live database.

## Code Style

Python conventions visible in the codebase:

- Use `from __future__ import annotations`, four-space indentation, type annotations, `Path` for filesystem paths, and snake_case functions/variables. Classes and Pydantic models use PascalCase; constants use UPPER_SNAKE_CASE.
- Keep lines within the configured 100-character Ruff limit where practical. Imports are grouped standard library, third party, then local modules.
- Prefer narrow typed boundaries: immutable `Settings`, Pydantic API models, dataclasses for internal value objects, and `Protocol` for replaceable segmenter/body-motion implementations.
- Raise `ServiceError` with a stable machine-readable code for expected API/worker failures. Preserve the shared JSON error envelope and request ID behavior. Do not leak raw exception text through HTTP contracts.
- Keep job and video state transitions explicit and persisted. Do not hold a SQLite transaction open while reading video, running FFmpeg, or performing inference.
- Keep the real and mock implementations behaviorally aligned so the mock pipeline remains a useful end-to-end test path.

Frontend conventions visible in `apps/web/src`:

- TypeScript is strict, ES2022, ESM-only, and uses React function components/hooks. Use PascalCase components/types, camelCase functions/values, and UPPER_SNAKE_CASE constants.
- Keep API request/response types and request helpers in `api.ts`; UI orchestration belongs in `App.tsx`, playback/overlay behavior in `Player.tsx`, and presentation in `styles.css`.
- Use relative `/api/v1/...` URLs so both the Vite proxy and the production same-origin FastAPI host work.

When changing result formats, update producer, API manifest, TypeScript types/consumer, tests, and the API guide together. Preserve `schema_version` and backward-compatible optional fields when feasible; old result chunks may lack newer fields such as exact frame indices.

## Testing

The committed tests use `unittest.TestCase` and `unittest.mock`; pytest can discover them because the `dev` extra includes pytest and `pyproject.toml` sets `testpaths = ["tests"]`. The documented full suite is:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

With the development environment installed, this is also valid:

```bash
uv run pytest
```

Testing conventions:

- Name files `tests/test_<area>.py`, classes `<Area>Test`, and methods `test_<behavior>`.
- Use temporary directories/databases and deterministic mock model/pose implementations. Tests must not require checkpoint downloads, network access, or a GPU.
- Patch external/model boundaries for focused tests. The pipeline integration test creates a tiny video with FFmpeg and exercises normalization, atomic queue claiming, mock segmentation, body-motion chunks, manifests, and rendering; FFmpeg/FFprobe must be installed for it.
- Add regression tests at the closest behavioral boundary. Geometry/event/export changes belong in `test_exporter.py`; model compatibility and mask fitting in `test_segmenter.py`; durable orchestration in `test_pipeline.py`; API middleware such as CORS in its focused test.
- Run the full Python suite after backend changes. Run `npm run build` after frontend or shared contract changes. No automated browser/frontend tests currently exist.

## Security

The current MVP has no application authentication, authorization, sessions, API keys, or TLS. Its explicit trust boundary is a private secure network. Do not expose port 8000 directly to an untrusted network; add TLS, authentication, owner-scoped authorization, and rate limiting first.

- Bind to a private interface where possible. CORS is disabled by default; if a separate browser origin is required, list exact origins in `SAM3_CORS_ALLOW_ORIGINS` rather than using a wildcard. CORS is not authentication.
- Never put Hugging Face/model credentials in the browser. Give checkpoint/model credentials only to the worker and load secrets from permission-restricted environment files. `.env` and `.env.*` are ignored; no dotenv loader is implemented, so export variables or use the systemd `EnvironmentFile`.
- Continue enforcing upload size, duration, type, contiguous parts, and optional part/whole-file checksums. Treat filenames as metadata only. Storage paths must be generated identifiers resolved beneath `SAM3_DATA_DIR`; never construct paths from user input or return raw storage paths.
- Keep API and worker unprivileged. The worker requires GPU device access but should not run as root. Protect the data directory because it contains uploaded videos, derived biometric/body-motion data, results, and job metadata.
- Avoid logging video contents, prompt text beyond operational need, filenames, access tokens, checkpoint credentials, or signed URLs. Do not use any of these as metric labels.
- Preserve retention/cleanup and free-space monitoring. Raw videos and derived results are currently local files with seven-day expiry metadata; storage exhaustion can stop both upload and queue processing.

## Configuration

Configuration is read directly from `SAM3_*` environment variables by `Settings.from_env()` in `config.py`. Paths are expanded and resolved. Defaults point at repository-local `data/` and `apps/web/dist`; `data/` and built frontend output are intentionally gitignored. API and worker must receive identical storage/database configuration.

Core service settings:

- `SAM3_DATA_DIR`, `SAM3_DATABASE_PATH`, `SAM3_STATIC_DIR`: shared runtime data, SQLite file, and built SPA paths.
- `SAM3_HOST`/`SAM3_PORT`: API bind address and port (used by the `sam3-api` entry point).
- `SAM3_CORS_ALLOW_ORIGINS`: comma-separated exact browser origins.
- `SAM3_MAX_UPLOAD_BYTES`, `SAM3_MAX_VIDEO_DURATION_SECONDS`, `SAM3_MAX_PROMPTS`, `SAM3_UPLOAD_CHUNK_BYTES`, `SAM3_RESULT_CHUNK_SECONDS`, `SAM3_POLL_INTERVAL_SECONDS`: validation, chunking, and worker polling limits.

Inference settings:

- `SAM3_SEGMENTER`: `mock` by default or `sam3` for real inference.
- `SAM3_CHECKPOINT_PATH`, `SAM3_OFFLINE`: local/gated checkpoint behavior.
- `SAM3_OFFLOAD_VIDEO_TO_CPU`, `SAM3_MAX_TRACKED_OBJECTS`, `SAM3_GROUNDING_BATCH_SIZE`, `SAM3_POSTPROCESS_BATCH_SIZE`: GPU memory/performance controls.
- `SAM3_ALLOW_UNSUPPORTED_BF16=1` enables an experimental slower T4 fallback. Do not enable it without deployment-like quality/performance validation.
- `SAM3_DEFAULT_REDETECT_INTERVAL_FRAMES`, `SAM3_DEFAULT_MAX_DETECTIONS_PER_FRAME`, `SAM3_MAX_DETECTIONS_PER_FRAME`, `SAM3_DEFAULT_DEDUPE_IOU_THRESHOLD`: default and hard detection controls. Keep backend defaults aligned with frontend presets and documentation.

Optional analysis/export settings:

- `SAM3_BODY_MOTION_ANALYZER`: `mediapipe` or deterministic `mock`; jobs opt into body motion separately.
- `SAM3_POSE_MODEL_PATH`: external MediaPipe Pose Landmarker `.task` asset.
- `SAM3_CENTERLINE_THICKNESS_PIXELS`: generated centerline/event-waterline thickness. `SAM3_SHAFT_THICKNESS_PIXELS` is only a deprecated fallback.
- `SAM3_PADDLE_COMPLETENESS_CHECKPOINT_PATH`, `SAM3_PADDLE_COMPLETENESS_PADDING_PERCENT`, `SAM3_PADDLE_COMPLETENESS_BATCH_SIZE`: optional completeness classifier. The committed default checkpoint is `src/checkpoints/best.pt` when present.
- `SAM3_EXPORT_FONT_PATH`: optional export-label font override.

Before changing production limits or GPU tuning, benchmark the target host with representative videos. Read `README.md` for the authoritative defaults and host setup, and update its configuration table whenever a supported environment variable or default changes.
