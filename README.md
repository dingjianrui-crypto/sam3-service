# SAM 3 Paddle Review

Upload a video, submit text prompts such as `paddle shaft`, process it asynchronously, and review synchronized segmentation overlays in a browser.

The application currently supports a complete mock-mode vertical slice. The SAM 3.1 adapter is included behind the same worker interface and is enabled on the target GPU after the environment feasibility check.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- FFmpeg and ffprobe
- For real inference: a CUDA host with SAM 3.1 installed and checkpoint access

## Local development

Install Python dependencies:

```bash
UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --extra dev
```

Install and build the web client:

```bash
cd apps/web
npm install
npm run build
cd ../..
```

Start the API:

```bash
SAM3_SEGMENTER=mock uv run sam3-api
```

Start the worker in a second terminal:

```bash
SAM3_SEGMENTER=mock uv run sam3-worker
```

Open `http://127.0.0.1:8000`. API documentation is available at `/docs`.

For live web development, run `npm run dev` under `apps/web`; Vite proxies `/api` to port 8000.

## Real SAM 3.1 worker

SAM 3 is an optional dependency because the API and mock worker do not need its large GPU stack. Install the pinned upstream package into this project's `.venv` with:

```bash
UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --extra sam3
```

The `sam3` extra in `pyproject.toml` pins an upstream commit compatible with the current adapter. On Linux it also installs PyTorch 2.10.0 and torchvision from the CUDA 12.8 PyTorch index used by the current upstream instructions. It includes setuptools because upstream SAM 3 imports its legacy `pkg_resources` module at runtime without declaring that dependency.

Confirm which package is imported:

```bash
uv run python -c "import sam3; print(sam3.__file__)"
```

If the server may download from Hugging Face and has already authenticated for the gated repository, run:

```bash
SAM3_SEGMENTER=sam3 uv run sam3-worker
```

With no checkpoint path, the upstream builder downloads the current SAM 3.1 checkpoint from `facebook/sam3.1` into the Hugging Face cache.

### Offline checkpoint

On an internet-connected machine with approved Hugging Face access:

```bash
hf auth login
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir ./sam3.1
```

Copy `sam3.1/sam3.1_multiplex.pt` to the GPU server, for example `/opt/models/sam3.1_multiplex.pt`, and start the worker with:

```bash
SAM3_SEGMENTER=sam3 \
SAM3_OFFLINE=1 \
SAM3_CHECKPOINT_PATH=/opt/models/sam3.1_multiplex.pt \
uv run sam3-worker
```

When `SAM3_CHECKPOINT_PATH` is supplied, the upstream builder loads that file and does not download the checkpoint. `SAM3_OFFLINE=1` also sets Hugging Face offline mode and refuses to start unless the local checkpoint exists. The tokenizer vocabulary is included in the installed SAM 3 package.

Current upstream video code uses BF16 autocast. NVIDIA T4 compatibility must be established before real inference is enabled. The worker fails early with `MODEL_UNSUPPORTED_GPU` when CUDA reports no BF16 support. Setting `SAM3_ALLOW_UNSUPPORTED_BF16=1` enables an experimental T4 path that replaces the upstream flash-only SDPA call with PyTorch's slower math kernel; output quality and performance still require validation.

On BF16-capable Ampere GPUs such as the NVIDIA A30, the worker enables PyTorch
Flash SDPA and memory-efficient SDPA, with the math implementation retained as a
fallback. `use_fa3` remains disabled because FlashAttention 3 targets
Hopper-class GPUs rather than the A30.

### Optional inference dependencies

The upstream README groups `einops`, `ninja`, `flash-attn-3`, and `cc_torch` under optional acceleration packages. However, the current SAM 3.1 multiplex implementation imports `einops` during normal startup. Its video predictor also imports `psutil`, its inference import chain reaches a training data module that imports `pycocotools`, and its default video loader imports `cv2` when processing begins. These are not all declared by the upstream base package, so our `sam3` extra installs them as required compatibility dependencies.

For the NVIDIA T4 deployment:

- do not install `flash-attn-3`; it requires an H100/H800-class Hopper GPU;
- `einops` is required by the current SAM 3.1 multiplex implementation;
- `psutil` is required by the upstream video predictor;
- `pycocotools` is required because the inference import chain loads an upstream COCO data module;
- `opencv-python-headless` 4.11 is required by the default server-side video loader; 4.12 is intentionally avoided because its NumPy 2 requirement conflicts with SAM 3's NumPy constraint;
- `ninja` is not required while FlashAttention 3 is disabled;
- leave `cc_torch` out of the initial deployment and benchmark the supported fallback first;
- consider `cc_torch` only as a later, separately tested optimization.

The service deliberately constructs the SAM 3.1 predictor with `use_fa3=False`, so installing FlashAttention 3 would not affect inference unless that code setting were also changed.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SAM3_DATA_DIR` | `./data` | SQLite, uploads, normalized videos, and results |
| `SAM3_DATABASE_PATH` | `$SAM3_DATA_DIR/sam3.sqlite3` | SQLite database |
| `SAM3_STATIC_DIR` | `./apps/web/dist` | Built web client |
| `SAM3_SEGMENTER` | `mock` | `mock` or `sam3` |
| `SAM3_CHECKPOINT_PATH` | unset | Local SAM 3.1 checkpoint; skips automatic download |
| `SAM3_OFFLINE` | `0` | Require local checkpoint and disable Hugging Face network access |
| `SAM3_OFFLOAD_VIDEO_TO_CPU` | `1` | Keep decoded video frames in CPU memory to reduce GPU use |
| `SAM3_MAX_TRACKED_OBJECTS` | `4` | Cap tracked instances to reduce inference memory |
| `SAM3_GROUNDING_BATCH_SIZE` | `1` | Frames processed together during grounding; higher values need more VRAM |
| `SAM3_POSTPROCESS_BATCH_SIZE` | `1` | Frames accumulated for mask post-processing |
| `SAM3_HOST` | `127.0.0.1` | Private bind address |
| `SAM3_PORT` | `8000` | HTTP port |
| `SAM3_MAX_UPLOAD_BYTES` | `524288000` | Maximum upload size |
| `SAM3_MAX_VIDEO_DURATION_SECONDS` | `300` | Maximum duration |
| `SAM3_MAX_PROMPTS` | `3` | Maximum prompts per job |
| `SAM3_UPLOAD_CHUNK_BYTES` | `8388608` | Upload part size |

## Linux daemon deployment with systemd

The deployment runs two long-lived services:

- `sam3-api.service`: FastAPI, the built web client, uploads, and result playback;
- `sam3-worker.service`: the single SQLite queue consumer and GPU model process.

Both services must run on the same host and share `/var/lib/sam3`. The commands below assume Ubuntu or another systemd-based Linux distribution, the application is installed at `/opt/sam3`, and the checkpoint is stored at `/opt/models/sam3.1_multiplex.pt`.

### 1. Install host prerequisites

Install these through the operating system or your standard server provisioning:

- Python 3.12 or newer;
- uv;
- Node.js 20 or newer and npm;
- FFmpeg with ffprobe;
- the SQLite command-line tool for operational backups;
- the NVIDIA driver and CUDA/PyTorch versions required by SAM 3.1;
- NVIDIA T4 access visible through `nvidia-smi`.

Do not run the services as root.

### 2. Create the service account and directories

```bash
sudo useradd \
  --system \
  --home-dir /var/lib/sam3 \
  --create-home \
  --shell /usr/sbin/nologin \
  sam3

sudo install -d -o root -g root -m 0755 /opt/sam3
sudo install -d -o root -g root -m 0755 /opt/models
sudo install -d -o sam3 -g sam3 -m 0750 /var/lib/sam3
sudo install -d -o root -g sam3 -m 0750 /etc/sam3
```

On distributions that control NVIDIA device access through groups, add the service account to the applicable groups:

```bash
sudo usermod -aG video,render sam3
```

Some distributions do not have a `render` group; omit it if it is absent.

### 3. Install the application

Copy or clone this repository into `/opt/sam3`, then install the application and pinned SAM 3 package:

```bash
cd /opt/sam3
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --no-dev --extra sam3
```

For a fully offline installation, copy a checkout of the pinned SAM 3 commit and the required CUDA/Python wheel files to the host. Install the local checkout into the same environment after the base `uv sync`. The example below assumes it was copied to `/opt/sam3-upstream`:

```bash
cd /opt/sam3
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache \
  uv pip install --python /opt/sam3/.venv/bin/python \
  -e /opt/sam3-upstream
```

Verify the import location and version before creating the services:

```bash
/opt/sam3/.venv/bin/python -c \
  "import sam3; print(sam3.__version__); print(sam3.__file__)"
```

Build the web client:

```bash
cd /opt/sam3/apps/web
sudo npm install
sudo npm run build
```

Keep `/opt/sam3` and `/opt/models` root-owned and read-only to the service account. Only `/var/lib/sam3` needs application write access:

```bash
sudo chown -R root:root /opt/sam3
sudo chmod -R a+rX /opt/sam3
sudo chown root:root /opt/models/sam3.1_multiplex.pt
sudo chmod 0644 /opt/models/sam3.1_multiplex.pt
```

For an offline deployment, copy the checkpoint and all Python/npm dependencies to the host before running these installation commands. `uv sync` and `npm install` otherwise require their configured package registries.

### 4. Configure the services

Create `/etc/sam3/sam3.env`:

```bash
sudo tee /etc/sam3/sam3.env >/dev/null <<'EOF'
SAM3_DATA_DIR=/var/lib/sam3
SAM3_DATABASE_PATH=/var/lib/sam3/sam3.sqlite3
SAM3_STATIC_DIR=/opt/sam3/apps/web/dist

# Bind to the server's private address where possible.
SAM3_HOST=0.0.0.0
SAM3_PORT=8000

SAM3_SEGMENTER=sam3
SAM3_OFFLINE=1
SAM3_CHECKPOINT_PATH=/opt/models/sam3.1_multiplex.pt
HF_HOME=/var/lib/sam3/huggingface

SAM3_MAX_UPLOAD_BYTES=524288000
SAM3_MAX_VIDEO_DURATION_SECONDS=300
SAM3_MAX_PROMPTS=3
SAM3_UPLOAD_CHUNK_BYTES=8388608
EOF

sudo chown root:sam3 /etc/sam3/sam3.env
sudo chmod 0640 /etc/sam3/sam3.env
```

Use `SAM3_SEGMENTER=mock` and remove the offline/checkpoint variables to validate the application plumbing without loading the model.

This MVP serves plain HTTP. Keep port 8000 inside the trusted network and restrict it with the host firewall or environment security group. Do not expose it directly to the public internet.

### 5. Create the API service

Create `/etc/systemd/system/sam3-api.service`:

```ini
[Unit]
Description=SAM 3 Paddle Review API
After=network.target local-fs.target

[Service]
Type=simple
User=sam3
Group=sam3
WorkingDirectory=/opt/sam3
EnvironmentFile=/etc/sam3/sam3.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/sam3/.venv/bin/sam3-api
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
UMask=0027

NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/sam3

[Install]
WantedBy=multi-user.target
```

### 6. Create the GPU worker service

Create `/etc/systemd/system/sam3-worker.service`:

```ini
[Unit]
Description=SAM 3 Paddle Review GPU Worker
After=network.target local-fs.target sam3-api.service
Wants=sam3-api.service

[Service]
Type=simple
User=sam3
Group=sam3
SupplementaryGroups=video render
WorkingDirectory=/opt/sam3
EnvironmentFile=/etc/sam3/sam3.env
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTORCH_ALLOC_CONF=expandable_segments:True
ExecStart=/opt/sam3/.venv/bin/sam3-worker
Restart=on-failure
RestartSec=10
TimeoutStopSec=120
KillSignal=SIGTERM
UMask=0027

NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/sam3

[Install]
WantedBy=multi-user.target
```

If the host has no `render` group, remove it from both `SupplementaryGroups` and the earlier `usermod` command. Do not add `PrivateDevices=true`; the worker needs access to `/dev/nvidia*`.

### 7. Enable and verify

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sam3-api.service
sudo systemctl enable --now sam3-worker.service

sudo systemctl status sam3-api.service
sudo systemctl status sam3-worker.service
curl http://127.0.0.1:8000/api/v1/health/live
curl http://127.0.0.1:8000/api/v1/health/ready
```

Open `http://SERVER_PRIVATE_IP:8000` from a browser inside the secure environment.

Follow logs:

```bash
sudo journalctl -u sam3-api.service -f
sudo journalctl -u sam3-worker.service -f
```

View logs from the current boot:

```bash
sudo journalctl -b -u sam3-api.service -u sam3-worker.service
```

### 8. Routine operations

Restart one or both services:

```bash
sudo systemctl restart sam3-api.service
sudo systemctl restart sam3-worker.service
```

Stop job processing while leaving uploaded results available:

```bash
sudo systemctl stop sam3-worker.service
```

Before an upgrade, stop both services and back up SQLite with its online backup command:

```bash
sudo systemctl stop sam3-worker.service sam3-api.service
sudo -u sam3 sqlite3 /var/lib/sam3/sam3.sqlite3 \
  ".backup '/var/lib/sam3/sam3-backup.sqlite3'"

cd /opt/sam3
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --no-dev --extra sam3
cd /opt/sam3/apps/web
sudo npm install
sudo npm run build

sudo systemctl start sam3-api.service sam3-worker.service
```

Check that the service account can see the GPU and checkpoint:

```bash
sudo -u sam3 nvidia-smi
sudo -u sam3 test -r /opt/models/sam3.1_multiplex.pt
```

If the API starts but jobs remain queued, inspect the worker log first. Common causes are missing checkpoint permissions, CUDA/PyTorch incompatibility, lack of `/dev/nvidia*` access, or the current T4/BF16 compatibility constraint.

If the worker reports `ModuleNotFoundError: No module named 'pkg_resources'`, update the source and resynchronize the `sam3` extra:

```bash
cd /opt/sam3
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache uv sync --no-dev --extra sam3
sudo systemctl restart sam3-worker.service
```

For an immediate repair before updating the source:

```bash
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache \
  uv pip install --python /opt/sam3/.venv/bin/python \
  "setuptools>=70,<82"
sudo systemctl restart sam3-worker.service
```

If an older deployment reports a missing `einops`, `psutil`, `pycocotools`, or `cv2` module, the equivalent immediate repair is:

```bash
sudo UV_CACHE_DIR=/tmp/sam3-uv-cache \
  uv pip install --python /opt/sam3/.venv/bin/python \
  "einops>=0.8,<1" "psutil>=5.9,<8" "pycocotools>=2.0.11,<3" \
  "opencv-python-headless==4.11.0.86"
sudo systemctl restart sam3-worker.service
```

## Tests

The core worker pipeline uses the standard-library test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The test generates a small video with FFmpeg, normalizes it, claims a SQLite job, runs mock segmentation, and verifies the result manifest and chunks.

## Documents

- [Product requirements](./PRD.md)
- [Technical design](./TECHNICAL_DESIGN.md)
