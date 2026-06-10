# Deployment

FAQCluster is deployed as two Docker containers orchestrated by Docker Compose, with a planned third container for POP-Translation running on a separate VM.

---

## Docker Compose

**File:** `docker-compose.yml`

### Services

#### `pipeline` — ML backend

```yaml
image: vicharanashala/faqcluster-pipeline:latest
network_mode: host
expose:
  - "8031"
```

- Runs with `network_mode: host` so the pipeline can reach the LLM inference server at `100.100.108.44:8013` over Tailscale without NAT.
- Requires GPU: reserved via `deploy.resources.reservations.devices` (driver `nvidia`, capabilities `[gpu]`).
- All secrets injected from `.env` file (not baked into the image).
- Healthcheck: `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8031/')"` every 15 seconds, 5 retries, 30s start period.
- Mounts `./pipeline_server/app-data:/app/pipeline_server/app-data` as a named volume for local scratch.

#### `frontend` — React UI

```yaml
image: vicharanashala/faqcluster-frontend:latest
ports:
  - "8030:80"
depends_on:
  pipeline:
    condition: service_healthy
```

- Nginx container serving the compiled React SPA.
- Only starts after the pipeline healthcheck passes.
- Proxies `/api/` to `http://localhost:8031` (pipeline server).

---

## Environment Variables (`.env`)

The `.env` file at the repo root is loaded by the pipeline container at startup (via `python-dotenv`). It is **not committed to git** — copy and fill in before deploying.

```bash
# Zoho WorkDrive — OAuth2 credentials
ZOHO_CLIENT_ID=<from Zoho API Console>
ZOHO_CLIENT_SECRET=<from Zoho API Console>
ZOHO_REFRESH_TOKEN=<exchange code for this once>
ZOHO_ROOT_FOLDER_ID=<the long ID from your WorkDrive folder link>

# LLM inference (optional overrides — defaults work for prod VM)
LLM_API_URL=http://100.100.108.44:8013/v1/chat/completions
LLM_MODEL=google/gemma-4-26B-A4B-it
LLM_API_KEY=
LLM_THINKING_ENABLED=false
LLM_GROUPING_STRICTNESS=strict
```

### Getting Zoho credentials

1. Go to [https://api-console.zoho.in](https://api-console.zoho.in) and create a Self Client.
2. Generate a code with scope `WorkDrive.files.ALL`.
3. Exchange the code for a refresh token using:
   ```bash
   curl -X POST "https://accounts.zoho.in/oauth/v2/token" \
     -d "grant_type=authorization_code&client_id=...&client_secret=...&code=..."
   ```
4. Copy `refresh_token` from the response into `.env`.
5. The `ZOHO_ROOT_FOLDER_ID` is the long alphanumeric ID in the URL when you navigate to your WorkDrive folder.

---

## Production Deployment Steps

```bash
# 1. SSH into the GPU VM
ssh <vm-ip>

# 2. Clone the repo (first time only)
git clone <repo-url>
cd FAQCluster

# 3. Create .env with credentials
cp .env.example .env
nano .env

# 4. Pull latest images
docker compose pull

# 5. Start services
docker compose up -d

# 6. Check logs
docker compose logs -f pipeline
```

After startup:
- Frontend: `http://<vm-ip>:8030`
- Pipeline API: `http://<vm-ip>:8031`

---

## Dockerfile (pipeline service)

**File:** `pipeline_server/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app/pipeline_server
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

EXPOSE 8031
CMD ["uvicorn", "pipeline_server:app", "--host", "0.0.0.0", "--port", "8031"]
```

Key notes:
- Uses `python:3.11-slim` — the SentenceTransformer / HDBSCAN GPU dependencies are installed via `requirements.lock` (pinned versions).
- No `--reload` in production CMD — the Dockerfile CMD is for production; use `--reload` only in dev.
- The GPU is accessed via the CUDA toolkit from the host NVIDIA driver (not installed in the image itself — NVIDIA Container Toolkit on the host handles this).

---

## GitHub Actions CI/CD

**File:** `.github/workflows/build-pipeline.yml`

Triggers on push to `main` or `server` when any file under `pipeline_server/**` changes.

Steps:
1. Log in to Docker Hub using `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repository secrets.
2. Build `pipeline_server/Dockerfile`.
3. Push two tags:
   - `vicharanashala/faqcluster-pipeline:latest`
   - `vicharanashala/faqcluster-pipeline:<git-sha>`

The frontend is built and pushed by a separate workflow (not in this repo).

---

## Hardware Requirements

The pipeline container requires an NVIDIA GPU with CUDA support.

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 24 GB+ |
| RAM | 16 GB | 32 GB+ |
| Disk (scratch) | 20 GB | 50 GB |
| Network | Tailscale VPN to LLM host | Same |

Stages 1–3 use the GPU intensively (SentenceTransformer + UMAP). The LLM inference server (`100.100.108.44:8013`) runs separately and handles Stages 2, 3B–D, 4 (local mode), and 7.

---

## Local Development

```bash
cd pipeline_server

# Create venv
python -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.lock

# Set env vars (or create .env in FAQCluster/ root)
export ZOHO_CLIENT_ID=...
export ZOHO_REFRESH_TOKEN=...
...

# Run server with auto-reload
uvicorn pipeline_server:app --host 0.0.0.0 --port 8031 --reload
```

The server will try to connect to Zoho on first request. If Zoho is unreachable, all file/run routes return HTTP 503.
