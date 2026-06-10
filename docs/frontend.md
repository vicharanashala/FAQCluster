# Frontend

The frontend is a **React 18 + Vite** single-page application that provides a UI for running pipelines, monitoring jobs, browsing outputs, and managing POP translations.

It runs as a separate Docker container (`vicharanashala/faqcluster-frontend:latest`) on port 8030, served by Nginx. All API calls go to the pipeline server on port 8031.

---

## Tech Stack

- **React 18** — UI framework
- **Vite** — build tool
- **Nginx** — serves the compiled SPA and proxies `/api/*` to port 8031

---

## API Contract

The frontend talks to the pipeline server exclusively via HTTP. The base URL is configured at build time (typically `http://localhost:8031` in development, proxied via Nginx in production).

### Pipeline trigger calls

```
POST /run/full
POST /run/pre
POST /run/pipeline
POST /run/post
```

All return `{job_id, job_type, status: "pending"}` immediately. The frontend polls the job status endpoint.

### Job polling

```
GET /jobs                    → list all jobs
GET /jobs/{job_id}           → single job with stdout + stderr
POST /jobs/{job_id}/stop     → cancel
DELETE /jobs/{job_id}        → remove from history
```

The frontend polls `GET /jobs/{job_id}` every few seconds to stream job output to the UI. `status` transitions: `pending → running → done | failed | stopped`.

### State table

```
GET /app/state-table          → {status, rows}
GET /app/state-table?refresh  → triggers Zoho rebuild, returns {status: "loading"}
```

The state table powers the main dashboard showing which state/district/crop combinations have been processed, audited, and downloaded.

### File operations

```
GET  /files/tree
GET  /files/download/{path}
POST /files/upload
POST /files/upload-chunk
POST /files/rename/{path}
DELETE /files/{path}
DELETE /folders/{path}
POST /files/folders
POST /files/upload-audited
```

The file browser component uses `/files/tree` to render the Zoho WorkDrive tree and the download/delete/rename/upload endpoints for file management.

### Output download

```
GET /app/output/{state}/{district}/{crop}
```

Downloads the final FAQ CSV for a state/district/crop. Marks `downloaded=True` in master.json.

### Next-state helper

```
GET /app/next-state?state=&district=&domains[]=
```

Returns `{name, state, is_new, existing_crops}` — used to pre-fill the output folder name in the "Run Pipeline" form before submission.

---

## Main UI Views

### Dashboard / State Table
Shows `master.json` rows as a table with columns: state, district, crop, processed, audited, downloaded. Allows downloading output CSVs and uploading audit files.

### Run Pipeline Form
Form for triggering `/run/full`. Fields map to `FullRequest`:
- State, district, domains (multi-select from a known list)
- Crops (multi-select or typed)
- Model, grid mode, GPU ID
- Flags: skip pre/post pipeline, skip QA gen

The "Next State" button calls `/app/next-state` to auto-fill the output folder name.

### Job Monitor
Live log view. Shows all jobs from `/jobs`, with color-coded status and streaming stdout. Stop button calls `/jobs/{id}/stop`.

### File Browser
Tree view of Zoho WorkDrive via `/files/tree`. Supports upload (chunked for large files), download, rename, move, and delete. The "upload-audited" flow uses `/files/upload-audited` which automatically names the file `audit_<original>` and updates master.json.

---

## Nginx Configuration

Nginx serves the SPA from `/usr/share/nginx/html` and proxies API calls:

```nginx
location /api/ {
    proxy_pass http://localhost:8031/;
}
```

All other routes return `index.html` (SPA client-side routing).

---

## Development

The frontend source is in a separate repository / Docker image (`vicharanashala/faqcluster-frontend`). To run against a local pipeline server:

```bash
# In the frontend repo
VITE_API_BASE=http://localhost:8031 npm run dev
```

For local development of the full stack:
```bash
# Start pipeline server
cd pipeline_server
uvicorn pipeline_server:app --host 0.0.0.0 --port 8031 --reload

# Start frontend dev server (separate terminal, frontend repo)
npm run dev
# → http://localhost:5173
```
