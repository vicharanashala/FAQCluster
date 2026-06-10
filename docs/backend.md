# Pipeline Server

The pipeline server is a **FastAPI** application that orchestrates all pipeline runs, manages files, and exposes the REST API consumed by the React frontend. All pipeline logic (pre, main, post) lives inside `pipeline_server/`.

- **Port:** 8031 (host network inside Docker)
- **Entry point:** `uvicorn pipeline_server:app --host 0.0.0.0 --port 8031`
- **Source:** `pipeline_server/pipeline_server.py` (~1600 lines)

---

## Startup

At startup (FastAPI `lifespan`), a daemon thread calls `_load_or_build_master()`:
1. Tries to download and parse `master.json` from the Zoho WorkDrive root folder.
2. If `master.json` doesn't exist or is corrupt, walks the entire `outputs/repair/` tree in Zoho and rebuilds it.
3. Sets the `_master_ready` event so state-table requests don't block indefinitely.

`_master_ready.is_set()` gates the `/app/state-table` response — while building, the endpoint returns `{"status": "loading", "rows": []}`.

---

## CORS

All origins, methods, and headers are allowed (`allow_origins=["*"]`). This is intentional — the frontend can be served from any machine on the VPN.

---

## Error handling

`RetryError` and `ConnectionError` from the `requests` library are caught globally and returned as HTTP 503 with a JSON body: `{"detail": "Zoho WorkDrive unavailable: ..."}`. All other unhandled exceptions return FastAPI's default 500 JSON.

---

## Path Safety

Two helpers prevent directory traversal in all file routes:

```python
_resolve_any_safe(base: Path, user_str: str) -> Path
_resolve_safe(user_str: str) -> Path  # wraps _resolve_any_safe with APP_DATA as base
```

Both reject:
- Absolute paths (any string starting with `/`)
- `..` in any path component
- Null bytes (`\x00`)
- Paths that resolve outside the `APP_DATA` sandbox (symlink escapes)

All Pydantic request models with path fields call `_resolve_safe` in a `@field_validator`.

---

## Job System

### Store
`jobs: dict[str, dict]` is an in-memory dict keyed by UUID job ID. Each entry:

```json
{
  "job_id": "uuid",
  "job_type": "pre | pipeline | post | full",
  "job_type_id": 1,
  "status": "pending | running | done | failed | stopped",
  "stdout": "...",
  "stderr": "...",
  "created_at": "2025-06-01T10:00:00Z"
}
```

`JOB_TYPE_IDS = {"pre": 1, "pipeline": 2, "post": 3, "full": 4}` — used by the frontend to render job type labels.

### Thread-local routing
`sys.stdout` is replaced with `_JobStdout`. On every `write()`, it checks the thread-local `_tl.job_id` and appends output to `jobs[job_id]["stdout"]`. This captures all `print()` calls from pipeline code running in executor threads.

### Submission
`_submit(fn, background, job_type)` creates a UUID job entry, registers a cancel event in `_job_ctl`, and dispatches `_run_job(job_id, fn)` as a FastAPI background task.

### Lifecycle (`_run_job`)
1. Sets `status = "running"`, wraps `fn` with thread-local `job_id` binding.
2. Runs `fn` via `loop.run_in_executor(EXECUTOR)`.
3. On normal completion: `status = "done"`.
4. On `JobCancelled` or if `is_cancelled(job_id)`: `status = "stopped"`.
5. On `SystemExit(0)`: `status = "done"`.
6. On `SystemExit(non-zero)`, `CalledProcessError`, or `Exception`: `status = "failed"`, traceback in `stderr`.
7. Always calls `_job_ctl.cleanup(job_id)` in `finally`.

### Cancellation
`_job_ctl.cancel(job_id)` sets the per-job `threading.Event`. Cancellable code calls `_job_ctl.check_cancel()`, which raises `JobCancelled(BaseException)` — a `BaseException` so it bypasses `except Exception` handlers in pipeline code. For subprocesses, `_job_ctl.cancel` also kills the entire process group via `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)`.

---

## Zoho Sync Helpers

All helpers work against the `ZohoWorkDrive` singleton (`_get_zoho()`, lazily initialized on first use with double-checked locking).

| Helper | Signature | Behaviour |
|--------|-----------|-----------|
| `_zoho_sync_down` | `(rel_path: str) → bool` | Downloads one file from Zoho to `APP_DATA/<rel_path>`. Returns False if not found. |
| `_zoho_sync_up` | `(local_path: Path) → Optional[str]` | Uploads one local file at its relative path under `APP_DATA` to Zoho. Returns file ID. |
| `_zoho_sync_up_dir` | `(local_dir: Path)` | Recursively uploads all files in a local directory. |
| `_stream_zoho_to_tmp` | `(zoho_path: str) → ContextManager[Path]` | Streams a Zoho file to a `NamedTemporaryFile` in `/tmp`. Context manager — auto-deletes on exit. Used for `cleaned_data.csv` (can be several GB). |
| `_zoho_walk_down` | `(zoho_path: str, local_base: Path)` | Recursively downloads all files from a Zoho folder tree to `local_base`, skipping already-present files. |

---

## master.json

A persistent JSON file at the Zoho root. The server loads it once at startup and updates it incrementally after each pipeline run. The in-memory copy is `_master_data: dict`.

### Shape
```json
{
  "built_at": "<iso8601>",
  "data": {
    "<state_slug>": {
      "<district_folder>": {
        "<crop_slug>": {
          "output_file": "outputs/repair/…/<district>_<crop>.csv",
          "audit_file":  "outputs/repair/…/audit_….csv",
          "downloaded":  false,
          "audited":     false,
          "processed":   true,
          "finished_at": "2025-06-01T10:05:00Z"
        }
      }
    }
  }
}
```

### Operations
| Function | Description |
|----------|-------------|
| `_load_or_build_master()` | Startup: load or rebuild. Runs in daemon thread. |
| `_rebuild_master()` | Full Zoho walk → in-memory update → upload. |
| `_update_master_crop(state, district, crop, **updates)` | Single-entry update + upload. Sets `finished_at` on first `processed=True`. |
| `_master_to_rows()` | Flattens `_master_data` to a list of row dicts for the API. |
| `_master_remove_folder(path)` | Removes entries for a deleted crop/district/state folder + uploads. |
| `_master_clear_file(path)` | Clears `output_file` or `audit_file` when a file is deleted. |
| `_upload_master_json()` | Serializes `_master_data` and uploads. Caller must hold `_master_lock`. |

---

## Request Models

### `PreRequest`
```
state: str           ← required
district: str        ← optional
crops: List[str]     ← optional (crops or domains required)
domains: List[str]   ← optional
output: str          ← relative path for pre-pipeline output CSV
keep_intermediate: bool = True
```

### `PipelineRequest`
```
state: str           ← optional (used for slug)
input: str = "cleaned_data.csv"
crops: List[str]     ← optional
domains: List[str]   ← optional
output_dir: str = "outputs/repair"
model: str           ← defaults to LLM_MODEL env var
api_key: str         ← for Claude Batch API (Stage 4)
gpu_id: int = 1
batch_size: int = 8
grid_mode: str = "quick"
skip_phase1/2/repair/unique_q/corpus_filter/qa_gen: bool
```

### `PostRequest`
```
input: str = "outputs/repair"
crops: List[str]     ← optional filter
skip_dedup: bool = False
```

### `FullRequest`
```
state: str           ← required
district: str        ← optional
crops: List[str]     ← optional
crops_file: str      ← path to crops.yaml (for corpus filter)
domains: List[str]   ← optional
output_dir: str = "outputs/repair"
pre_output: str      ← optional explicit path for pre-pipeline output
model: str
api_key: str
gpu_id: int = 1
batch_size: int = 8
grid_mode: str = "quick"
skip_pre_pipeline / skip_qa_gen / skip_post_pipeline: bool
```

---

## Routes Reference

### Health / Root

```
GET  /
```
Returns `{"status": "ok"}`. Used by Docker healthcheck.

---

### Pipeline Run Routes

All four accept `BackgroundTasks` — the job starts asynchronously and the response returns immediately with the job ID.

```
POST /run/pre
```
Body: `PreRequest`. Runs pre-pipeline (state filter + crop normalization). Streams `cleaned_data.csv` from Zoho, writes output to Zoho. Returns `{job_id, job_type, status}`.

```
POST /run/pipeline
```
Body: `PipelineRequest`. Runs the 7-stage pipeline for one or more crops. Launches `run_pipeline.py` as a subprocess per crop. Returns `{job_id, ...}`.

```
POST /run/post
```
Body: `PostRequest`. Runs post-pipeline dedup and review generation. Returns `{job_id, ...}`.

```
POST /run/full
```
Body: `FullRequest`. Runs pre → per-crop pipeline → post in sequence. Most complex route. Handles pre-pipeline caching (reuses if same state/district/domains, versions if changed), crop discovery, and Zoho upload at each stage. Returns `{job_id, ...}`.

---

### File Routes

```
GET  /files/tree
```
Returns a nested JSON tree of all files in the Zoho WorkDrive, built by `walk_folder` on the root. Used by the frontend file browser.

```
GET  /files/download/{path}
```
Streams file content from Zoho as `application/octet-stream`.

```
POST /files/upload
```
Form data: `file` (multipart), `dest` (optional folder path). Uploads to Zoho. Returns `{uploaded: path, id: zoho_file_id}`.

```
POST /files/upload-chunk
```
Query params: `upload_id`, `chunk_index`, `total_chunks`, `filename`, `dest`. Reassembles chunks in `_CHUNK_TMP/<upload_id>/` and uploads the complete file to Zoho once all chunks arrive.

```
POST /files/rename/{path}
```
Body: `{"to": "<new_path>"}`. Renames and/or moves a file in Zoho. Returns 409 if destination already exists.

```
DELETE /files/{path}
```
Deletes a file from Zoho and updates `master.json`. Also deletes the linked `dedup_` or `phase_` prefixed counterpart if present.

```
DELETE /folders/{path}
```
Deletes a folder from Zoho and removes its `master.json` entries.

```
POST /files/folders
```
Body: `{"path": "<relative_path>"}`. Creates a folder (recursively) in Zoho. Returns `{created, id}`.

```
POST /files/upload-audited
```
Form data: `state`, `district`, `crop`, `file`. Uploads as `audit_<filename>` to `outputs/repair/<state>/<district>/<crop>/` and marks `audited=True` in master.json.

---

### App Utility Routes

```
GET  /app/next-state?state=&district=&domains[]=
```
Checks Zoho for existing versioned district folders (e.g. `dist_0`, `dist_1`) and finds the one matching `district + domains`. Returns `{name, state, is_new, existing_crops}`. Used by the frontend to pre-fill the "output" field before starting a new run.

```
GET  /app/state-table?refresh=false
```
Returns `{status: "loading"|"ready", rows: [...]}`. Each row: `{state, district, crop, output_file, audit_file, downloaded, audited, processed, finished_at}`. Pass `?refresh=true` to trigger an async Zoho rebuild.

```
GET  /app/output/{state}/{district}/{crop}
```
Streams the final deduped FAQ CSV for a crop as `text/csv`. Marks `downloaded=True` in master.json.

---

### Job Routes

```
GET  /jobs
```
Returns array of all job objects.

```
GET  /jobs/{job_id}
```
Returns a single job object. 404 if not found.

```
POST /jobs/{job_id}/stop
```
Cancels a running job. Returns 409 if job is not running. Sets status to `stopped`.

```
DELETE /jobs/{job_id}
```
Removes a job from the in-memory store (does not stop it).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_URL` | `http://100.100.108.44:8013/v1/chat/completions` | Remote inference endpoint |
| `LLM_MODEL` | `google/gemma-4-26B-A4B-it` | Model used in `PipelineRequest.model` default |
| `LLM_API_KEY` | `""` | Bearer token for LLM API |
| `LLM_THINKING_ENABLED` | `false` | Gemma thinking tokens |
| `LLM_GROUPING_STRICTNESS` | `strict` | Stage 4 grouping strictness (`strict` or `loose`) |
| `ZOHO_CLIENT_ID` | — | Zoho OAuth2 client ID |
| `ZOHO_CLIENT_SECRET` | — | Zoho OAuth2 client secret |
| `ZOHO_REFRESH_TOKEN` | — | Zoho OAuth2 refresh token |
| `ZOHO_ROOT_FOLDER_ID` | — | Zoho folder ID (from WorkDrive link) |
