"""
FAQCluster Pipeline Server — consolidated FastAPI app.
Runs inside the pipeline container on port 8031.

Routes:
  GET  /                          health check
  GET  /app/tree                  FAQ file tree

  POST /run/pre                   run pre-pipeline
  POST /run/pipeline              run pipeline for one or more crops
  POST /run/post                  run post-pipeline
  POST /run/full                  run full pipeline (pre → pipeline → post)

  GET  /files/tree                filtered view of app-data/
  GET  /files/download/{path}     stream file
  POST /files/upload              upload single file (≤ chunk size)
  POST /files/upload-chunk        chunked upload
  POST /files/rename/{path}       rename/move
  DELETE /files/{path}            delete file
  DELETE /folders/{path}          delete folder
  POST /files/folders             create directory
  POST /files/upload-audited      upload audited CSV

  GET  /app/next-state                        get next versioned district folder name
  GET  /app/state-table                       table of all state/district/crop outputs
  GET  /app/output/{state}/{district}/{crop}  download output CSV

  GET  /jobs                      list all jobs
  GET  /jobs/{job_id}             get job details
  POST /jobs/{job_id}/stop        stop a running job
  DELETE /jobs/{job_id}           remove job from history
"""

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator, model_validator

import _job_ctl

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

APP_DATA = SCRIPT_DIR / "app-data"
APP_DATA.mkdir(exist_ok=True)  # local scratch only — persistent data lives in Zoho WorkDrive

# ── Zoho WorkDrive integration ────────────────────────────────────────────────

_zoho_instance = None
_zoho_init_lock = threading.Lock()


def _get_zoho():
    global _zoho_instance
    if _zoho_instance is None:
        with _zoho_init_lock:
            if _zoho_instance is None:
                from helpers.zoho_workdrive import ZohoWorkDrive
                _zoho_instance = ZohoWorkDrive()
    return _zoho_instance


def _zoho_sync_down(rel_path: str) -> bool:
    """Download a file from Zoho to APP_DATA/<rel_path>. Returns True if found."""
    try:
        zwd = _get_zoho()
        result = zwd.resolve_path(rel_path)
        if result is None:
            return False
        file_id, ftype = result
        if ftype == "folder":
            return False
        local = APP_DATA / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(zwd.download_file(file_id))
        print(f"[ZOHO] Downloaded {rel_path}")
        return True
    except Exception as e:
        print(f"[ZOHO] sync_down failed for {rel_path}: {e}")
        return False


def _zoho_sync_up(local_path: Path) -> Optional[str]:
    """Upload a local file to Zoho at the same relative path under root. Returns file ID."""
    if not local_path.exists():
        return None
    try:
        rel = local_path.relative_to(APP_DATA)
        parts = rel.parts
        zoho_folder = "/".join(parts[:-1]) if len(parts) > 1 else ""
        zwd = _get_zoho()
        parent_id = zwd.ensure_path(zoho_folder) if zoho_folder else zwd.root_folder_id
        fid = zwd.upload_file(local_path.name, local_path.read_bytes(), parent_id)
        print(f"[ZOHO] Uploaded {rel}")
        return fid
    except Exception as e:
        print(f"[ZOHO] sync_up failed for {local_path}: {e}")
        return None


def _zoho_sync_up_dir(local_dir: Path) -> None:
    """Upload all files in a local directory tree to Zoho."""
    if not local_dir.exists():
        return
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            _zoho_sync_up(f)


@contextmanager
def _stream_zoho_to_tmp(zoho_path: str) -> Iterator[Path]:
    """Stream a Zoho file into a NamedTemporaryFile and yield its path. Auto-deleted on exit."""
    zwd = _get_zoho()
    result = zwd.resolve_path(zoho_path)
    if result is None:
        raise FileNotFoundError(f"{zoho_path!r} not found in Zoho WorkDrive")
    file_id = result[0]
    resp = zwd.download_file_stream(file_id)
    with tempfile.NamedTemporaryFile(suffix=".csv", dir="/tmp", delete=True) as tmp:
        try:
            for chunk in resp.iter_content(chunk_size=4 << 20):
                tmp.write(chunk)
            tmp.flush()
        finally:
            resp.close()
        print(f"[ZOHO] Streamed {zoho_path} → /tmp ({Path(tmp.name).stat().st_size >> 20} MB)")
        yield Path(tmp.name)
    # NamedTemporaryFile auto-deletes here


def _zoho_walk_down(zoho_path: str, local_base: Path) -> None:
    """Recursively download all files under a Zoho path into local_base."""
    try:
        zwd = _get_zoho()
        result = zwd.resolve_path(zoho_path)
        if result is None:
            return
        folder_id, ftype = result
        if ftype != "folder":
            return
        for path, item in zwd.walk_folder(folder_id, prefix=zoho_path):
            if item["type"] == "folder":
                continue
            local_dest = local_base / path
            if local_dest.exists():
                continue
            try:
                content = zwd.download_file(item["id"])
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                local_dest.write_bytes(content)
            except Exception as e:
                print(f"[ZOHO] walk_down failed for {path}: {e}")
    except Exception as e:
        print(f"[ZOHO] walk_down failed for {zoho_path}: {e}")

_CHUNK_TMP = Path(tempfile.gettempdir()) / "faq_chunks"
_RUN_PIPELINE = SCRIPT_DIR / "run_pipeline.py"

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_or_build_master, daemon=True, name="master-json-init").start()
    yield

app = FastAPI(title="FAQCluster Pipeline Service", redirect_slashes=False, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Return 503 instead of crashing when Zoho WorkDrive is unreachable or rate-limits us
import requests as _requests

@app.exception_handler(_requests.exceptions.RetryError)
@app.exception_handler(_requests.exceptions.ConnectionError)
async def zoho_unavailable_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Zoho WorkDrive unavailable: {exc}"},
    )

# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _resolve_any_safe(base: Path, user_str: str) -> Path:
    """Resolve user_str relative to base; reject absolute paths, .., null bytes, symlink escapes."""
    p = Path(user_str)
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {user_str!r}")
    for part in p.parts:
        if part == "..":
            raise ValueError(f"path traversal not allowed: {user_str!r}")
        if "\x00" in part:
            raise ValueError(f"null bytes not allowed in path: {user_str!r}")
    resolved = (base / p).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes sandbox: {user_str!r}")
    return resolved


def _resolve_safe(user_str: str) -> Path:
    """Resolve a user-supplied relative path inside APP_DATA."""
    return _resolve_any_safe(APP_DATA, user_str)


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

EXECUTOR = ThreadPoolExecutor()

jobs: dict[str, dict] = {}

JOB_TYPE_IDS: dict[str, int] = {
    "pre": 1,
    "pipeline": 2,
    "post": 3,
    "full": 4,
}

_tl = threading.local()


class _JobStdout:
    """Routes print() output to the current job's stdout buffer, thread-safely."""
    def __init__(self, original):
        self._orig = original

    def write(self, data):
        job_id = getattr(_tl, "job_id", None)
        if job_id and job_id in jobs:
            jobs[job_id]["stdout"] += data
        self._orig.write(data)

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


sys.stdout = _JobStdout(sys.stdout)


async def _run_job(job_id: str, fn: Callable[[], None]) -> None:
    jobs[job_id]["status"] = "running"
    loop = asyncio.get_running_loop()

    def _wrapped():
        _tl.job_id = job_id
        _job_ctl.set_job_id(job_id)
        try:
            fn()
        finally:
            _tl.job_id = None
            _job_ctl.set_job_id(None)
            _job_ctl.deregister_proc()

    try:
        await loop.run_in_executor(EXECUTOR, _wrapped)
        if _job_ctl.is_cancelled(job_id):
            jobs[job_id]["status"] = "stopped"
            jobs[job_id]["stderr"] = "stopped by user"
        else:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["stderr"] = ""
    except _job_ctl.JobCancelled:
        jobs[job_id]["status"] = "stopped"
        jobs[job_id]["stderr"] = "stopped by user"
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        if _job_ctl.is_cancelled(job_id):
            jobs[job_id]["status"] = "stopped"
            jobs[job_id]["stderr"] = "stopped by user"
        else:
            jobs[job_id]["status"] = "done" if code == 0 else "failed"
            jobs[job_id]["stderr"] = f"SystemExit({code})"
    except subprocess.CalledProcessError as exc:
        if _job_ctl.is_cancelled(job_id):
            jobs[job_id]["status"] = "stopped"
            jobs[job_id]["stderr"] = "stopped by user"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["stderr"] = (
                f"CalledProcessError (returncode={exc.returncode}): "
                f"{exc.cmd!r}\n{traceback.format_exc()}"
            )
    except Exception:
        if _job_ctl.is_cancelled(job_id):
            jobs[job_id]["status"] = "stopped"
            jobs[job_id]["stderr"] = "stopped by user"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["stderr"] = traceback.format_exc()
    finally:
        _job_ctl.cleanup(job_id)


def _submit(fn: Callable[[], None], background: BackgroundTasks, job_type: str) -> dict:
    job_id = str(uuid.uuid4())
    _job_ctl.make_event(job_id)
    jobs[job_id] = {
        "job_id": job_id,
        "job_type": job_type,
        "job_type_id": JOB_TYPE_IDS[job_type],
        "status": "pending",
        "stdout": "",
        "stderr": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    background.add_task(_run_job, job_id, fn)
    return {
        "job_id": job_id,
        "job_type": job_type,
        "job_type_id": JOB_TYPE_IDS[job_type],
        "status": "pending",
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class PreRequest(BaseModel):
    state: str
    district: Optional[str] = None
    crops: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    output: str
    keep_intermediate: bool = True

    @field_validator("output")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        _resolve_safe(v)
        return v

    @model_validator(mode="after")
    def require_crops_or_domains(self) -> "PreRequest":
        if not self.crops and not self.domains:
            raise ValueError("at least one of 'crops' or 'domains' must be provided")
        return self


class PipelineRequest(BaseModel):
    state: Optional[str] = None
    input: str = "cleaned_data.csv"
    crops: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    output_dir: str = "outputs/repair"
    model: str = os.environ.get("LLM_MODEL", "google/gemma-4-26B-A4B-it")
    api_key: Optional[str] = None
    gpu_id: int = 1
    batch_size: int = 8
    grid_mode: str = "quick"
    skip_phase1: bool = False
    skip_phase2: bool = False
    skip_repair: bool = False
    skip_unique_q: bool = False
    skip_corpus_filter: bool = False
    skip_qa_gen: bool = False

    @field_validator("output_dir", "input")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        _resolve_safe(v)
        return v


class PostRequest(BaseModel):
    input: str = "outputs/repair"
    crops: Optional[List[str]] = None
    skip_dedup: bool = False

    @field_validator("input")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        _resolve_safe(v)
        return v


class FullRequest(BaseModel):
    state: str
    district: Optional[str] = None
    crops: Optional[List[str]] = None
    crops_file: Optional[str] = None
    domains: Optional[List[str]] = None
    output_dir: str = "outputs/repair"
    pre_output: Optional[str] = None
    model: str = os.environ.get("LLM_MODEL", "google/gemma-4-26B-A4B-it")
    api_key: Optional[str] = None
    gpu_id: int = 1
    batch_size: int = 8
    grid_mode: str = "quick"
    skip_pre_pipeline: bool = False
    skip_qa_gen: bool = False
    skip_post_pipeline: bool = False

    @field_validator("output_dir", "crops_file", "pre_output")
    @classmethod
    def _safe_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            _resolve_safe(v)
        return v


class RenameRequest(BaseModel):
    to: str

    @field_validator("to")
    @classmethod
    def _safe_to(cls, v: str) -> str:
        _resolve_safe(v)
        return v


# ---------------------------------------------------------------------------
# Pipeline sync functions
# ---------------------------------------------------------------------------

def _run_pre_sync(r: PreRequest) -> None:
    from run_pre_pipeline import run_state_filter, run_crop_normalizer

    output_path = _resolve_safe(r.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output_path.parent / f"{output_path.stem}_state_rows.csv"

    print("[ZOHO] Streaming cleaned_data.csv from WorkDrive...")
    with _stream_zoho_to_tmp("cleaned_data.csv") as input_path:
        run_state_filter(input_path, r.state, intermediate, domains=r.domains or [], district=r.district)
        _job_ctl.check_cancel()
        if r.crops:
            run_crop_normalizer(intermediate, output_path, r.crops)
        else:
            shutil.copy2(intermediate, output_path)

    if intermediate.exists() and not r.keep_intermediate:
        intermediate.unlink()

    _zoho_sync_up(output_path)


def _run_pipeline_sync(r: PipelineRequest) -> None:
    import pandas as pd

    # Ensure input CSV is present locally (download from Zoho if needed)
    local_input = APP_DATA / r.input
    if not local_input.exists():
        print(f"[ZOHO] {r.input} not local — downloading from WorkDrive...")
        if not _zoho_sync_down(r.input):
            raise FileNotFoundError(f"Input file not found locally or in Zoho: {r.input}")

    resolved_raw  = str(APP_DATA / r.input)
    input_parts   = Path(r.input).parts
    district_folder = Path(r.input).stem  # e.g. "bengaluru_0" from "karnataka/bengaluru_0/bengaluru_0.csv"

    # Derive state slug: explicit field > first component of a 3-part path > fallback
    if r.state:
        state_slug = slug(r.state)
    elif len(input_parts) >= 3:
        state_slug = input_parts[0]
    else:
        state_slug = district_folder  # backward-compat for flat paths

    out_base     = _resolve_safe(r.output_dir) / state_slug / district_folder
    failed       = []

    if r.crops or r.domains:
        _df = pd.read_csv(resolved_raw, low_memory=False)
        if r.crops:
            _df = _df[_df["Crop"].dropna().str.strip().isin(r.crops)]
        if r.domains:
            _df = _df[_df["QueryType"].dropna().str.strip().isin(r.domains)]
        crops = _df["Crop"].dropna().str.strip().unique().tolist()
        print(f"[INFO] Filtered to {len(crops)} unique crop(s): {', '.join(crops)}")
    else:
        print("[INFO] No crops/domains provided — discovering unique crops from input CSV...")
        _df = pd.read_csv(resolved_raw, low_memory=False)
        crops = _df["Crop"].dropna().str.strip().unique().tolist()
        print(f"[INFO] Found {len(crops)} unique crop(s): {', '.join(crops)}")

    # Pass repair/state_slug as --output-dir so run_pipeline.py puts outputs at
    # repair/state_slug/district_folder/crop_slug (run_pipeline appends stem + crop_slug)
    out_dir_str = str(_resolve_safe(r.output_dir) / state_slug)

    for crop in crops:
        _job_ctl.check_cancel()
        crop_out = out_base / slug(crop)
        crop_out.mkdir(parents=True, exist_ok=True)

        # Pull any existing intermediates from Zoho so stage-skipping survives restarts
        _zoho_walk_down(f"outputs/repair/{state_slug}/{district_folder}/{slug(crop)}", APP_DATA)

        print(f"[INFO] Starting pipeline for '{crop}'...")

        cmd = [
            sys.executable, "-u", str(_RUN_PIPELINE),
            "--raw-file",   resolved_raw,
            "--crop",       crop,
            "--model",      r.model,
            "--gpu-id",     str(r.gpu_id),
            "--batch-size", str(r.batch_size),
            "--grid-mode",  r.grid_mode,
            "--output-dir", out_dir_str,
        ]
        if r.api_key:          cmd += ["--api-key", r.api_key]
        if r.skip_phase1:      cmd += ["--skip-phase1"]
        if r.skip_phase2:      cmd += ["--skip-phase2"]
        if r.skip_repair:      cmd += ["--skip-repair"]
        if r.skip_unique_q:    cmd += ["--skip-unique-q"]
        if r.skip_corpus_filter: cmd += ["--skip-corpus-filter"]
        if r.skip_qa_gen:      cmd += ["--skip-qa-gen"]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, preexec_fn=os.setsid, bufsize=1,
        )
        _job_ctl.register_proc(proc)
        for line in proc.stdout:
            print(line, end="", flush=True)
            if _job_ctl.is_cancelled(_job_ctl.current_job_id()):
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                break
        proc.wait()
        _job_ctl.deregister_proc()
        rc = proc.returncode

        _job_ctl.check_cancel()

        if rc != 0:
            print(f"[WARN] Crop '{crop}' failed (returncode={rc})")
            failed.append(crop)
        else:
            # Upload any outputs not already pushed by run_pipeline.py per-stage
            _zoho_sync_up_dir(crop_out)
            shutil.rmtree(crop_out, ignore_errors=True)
            print(f"[INFO] Cleaned up local folder: {crop_out.relative_to(APP_DATA)}")

    if failed:
        raise RuntimeError(f"The following crops failed: {', '.join(failed)}")


def _run_post_sync(r: PostRequest) -> None:
    from run_post_pipeline import run_dedup as post_run_dedup

    input_dir = _resolve_safe(r.input)

    # Download the repair folder from Zoho if not present locally
    if not input_dir.exists():
        print(f"[ZOHO] {r.input} not local — downloading from WorkDrive...")
        _zoho_walk_down(r.input, APP_DATA)

    if not input_dir.exists():
        raise FileNotFoundError(f"input folder not found locally or in Zoho: {input_dir}")

    if not r.skip_dedup:
        post_run_dedup(input_dir, r.crops or None)

    print("[ZOHO] Uploading post-pipeline outputs to WorkDrive...")
    _zoho_sync_up_dir(input_dir)


def _next_versioned_path(current_path: Path, zwd=None) -> Path:
    """Return the next non-existent versioned path, checking Zoho (authoritative) if zwd given."""
    m = re.match(r"^(.+)_(\d+)$", current_path.stem)
    base, ver = (m.group(1), int(m.group(2))) if m else (current_path.stem, 0)
    next_ver = ver + 1
    while True:
        candidate = current_path.parent / f"{base}_{next_ver}{current_path.suffix}"
        if zwd is not None:
            try:
                rel = str(candidate.relative_to(APP_DATA))
                exists = zwd.resolve_path(rel) is not None
            except ValueError:
                exists = False
        else:
            exists = candidate.exists()
        if not exists:
            return candidate
        next_ver += 1


def _run_full_sync(r: FullRequest) -> None:
    import pandas as pd
    from run_pre_pipeline import run_state_filter, run_crop_normalizer
    from run_post_pipeline import run_dedup as post_run_dedup

    if r.crops:
        crops = r.crops
    elif r.crops_file:
        crops_file = _resolve_safe(r.crops_file)
        if not crops_file.exists():
            _zoho_sync_down(r.crops_file)
        if not crops_file.exists():
            raise FileNotFoundError(f"crops file not found locally or in Zoho: {crops_file}")
        lines = crops_file.read_text().splitlines()
        crops = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        if not crops:
            raise ValueError("no crops found in crops_file")
    else:
        crops = None

    _norm_is_temp = False

    state_slug = slug(r.state)

    if r.skip_pre_pipeline:
        # Needed for the full duration (passed to per-crop subprocesses as --raw-file)
        print("[ZOHO] Streaming cleaned_data.csv from WorkDrive...")
        _cleaned_ctx = _stream_zoho_to_tmp("cleaned_data.csv")
        _cleaned_tmp = _cleaned_ctx.__enter__()
        effective_raw = str(_cleaned_tmp)
        if crops is None:
            _df = pd.read_csv(effective_raw, low_memory=False)
            if r.domains:
                _df = _df[_df["QueryType"].dropna().str.strip().isin(r.domains)]
            crops = _df["Crop"].dropna().str.strip().unique().tolist()
            print(f"[INFO] Auto-discovered {len(crops)} crop(s) from input CSV")
    else:
        # ── Pre-check: read Zoho meta BEFORE streaming the 4 GB cleaned_data.csv ──
        # Zoho is authoritative — fetch state/district/meta.json first so we only
        # download cleaned_data.csv when we actually need to run the pre-pipeline.
        _zwd_pre = _get_zoho()
        existing_meta: dict = {}
        _pre_can_reuse_fully = False
        pre_rel = r.pre_output  # may be None
        file_exists = same_state = same_district = same_domains = False
        existing_crops_meta: set = set()

        if r.pre_output:
            _meta_zoho = _zwd_pre.resolve_path(str(Path(pre_rel).parent / "meta.json"))
            if _meta_zoho:
                try:
                    existing_meta = json.loads(_zwd_pre.download_file(_meta_zoho[0]))
                except Exception:
                    pass

            existing_domains    = set(existing_meta.get("domains", []))
            existing_crops_meta = set(existing_meta.get("crops", []))
            requested_domains   = set(r.domains or [])
            file_exists         = _zwd_pre.resolve_path(pre_rel) is not None and bool(existing_meta)
            same_state          = existing_meta.get("state") == r.state if existing_meta else False
            same_district       = existing_meta.get("district", "") == (r.district or "") if existing_meta else False
            same_domains        = existing_domains == requested_domains

            if file_exists and same_state and same_district and same_domains:
                missing = [c for c in (crops or []) if c not in existing_crops_meta]
                if crops is None or not missing:
                    _pre_can_reuse_fully = True
                    print(f"[INFO] Pre-pipeline output found in Zoho — skipping cleaned_data.csv download")

        if _pre_can_reuse_fully:
            # All required crops already processed — just pull the small pre-pipeline CSV
            norm_file = _resolve_safe(pre_rel)
            norm_file.parent.mkdir(parents=True, exist_ok=True)
            _norm_is_temp = False
            meta_path = norm_file.parent / "meta.json"
            if not norm_file.exists():
                _zoho_sync_down(pre_rel)
            if crops is None:
                crops = pd.read_csv(norm_file)["Crop"].dropna().str.strip().unique().tolist()
                print(f"[INFO] {len(crops)} crop(s) from cached pre-pipeline output (Zoho)")
            else:
                print(f"[INFO] Reusing pre-pipeline output: {pre_rel} (from Zoho)")
            effective_raw = str(norm_file)
        else:
            # Need to (partially) re-run pre-pipeline — stream cleaned_data.csv now
            print("[ZOHO] Streaming cleaned_data.csv from WorkDrive...")
            with _stream_zoho_to_tmp("cleaned_data.csv") as raw_file:
                if r.pre_output:
                    norm_file = _resolve_safe(r.pre_output)
                    norm_file.parent.mkdir(parents=True, exist_ok=True)
                    _norm_is_temp = False

                    meta_path = norm_file.parent / "meta.json"

                    # Reuse already-fetched metadata from the pre-check above
                    if file_exists and same_state and same_district and same_domains:
                        if crops is None:
                            print(f"[INFO] Reusing existing pre-pipeline output: {pre_rel} (from Zoho)")
                            if not norm_file.exists():
                                _zoho_sync_down(pre_rel)
                            crops = pd.read_csv(norm_file)["Crop"].dropna().str.strip().unique().tolist()
                            print(f"[INFO] {len(crops)} crop(s) from cached pre-pipeline output")
                        else:
                            missing_crops = [c for c in crops if c not in existing_crops_meta]
                            if not missing_crops:
                                print(f"[INFO] Reusing existing pre-pipeline output: {pre_rel} (from Zoho)")
                                if not norm_file.exists():
                                    _zoho_sync_down(pre_rel)
                            else:
                                print(
                                    f"[INFO] {len(crops) - len(missing_crops)} crop(s) reused; "
                                    f"running pre-pipeline for {len(missing_crops)} new crop(s): {', '.join(missing_crops)}"
                                )
                                if not norm_file.exists():
                                    _zoho_sync_down(pre_rel)
                                intermediate = norm_file.parent / f"{norm_file.stem}_state_rows.csv"
                                run_state_filter(raw_file, r.state, intermediate,
                                                 domains=r.domains or [], district=r.district)
                                fd2, tmp2 = tempfile.mkstemp(suffix="_extra.csv", dir=str(norm_file.parent))
                                os.close(fd2)
                                tmp2_path = Path(tmp2)
                                try:
                                    run_crop_normalizer(intermediate, tmp2_path, missing_crops)
                                    _df_extra = pd.read_csv(tmp2_path, low_memory=False)
                                    if "domain" not in _df_extra.columns and "QueryType" in _df_extra.columns:
                                        _df_extra.insert(12, "domain", _df_extra["QueryType"])
                                    _df_existing = pd.read_csv(norm_file, low_memory=False)
                                    pd.concat([_df_existing, _df_extra], ignore_index=True).to_csv(norm_file, index=False)
                                    print(f"[INFO] Appended {len(missing_crops)} new crop(s) to {norm_file}")
                                finally:
                                    if tmp2_path.exists():
                                        tmp2_path.unlink()
                                if intermediate.exists():
                                    intermediate.unlink()
                                meta_path.write_text(json.dumps({
                                    "state": r.state,
                                    "district": r.district or "",
                                    "domains": r.domains or [],
                                    "crops": sorted(existing_crops_meta | set(missing_crops)),
                                }))
                    else:
                        if file_exists:
                            norm_file = _next_versioned_path(norm_file, _zwd_pre)
                            norm_file.parent.mkdir(parents=True, exist_ok=True)
                            meta_path = norm_file.parent / "meta.json"
                            print(f"[INFO] Domain/state/district change — new pre-pipeline output: {norm_file}")

                        intermediate = norm_file.parent / f"{norm_file.stem}_state_rows.csv"
                        run_state_filter(raw_file, r.state, intermediate,
                                         domains=r.domains or [], district=r.district)
                        if crops is None:
                            run_crop_normalizer(intermediate, norm_file)
                            crops = pd.read_csv(norm_file)["Crop"].dropna().str.strip().unique().tolist()
                            print(f"[INFO] Auto-discovered {len(crops)} canonical crop(s) after normalization")
                        else:
                            run_crop_normalizer(intermediate, norm_file, crops)
                        if intermediate.exists():
                            intermediate.unlink()
                        meta_path.write_text(json.dumps({
                            "state": r.state,
                            "district": r.district or "",
                            "domains": r.domains or [],
                            "crops": sorted(crops) if crops else [],
                        }))
                        _df_norm = pd.read_csv(norm_file, low_memory=False)
                        if "domain" not in _df_norm.columns and "QueryType" in _df_norm.columns:
                            _df_norm.insert(12, "domain", _df_norm["QueryType"])
                            _df_norm.to_csv(norm_file, index=False)
                else:
                    fd, tmp_path = tempfile.mkstemp(suffix="_norm.csv", dir=str(APP_DATA))
                    os.close(fd)
                    norm_file = Path(tmp_path)
                    _norm_is_temp = True
                    print("[INFO] No pre-pipeline output path provided — temp file deleted after use")
                    intermediate = norm_file.parent / f"{norm_file.stem}_state_rows.csv"
                    run_state_filter(raw_file, r.state, intermediate,
                                     domains=r.domains or [], district=r.district)
                    if crops is None:
                        run_crop_normalizer(intermediate, norm_file)
                        crops = pd.read_csv(norm_file)["Crop"].dropna().str.strip().unique().tolist()
                        print(f"[INFO] Auto-discovered {len(crops)} canonical crop(s) after normalization")
                    else:
                        run_crop_normalizer(intermediate, norm_file, crops)
                    if intermediate.exists():
                        intermediate.unlink()
                    _df_norm = pd.read_csv(norm_file, low_memory=False)
                    if "domain" not in _df_norm.columns and "QueryType" in _df_norm.columns:
                        _df_norm.insert(12, "domain", _df_norm["QueryType"])
                        _df_norm.to_csv(norm_file, index=False)
            effective_raw = str(norm_file)

            # Upload pre-pipeline output to Zoho immediately after writing.
            # Use norm_file (not r.pre_output) — it may have been versioned.
            if r.pre_output and not _norm_is_temp:
                if norm_file.exists():
                    _zoho_sync_up(norm_file)
                if meta_path.exists():
                    _zoho_sync_up(meta_path)

    # Output structure: repair/state_slug/district_folder/crop_slug
    district_folder = Path(effective_raw).stem
    out_base = _resolve_safe(r.output_dir) / state_slug / district_folder

    def _output_done(crop_slug: str) -> bool:
        zoho_folder = f"outputs/repair/{state_slug}/{district_folder}/{crop_slug}"
        _zwd = _get_zoho()
        result = _zwd.resolve_path(zoho_folder)
        if result:
            children = {f["name"] for f in _zwd.list_folder(result[0])}
            if f"{district_folder}_{crop_slug}.csv" in children or "dedup_faq.csv" in children:
                return True
        return False

    crops_to_run  = [c for c in crops if not _output_done(slug(c))]
    skipped_crops = [c for c in crops if _output_done(slug(c))]
    if skipped_crops:
        print(f"[INFO] Skipping {len(skipped_crops)} already-completed crop(s): {', '.join(skipped_crops)}")
        with _master_lock:
            for _sc in skipped_crops:
                _sc_slug = slug(_sc)
                entry = _master_data.setdefault(state_slug, {}).setdefault(district_folder, {}).setdefault(_sc_slug, {})
                entry.update({
                    "output_file": f"outputs/repair/{state_slug}/{district_folder}/{_sc_slug}/{district_folder}_{_sc_slug}.csv",
                    "processed": True,
                })
            try:
                _upload_master_json()
            except Exception as e:
                print(f"[MASTER] Failed to upload master.json for skipped crops: {e}")
    print(f"[INFO] Found {len(crops_to_run)} unique crop(s): {', '.join(crops_to_run)}")

    # Pass repair/state_slug so run_pipeline.py appends district_folder/crop_slug
    out_dir_str = str(_resolve_safe(r.output_dir) / state_slug)
    failed = []

    for crop in crops_to_run:
        _job_ctl.check_cancel()
        crop_out = out_base / slug(crop)
        crop_out.mkdir(parents=True, exist_ok=True)

        # Pull any existing intermediates from Zoho so stage-skipping survives restarts
        _zoho_walk_down(f"outputs/repair/{state_slug}/{district_folder}/{slug(crop)}", APP_DATA)

        print(f"[INFO] Starting pipeline for '{crop}'...")

        cmd = [
            sys.executable, "-u", str(_RUN_PIPELINE),
            "--raw-file",   effective_raw,
            "--crop",       crop,
            "--model",      r.model,
            "--gpu-id",     str(r.gpu_id),
            "--batch-size", str(r.batch_size),
            "--grid-mode",  r.grid_mode,
            "--output-dir", out_dir_str,
        ]
        if r.api_key:    cmd += ["--api-key", r.api_key]
        if r.skip_qa_gen: cmd += ["--skip-qa-gen"]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, preexec_fn=os.setsid, bufsize=1,
        )
        _job_ctl.register_proc(proc)
        for line in proc.stdout:
            print(line, end="", flush=True)
            if _job_ctl.is_cancelled(_job_ctl.current_job_id()):
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                break
        proc.wait()
        _job_ctl.deregister_proc()
        rc = proc.returncode

        _job_ctl.check_cancel()

        if rc != 0:
            print(f"[WARN] Crop '{crop}' failed (returncode={rc})")
            failed.append(crop)
        else:
            if not r.skip_post_pipeline:
                try:
                    post_run_dedup(out_base, [crop])
                except Exception as exc:
                    print(f"[WARN] Post-pipeline for '{crop}' failed: {exc}")
            # run_pipeline.py already uploads all intermediate files per-stage.
            # Only upload the 2 files post-pipeline writes.
            crop_slug = slug(crop)
            for fname in ["phase_data_faq.csv", f"{district_folder}_{crop_slug}.csv"]:
                _zoho_sync_up(crop_out / fname)
            shutil.rmtree(crop_out, ignore_errors=True)
            print(f"[INFO] Cleaned up local folder: {crop_out.relative_to(APP_DATA)}")
            _update_master_crop(
                state_slug, district_folder, crop_slug,
                output_file=f"outputs/repair/{state_slug}/{district_folder}/{crop_slug}/{district_folder}_{crop_slug}.csv",
                processed=True,
            )

    if r.skip_pre_pipeline:
        _cleaned_ctx.__exit__(None, None, None)

    if not r.skip_pre_pipeline and _norm_is_temp and norm_file.exists():
        norm_file.unlink()
        print("[INFO] Temporary pre-pipeline file removed from disk")

    if r.pre_output and not r.skip_pre_pipeline:
        _zwd_meta = _get_zoho()
        _meta_rel = str(Path(r.pre_output).parent / "meta.json")
        cur_meta: dict = {}
        _cm_result = _zwd_meta.resolve_path(_meta_rel)
        if _cm_result:
            try:
                cur_meta = json.loads(_zwd_meta.download_file(_cm_result[0]))
            except Exception:
                pass
        newly_done = [c for c in crops_to_run if c not in failed]
        all_crops  = sorted(set(cur_meta.get("crops", [])) | set(newly_done))
        meta_path.write_text(json.dumps({
            "state": r.state,
            "district": r.district or "",
            "domains": r.domains or [],
            "crops": all_crops,
        }))
        _zoho_sync_up(meta_path)

    if not crops_to_run:
        print("[INFO] No new crops processed — skipping Zoho upload")

    # Clean up the district-level local directories — everything is safely in Zoho
    if r.pre_output and not _norm_is_temp:
        _norm_parent = _resolve_safe(r.pre_output).parent
        if _norm_parent.exists():
            shutil.rmtree(_norm_parent, ignore_errors=True)
            print(f"[INFO] Cleaned up local pre-pipeline dir: {_norm_parent.relative_to(APP_DATA)}")
    if out_base.exists():
        shutil.rmtree(out_base, ignore_errors=True)
        print(f"[INFO] Cleaned up local repair dir: {out_base.relative_to(APP_DATA)}")

    if failed:
        raise RuntimeError(f"The following crops failed: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Health + app tree
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/app/tree")
def app_tree():
    return _files_tree()


# ---------------------------------------------------------------------------
# Run routes
# ---------------------------------------------------------------------------

@app.post("/run/pre")
def run_pre(req: PreRequest, background: BackgroundTasks):
    return _submit(lambda: _run_pre_sync(req), background, "pre")


@app.post("/run/pipeline")
def run_pipeline(req: PipelineRequest, background: BackgroundTasks):
    return _submit(lambda: _run_pipeline_sync(req), background, "pipeline")


@app.post("/run/post")
def run_post(req: PostRequest, background: BackgroundTasks):
    return _submit(lambda: _run_post_sync(req), background, "post")


@app.post("/run/full")
def run_full(req: FullRequest, background: BackgroundTasks):
    return _submit(lambda: _run_full_sync(req), background, "full")


# ---------------------------------------------------------------------------
# File routes
# ---------------------------------------------------------------------------

def _files_tree():
    """Walk Zoho WorkDrive root and return the same structure the frontend expects."""
    zwd = _get_zoho()

    all_csvs: list[dict] = []
    crop_qa_files: list[dict] = []
    final_csvs: list[dict] = []

    for path, item in zwd.walk_folder(zwd.root_folder_id):
        if item["type"] == "folder":
            continue

        parts = path.split("/")
        in_outputs = parts[0] == "outputs"
        in_repair  = in_outputs and len(parts) > 1 and parts[1] == "repair"
        in_final   = in_outputs and len(parts) > 1 and parts[1] == "final"

        entry = {"name": item["name"], "path": path, "size": item["size"], "id": item["id"]}

        if not in_outputs and not in_final and path.endswith(".csv"):
            all_csvs.append(entry)
            continue

        # outputs/repair/<state>/<district>/<crop>/unique_questions_freq_qa.csv
        if in_repair and len(parts) == 6 and item["name"] == "unique_questions_freq_qa.csv":
            state, district, crop = parts[2], parts[3], parts[4]
            all_csvs_entry = dict(entry)
            all_csvs_entry.update({
                "crop": crop,
                "district": district,
                "state": state,
                "displayName": crop,
            })
            crop_qa_files.append(all_csvs_entry)
            continue

        # outputs/repair/<state>/<district>/<crop>/{dedup_,phase_}*.csv
        if (in_repair and len(parts) == 6 and
                (item["name"].startswith("dedup_") or item["name"].startswith("phase_"))):
            state, district, crop = parts[2], parts[3], parts[4]
            folder_entry = dict(entry)
            folder_entry.update({
                "state": state,
                "district": district,
                "crop": crop,
                "folderPath": "/".join(parts[:5]),
            })
            final_csvs.append(folder_entry)

    all_csvs.sort(key=lambda e: e["path"])
    crop_qa_files.sort(key=lambda e: e["path"])
    final_csvs.sort(key=lambda e: e["path"])

    return {
        "all_csvs": all_csvs,
        "crop_qa_files": crop_qa_files,
        "final_csvs": final_csvs,
    }


@app.get("/files/tree")
def get_files_tree():
    return _files_tree()


@app.get("/files/download/{path:path}")
def download_file(path: str):
    # Validate path safety (no traversal, no absolute)
    _resolve_safe(path)
    zwd = _get_zoho()
    result = zwd.resolve_path(path)
    if result is None:
        raise HTTPException(status_code=404, detail="file not found")
    file_id, ftype = result
    if ftype == "folder":
        raise HTTPException(status_code=400, detail="path is a directory")
    filename = Path(path).name

    def _stream():
        resp = zwd.download_file_stream(file_id)
        for chunk in resp.iter_content(chunk_size=1 << 20):
            yield chunk
        resp.close()

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _master_remove_folder(path: str) -> None:
    """Remove entries from _master_data for an outputs/repair path and re-upload master.json.

    Handles crop, district, or state level deletions.
    Caller must NOT hold _master_lock.
    """
    parts = [p for p in path.split("/") if p]
    # Must be under outputs/repair/
    if len(parts) < 3 or parts[0] != "outputs" or parts[1] != "repair":
        return
    with _master_lock:
        if len(parts) == 5:  # outputs/repair/<state>/<district>/<crop>
            state, district, crop = parts[2], parts[3], parts[4]
            if state in _master_data and district in _master_data[state]:
                _master_data[state][district].pop(crop, None)
                if not _master_data[state][district]:
                    _master_data[state].pop(district, None)
                if not _master_data[state]:
                    _master_data.pop(state, None)
        elif len(parts) == 4:  # outputs/repair/<state>/<district>
            state, district = parts[2], parts[3]
            if state in _master_data:
                _master_data[state].pop(district, None)
                if not _master_data[state]:
                    _master_data.pop(state, None)
        elif len(parts) == 3:  # outputs/repair/<state>
            _master_data.pop(parts[2], None)
        else:
            return  # nothing to do
        try:
            _upload_master_json()
        except Exception as e:
            print(f"[MASTER] Failed to upload master.json after folder delete: {e}")


def _master_clear_file(path: str) -> None:
    """Clear output_file or audit_file references in _master_data when a file is deleted."""
    parts = [p for p in path.split("/") if p]
    # outputs/repair/<state>/<district>/<crop>/<filename>
    if len(parts) != 6 or parts[0] != "outputs" or parts[1] != "repair":
        return
    state, district, crop, fname = parts[2], parts[3], parts[4], parts[5]
    with _master_lock:
        entry = _master_data.get(state, {}).get(district, {}).get(crop)
        if entry is None:
            return
        changed = False
        if entry.get("output_file", "").endswith(f"/{fname}"):
            entry["output_file"] = None
            entry["processed"] = False
            changed = True
        if entry.get("audit_file", "").endswith(f"/{fname}"):
            entry["audit_file"] = None
            entry["audited"] = False
            changed = True
        if changed:
            try:
                _upload_master_json()
            except Exception as e:
                print(f"[MASTER] Failed to upload master.json after file delete: {e}")


@app.delete("/files/{path:path}")
def delete_file(path: str):
    _resolve_safe(path)
    zwd = _get_zoho()
    result = zwd.resolve_path(path)
    if result is None:
        raise HTTPException(status_code=404, detail="file not found")
    file_id, ftype = result
    if ftype == "folder":
        raise HTTPException(status_code=400, detail="path is a directory")
    if not zwd.delete(file_id):
        raise HTTPException(status_code=502, detail="Zoho delete failed")

    _master_clear_file(path)

    # Also delete linked base file when a dedup_/phase_ prefixed file is deleted
    name = Path(path).name
    for prefix in ("dedup_", "phase_"):
        if name.startswith(prefix):
            base_path = str(Path(path).parent / name[len(prefix):])
            base_result = zwd.resolve_path(base_path)
            if base_result:
                zwd.delete(base_result[0])
            _master_clear_file(base_path)
            break

    return {"deleted": path}


@app.delete("/folders/{path:path}")
def delete_folder(path: str):
    _resolve_safe(path)
    zwd = _get_zoho()
    result = zwd.resolve_path(path)
    if result is None:
        raise HTTPException(status_code=404, detail="folder not found")
    file_id, ftype = result
    if ftype != "folder":
        raise HTTPException(status_code=400, detail="path is not a directory")
    if not zwd.delete(file_id):
        raise HTTPException(status_code=502, detail="Zoho delete failed")
    _master_remove_folder(path)
    return {"deleted": path}


@app.post("/files/rename/{path:path}")
def rename_file(path: str, body: RenameRequest):
    _resolve_safe(path)
    _resolve_safe(body.to)
    zwd = _get_zoho()
    src_result = zwd.resolve_path(path)
    if src_result is None:
        raise HTTPException(status_code=404, detail="source not found")
    dst_result = zwd.resolve_path(body.to)
    if dst_result is not None:
        raise HTTPException(status_code=409, detail="destination already exists")

    file_id = src_result[0]
    src_parts = [p for p in path.split("/") if p]
    dst_parts = [p for p in body.to.split("/") if p]

    src_parent = "/".join(src_parts[:-1])
    dst_parent = "/".join(dst_parts[:-1])
    dst_name   = dst_parts[-1] if dst_parts else path

    if src_parts[-1] != dst_name:
        zwd.rename(file_id, dst_name)
    if src_parent != dst_parent:
        new_parent_id = zwd.ensure_path(dst_parent) if dst_parent else zwd.root_folder_id
        zwd.move(file_id, new_parent_id)

    return {"from": path, "to": body.to}


@app.post("/files/folders")
def create_folder(body: dict):
    rel_path = body.get("path", "")
    if not rel_path:
        raise HTTPException(status_code=400, detail="path is required")
    _resolve_safe(rel_path)
    zwd = _get_zoho()
    folder_id = zwd.ensure_path(rel_path)
    return {"created": rel_path, "id": folder_id}


@app.post("/files/upload-chunk")
async def upload_file_chunk(
    request: Request,
    upload_id: str,
    chunk_index: int,
    total_chunks: int,
    filename: str,
    dest: str = "",
):
    tmp_dir = _CHUNK_TMP / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / f"{chunk_index:06d}").write_bytes(await request.body())

    if not all((tmp_dir / f"{i:06d}").exists() for i in range(total_chunks)):
        return {"chunk": chunk_index, "total": total_chunks}

    if dest:
        _resolve_safe(dest)
    zoho_folder = dest if dest else ""

    content = b""
    for i in range(total_chunks):
        content += (tmp_dir / f"{i:06d}").read_bytes()

    zwd = _get_zoho()
    parent_id = zwd.ensure_path(zoho_folder) if zoho_folder else zwd.root_folder_id
    file_id = zwd.upload_file(filename, content, parent_id)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    zoho_path = f"{zoho_folder}/{filename}" if zoho_folder else filename
    return {"uploaded": zoho_path, "id": file_id}


@app.post("/files/upload")
async def upload_file(dest: str = "", file: UploadFile = File(...)):
    if dest:
        _resolve_safe(dest)
    content = await file.read()
    zwd = _get_zoho()
    parent_id = zwd.ensure_path(dest) if dest else zwd.root_folder_id
    file_id = zwd.upload_file(file.filename, content, parent_id)
    zoho_path = f"{dest}/{file.filename}" if dest else file.filename
    return {"uploaded": zoho_path, "id": file_id}


@app.post("/files/upload-audited")
async def upload_audited(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    file: UploadFile = File(...),
):
    audit_filename = f"audit_{file.filename}"
    zoho_folder = f"outputs/repair/{state}/{district}/{crop}"
    content = await file.read()

    zwd = _get_zoho()

    # Verify crop folder exists in Zoho before accepting the upload
    if zwd.resolve_path(zoho_folder) is None:
        raise HTTPException(status_code=404, detail="crop folder not found in WorkDrive")

    parent_id = zwd.ensure_path(zoho_folder)
    zwd.upload_file(audit_filename, content, parent_id)

    _update_master_crop(state, district, crop, audited=True,
                        audit_file=f"{zoho_folder}/{audit_filename}")

    return {"uploaded": f"{zoho_folder}/{audit_filename}"}


# ---------------------------------------------------------------------------
# App utility routes
# ---------------------------------------------------------------------------

@app.get("/app/next-state")
def get_next_state(
    state: str = "",
    district: str = "",
    domains: List[str] = Query(default=[]),
):
    state_slug  = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_") if state else "state"
    dist_slug   = re.sub(r"[^a-z0-9]+", "_", district.lower()).strip("_") if district else state_slug
    pattern     = re.compile(rf"^{re.escape(dist_slug)}_(\d+)$")
    sorted_domains = sorted(domains)

    zwd = _get_zoho()
    state_folder = zwd.resolve_path(state_slug)
    existing: list[tuple[int, str, str]] = []  # (idx, folder_name, folder_id)

    if state_folder:
        state_folder_id = state_folder[0]
        for item in zwd.list_folder(state_folder_id):
            if item["type"] != "folder":
                continue
            m = pattern.match(item["name"])
            if m:
                existing.append((int(m.group(1)), item["name"], item["id"]))

    for idx, folder_name, folder_id in sorted(existing):
        meta_result = zwd.find_child(folder_id, "meta.json")
        if meta_result:
            try:
                meta = json.loads(zwd.download_file(meta_result["id"]))
                if (meta.get("district", "") == district and
                        sorted(meta.get("domains", [])) == sorted_domains):
                    return {
                        "name": folder_name,
                        "state": state_slug,
                        "is_new": False,
                        "existing_crops": meta.get("crops", []),
                    }
            except Exception:
                pass

    next_idx = max((i for i, _, _ in existing), default=-1) + 1
    return {
        "name": f"{dist_slug}_{next_idx}",
        "state": state_slug,
        "is_new": True,
        "existing_crops": [],
    }


# ---------------------------------------------------------------------------
# master.json — persistent state-table cache in Zoho root
#
# Shape:
#   { "built_at": "<iso>",
#     "data": { "<state>": { "<district>": { "<crop>": {
#       "output_file": str|null, "audit_file": str|null,
#       "downloaded": bool, "audited": bool, "processed": bool
#     }}}}}
# ---------------------------------------------------------------------------

_master_ready  = threading.Event()
_master_lock   = threading.Lock()
_master_data: dict = {}   # in-memory nested copy of master.json["data"]

MASTER_JSON_NAME = "master.json"


def _upload_master_json() -> None:
    """Write current _master_data to master.json in Zoho root. Caller must hold _master_lock."""
    zwd = _get_zoho()
    payload = json.dumps({"built_at": datetime.now(timezone.utc).isoformat(), "data": _master_data})
    zwd.upload_file(MASTER_JSON_NAME, payload.encode(), zwd.root_folder_id)
    print("[MASTER] master.json uploaded to Zoho root")


def _build_master_data_from_zoho() -> dict:
    """Walk outputs/repair tree in Zoho and return a fresh nested data dict."""
    zwd = _get_zoho()
    data: dict = {}

    repair_result = zwd.resolve_path("outputs/repair")
    if repair_result is None:
        return data
    repair_id = repair_result[0]

    for state_item in sorted(zwd.list_folder(repair_id), key=lambda x: x["name"]):
        if state_item["type"] != "folder" or state_item["name"].startswith("."):
            continue
        state_name = state_item["name"]

        district_items = sorted(
            zwd.list_folder(state_item["id"]),
            key=lambda d: (0 if d["name"].startswith(state_name) else 1, d["name"]),
        )

        for district_item in district_items:
            if district_item["type"] != "folder" or district_item["name"].startswith("."):
                continue
            district_name = district_item["name"]

            for crop_item in sorted(zwd.list_folder(district_item["id"]), key=lambda x: x["name"]):
                if crop_item["type"] != "folder" or crop_item["name"] in ("final",) or crop_item["name"].startswith("."):
                    continue
                crop_name = crop_item["name"]

                # Last-wins on Zoho duplicate filenames
                crop_files: dict[str, dict] = {}
                for f in zwd.list_folder(crop_item["id"]):
                    crop_files[f["name"]] = f

                output_file = None
                for candidate in (f"{district_name}_{crop_name}.csv", "dedup_faq.csv"):
                    if candidate in crop_files:
                        output_file = f"outputs/repair/{state_name}/{district_name}/{crop_name}/{candidate}"
                        break

                audit_file = None
                for fname in sorted(crop_files):
                    if fname.startswith("audit_") and fname.endswith(".csv"):
                        audit_file = f"outputs/repair/{state_name}/{district_name}/{crop_name}/{fname}"
                        break

                data.setdefault(state_name, {}).setdefault(district_name, {})[crop_name] = {
                    "output_file": output_file,
                    "audit_file": audit_file,
                    "downloaded": False,
                    "audited": False,
                    "processed": output_file is not None,
                }

    return data


def _load_or_build_master() -> None:
    """Called once at startup: load master.json from Zoho root if present, else build it."""
    global _master_data
    zwd = _get_zoho()
    existing = zwd.find_child(zwd.root_folder_id, MASTER_JSON_NAME)
    if existing:
        try:
            payload = json.loads(zwd.download_file(existing["id"]))
            with _master_lock:
                _master_data = payload.get("data", {})
            print("[MASTER] Loaded master.json from Zoho root")
            _master_ready.set()
            return
        except Exception as e:
            print(f"[MASTER] Failed to load existing master.json ({e}), rebuilding")

    _rebuild_master()


def _rebuild_master() -> None:
    """Full Zoho walk → rebuild _master_data → upload master.json → set ready."""
    global _master_data
    print("[MASTER] Building master.json from Zoho tree …")
    try:
        new_data = _build_master_data_from_zoho()
        with _master_lock:
            _master_data = new_data
            _upload_master_json()
        _master_ready.set()
        print("[MASTER] master.json ready")
    except Exception as e:
        print(f"[MASTER] Build failed: {e}")
        _master_ready.set()  # unblock requests even on failure


def _update_master_crop(state: str, district: str, crop: str, **updates) -> None:
    """Update a single crop entry in memory and re-upload master.json."""
    with _master_lock:
        entry = _master_data.setdefault(state, {}).setdefault(district, {}).setdefault(crop, {})
        entry.update(updates)
        try:
            _upload_master_json()
        except Exception as e:
            print(f"[MASTER] Failed to upload master.json after update: {e}")


def _master_to_rows() -> list[dict]:
    rows = []
    with _master_lock:
        snapshot = json.loads(json.dumps(_master_data))  # shallow-safe copy under lock
    for state_name, districts in sorted(snapshot.items()):
        for district_name, crops in sorted(districts.items()):
            for crop_name, d in sorted(crops.items()):
                rows.append({
                    "state": state_name,
                    "district": district_name,
                    "crop": crop_name,
                    "output_file": d.get("output_file"),
                    "audit_file": d.get("audit_file"),
                    "downloaded": d.get("downloaded", False),
                    "audited": d.get("audited", False),
                    "processed": d.get("processed", False),
                })
    return rows


@app.get("/app/state-table")
def get_state_table(refresh: bool = False):
    if refresh:
        _master_ready.clear()
        threading.Thread(target=_rebuild_master, daemon=True, name="master-json-refresh").start()
        return {"status": "loading", "rows": []}

    if not _master_ready.is_set():
        return {"status": "loading", "rows": []}

    return {"status": "ready", "rows": _master_to_rows()}


@app.get("/app/output/{state}/{district}/{crop}")
def download_output(state: str, district: str, crop: str):
    """Download the output CSV for a state/district/crop and mark it downloaded."""
    zwd = _get_zoho()
    crop_folder_path = f"outputs/repair/{state}/{district}/{crop}"
    crop_folder_result = zwd.resolve_path(crop_folder_path)
    if crop_folder_result is None:
        raise HTTPException(status_code=404, detail="output not found")

    crop_folder_id = crop_folder_result[0]
    crop_files = {f["name"]: f for f in zwd.list_folder(crop_folder_id)}

    # Find the dedup output
    output_filename = None
    for candidate in (f"{district}_{crop}.csv", "dedup_faq.csv"):
        if candidate in crop_files:
            output_filename = candidate
            break
    if output_filename is None:
        raise HTTPException(status_code=404, detail="output not found")

    file_id = crop_files[output_filename]["id"]

    _update_master_crop(state, district, crop, downloaded=True)

    content = zwd.download_file(file_id)
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{district}_{crop}.csv"'},
    )


# ---------------------------------------------------------------------------
# Job routes
# ---------------------------------------------------------------------------

@app.get("/jobs")
def list_jobs():
    return list(jobs.values())


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "running":
        raise HTTPException(status_code=409, detail=f"job is not running (status={job['status']})")
    _job_ctl.cancel(job_id)
    jobs[job_id]["status"] = "stopped"
    return {"job_id": job_id, "status": "stopped"}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="cannot delete a running job")
    del jobs[job_id]
    return {"deleted": job_id}
