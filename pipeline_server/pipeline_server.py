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

  GET  /app/next-state            get next versioned state folder name
  GET  /app/state-table           table of all state/crop outputs
  GET  /app/output/{state}/{crop} download output CSV

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
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator, model_validator

import _job_ctl

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

APP_DATA = SCRIPT_DIR / "app-data"
APP_DATA.mkdir(exist_ok=True)
(APP_DATA / "outputs" / "repair").mkdir(parents=True, exist_ok=True)
(APP_DATA / "outputs" / "hyperparameter_tuning").mkdir(parents=True, exist_ok=True)

_CHUNK_TMP = Path(tempfile.gettempdir()) / "faq_chunks"
_RUN_PIPELINE = SCRIPT_DIR / "run_pipeline.py"

app = FastAPI(title="FAQCluster Pipeline Service", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

    input_path  = APP_DATA / "cleaned_data.csv"
    output_path = _resolve_safe(r.output)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output_path.parent / f"{output_path.stem}_state_rows.csv"

    run_state_filter(input_path, r.state, intermediate, domains=r.domains or [])
    _job_ctl.check_cancel()
    if r.crops:
        run_crop_normalizer(intermediate, output_path, r.crops)
    else:
        shutil.copy2(intermediate, output_path)

    if intermediate.exists() and not r.keep_intermediate:
        intermediate.unlink()


def _run_pipeline_sync(r: PipelineRequest) -> None:
    import pandas as pd

    resolved_raw = str(APP_DATA / r.input)
    state_folder = Path(r.input).stem
    out_base     = _resolve_safe(r.output_dir) / state_folder
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

    out_dir_str = str(_resolve_safe(r.output_dir))

    for crop in crops:
        _job_ctl.check_cancel()
        (out_base / slug(crop)).mkdir(parents=True, exist_ok=True)
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

    if failed:
        raise RuntimeError(f"The following crops failed: {', '.join(failed)}")


def _run_post_sync(r: PostRequest) -> None:
    from run_post_pipeline import run_dedup as post_run_dedup

    input_dir = _resolve_safe(r.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"input folder not found: {input_dir}")

    if not r.skip_dedup:
        post_run_dedup(input_dir, r.crops or None)


def _next_versioned_path(current_path: Path) -> Path:
    """Return the next non-existent versioned path, e.g. maharashtra_1.csv → maharashtra_2.csv."""
    m = re.match(r"^(.+)_(\d+)$", current_path.stem)
    base, ver = (m.group(1), int(m.group(2))) if m else (current_path.stem, 0)
    next_ver = ver + 1
    while True:
        candidate = current_path.parent / f"{base}_{next_ver}{current_path.suffix}"
        if not candidate.exists():
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
            raise FileNotFoundError(f"crops file not found: {crops_file}")
        lines = crops_file.read_text().splitlines()
        crops = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        if not crops:
            raise ValueError("no crops found in crops_file")
    else:
        crops = None

    _norm_is_temp = False

    if r.skip_pre_pipeline:
        effective_raw = str(APP_DATA / "cleaned_data.csv")
        if crops is None:
            _df = pd.read_csv(effective_raw, low_memory=False)
            if r.domains:
                _df = _df[_df["QueryType"].dropna().str.strip().isin(r.domains)]
            crops = _df["Crop"].dropna().str.strip().unique().tolist()
            print(f"[INFO] Auto-discovered {len(crops)} crop(s) from input CSV")
    else:
        raw_file = APP_DATA / "cleaned_data.csv"
        if r.pre_output:
            norm_file = _resolve_safe(r.pre_output)
            norm_file.parent.mkdir(parents=True, exist_ok=True)
            _norm_is_temp = False

            meta_path = norm_file.parent / "meta.json"
            existing_meta: dict = {}
            for _mp in (meta_path, norm_file.with_suffix(".json")):
                if _mp.exists():
                    try:
                        existing_meta = json.loads(_mp.read_text())
                        break
                    except Exception:
                        pass

            existing_domains    = set(existing_meta.get("domains", []))
            existing_crops_meta = set(existing_meta.get("crops", []))
            requested_domains   = set(r.domains or [])
            file_exists         = norm_file.exists() and bool(existing_meta)
            same_state          = existing_meta.get("state") == r.state if existing_meta else False
            same_domains        = existing_domains == requested_domains

            if file_exists and same_state and same_domains:
                if crops is None:
                    print(f"[INFO] Reusing existing pre-pipeline output: {norm_file}")
                    crops = pd.read_csv(norm_file)["Crop"].dropna().str.strip().unique().tolist()
                    print(f"[INFO] {len(crops)} crop(s) from cached pre-pipeline output")
                else:
                    missing_crops = [c for c in crops if c not in existing_crops_meta]
                    if not missing_crops:
                        print(f"[INFO] Reusing existing pre-pipeline output: {norm_file}")
                    else:
                        print(
                            f"[INFO] {len(crops) - len(missing_crops)} crop(s) reused; "
                            f"running pre-pipeline for {len(missing_crops)} new crop(s): {', '.join(missing_crops)}"
                        )
                        intermediate = norm_file.parent / f"{norm_file.stem}_state_rows.csv"
                        run_state_filter(raw_file, r.state, intermediate, domains=r.domains or [])
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
                            "domains": r.domains or [],
                            "crops": sorted(existing_crops_meta | set(missing_crops)),
                        }))
            else:
                if file_exists:
                    norm_file = _next_versioned_path(norm_file)
                    norm_file.parent.mkdir(parents=True, exist_ok=True)
                    meta_path = norm_file.with_suffix(".json")
                    print(f"[INFO] Domain/state change — new pre-pipeline output: {norm_file}")

                intermediate = norm_file.parent / f"{norm_file.stem}_state_rows.csv"
                run_state_filter(raw_file, r.state, intermediate, domains=r.domains or [])
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
            run_state_filter(raw_file, r.state, intermediate, domains=r.domains or [])
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

    state_folder = Path(effective_raw).stem
    out_base     = _resolve_safe(r.output_dir) / state_folder

    def _output_done(crop_slug: str) -> bool:
        d = out_base / crop_slug
        return (d / f"{out_base.name}_{crop_slug}.csv").exists() or (d / "dedup_faq.csv").exists()

    crops_to_run  = [c for c in crops if not _output_done(slug(c))]
    skipped_crops = [c for c in crops if _output_done(slug(c))]
    if skipped_crops:
        print(f"[INFO] Skipping {len(skipped_crops)} already-completed crop(s): {', '.join(skipped_crops)}")
    print(f"[INFO] Found {len(crops_to_run)} unique crop(s): {', '.join(crops_to_run)}")

    out_dir_str = str(_resolve_safe(r.output_dir))
    failed = []

    for crop in crops_to_run:
        _job_ctl.check_cancel()
        (out_base / slug(crop)).mkdir(parents=True, exist_ok=True)
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
        elif not r.skip_post_pipeline:
            try:
                post_run_dedup(out_base, [crop])
            except Exception as exc:
                print(f"[WARN] Post-pipeline for '{crop}' failed: {exc}")

    if not r.skip_pre_pipeline and _norm_is_temp and norm_file.exists():
        norm_file.unlink()
        print("[INFO] Temporary pre-pipeline file removed from disk")

    if r.pre_output and not r.skip_pre_pipeline:
        try:
            cur_meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            cur_meta = {}
        newly_done = [c for c in crops_to_run if c not in failed]
        all_crops  = sorted(set(cur_meta.get("crops", [])) | set(newly_done))
        meta_path.write_text(json.dumps({
            "state": r.state,
            "domains": r.domains or [],
            "crops": all_crops,
        }))

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
    """Filtered view of app-data/ for the frontend."""
    def _file_entry(p: Path) -> dict:
        return {
            "name": p.name,
            "path": str(p.relative_to(APP_DATA)),
            "size": p.stat().st_size,
        }

    outputs_dir = APP_DATA / "outputs"
    repair_dir  = outputs_dir / "repair"
    final_root  = APP_DATA / "final"

    all_csvs: list[dict] = []
    csv_ancestor_dirs: set[Path] = set()

    if APP_DATA.exists():
        for p in APP_DATA.rglob("*.csv"):
            if ".ipynb_checkpoints" in p.parts:
                continue
            try:
                p.relative_to(outputs_dir)
                continue
            except ValueError:
                pass
            try:
                p.relative_to(final_root)
                continue
            except ValueError:
                pass
            all_csvs.append(_file_entry(p))
            for ancestor in p.parents:
                if ancestor == APP_DATA:
                    break
                csv_ancestor_dirs.add(ancestor)
    all_csvs.sort(key=lambda e: e["path"])

    def _add_empty_dirs(d: Path) -> None:
        if ".ipynb_checkpoints" in d.parts or d.name.startswith("."):
            return
        try:
            d.relative_to(outputs_dir)
            return
        except ValueError:
            pass
        try:
            d.relative_to(final_root)
            return
        except ValueError:
            pass
        if d not in csv_ancestor_dirs:
            all_csvs.append({
                "name": d.name,
                "path": str(d.relative_to(APP_DATA)),
                "size": 0,
                "isDir": True,
            })
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    _add_empty_dirs(sub)

    if APP_DATA.exists():
        for d in sorted(APP_DATA.iterdir()):
            if d.is_dir():
                _add_empty_dirs(d)

    crop_qa_files: list[dict] = []
    if repair_dir.exists():
        for qa_file in sorted(repair_dir.rglob("unique_questions_freq_qa.csv")):
            crop_slug  = qa_file.parent.name
            state_name = qa_file.parent.parent.name
            entry = _file_entry(qa_file)
            entry["crop"]        = crop_slug
            entry["state"]       = state_name
            entry["displayName"] = crop_slug
            crop_qa_files.append(entry)

    final_csvs: list[dict] = []
    if repair_dir.exists():
        for state_dir in sorted(repair_dir.iterdir()):
            if not state_dir.is_dir():
                continue
            for crop_dir in sorted(state_dir.iterdir()):
                if not crop_dir.is_dir() or crop_dir.name == "final":
                    continue
                for p in sorted(crop_dir.iterdir()):
                    if not p.is_file():
                        continue
                    if p.name.startswith("dedup_") or p.name.startswith("phase_"):
                        entry = _file_entry(p)
                        entry["state"]      = state_dir.name
                        entry["crop"]       = crop_dir.name
                        entry["folderPath"] = str(crop_dir.relative_to(APP_DATA))
                        final_csvs.append(entry)

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
    target = _resolve_safe(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(target), filename=target.name)


@app.delete("/files/{path:path}")
def delete_file(path: str):
    target = _resolve_safe(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="path is a directory")
    target.unlink()
    parent = target.parent
    name = target.name
    for prefix in ("dedup_", "phase_"):
        if name.startswith(prefix):
            base = parent / name[len(prefix):]
            if base.exists() and base.is_file():
                base.unlink()
            break
    return {"deleted": path}


@app.delete("/folders/{path:path}")
def delete_folder(path: str):
    target = _resolve_safe(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="folder not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")
    shutil.rmtree(target)
    return {"deleted": path}


@app.post("/files/rename/{path:path}")
def rename_file(path: str, body: RenameRequest):
    source = _resolve_safe(path)
    dest   = _resolve_safe(body.to)
    if not source.exists():
        raise HTTPException(status_code=404, detail="source not found")
    if dest.exists():
        raise HTTPException(status_code=409, detail="destination already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source.rename(dest)
    return {"from": path, "to": body.to}


@app.post("/files/folders")
def create_folder(body: dict):
    rel_path = body.get("path", "")
    if not rel_path:
        raise HTTPException(status_code=400, detail="path is required")
    target = _resolve_safe(rel_path)
    target.mkdir(parents=True, exist_ok=True)
    return {"created": rel_path}


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

    save_dir = _resolve_safe(dest) if dest else APP_DATA
    save_dir.mkdir(parents=True, exist_ok=True)
    target = (save_dir / filename).resolve()
    if not target.is_relative_to(APP_DATA.resolve()):
        shutil.rmtree(tmp_dir)
        raise HTTPException(status_code=400, detail="path escapes sandbox")

    with target.open("wb") as f:
        for i in range(total_chunks):
            f.write((tmp_dir / f"{i:06d}").read_bytes())
    shutil.rmtree(tmp_dir)
    return {"uploaded": str(target.relative_to(APP_DATA))}


@app.post("/files/upload")
async def upload_file(dest: str = "", file: UploadFile = File(...)):
    if dest:
        _resolve_safe(dest)
        save_dir = (APP_DATA / dest).resolve()
    else:
        save_dir = APP_DATA
    save_dir.mkdir(parents=True, exist_ok=True)
    target = (save_dir / file.filename).resolve()
    if not target.is_relative_to(APP_DATA.resolve()):
        raise HTTPException(status_code=400, detail="path escapes sandbox")
    content = await file.read()
    target.write_bytes(content)
    return {"uploaded": str(target.relative_to(APP_DATA))}


@app.post("/files/upload-audited")
async def upload_audited(
    state: str = Form(...),
    crop: str = Form(...),
    file: UploadFile = File(...),
):
    target = _resolve_safe(f"outputs/repair/{state}/{crop}/audit_{file.filename}")
    if not target.parent.exists():
        raise HTTPException(status_code=404, detail="crop folder not found")
    content = await file.read()
    target.write_bytes(content)

    meta_path = target.parent / "meta.json"
    meta = {"download": False, "audit": False}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta["audit"] = True
    meta_path.write_text(json.dumps(meta))
    return {"uploaded": str(target.relative_to(APP_DATA))}


# ---------------------------------------------------------------------------
# App utility routes
# ---------------------------------------------------------------------------

@app.get("/app/next-state")
def get_next_state(state: str = "", domains: List[str] = Query(default=[])):
    _slug = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_") if state else "state"
    pattern = re.compile(rf"^{re.escape(_slug)}_(\d+)$")
    sorted_domains = sorted(domains)

    existing: list[tuple[int, Path]] = []
    if APP_DATA.exists():
        for p in APP_DATA.iterdir():
            m = pattern.match(p.name)
            if m and p.is_dir():
                existing.append((int(m.group(1)), p))

    for idx, folder in sorted(existing):
        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if sorted(meta.get("domains", [])) == sorted_domains:
                    return {
                        "name": folder.name,
                        "is_new": False,
                        "existing_crops": meta.get("crops", []),
                    }
            except Exception:
                pass

    next_idx = max((i for i, _ in existing), default=-1) + 1
    return {
        "name": f"{_slug}_{next_idx}",
        "is_new": True,
        "existing_crops": [],
    }


@app.get("/app/state-table")
def get_state_table():
    repair_dir = APP_DATA / "outputs" / "repair"
    rows = []

    if not repair_dir.exists():
        return {"rows": rows}

    for state_dir in sorted(repair_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        state_name = state_dir.name

        domains: list[str] = []
        meta_path = APP_DATA / state_name / "meta.json"
        if meta_path.exists():
            try:
                domains = json.loads(meta_path.read_text()).get("domains", [])
            except Exception:
                pass

        for crop_dir in sorted(state_dir.iterdir()):
            if not crop_dir.is_dir() or crop_dir.name == "final":
                continue

            dedup = crop_dir / f"{state_name}_{crop_dir.name}.csv"
            if not dedup.exists():
                dedup = crop_dir / "dedup_faq.csv"
            output_file = str(dedup.relative_to(APP_DATA)) if dedup.exists() else None

            audit_file = None
            for f in sorted(crop_dir.iterdir()):
                if f.name.startswith("audit_") and f.suffix == ".csv":
                    audit_file = str(f.relative_to(APP_DATA))
                    break

            crop_meta = {"download": False, "audit": False}
            crop_meta_path = crop_dir / "meta.json"
            if crop_meta_path.exists():
                try:
                    crop_meta = json.loads(crop_meta_path.read_text())
                except Exception:
                    pass
            elif output_file:
                crop_meta_path.write_text(json.dumps(crop_meta))

            rows.append({
                "state": state_name,
                "crop": crop_dir.name,
                "domains": domains,
                "output_file": output_file,
                "audit_file": audit_file,
                "downloaded": crop_meta.get("download", False),
                "audited": crop_meta.get("audit", False),
            })

    return {"rows": rows}


@app.get("/app/output/{state}/{crop}")
def download_output(state: str, crop: str):
    """Download the output CSV for a state/crop and mark it downloaded."""
    crop_dir = _resolve_safe(f"outputs/repair/{state}/{crop}")
    dedup = crop_dir / f"{state}_{crop}.csv"
    if not dedup.exists():
        dedup = crop_dir / "dedup_faq.csv"
    if not dedup.exists():
        raise HTTPException(status_code=404, detail="output not found")

    meta_path = crop_dir / "meta.json"
    meta = {"download": False, "audit": False}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta["download"] = True
    meta_path.write_text(json.dumps(meta))
    return FileResponse(str(dedup), filename=f"{state}_{crop}.csv")


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
