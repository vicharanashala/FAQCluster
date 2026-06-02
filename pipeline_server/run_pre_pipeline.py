#!/usr/bin/env python3
"""
run_pre_pipeline.py — Pre-Pipeline: State filter + Crop normalization

Prepares a raw KCC CSV for the main pipeline by:
  1. Filter  — keep only rows for the specified state (get_state_crop_rows.py)
  2. Normalize — map raw crop variants to canonical names, drop others (crop_normalizer.py)

Usage:
    python run_pre_pipeline.py \\
        --input  zoho_raw.csv \\
        --state  Karnataka \\
        --crops  Cotton Sugarcane "Sugar Beet" \\
        --output karna_norm.csv

Output:
    <output>  — normalized CSV ready to pass as --raw-file to run_pipeline.py / run_full.py
"""

import os
import signal
import sys
import subprocess
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

SCRIPT_DIR        = Path(__file__).resolve().parent
PRE_PIPELINE_DIR  = SCRIPT_DIR / 'pre_pipeline'

try:
    import _job_ctl as _ctl
except ImportError:
    _ctl = None


def banner(msg: str):
    width = 66
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


def _stream(cmd: list) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid,
    )
    if _ctl:
        _ctl.register_proc(proc)
    cancelled = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        if _ctl and _ctl.is_cancelled(_ctl.current_job_id()):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
            cancelled = True
            break
    proc.wait()
    if _ctl:
        _ctl.deregister_proc()
    if cancelled:
        raise _ctl.JobCancelled(_ctl.current_job_id())
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_state_filter(input_path: Path, state: str, intermediate: Path,
                     domains: list[str] = None, district: str = None):
    banner("Stage 1/2 — State Filter")
    cmd = [
        sys.executable,
        str(PRE_PIPELINE_DIR / 'get_state_crop_rows.py'),
        '--input',  str(input_path),
        '--state',  state,
        '--output', str(intermediate),
    ]
    if district:
        cmd += ['--district', district]
    if domains:
        cmd += ['--domains', ','.join(domains)]
    print(f"  Input    : {input_path}")
    print(f"  State    : {state}")
    if district:
        print(f"  District : {district}")
    if domains:
        print(f"  Domains  : {', '.join(domains)}")
    print(f"  Output   : {intermediate}")
    _stream(cmd)
    print(f"\n  ✓ State filter complete")


def run_crop_normalizer(intermediate: Path, output_path: Path, crops: list[str] = None):
    banner("Stage 2/2 — Crop Normalization")
    cmd = [
        sys.executable,
        str(PRE_PIPELINE_DIR / 'crop_normalizer.py'),
        '--input',  str(intermediate),
        '--output', str(output_path),
    ]
    if crops:
        cmd += ['--crops', *crops]
    if crops:
        print(f"  Primary crops : {', '.join(crops)}")
    else:
        print(f"  Primary crops : (all, auto-mapped)")
    print(f"  Output        : {output_path}")
    _stream(cmd)
    print(f"\n  ✓ Crop normalization complete")


def parse_args():
    parser = argparse.ArgumentParser(
        prog='run_pre_pipeline.py',
        description=textwrap.dedent("""\
            Pre-pipeline runner: filter by state then normalize crop names.
            Produces a clean CSV ready for the main FAQ pipeline.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input',  required=True,
                        help='Raw KCC CSV file (e.g. zoho_downloaded_file.csv)')
    parser.add_argument('--state',  required=True,
                        help='State name to filter rows by (e.g. Karnataka)')
    parser.add_argument('--crops',  required=True, nargs='+', metavar='CROP',
                        help='Primary crop names to keep and normalize '
                             '(e.g. Cotton Sugarcane "Sugar Beet")')
    parser.add_argument('--output', required=True,
                        help='Path for the final normalized CSV output')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep the state-filtered intermediate CSV (do not delete it)')
    return parser.parse_args()


def main():
    args = parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    # Intermediate file sits alongside the output, cleaned up on success
    intermediate = output_path.parent / f"{output_path.stem}_state_rows.csv"

    start_time = datetime.now()
    banner("KCC FAQ Pre-Pipeline")
    print(f"  Input    : {input_path}")
    print(f"  State    : {args.state}")
    print(f"  Crops    : {', '.join(args.crops)}")
    print(f"  Output   : {output_path}")
    print(f"  Started  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    run_state_filter(input_path, args.state, intermediate)
    run_crop_normalizer(intermediate, output_path, args.crops)

    if intermediate.exists() and not args.keep_intermediate:
        intermediate.unlink()

    elapsed = datetime.now() - start_time
    banner("Pre-Pipeline Complete!")
    print(f"  Output  : {output_path}")
    print(f"  Elapsed : {elapsed}")
    print()


if __name__ == '__main__':
    main()
