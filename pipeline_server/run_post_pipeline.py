#!/usr/bin/env python3
"""
run_post_pipeline.py — Post-Pipeline: Deduplicate FAQ outputs in-place

Runs LLM deduplication directly on unique_questions_freq_qa.csv in each
crop subfolder.  No collect/copy step is needed.

Usage:
    python run_post_pipeline.py \\
        --input outputs/repair/tamilnadu_norm \\
        [--crops Rice Wheat "Black Gram"] \\
        [--skip-dedup]

Output (per crop folder):
    <input>/<crop>/<state>_<crop>.csv   (deduplicated FAQs)
    <input>/<crop>/phase_data_faq.csv  (LLM matching phase log)
"""

import re
import sys
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

SCRIPT_DIR        = Path(__file__).resolve().parent
POST_PIPELINE_DIR = SCRIPT_DIR / 'post_pipeline'
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import _job_ctl as _ctl
except ImportError:
    _ctl = None

try:
    from helpers.zoho_workdrive import ZohoWorkDrive as _ZohoWorkDrive
    _ZOHO_AVAILABLE = True
except Exception:
    _ZOHO_AVAILABLE = False

SOURCE_FILE  = "unique_questions_freq_qa.csv"
MAPPING_FILE = "unique_question_mapping.csv"
PHASE_OUT    = "phase_data_faq.csv"


def _zoho_push(zwd, zoho_rel: str, *files) -> None:
    """Upload specific files to a Zoho folder path."""
    if zwd is None or not zoho_rel:
        return
    try:
        parent_id = zwd.ensure_path(zoho_rel)
    except Exception as e:
        print(f"  [ZOHO] ensure_path failed for {zoho_rel}: {e}")
        return
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        try:
            zwd.upload_file(f.name, f.read_bytes(), parent_id)
            print(f"  [ZOHO] ↑ {f.name}")
        except Exception as e:
            print(f"  [ZOHO] upload failed for {f.name}: {e}")


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def banner(msg: str):
    width = 66
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


def run_review(input_dir: Path, crops: list | None = None, zwd=None) -> None:
    """Generate review CSVs (Generated_Question + all farmer questions) per crop and upload to Zoho."""
    from post_pipeline.generate_review_file import generate_review

    banner("Review File Generation")
    crop_slugs = {_slug(c) for c in crops} if crops else None

    crop_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name != 'final'
        and (crop_slugs is None or d.name in crop_slugs)
    )

    if not crop_dirs:
        print(f"  No matching crop folders found in {input_dir}")
        return

    for crop_dir in crop_dirs:
        # Prefer the final deduped file; fall back to unique_questions_freq_qa.csv
        dedup_name = f"{input_dir.name}_{crop_dir.name}.csv"
        qa_csv = crop_dir / dedup_name
        if not qa_csv.exists():
            qa_csv = crop_dir / SOURCE_FILE
        if not qa_csv.exists():
            print(f"  [SKIP] {crop_dir.name}: no QA csv found")
            continue

        map_csv = crop_dir / MAPPING_FILE
        if not map_csv.exists():
            print(f"  [SKIP] {crop_dir.name}: {MAPPING_FILE} not found")
            continue

        review_name = f"{input_dir.name}_{crop_dir.name}_review.csv"
        review_csv  = crop_dir / review_name

        try:
            generate_review(qa_csv, map_csv, review_csv)
        except Exception as e:
            print(f"  [ERROR] {crop_dir.name}: review generation failed: {e}")
            continue

        zoho_rel = f"outputs/repair/{input_dir.name}/{crop_dir.name}"
        _zoho_push(zwd, zoho_rel, review_csv)

    print(f"\n  ✓ Review files complete")


def run_dedup(input_dir: Path, crops: list | None = None, zwd=None):
    """Run LLM dedup on each crop subfolder of *input_dir*.

    Args:
        input_dir: state-level folder (e.g. outputs/repair/tamilnadu_norm).
        crops:     optional list of crop names (original form, e.g. ["Rice"]).
                   When given, only folders whose slug matches are processed.
                   When None/empty, all crop subfolders are processed.
        zwd:       ZohoWorkDrive instance (optional); uploads dedup output if provided.
    """
    banner("LLM Deduplication (Gemma-4-26B)")
    import pandas as pd
    from post_pipeline.post_processing_dedup import deduplicate_and_aggregate

    crop_slugs = {_slug(c) for c in crops} if crops else None

    crop_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name != 'final'
        and (crop_slugs is None or d.name in crop_slugs)
    )

    if not crop_dirs:
        print(f"  No matching crop folders found in {input_dir}")
        return

    for crop_dir in crop_dirs:
        source = crop_dir / SOURCE_FILE
        if not source.exists():
            print(f"  [SKIP] {crop_dir.name}: {SOURCE_FILE} not found")
            continue

        if _ctl:
            _ctl.check_cancel()

        _cancel_ev = None
        if _ctl:
            _jid = _ctl.current_job_id()
            if _jid:
                _cancel_ev = _ctl.get_cancel_event(_jid)

        print(f"\n  Processing: {crop_dir.name}/{SOURCE_FILE}")
        try:
            df = pd.read_csv(source, low_memory=False)
            df, df_phase = deduplicate_and_aggregate(df, cancel_event=_cancel_ev)
        except Exception as e:
            print(f"  [ERROR] {crop_dir.name}: {e}")
            continue

        dedup_out_name = f"{crop_dir.parent.name}_{crop_dir.name}.csv"
        df_phase.to_csv(crop_dir / PHASE_OUT, index=False)
        df.to_csv(crop_dir / dedup_out_name, index=False)
        print(f"  Saved: {crop_dir.name}/{PHASE_OUT}")
        print(f"  Saved: {crop_dir.name}/{dedup_out_name}")

        zoho_rel = f"outputs/repair/{input_dir.name}/{crop_dir.name}"
        _zoho_push(zwd, zoho_rel, crop_dir / dedup_out_name, crop_dir / PHASE_OUT)

    print(f"\n  ✓ Deduplication complete")


def parse_args():
    parser = argparse.ArgumentParser(
        prog='run_post_pipeline.py',
        description=textwrap.dedent("""\
            Post-pipeline runner: deduplicate per-crop FAQ CSVs in-place.
            Run this after the main pipeline has finished for all crops.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io = parser.add_argument_group('I/O')
    io.add_argument(
        '--input', required=True,
        help='State folder containing crop subfolders '
             '(e.g. outputs/repair/tamilnadu_norm)',
    )
    io.add_argument(
        '--crops', nargs='*', metavar='CROP',
        help='Crop names to process (default: all). '
             'E.g. --crops Rice Wheat "Black Gram"',
    )
    ctrl = parser.add_argument_group('Pipeline control')
    ctrl.add_argument('--skip-dedup', action='store_true',
                      help='Skip LLM deduplication')
    ctrl.add_argument('--skip-review', action='store_true',
                      help='Skip review file generation (and Zoho upload)')
    ctrl.add_argument('--no-zoho', action='store_true',
                      help='Disable Zoho upload even if credentials are available')
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        sys.exit(f"ERROR: input folder not found: {input_dir}")

    start_time = datetime.now()
    banner("KCC FAQ Post-Pipeline")
    print(f"  Input dir  : {input_dir}")
    if args.crops:
        print(f"  Crops      : {', '.join(args.crops)}")
    else:
        print(f"  Crops      : (all)")
    print(f"  Started    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Init Zoho once so both dedup and review can upload
    zwd = None
    if _ZOHO_AVAILABLE and not args.no_zoho:
        try:
            zwd = _ZohoWorkDrive()
            print("  Zoho       : connected")
        except Exception as ze:
            print(f"  Zoho       : init failed — running local only ({ze})")

    if args.skip_dedup:
        print("\n[--skip-dedup] Skipping LLM deduplication")
    else:
        run_dedup(input_dir, args.crops or None, zwd=zwd)

    if args.skip_review:
        print("\n[--skip-review] Skipping review file generation")
    else:
        run_review(input_dir, args.crops or None, zwd=zwd)

    elapsed = datetime.now() - start_time
    banner("Post-Pipeline Complete!")
    print(f"  Elapsed    : {elapsed}")
    print()


if __name__ == '__main__':
    main()
