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

SOURCE_FILE = "unique_questions_freq_qa.csv"
PHASE_OUT   = "phase_data_faq.csv"


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def banner(msg: str):
    width = 66
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


def run_dedup(input_dir: Path, crops: list | None = None):
    """Run LLM dedup on each crop subfolder of *input_dir*.

    Args:
        input_dir: state-level folder (e.g. outputs/repair/tamilnadu_norm).
        crops:     optional list of crop names (original form, e.g. ["Rice"]).
                   When given, only folders whose slug matches are processed.
                   When None/empty, all crop subfolders are processed.
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

    if args.skip_dedup:
        print("\n[--skip-dedup] Skipping LLM deduplication")
    else:
        run_dedup(input_dir, args.crops or None)

    elapsed = datetime.now() - start_time
    banner("Post-Pipeline Complete!")
    print(f"  Elapsed    : {elapsed}")
    print()


if __name__ == '__main__':
    main()
