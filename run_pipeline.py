#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end KCC FAQ Generation Pipeline

Self-contained entry point for the entire KCC FAQ pipeline.
Runs clustering → LLM evaluation → cluster repair → unique question
extraction → deduplication → irrelevant corpus filtering.

Usage (from the KCC Analysis/ root):
    python kcc_faq/run_pipeline.py \\
        --raw-file data/raw/punjab_maize_raw.csv \\
        --crop "Maize Makka" \\
        --api-key sk-ant-...          # Anthropic Claude for Stage 4
        [--model /path/to/qwen]      # local Qwen 7B (Stages 2-3)
        [--grid-mode medium] \\
        [--output-dir outputs/repair]

Pipeline Stages:
    1. Phase 1 — Hyperparameter screening (HDBSCAN + UMAP grid search)
    2. Phase 2 — LLM evaluation of top candidate configs (Qwen 7B local)
    3. Repair  — LLM-based cluster repair: diverse reps, cross-crop filter,
                 coherence/split, merge, raw row back-mapping
    4. Unique  — Unique question extraction (Claude Haiku or local Qwen)
    5. Dedup   — Final deduplication
    6. Filter  — Irrelevant corpus filtering (removes contact/weather/market queries)

Output:
    outputs/repair/<crop_slug>/unique_questions_freq.csv
    outputs/repair/<crop_slug>/corpus_filtered_out.csv  (removed rows)
"""

import sys
import re
import os
import argparse
import subprocess
import pickle
import textwrap
from pathlib import Path
from datetime import datetime

SCRIPT_DIR   = Path(__file__).resolve().parent          # kcc_faq/
PIPELINE_DIR = SCRIPT_DIR / 'pipeline'                 # kcc_faq/pipeline/
# Corpus config is local to this folder — no external project dependency
DEFAULT_CORPUS = SCRIPT_DIR / 'config' / 'irrelevant_corpus.yaml'
sys.path.insert(0, str(SCRIPT_DIR))  # so `from pipeline.X import Y` works


def banner(msg: str):
    width = 66
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def run_phase1(args, out_dir: Path):
    banner("Stage 1/6 — Phase 1: Hyperparameter Screening")
    from pipeline.hyperparameter_tuning import (
        generate_param_grid, phase1_fast_screening, load_stopwords,
        ClusteringResult,
    )
    from sentence_transformers import SentenceTransformer
    import pandas as pd

    print(f"  Raw file : {args.raw_file}")
    print(f"  Crop     : {args.crop}")
    print(f"  Grid mode: {args.grid_mode}")

    df = pd.read_csv(args.raw_file, low_memory=False)
    if 'query_text' not in df.columns:
        df['query_text'] = df['QueryText'] if 'QueryText' in df.columns else df.iloc[:, 8]

    # Flexible crop filter (strip parentheses)
    crop_norm = re.sub(r'[\(\)]', '', args.crop).strip()
    mask = df['Crop'].str.replace(r'[\(\)]', '', regex=True).str.strip() == crop_norm
    df = df[mask].copy()
    print(f"  Rows for '{args.crop}': {len(df)}")

    # Deduplicate
    df = df.groupby('query_text', as_index=False).size().rename(columns={'size': 'count'})
    print(f"  Unique queries: {len(df)}")

    if len(df) > args.max_queries:
        df = df.sample(n=args.max_queries, random_state=42)
        print(f"  Sampled: {args.max_queries}")

    print("\n  Loading sentence transformer...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    stop_words = load_stopwords()
    configs = generate_param_grid(mode=args.grid_mode)

    print(f"\n  Running Phase 1: {len(configs)} configs...")
    candidates = phase1_fast_screening(df, configs, model, stop_words)

    # Save pickle
    pkl_path = out_dir / 'phase1_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(candidates, f)

    cand_rows = [{'config': str(r.config), **r.config.to_dict(), **r.metrics} for r in candidates]
    pd.DataFrame(cand_rows).to_csv(out_dir / 'phase1_candidates.csv', index=False)

    print(f"\n  ✓ Phase 1 complete — {len(candidates)} viable candidates")
    print(f"  Saved to: {pkl_path}")
    return candidates


def run_phase2(args, out_dir: Path, candidates: list) -> str:
    banner("Stage 2/6 — Phase 2: LLM Evaluation")
    from pipeline.cluster_repair import RepairJudge, run_phase2 as _run_phase2

    best_cfg = _run_phase2(
        candidates, out_dir, args.model, args.gpu_id,
        top_k=args.phase2_top_k, batch_size=args.batch_size,
        coverage_cap=args.coverage_cap,
    )
    print(f"\n  ✓ Phase 2 complete — best config: {best_cfg}")
    return best_cfg


def run_repair(args, out_dir: Path, candidates: list, best_cfg: str):
    banner("Stage 3/6 — Cluster Repair (Steps A–E)")
    import copy
    from pipeline.cluster_repair import (
        RepairJudge, step_a_diverse_reps, step_b_cross_crop,
        step_c_split, step_d_merge, step_e_raw_mapping,
    )
    from sentence_transformers import SentenceTransformer

    result = next((r for r in candidates if str(r.config) == best_cfg), None)
    if result is None:
        sys.exit(f"ERROR: config '{best_cfg}' not found in Phase 1 results.")

    n_original = len(result.clusters)
    print(f"  Config  : {best_cfg}")
    print(f"  Clusters: {n_original}")

    clusters = copy.deepcopy(result.clusters)

    # Step A: diverse reps
    print(f"\n  [A] Loading sentence transformer for embeddings...")
    st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    texts    = result.df['query_text'].tolist()
    import numpy as np
    all_embs = st_model.encode(
        texts, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    clusters, diverse_reps = step_a_diverse_reps(clusters, result.df, all_embs, k=args.diverse_k)
    print(f"  [A] Done — {len(clusters)} clusters have diverse reps")

    # Load LLM for B–D
    print(f"\n  Loading Qwen 7B on GPU {args.gpu_id}...")
    judge = RepairJudge(args.model, batch_size=args.batch_size, gpu_id=args.gpu_id)

    # Step B: cross-crop filter
    clusters = step_b_cross_crop(clusters, judge, args.crop)

    # Refresh diverse_reps after B
    for cid in list(diverse_reps.keys()):
        if cid in clusters:
            q_set = set(clusters[cid]['queries'])
            diverse_reps[cid] = [q for q in diverse_reps[cid] if q in q_set]
            if not diverse_reps[cid]:
                diverse_reps[cid] = clusters[cid]['queries'][:1]
        else:
            del diverse_reps[cid]

    # Step C: coherence + split
    clusters, n_splits = step_c_split(
        clusters, diverse_reps, result.df, all_embs,
        judge, args.crop, flag_on=args.coherence_flag,
    )

    # Step D: merge
    clusters, n_merges = step_d_merge(
        clusters, st_model, judge, args.crop, sim_thresh=args.merge_sim,
    )

    # Step E: raw row back-mapping
    step_e_raw_mapping(clusters, result.df, Path(args.raw_file), args.crop, n_original, out_dir)

    print(f"\n  ✓ Repair complete — {len(clusters)} final clusters")
    print(f"    Splits: {n_splits}  Merges: {n_merges}")


def run_unique_questions(args, out_dir: Path):
    banner("Stage 4/6 — Unique Question Extraction")
    cluster_file = out_dir / 'cluster_questions.csv'
    if not cluster_file.exists():
        sys.exit(f"ERROR: {cluster_file} not found. Run repair stage first.")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / 'unique_question_finder.py'),
        '--cluster-file', str(cluster_file),
        '--raw-file',     str(args.raw_file),
        '--crop',         args.crop,
        '--output-dir',   str(out_dir),
    ]

    if args.api_key:
        cmd += ['--api-provider', 'anthropic', '--api-key', args.api_key]
    else:
        cmd += [
            '--api-provider', 'local',
            '--model',        args.model,
            '--gpu-id',       str(args.gpu_id),
            '--batch-size',   str(args.batch_size),
        ]

    print(f"  Running: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, check=True)
    print(f"\n  ✓ Unique question extraction complete")


def run_dedup(out_dir: Path):
    banner("Stage 5/6 — Final Deduplication")
    freq_csv = out_dir / 'unique_questions_freq.csv'
    if not freq_csv.exists():
        sys.exit(f"ERROR: {freq_csv} not found. Run unique-question stage first.")

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / 'dedup_freq_csv.py'),
        '--input', str(freq_csv),
        '--output', str(freq_csv),
        '--drop-rank',
    ]
    subprocess.run(cmd, check=True)
    print(f"\n  ✓ Dedup complete — final FAQ: {freq_csv}")


def run_corpus_filter(out_dir: Path, corpus_file: str, fuzz_thresh: int = 85):
    """Stage 6: Remove irrelevant representative questions via corpus filter."""
    banner("Stage 6/6 — Irrelevant Corpus Filtering")
    freq_csv = out_dir / 'unique_questions_freq.csv'
    if not freq_csv.exists():
        sys.exit(f"ERROR: {freq_csv} not found. Run dedup stage first.")

    print(f"  Corpus   : {corpus_file}")
    print(f"  Fuzz thr : {fuzz_thresh}")

    from pipeline.filter_faq_corpus import filter_faq, load_corpus
    kept_df, removed_df = filter_faq(
        input_path  = freq_csv,
        corpus_path = corpus_file,
        output_path = freq_csv,
        fuzz_thresh = fuzz_thresh,
        dry_run     = False,
    )
    print(f"\n  ✓ Corpus filter complete — {len(kept_df)} rows kept, "
          f"{len(removed_df)} rows removed")


def load_candidates(out_dir: Path) -> list:
    """Load Phase 1 candidates from existing pickle."""
    pkl = out_dir / 'phase1_results.pkl'
    if not pkl.exists():
        sys.exit(f"ERROR: {pkl} not found. Run without --skip-phase1 first.")
    print(f"  Loading existing Phase 1 results from {pkl} ...")
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_best_cfg(out_dir: Path, candidates: list) -> str:
    """Determine best config from Phase 2 CSV or Phase 1 metric."""
    import pandas as pd
    p2_csv = out_dir / 'phase2_scores.csv'
    if p2_csv.exists():
        p2 = pd.read_csv(p2_csv)
        cfg = p2.sort_values('composite_score', ascending=False).iloc[0]['config']
        print(f"  Best config (Phase 2): {cfg}")
        return cfg
    # Fallback: lowest coverage_efficiency (tightest)
    cfg = str(min(candidates, key=lambda r: r.metrics.get('coverage_efficiency', 1)).config)
    print(f"  Best config (Phase 1 fallback): {cfg}")
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        prog='kcc_faq_pipeline.py',
        description=textwrap.dedent("""\
            End-to-end KCC FAQ pipeline.
            Runs clustering, LLM tuning, cluster repair, and unique question extraction.
            Final output: outputs/repair/<crop>/unique_questions_freq.csv
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    io = parser.add_argument_group('I/O (required)')
    io.add_argument('--raw-file', required=True,
                    help='Path to raw KCC CSV file (e.g. data/raw/punjab_maize.csv)')
    io.add_argument('--crop', required=True,
                    help='Crop name as it appears in the Crop column (e.g. "Maize Makka")')

    # ── Model ─────────────────────────────────────────────────────────────────
    mdl = parser.add_argument_group('Model / API')
    mdl.add_argument('--model',
                     default='/home/kshitij/models/qwen2.5-7b-instruct',
                     help='Local Qwen 7B model path for LLM evaluation and repair')
    mdl.add_argument('--api-key', default=None,
                     help='Anthropic API key for Claude Haiku unique-question stage. '
                          'If omitted, local Qwen is used instead.')
    mdl.add_argument('--gpu-id', type=int, default=0,
                     help='CUDA device index (default: 0)')
    mdl.add_argument('--batch-size', type=int, default=8,
                     help='LLM batch size (default: 8)')

    # ── Pipeline control ──────────────────────────────────────────────────────
    ctrl = parser.add_argument_group('Pipeline control')
    ctrl.add_argument('--output-dir', default='outputs/repair',
                      help='Base output directory (default: outputs/repair)')
    ctrl.add_argument('--skip-phase1', action='store_true',
                      help='Skip Phase 1 — load existing phase1_results.pkl')
    ctrl.add_argument('--skip-phase2', action='store_true',
                      help='Skip Phase 2 — load best config from phase2_scores.csv or auto-detect')
    ctrl.add_argument('--skip-repair', action='store_true',
                      help='Skip cluster repair — use existing cluster_questions.csv')
    ctrl.add_argument('--skip-unique-q', action='store_true',
                      help='Skip unique question extraction — use existing unique_questions_freq.csv')
    ctrl.add_argument('--skip-corpus-filter', action='store_true',
                      help='Skip corpus-based irrelevant query removal (Stage 6)')
    ctrl.add_argument('--corpus-file',
                      default=str(DEFAULT_CORPUS),
                      help=('Path to irrelevant_corpus.yaml '
                            f'(default: {DEFAULT_CORPUS})'))
    ctrl.add_argument('--fuzz-threshold', type=int, default=100,
                      help='Fuzzy match threshold for corpus filter 0–100 (default: 100 - disabled)')

    # ── Tuning ────────────────────────────────────────────────────────────────
    tune = parser.add_argument_group('Tuning parameters')
    tune.add_argument('--max-queries', type=int, default=20000,
                      help='Max unique queries to cluster (default: 20000)')
    tune.add_argument('--grid-mode', default='medium',
                      choices=['quick', 'medium', 'full', 'exhaustive'],
                      help='HP search grid size: quick(18), medium(108), full(240), exhaustive(480)')
    tune.add_argument('--phase2-top-k', type=int, default=5,
                      help='Number of Phase 1 candidates to evaluate with LLM (default: 5)')
    tune.add_argument('--coverage-cap', type=float, default=0.80,
                      help='Fraction of query volume to evaluate in Phase 2 (default: 0.80)')

    # ── Repair ────────────────────────────────────────────────────────────────
    rep = parser.add_argument_group('Repair parameters')
    rep.add_argument('--diverse-k', type=int, default=3,
                     help='Max-diverse reps per cluster in Step A (default: 3)')
    rep.add_argument('--coherence-flag', default='C', choices=['B', 'C'],
                     help='LLM coherence rating that triggers a split (default: C)')
    rep.add_argument('--merge-sim', type=float, default=0.82,
                     help='Cosine sim threshold for merge candidates in Step D (default: 0.82)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = SCRIPT_DIR
    args.raw_file = str(Path(args.raw_file).expanduser())
    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = project_root / out_base
    out_dir = out_base / slug(args.crop)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    banner(f"KCC FAQ Pipeline — {args.crop}")
    print(f"  Output dir : {out_dir}")
    print(f"  Raw file   : {args.raw_file}")
    print(f"  LLM model  : {args.model}")
    print(f"  API        : {'Anthropic (Claude Haiku)' if args.api_key else 'Local Qwen 7B'}")
    print(f"  Grid mode  : {args.grid_mode}")
    print(f"  Started    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Stage 1: Phase 1 ─────────────────────────────────────────────────────
    if args.skip_phase1:
        print("\n[--skip-phase1] Loading existing Phase 1 results...")
        candidates = load_candidates(out_dir)
    else:
        candidates = run_phase1(args, out_dir)

    # ── Stage 2: Phase 2 ─────────────────────────────────────────────────────
    if args.skip_phase2:
        print("\n[--skip-phase2] Detecting best config...")
        best_cfg = load_best_cfg(out_dir, candidates)
    else:
        best_cfg = run_phase2(args, out_dir, candidates)

    # ── Stage 3: Repair ───────────────────────────────────────────────────────
    if args.skip_repair:
        print("\n[--skip-repair] Skipping cluster repair (using existing cluster_questions.csv)")
    else:
        run_repair(args, out_dir, candidates, best_cfg)

    # ── Stage 4: Unique questions ──────────────────────────────────────────────
    if args.skip_unique_q:
        print("\n[--skip-unique-q] Skipping unique question extraction")
    else:
        run_unique_questions(args, out_dir)

    # ── Stage 5: Dedup ────────────────────────────────────────────────────────
    run_dedup(out_dir)

    # ── Stage 6: Corpus filter ────────────────────────────────────────────────
    if args.skip_corpus_filter:
        print("\n[--skip-corpus-filter] Skipping irrelevant corpus filtering")
    else:
        if not Path(args.corpus_file).exists():
            print(f"\n  WARNING: corpus file not found: {args.corpus_file}")
            print("  Skipping corpus filter (use --corpus-file to specify a valid path)")
        else:
            run_corpus_filter(out_dir, args.corpus_file, args.fuzz_threshold)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    banner("Pipeline Complete!")
    faq = out_dir / 'unique_questions_freq.csv'
    print(f"  ⭐ Final FAQ  : {faq}")
    print(f"  Elapsed      : {elapsed}")
    print(f"\n  Output columns:")
    print("    unique_q_id, representative_question, raw_frequency,")
    print("    pct_of_total, cluster_id, cluster_label, answer_label,")
    print("    n_questions_in_group, was_cluster_split, parent_cluster,")
    print("    merged_from, merged_cross_cluster")
    print()


if __name__ == '__main__':
    main()
