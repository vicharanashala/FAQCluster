#!/usr/bin/env python3
"""
Generate a human-readable cluster mapping for the best hyperparameter config.

Outputs:
  outputs/hyperparameter_tuning/best_config_cluster_mapping.csv  — full detail
  outputs/hyperparameter_tuning/best_config_cluster_summary.csv  — one row per cluster

Usage:
  python analysis/cluster_mapping.py [--config "α=0.1_mcs=8_ms=3_nn=8"]
"""
import sys, argparse
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR   = PROJECT_ROOT / 'outputs' / 'hyperparameter_tuning'

sys.path.insert(0, str(SCRIPT_DIR))
from hyperparameter_tuning import ClusteringResult, ClusteringConfig  # noqa


def dedup_queries(queries):
    seen, out = set(), []
    for q in queries:
        k = q.strip().lower()
        if k not in seen:
            seen.add(k)
            out.append(q.strip())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Config string to visualise (default: auto-pick best from phase2 CSV)')
    parser.add_argument('--pickle', type=str, default='phase1_results.pkl')
    parser.add_argument('--phase2-csv', type=str, default='phase2_hf_evaluation_final.csv')
    args = parser.parse_args()

    # --- pick config string ---
    if args.config:
        config_str = args.config
    else:
        phase2_file = OUTPUT_DIR / args.phase2_csv
        if phase2_file.exists():
            df2 = pd.read_csv(phase2_file)
            config_str = df2.sort_values('composite_score', ascending=False).iloc[0]['config']
            score = df2.sort_values('composite_score', ascending=False).iloc[0]['composite_score']
            print(f"Auto-selected best config from {phase2_file.name}: {config_str}  (composite={score:.4f})")
        else:
            config_str = 'α=0.1_mcs=8_ms=3_nn=8'
            print(f"No Phase 2 CSV found — using known winner: {config_str}")

    # --- load pickle ---
    import pickle
    pkl_path = OUTPUT_DIR / args.pickle
    print(f"Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    result = next((r for r in all_results if str(r.config) == config_str), None)
    if result is None:
        print(f"ERROR: config '{config_str}' not found in pickle.")
        print("Available configs (first 5):", [str(r.config) for r in all_results[:5]])
        sys.exit(1)

    clusters = result.clusters
    print(f"Config: {config_str}")
    print(f"Total clusters: {len(clusters)}")

    # Sort clusters by size (largest first — most asked questions first)
    all_ids = sorted(clusters.keys(), key=lambda c: clusters[c]['size'], reverse=True)
    total_volume = sum(clusters[c]['size'] for c in all_ids)

    # --- SUMMARY CSV (one row per cluster) ---
    summary_rows = []
    cumulative = 0
    for rank, cid in enumerate(all_ids, 1):
        cl = clusters[cid]
        uqs = dedup_queries(cl['queries'])
        cumulative += cl['size']
        summary_rows.append({
            'rank':                rank,
            'cluster_id':          cid,
            'n_unique_questions':  len(uqs),
            'total_query_volume':  cl['size'],
            'pct_of_total':        round(cl['size'] / total_volume * 100, 2),
            'cumulative_pct':      round(cumulative / total_volume * 100, 2),
            'representative_question': uqs[0] if uqs else '',
            'all_unique_questions': ' | '.join(uqs),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'best_config_cluster_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV → {summary_path}")
    print(f"  Columns: rank, cluster_id, n_unique_questions, total_query_volume, "
          f"pct_of_total, cumulative_pct, representative_question, all_unique_questions")

    # --- DETAIL CSV (one row per unique question) ---
    detail_rows = []
    for rank, cid in enumerate(all_ids, 1):
        cl = clusters[cid]
        uqs = dedup_queries(cl['queries'])
        for q_rank, q in enumerate(uqs, 1):
            detail_rows.append({
                'cluster_rank':       rank,
                'cluster_id':         cid,
                'cluster_volume':     cl['size'],
                'cluster_pct':        round(cl['size'] / total_volume * 100, 2),
                'question_rank':      q_rank,
                'question':           q,
                'is_representative':  q_rank == 1,
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = OUTPUT_DIR / 'best_config_cluster_mapping.csv'
    detail_df.to_csv(detail_path, index=False)
    print(f"Detail CSV  → {detail_path}")
    print(f"  {len(detail_df):,} total rows ({len(all_ids)} clusters × avg {len(detail_df)/len(all_ids):.1f} unique Qs)")

    # --- Coverage stats ---
    print(f"\nCoverage stats:")
    print(f"  {'Rank':>5}  {'Clusters':>9}  {'Vol':>8}  {'Cum%':>7}  {'Representative question'}")
    print(f"  {'-'*80}")
    highlighted = [1, 5, 10, 25, 52, 100, len(all_ids)]  # 52 = ~80% coverage for winner
    prev_cum = 0
    shown = set()
    cumulative = 0
    for rank, cid in enumerate(all_ids, 1):
        cumulative += clusters[cid]['size']
        cum_pct = cumulative / total_volume * 100
        show = False
        for h in highlighted:
            if rank == h:
                show = True
        # also show the 80%, 90%, 95% threshold crossings
        for thresh in [50, 70, 80, 90, 95]:
            if prev_cum < thresh <= cum_pct and thresh not in shown:
                show = True
                shown.add(thresh)
        if show:
            uqs = dedup_queries(clusters[cid]['queries'])
            rep = (uqs[0][:55] + '…') if uqs and len(uqs[0]) > 55 else (uqs[0] if uqs else '')
            print(f"  {rank:>5}  {rank:>9}  {clusters[cid]['size']:>8,}  {cum_pct:>6.1f}%  {rep}")
        prev_cum = cum_pct

    print(f"\nDone. Open the CSVs in Excel/Calc for manual verification.")
    print(f"  Tip: sort 'best_config_cluster_summary.csv' by 'total_query_volume' to see what farmers ask most.")


if __name__ == '__main__':
    main()
