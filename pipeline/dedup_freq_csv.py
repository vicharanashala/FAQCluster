#!/usr/bin/env python3
"""
dedup_freq_csv.py — Remove duplicate representative questions from a
unique_questions_freq.csv file (or any CSV with a representative_question column).

Comparison is case-insensitive and strip-whitespace normalised.
For each group of duplicates, the row with the highest raw_frequency is kept.

Usage:
    python pipeline/dedup_freq_csv.py --input outputs/repair/maize_makka/unique_questions_freq.csv

    # Write to a different file:
    python pipeline/dedup_freq_csv.py \
        --input outputs/repair/maize_makka/unique_questions_freq.csv \
        --output outputs/repair/maize_makka/faq.csv

    # Drop the rank column (pipeline default):
    python pipeline/dedup_freq_csv.py --input ... --drop-rank

    # Dry-run (report only, do not write):
    python pipeline/dedup_freq_csv.py --input ... --dry-run
"""

import argparse
import pandas as pd
from pathlib import Path


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    norm = df["representative_question"].str.strip().str.lower()
    dup_mask = norm.duplicated(keep=False)
    return df[dup_mask].copy()


def remove_duplicates(df: pd.DataFrame, drop_rank: bool = False) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.copy()
    df["_norm"] = df["representative_question"].str.strip().str.lower()
    df = df.sort_values("raw_frequency", ascending=False)
    df = df.drop_duplicates(subset="_norm", keep="first")
    df = df.drop(columns="_norm")
    df = df.sort_values("raw_frequency", ascending=False).reset_index(drop=True)
    if drop_rank:
        df = df.drop(columns=["rank"], errors="ignore")
    else:
        df["rank"] = range(1, len(df) + 1)
    return df, before - len(df)


def main():
    ap = argparse.ArgumentParser(
        description="Detect and remove duplicate representative questions.")
    # Prefer --input/--output but also accept legacy --file
    ap.add_argument("--input", "--file", dest="input", required=True,
                    help="Path to unique_questions_freq.csv")
    ap.add_argument("--output", default=None,
                    help="Output path (default: overwrite --input)")
    ap.add_argument("--drop-rank", action="store_true",
                    help="Remove the 'rank' column from the output (used by pipeline)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report duplicates without modifying the file")
    args = ap.parse_args()

    fp = Path(args.input)
    out_fp = Path(args.output) if args.output else fp
    if not fp.exists():
        ap.error(f"File not found: {fp}")

    df = pd.read_csv(fp)
    if "representative_question" not in df.columns:
        ap.error("Column 'representative_question' not found in file.")

    dups = find_duplicates(df)

    if dups.empty:
        print(f"✓ No duplicates found in {fp.name} ({len(df)} rows).")
        if args.drop_rank and not args.dry_run:
            df = df.drop(columns=["rank"], errors="ignore")
            df.to_csv(out_fp, index=False)
            print(f"  Dropped 'rank' column and saved to {out_fp}")
        return

    # Group and display duplicates
    dups["_norm"] = dups["representative_question"].str.strip().str.lower()
    print(f"Found {dups['_norm'].nunique()} duplicate group(s) "
          f"({len(dups)} rows affected):\n")
    for norm_q, grp in dups.groupby("_norm"):
        grp_sorted = grp.sort_values("raw_frequency", ascending=False)
        print(f"  Question : \"{grp_sorted['representative_question'].iloc[0]}\"")
        for _, row in grp_sorted.iterrows():
            rank_val = int(row['rank']) if 'rank' in row.index else '?'
            marker = "KEEP" if row["raw_frequency"] == grp_sorted["raw_frequency"].iloc[0] else "DROP"
            print(f"    [{marker}] rank={rank_val}  "
                  f"unique_q_id={row.get('unique_q_id', '?')}  "
                  f"freq={int(row['raw_frequency'])}")
        print()

    if args.dry_run:
        print("Dry-run mode — file not modified.")
        return

    clean_df, n_removed = remove_duplicates(df, drop_rank=args.drop_rank)
    clean_df.to_csv(out_fp, index=False)
    print(f"Removed {n_removed} duplicate row(s). "
          f"{len(df)} → {len(clean_df)} rows. Saved to {out_fp}")


if __name__ == "__main__":
    main()
