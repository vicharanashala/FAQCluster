#!/usr/bin/env python3
"""
filter_faq_corpus.py — Irrelevant-corpus filtering for FAQ output

Removes rows from unique_questions_freq.csv whose representative_question
contains keywords from an irrelevant_corpus.yaml file.

The filtering logic is identical to the one used in kcc-dedup-app:
  - Exact word-level match against all keywords + typos from the YAML
  - Fuzzy match fallback (fuzz.ratio >= 85) when word lengths are close
  - Minimum word length: 3 characters (configurable via YAML min_word_length)

Usage (standalone):
    python pipeline/filter_faq_corpus.py \\
        --input  outputs/repair/maize_makka/unique_questions_freq.csv \\
        --corpus /home/kshitij/Kshitij/kcc-dedup-app/config/irrelevant_corpus.yaml \\
        [--output outputs/repair/maize_makka/unique_questions_freq.csv]  # overwrites by default
        [--dry-run]    # report only, do not write
        [--fuzz-threshold 85]

Output columns: same as input, no rank column.
Removed rows are saved to: <output_dir>/corpus_filtered_out.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from rapidfuzz import fuzz
from tqdm import tqdm


def load_corpus(corpus_file: str) -> tuple[list[str], int]:
    """Load all keywords + typos from the YAML corpus file."""
    with open(corpus_file, 'r') as f:
        data = yaml.safe_load(f)

    keywords = []
    for cat, body in data.get('categories', {}).items():
        keywords.extend(body.get('keywords', []))
        keywords.extend(body.get('typos', []))

    # Lowercase, deduplicate, keep only non-empty strings
    keywords = list(set(
        k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()
    ))
    min_len = int(data.get('min_word_length', 3))
    return keywords, min_len


def is_irrelevant(question: str, keywords: list[str],
                  min_word_len: int, fuzz_thresh: int) -> bool:
    """
    Return True if the question text matches any corpus keyword.

    Matching strategy (same as kcc-dedup-app):
    1. Tokenise the question into words, strip punctuation.
    2. For each word (length >= min_word_len):
       a. Exact match against keyword set.
       b. Fuzzy match (fuzz.ratio >= fuzz_thresh) for keywords with
          similar length (|len_word – len_kw| <= 2).
    Also checks full n-gram phrases (up to 4 words) for multi-word keywords.
    """
    if not question or pd.isna(question):
        return False

    text = str(question).lower()
    punct = '.,!?;:()[]{}"\''

    # ── Multi-word keyword check ───────────────────────────────────────────────
    for kw in keywords:
        if ' ' in kw and kw in text:          # fast substring match for phrases
            return True

    # ── Single-word token check ────────────────────────────────────────────────
    words = text.split()
    keyword_set = set(keywords)

    for word in words:
        word = word.strip(punct)
        if len(word) < min_word_len:
            continue

        # Exact
        if word in keyword_set:
            return True

        # Fuzzy fallback
        for kw in keywords:
            if ' ' in kw:
                continue                       # phrases handled above
            if abs(len(word) - len(kw)) <= 2:
                if fuzz.ratio(word, kw) >= fuzz_thresh:
                    return True

    return False


def filter_faq(input_path: Path, corpus_path: str,
               output_path: Path, fuzz_thresh: int,
               dry_run: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FAQ CSV, apply corpus filter, return (kept_df, removed_df).
    """
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} rows loaded")

    print(f"Loading corpus: {corpus_path} ...")
    keywords, min_word_len = load_corpus(corpus_path)
    print(f"  {len(keywords)} keywords/typos loaded")

    mask_irrelevant = []
    text_cols = ['representative_question', 'cluster_label', 'answer_label']
    available_cols = [c for c in text_cols if c in df.columns]

    if not available_cols:
        sys.exit(f"ERROR: None of {text_cols} found in {input_path}")

    print(f"Checking columns: {', '.join(available_cols)} ...")
    print("\nFiltering ...")

    # For each row, check if ANY of the text columns are irrelevant
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Corpus filter"):
        irrelevant = False
        for col in available_cols:
            if is_irrelevant(row[col], keywords, min_word_len, fuzz_thresh):
                irrelevant = True
                break
        mask_irrelevant.append(irrelevant)

    import numpy as np
    mask = pd.Series(mask_irrelevant, index=df.index)
    removed_df = df[mask].copy()
    kept_df    = df[~mask].copy()

    print(f"\n  Total rows : {len(df)}")
    print(f"  Removed    : {len(removed_df)}  ({len(removed_df)/len(df)*100:.1f}%)")
    print(f"  Kept       : {len(kept_df)}")

    if removed_df.empty:
        print("  ✓ No irrelevant rows found.")
    else:
        print("\n  Removed questions:")
        for q in removed_df[col].tolist():
            print(f"    – {q}")

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kept_df.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved filtered FAQ → {output_path}")

        removed_out = output_path.parent / 'corpus_filtered_out.csv'
        removed_df.to_csv(removed_out, index=False)
        print(f"  ✓ Removed rows saved  → {removed_out}")
    else:
        print("\n  [dry-run] No files written.")

    return kept_df, removed_df


def main():
    ap = argparse.ArgumentParser(
        description="Filter irrelevant questions from FAQ output using an irrelevant_corpus.yaml")
    ap.add_argument('--input',  required=True,
                    help='Path to unique_questions_freq.csv')
    ap.add_argument('--corpus', required=True,
                    help='Path to irrelevant_corpus.yaml')
    ap.add_argument('--output', default=None,
                    help='Output path (default: overwrite --input)')
    ap.add_argument('--fuzz-threshold', type=int, default=100,
                    help='Fuzzy match ratio threshold 0–100 (default: 100 - disabled)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report only, do not write output files')
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    filter_faq(
        input_path  = input_path,
        corpus_path = args.corpus,
        output_path = output_path,
        fuzz_thresh = args.fuzz_threshold,
        dry_run     = args.dry_run,
    )


if __name__ == '__main__':
    main()
