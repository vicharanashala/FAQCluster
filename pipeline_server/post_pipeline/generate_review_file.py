#!/usr/bin/env python3
"""
generate_review_file.py — Build an agri-team review CSV per crop.

For each generated FAQ entry, shows the full set of real farmer questions
that back it (from unique_question_mapping.csv), so reviewers can verify
the generated question actually represents its group.

Output columns:
  rank, Generated_Question, Generated_Category, Times_Asked,
  Num_Farmer_Questions, Farmer_Questions (one per line), Generated_Answer

Usage:
    python post_pipeline/generate_review_file.py \\
        --qa-csv  outputs/repair/kerala/coconut/unique_questions_freq_qa.csv \\
        --map-csv outputs/repair/kerala/coconut/unique_question_mapping.csv \\
        --output  outputs/repair/kerala/coconut/kerala_coconut_review.csv

Can also be called via generate_review(qa_csv, map_csv, output_csv).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def generate_review(qa_csv: Path, map_csv: Path, output_csv: Path) -> Path:
    """
    Join FAQ Q&A output with the question mapping to produce a review file.

    Args:
        qa_csv:     Path to unique_questions_freq_qa.csv (or final deduped csv).
                    Must have: unique_q_id, Generated_Question, Generated_Category,
                               Generated_Answer, raw_frequency, n_questions_in_group.
        map_csv:    Path to unique_question_mapping.csv.
                    Must have: unique_q_id, question.
        output_csv: Where to write the review file.

    Returns:
        output_csv path.
    """
    qa = pd.read_csv(qa_csv, low_memory=False)
    mapping = pd.read_csv(map_csv, low_memory=False)

    # Drop rows with no generated question (PARSE_ERROR / IRRELEVANT_CROP)
    qa = qa[qa['Generated_Question'].notna() & (qa['Generated_Question'].str.strip() != '')]
    qa = qa[~qa.get('Generated_Category', pd.Series(dtype=str)).isin({'PARSE_ERROR', 'IRRELEVANT_CROP'})]

    # Aggregate all farmer questions per unique_q_id from the mapping
    mapping['question'] = mapping['question'].astype(str).str.strip()
    # Sort by frequency descending so most common questions appear first
    if 'raw_freq_individual' in mapping.columns:
        mapping = mapping.sort_values('raw_freq_individual', ascending=False)

    all_questions = (
        mapping.groupby('unique_q_id')['question']
        .apply(lambda qs: '\n'.join(f'{i+1}. {q}' for i, q in enumerate(qs)))
        .reset_index()
        .rename(columns={'question': 'Farmer_Questions'})
    )

    # Merge into QA df
    merged = qa.merge(all_questions, on='unique_q_id', how='left')

    # Fall back to sample_questions if mapping join produced nothing
    if 'sample_questions' in merged.columns:
        missing_mask = merged['Farmer_Questions'].isna() | (merged['Farmer_Questions'] == '')
        merged.loc[missing_mask, 'Farmer_Questions'] = (
            merged.loc[missing_mask, 'sample_questions']
            .fillna('')
            .str.split('|')
            .apply(lambda qs: '\n'.join(f'{i+1}. {q.strip()}' for i, q in enumerate(qs) if q.strip()))
        )

    # Rename / select output columns
    freq_col = 'raw_frequency'
    n_col    = 'n_questions_in_group'

    col_map = {
        freq_col: 'Times_Asked',
        n_col:    'Num_Farmer_Questions',
    }
    merged = merged.rename(columns={k: v for k, v in col_map.items() if k in merged.columns})

    out_cols = [
        'rank' if 'rank' in merged.columns else None,
        'Generated_Question',
        'Generated_Category',
        'Times_Asked'         if 'Times_Asked'         in merged.columns else None,
        'Num_Farmer_Questions' if 'Num_Farmer_Questions' in merged.columns else None,
        'Farmer_Questions',
        'Generated_Answer',
    ]
    out_cols = [c for c in out_cols if c]

    # Re-rank by Times_Asked if rank column is missing or stale
    if 'Times_Asked' in merged.columns:
        merged = merged.sort_values('Times_Asked', ascending=False).reset_index(drop=True)
        merged['rank'] = range(1, len(merged) + 1)
        if 'rank' not in out_cols:
            out_cols.insert(0, 'rank')

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[out_cols].to_csv(output_csv, index=False)
    print(f"  [review] → {output_csv}  ({len(merged)} rows)")
    return output_csv


def main():
    ap = argparse.ArgumentParser(description="Generate agri-team review CSV from FAQ output.")
    ap.add_argument('--qa-csv',  required=True, help='Path to unique_questions_freq_qa.csv (or final deduped csv)')
    ap.add_argument('--map-csv', required=True, help='Path to unique_question_mapping.csv')
    ap.add_argument('--output',  required=True, help='Output review CSV path')
    args = ap.parse_args()

    generate_review(Path(args.qa_csv), Path(args.map_csv), Path(args.output))


if __name__ == '__main__':
    main()
