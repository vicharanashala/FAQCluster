import os
import requests
import pandas as pd
import re
import json
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from _job_ctl import JobCancelled as _JobCancelled
except ImportError:
    class _JobCancelled(BaseException):  # type: ignore[no-redef]
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass

_API_URL            = os.environ.get("LLM_API_URL",   "http://100.100.108.44:8013/v1/chat/completions")
_API_MODEL          = os.environ.get("LLM_MODEL",     "google/gemma-4-26B-A4B-it")
_API_KEY            = os.environ.get("LLM_API_KEY",   "")
_THINKING_ENABLED   = os.environ.get("LLM_THINKING_ENABLED", "false").lower() == "true"
_CATEGORY_WORKERS   = int(os.environ.get("LLM_CLUSTER_WORKERS", "4"))
_BATCH_WORKERS      = int(os.environ.get("LLM_BATCH_WORKERS",  "4"))
_EMBED_DEDUP_THRESH = float(os.environ.get("LLM_EMBED_DEDUP_THRESH", "0.95"))

crops_folder = Path('../outputs/repair/final')


def llm_completion(prompt, max_tokens=200, temperature=0.0, top_p=0.95, stop=None):
    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    data = {
        "model": _API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **( {} if _THINKING_ENABLED else {"thinking": {"type": "disabled"}} ),
    }
    if stop is not None:
        data["stop"] = stop

    response = requests.post(_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        msg = response.json()['choices'][0]['message']
        return msg.get('content') or msg.get('reasoning') or msg.get('reasoning_content') or ""
    else:
        raise RuntimeError(f"Failed to fetch completion: {response.status_code} - {response.text}")


def _process_category(category, cat_df, text_col, batch_size, cancel_event=None):
    """Process one category end-to-end. Returns (final_rows, phase_rows)."""
    tqdm.write(f"\n{'='*70}\n🚀 STARTING CATEGORY: {category}\n{'='*70}")
    tqdm.write(f"[INFO] Total rows to process in '{category}': {len(cat_df)}")

    phase_data = []
    final_cleaned_data = []

    while not cat_df.empty:
        if cancel_event and cancel_event.is_set():
            raise _JobCancelled("post-processing stopped by user")
        ref_phase = []
        reference_row = cat_df.iloc[0].copy()
        ref_phase.append({'representative_question': reference_row['representative_question']})
        ref_id       = reference_row['unique_q_id']
        ref_question = reference_row[text_col]
        current_freq = reference_row['raw_frequency']

        remaining_df = cat_df.iloc[1:].reset_index(drop=True)

        if remaining_df.empty:
            tqdm.write(f"\n[INFO] Only 1 row left in category. Saving Reference [{ref_id}] directly.")
            final_cleaned_data.append(reference_row.to_dict())
            break

        tqdm.write(f"\n🔍 Evaluating Reference [{ref_id}]: {ref_question} (Current Freq: {current_freq})")
        tqdm.write(f"   Candidates remaining in pool: {len(remaining_df)}")

        matched_ids, ref_phase = _get_verified_matches(
            reference_row, remaining_df, text_col, batch_size, ref_phase, cancel_event=cancel_event
        )

        if matched_ids:
            matches_df = remaining_df[remaining_df['unique_q_id'].isin(matched_ids)]

            ref_phase = pd.concat([
                ref_phase,
                matches_df[['representative_question', 'unique_q_id']].rename(columns={
                    'representative_question': 'phase_2',
                    'unique_q_id':             'phase_2_id',
                }).reset_index(drop=True),
            ], axis=1)

            phase_2_ids  = set(ref_phase['phase_2_id'].dropna())
            removed_mask = ~ref_phase['phase_1_id'].isin(phase_2_ids)
            ref_phase['false_positive'] = ref_phase.loc[removed_mask, 'phase_1']
            false_positives = ref_phase['false_positive'].dropna()
            ref_phase['false_positive'] = false_positives
            ref_phase.drop(columns=['phase_1_id', 'phase_2_id'], inplace=True)
            ref_phase = ref_phase[['representative_question', 'phase_1', 'false_positive', 'phase_2']]
            ref_phase = pd.concat([ref_phase, pd.DataFrame([{}])], ignore_index=True)

            phase_data.append(ref_phase)

            summed_frequency = matches_df['raw_frequency'].sum()
            tqdm.write(f"   ✅ SUCCESS: Found {len(matched_ids)} verified matches.")
            for _, m_row in matches_df.iterrows():
                tqdm.write(f"      -> Matched [{m_row['unique_q_id']}]: {m_row[text_col]} (Freq: {m_row['raw_frequency']})")
            tqdm.write(f"   📈 Aggregating Frequencies: {current_freq} + {summed_frequency} = {current_freq + summed_frequency}")

            reference_row['raw_frequency'] += summed_frequency
            cat_df = remaining_df[~remaining_df['unique_q_id'].isin(matched_ids)].reset_index(drop=True)
            tqdm.write(f"   🗑️  Removed {len(matched_ids)} matches from the pool. New pool size: {len(cat_df)}")
            final_cleaned_data.append(reference_row.to_dict())

        else:
            tqdm.write("   ❌ No matches found. Moving to next row.")
            cat_df = remaining_df
            final_cleaned_data.append(reference_row.to_dict())

    return final_cleaned_data, phase_data


def embedding_prepass(df, text_col='Generated_Question', sim_thresh=_EMBED_DEDUP_THRESH):
    """
    Within each Generated_Category, auto-merge semantically near-identical questions
    (cosine >= sim_thresh) using sentence embeddings — before the LLM pass runs.

    Higher raw_frequency row survives; its frequency is incremented by the absorbed row's.
    This catches cases where Q&A generation turned short/typo original questions into
    different-looking English sentences that are semantically identical.
    """
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        tqdm.write("  [embed-dedup] sentence_transformers not available — skipping pre-pass")
        return df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"  [embed-dedup] Loading sentence transformer on {device}...")
    st_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=device,
    )

    all_kept = []
    total_merged = 0

    for cat in df["Generated_Category"].unique():
        cat_df = df[df["Generated_Category"] == cat].copy().reset_index(drop=True)
        n = len(cat_df)
        if n <= 1:
            all_kept.append(cat_df)
            continue

        texts = cat_df[text_col].tolist()
        embs  = st_model.encode(texts, batch_size=64, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        sim   = embs @ embs.T

        # Process in descending frequency order so the most-asked question survives
        order    = cat_df["raw_frequency"].argsort()[::-1].tolist()
        absorbed = set()

        for rank_i, i in enumerate(order):
            if i in absorbed:
                continue
            for j in order[rank_i + 1:]:
                if j in absorbed:
                    continue
                if sim[i, j] >= sim_thresh:
                    cat_df.at[i, "raw_frequency"] += cat_df.at[j, "raw_frequency"]
                    absorbed.add(j)
                    total_merged += 1
                    tqdm.write(
                        f"  [embed-dedup] [{cat}] MERGE "
                        f"'{cat_df.at[j, text_col][:70]}' → "
                        f"'{cat_df.at[i, text_col][:70]}' "
                        f"(sim={sim[i, j]:.3f})"
                    )

        all_kept.append(cat_df[~cat_df.index.isin(absorbed)].copy())

    result = pd.concat(all_kept, ignore_index=True)
    tqdm.write(f"  [embed-dedup] {len(df)} → {len(result)} rows ({total_merged} auto-merged)")
    return result


def deduplicate_and_aggregate(df, text_col='Generated_Question', batch_size=100, cancel_event=None):
    """
    Iterates through categories in parallel, finds similar questions using LLM,
    aggregates 'raw_frequency', and removes duplicates with full logging.
    """
    df['raw_frequency'] = pd.to_numeric(df['raw_frequency'], errors='coerce').fillna(0)

    # Drop rows where QA generation failed or was out of scope before dedup
    n_before = len(df)
    bad_cats = {'PARSE_ERROR', 'IRRELEVANT_CROP'}
    df = df[~df['Generated_Category'].isin(bad_cats)].copy()
    df = df[df[text_col].notna() & (df[text_col].str.strip() != '')].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        tqdm.write(f"  Filtered {n_dropped} rows (PARSE_ERROR / IRRELEVANT_CROP / empty question) before dedup")

    tqdm.write("\n[Embedding pre-pass] Auto-merging near-identical questions by embedding similarity...")
    df = embedding_prepass(df, text_col=text_col)

    categories = df['Generated_Category'].unique()
    n_workers  = min(len(categories), _CATEGORY_WORKERS)
    tqdm.write(f"Processing {len(categories)} categories with {n_workers} parallel workers "
               f"(LLM_CLUSTER_WORKERS={_CATEGORY_WORKERS})...")

    all_final = []
    all_phase = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _process_category,
                cat,
                df[df['Generated_Category'] == cat].copy(),
                text_col,
                batch_size,
                cancel_event,
            ): cat
            for cat in categories
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Categories"):
            cat = futures[future]
            try:
                final_rows, phase_rows = future.result()
                all_final.extend(final_rows)
                all_phase.extend(phase_rows)
            except Exception as e:
                tqdm.write(f"[ERROR] Category '{cat}' failed: {e}")
            # _JobCancelled (BaseException) propagates naturally, bypassing except Exception

    tqdm.write(f"\n{'='*70}\n🎉 PROCESSING COMPLETE\n{'='*70}")
    tqdm.write(f"Original Dataset Size: {len(df)}")
    tqdm.write(f"Cleaned Dataset Size:  {len(all_final)}")

    phase_df = pd.concat(all_phase, ignore_index=True) if all_phase else pd.DataFrame()
    return pd.DataFrame(all_final), phase_df


def _get_verified_matches(reference_row, candidate_df, text_col, batch_size, ref_phase, cancel_event=None):
    """
    2-pass LLM matching. Pass 1 batches fire in parallel; Pass 2 is a single
    verification call over the candidates flagged by Pass 1.
    """
    original_id       = reference_row['unique_q_id']
    original_question = reference_row[text_col]
    batch_starts      = list(range(0, len(candidate_df), batch_size))

    tqdm.write("   [Pass 1] Scanning for initial candidates...")

    def _run_batch(start):
        if cancel_event and cancel_event.is_set():
            return []
        batch      = candidate_df.iloc[start:start + batch_size]
        batch_text = "\n".join(
            f"{row['unique_q_id']}: {row[text_col]}" for _, row in batch.iterrows()
        )
        prompt = (
            f"You are given one reference question and a list of candidate questions.\n"
            f"Task: Return ONLY the IDs of questions that are the same question as the reference "
            f"— same topic, same type of information requested, and same level of specificity.\n"
            f"A candidate is a match ONLY if a farmer asking the reference question and a farmer "
            f"asking the candidate question would both be satisfied by the exact same answer.\n"
            f"DO NOT flag a candidate if it differs in any of these ways:\n"
            f"- Adds or removes a qualifier (season, crop stage, growth stage, region)\n"
            f"- Asks about a specific chemical/variety/pest while the reference is general, or vice versa\n"
            f"- Asks for a different type of information (e.g. identification vs treatment vs dosage vs timing)\n"
            f"Rules:\n- Output ONLY a JSON list of matching IDs\n"
            f"- No explanation\n- If none match, return []\n\n"
            f"Reference:\n{original_id}: {original_question}\n\n"
            f"Candidates:\n{batch_text}\n\nOutput:"
        )
        response = llm_completion(prompt, max_tokens=500, temperature=0.0)
        m = re.search(r'\[.*\]', response, re.DOTALL)
        if m:
            try:
                found = json.loads(m.group(0))
                if found:
                    tqdm.write(f"      -> Batch flagged {len(found)} potential IDs: {found}")
                return found
            except json.JSONDecodeError:
                tqdm.write("      -> [WARNING] JSON decoding failed in Pass 1 for a batch.")
        return []

    all_candidate_ids = []
    n_workers = min(len(batch_starts), _BATCH_WORKERS)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_batch, s) for s in batch_starts]
        for f in as_completed(futures):
            all_candidate_ids.extend(f.result())

    ref_phase = pd.DataFrame(ref_phase)

    if not all_candidate_ids:
        return [], ref_phase

    if cancel_event and cancel_event.is_set():
        raise _JobCancelled("post-processing stopped by user")

    # --- Pass 2: Verification ---
    tqdm.write(f"   [Pass 2] Verifying {len(all_candidate_ids)} potential candidates...")
    candidate_rows = candidate_df[candidate_df['unique_q_id'].isin(all_candidate_ids)]

    ref_phase = pd.concat([
        ref_phase,
        candidate_rows[['representative_question', 'unique_q_id']].rename(columns={
            'representative_question': 'phase_1',
            'unique_q_id':             'phase_1_id',
        }).reset_index(drop=True),
    ], axis=1)

    verification_text = "\n".join(
        f"{row['unique_q_id']}: {row[text_col]}" for _, row in candidate_rows.iterrows()
    )
    verification_prompt = (
        f"Task: Strictly verify these candidates against the reference. Keep a candidate ONLY "
        f"if it is asking the exact same question with the same scope and specificity — "
        f"just worded differently.\n"
        f"Remove a candidate if it differs in ANY of the following:\n"
        f"- The season, crop stage, or growth stage (e.g. 'summer' vs 'Rabi', 'seedling' vs 'flowering')\n"
        f"- The level of specificity (one names a specific variety/chemical/pest, the other does not)\n"
        f"- The type of information asked (identification/symptoms vs treatment vs dosage vs timing vs selection)\n"
        f"Rules: Output ONLY a JSON list of matching IDs. No explanation. "
        f"If none match, return [].\n\n"
        f"Reference:\n{original_id}: {original_question}\n\n"
        f"Candidates to Verify:\n{verification_text}\n\nOutput:"
    )
    verification_response = llm_completion(verification_prompt, max_tokens=500, temperature=0.0)

    final_matched_ids = []
    final_match = re.search(r'\[.*\]', verification_response, re.DOTALL)
    if final_match:
        try:
            final_matched_ids = json.loads(final_match.group(0))
        except json.JSONDecodeError:
            tqdm.write("      -> [WARNING] JSON decoding failed in Pass 2.")

    final_matched_ids = [mid for mid in final_matched_ids if mid in all_candidate_ids]
    return final_matched_ids, ref_phase


if __name__ == '__main__':
    for csv_file in crops_folder.glob("*.csv"):
        print(f"Processing: {csv_file.name}")

        output_name_phase = f"phase_data_{csv_file.name}"
        output_name_final = f"dedup_{csv_file.name}"

        output_path_phase = crops_folder / output_name_phase
        output_path_final = crops_folder / output_name_final

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            df, df1 = deduplicate_and_aggregate(df)
        except Exception as e:
            print(f"[ERROR] Failed processing {csv_file.name}: {e}")
            continue

        df1.to_csv(output_path_phase, index=False)
        df.to_csv(output_path_final, index=False)

        print(f"Saved phase output: {output_name_phase}")
        print(f"Saved final output: {output_name_final}")
