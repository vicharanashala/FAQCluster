#!/usr/bin/env python3
"""
cluster_repair.py — Self-contained LLM-based post-clustering repair pipeline.

Modes:
  --mode phase1   Run clustering Phase 1 (fast screening) on a raw CSV.
  --mode phase2   Run LLM evaluation on Phase 1 results to pick best config.
  --mode repair   Run the full repair (B-E) on an existing Phase 1 pickle.
  --mode full     Run Phase 1 → Phase 2 → Repair end-to-end (default).

Repair Steps:
  A. Max-diversity rep selection  (embedding-based greedy, no LLM)
  B. Cross-crop contamination filter   (LLM JSON, all clusters)
  C. Coherence diagnostic + split      (LLM A/B/C + JSON, flagged clusters)
  D. Merge near-duplicate clusters     (LLM A/B, embedding-candidate pairs)
  E. Raw row back-mapping              (join with original raw CSV)

Outputs  (outputs/repair/<slug>/):
  phase1_results.pkl         Clustering results for all Phase 1 configs
  phase2_scores.csv          LLM composite scores per config
  repaired_clusters.csv      One row per final repaired cluster
  raw_row_mapping.csv        One row per raw CSV row with cluster assignment

Usage:
  python analysis/cluster_repair.py \\
      --raw-file data/raw/punjab_maize_raw.csv \\
      --crop "Maize Makka" \\
      --model /home/kshitij/models/qwen2.5-7b-instruct \\
      --gpu-id 0 \\
      --mode full
"""

import sys, re, json, pickle, argparse, copy
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPAIR_BASE  = PROJECT_ROOT / 'outputs' / 'repair'

sys.path.insert(0, str(SCRIPT_DIR))
# Import needed for pickle deserialization of ClusteringResult
from hyperparameter_tuning import (                        # noqa: F401
    ClusteringResult, ClusteringConfig,
    run_clustering, phase1_fast_screening,
    generate_param_grid, load_stopwords
)
from llm_evaluator_hf import LocalHFJudge, evaluate_config_with_hf  # noqa: F401


# ══════════════════════════════════════════════════════════════════════════════
# A. Max-diversity representative selection (pure embedding math, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _centroid_nearest(embs: np.ndarray) -> int:
    """Index of the point whose embedding is closest to the cluster centroid."""
    if len(embs) == 1:
        return 0
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embs / norms
    centroid = normed.mean(axis=0)
    return int(np.argmax(normed @ centroid))


def greedy_diverse_indices(embs: np.ndarray, k: int) -> list:
    """
    Greedy furthest-point selection from a set of embeddings.
    Returns list of k indices.
      index[0] = centroid-nearest (most typical / new representative)
      index[1..k-1] = maximally diverse from each other
    """
    n = len(embs)
    k = min(k, n)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embs / norms

    seed = _centroid_nearest(embs)
    selected = [seed]
    min_cos_dist = np.full(n, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        d = 1.0 - (normed @ normed[last])          # cosine distance to 'last'
        min_cos_dist = np.minimum(min_cos_dist, d)
        min_cos_dist[selected] = -np.inf            # exclude already-selected
        selected.append(int(np.argmax(min_cos_dist)))

    return selected


def _elect_rep(queries: list, q2idx: dict, all_embs: np.ndarray) -> str:
    """Return the centroid-nearest question from a list, using precomputed embeddings."""
    idxs = [q2idx[q] for q in queries if q in q2idx]
    if not idxs:
        return queries[0] if queries else ""
    best = _centroid_nearest(all_embs[idxs])
    return queries[best]


def step_a_diverse_reps(clusters: dict, result_df: pd.DataFrame,
                        all_embs: np.ndarray, k: int = 3):
    """
    For every cluster:
      - Compute k max-diverse representative questions.
      - Update clusters[cid]['representative'] to centroid-nearest (diverse[0]).
    Returns (clusters, diverse_reps_dict).
    """
    q2idx = {q: i for i, q in enumerate(result_df['query_text'].tolist())}
    diverse = {}

    for cid, data in clusters.items():
        queries  = data['queries']
        idx_list = [q2idx[q] for q in queries if q in q2idx]
        if not idx_list:
            diverse[cid] = queries[:k]
            continue
        embs = all_embs[idx_list]
        sel  = greedy_diverse_indices(embs, k)
        diverse[cid] = [queries[i] for i in sel if i < len(queries)]
        # Update representative: centroid-nearest is index 0
        clusters[cid]['representative'] = diverse[cid][0]

    return clusters, diverse


# ══════════════════════════════════════════════════════════════════════════════
# Extended LLM judge — adds JSON-output methods for repair operations
# ══════════════════════════════════════════════════════════════════════════════

class RepairJudge(LocalHFJudge):

    def _gen_long(self, user_text: str, max_new_tokens: int = 300) -> str:
        """Generate a longer response (for JSON outputs)."""
        import torch
        prompt = self._build_prompt(user_text)
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=4096).to(self.device)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_json_list(raw: str) -> list:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return []

    @staticmethod
    def _parse_json_obj(raw: str) -> dict:
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}

    # ── B: Cross-crop contamination ───────────────────────────────────────────

    def cross_crop_filter(self, queries: list, crop: str) -> list:
        """Returns 0-based indices of queries that are NOT about `crop`."""
        qstr = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
        prompt = (
            f"Target crop: {crop}.\n"
            f"Identify queries that clearly ask about a DIFFERENT crop (not {crop}).\n\n"
            f"{qstr}\n\n"
            f'Return ONLY JSON: {{"off": []}} where "off" is a list of 1-based '
            f"indices of queries NOT about {crop}. Empty list if all are about {crop}.\nJSON:"
        )
        raw  = self._gen_long(prompt, max_new_tokens=80)
        data = self._parse_json_obj(raw)
        off  = data.get("off", [])
        return [int(i) - 1 for i in off if 1 <= int(i) <= len(queries)]

    # ── C: Coherence diagnostic ───────────────────────────────────────────────

    def coherence_diagnostic(self, diverse_reps: list, crop: str) -> str:
        """
        Fast coherence check using max-diverse reps as diagnostic.
        Returns 'A' (coherent), 'B' (mostly), 'C' (needs split).
        """
        qstr = "\n".join(f"{i+1}. {q}" for i, q in enumerate(diverse_reps))
        prompt = (
            f"Crop: {crop}.\n"
            f"These {len(diverse_reps)} questions represent the RANGE of queries in "
            f"one FAQ cluster:\n{qstr}\n\n"
            f"Would an agricultural extension officer give the SAME specific practical "
            f"advice (same chemical, same method, same treatment) for ALL of them?\n"
            f"A) YES — identical advice for all\n"
            f"B) MOSTLY — very similar, minor variations only\n"
            f"C) NO — clearly different advice needed for different questions\n\n"
            f"Answer (A/B/C):"
        )
        return self._parse_abc(self._generate_one(prompt), default="A")

    # ── C: Split ─────────────────────────────────────────────────────────────

    def split_cluster(self, queries: list, crop: str) -> list:
        """
        Groups queries into sub-clusters that each require the same advice.
        Returns [{'label': str, 'indices': [0-based int]}].
        A single-group result means no split is needed.
        """
        qstr = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
        prompt = (
            f"You are an agricultural extension officer for {crop}.\n"
            f"Group these farmer queries so that ALL queries in each group need "
            f"the SAME specific practical advice.\n\n"
            f"Rules:\n"
            f"- Only split if groups need genuinely DIFFERENT treatments\n"
            f"- No singleton groups — merge lone questions into the closest group\n"
            f"- If all queries need the same advice, return one group with all indices\n"
            f'- Each "label" must be a 2-4 word topic name (e.g. "stem borer control", '
            f'"nutrient deficiency", "weed management") — never use the words '
            f'"short description" or "group"\n\n'
            f"Queries:\n{qstr}\n\n"
            f"Return ONLY valid JSON array, no text before or after:\n"
            f'[{{"group":1,"label":"stem borer control","indices":[1,2,3]}},'
            f'{{"group":2,"label":"nutrient deficiency","indices":[4,5]}}]\nJSON:'
        )
        raw       = self._gen_long(prompt, max_new_tokens=350)
        groups_raw = self._parse_json_list(raw)

        result, seen = [], set()
        for g in groups_raw:
            idxs = [int(i) - 1 for i in g.get("indices", [])
                    if 1 <= int(i) <= len(queries) and (int(i) - 1) not in seen]
            seen.update(idxs)
            if idxs:
                result.append({"label": str(g.get("label", ""))[:100], "indices": idxs})

        # Assign any untracked indices to the largest group
        unassigned = [i for i in range(len(queries)) if i not in seen]
        if unassigned and result:
            max(result, key=lambda x: len(x["indices"]))["indices"].extend(unassigned)

        return result or [{"label": queries[0][:100], "indices": list(range(len(queries)))}]

    # ── D: Merge check ────────────────────────────────────────────────────────

    def should_merge(self, rep_a: str, rep_b: str, crop: str) -> bool:
        """True if both cluster representatives would receive the same agricultural advice."""
        prompt = (
            f"Crop: {crop}.\n\n"
            f"Cluster 1 representative question: {rep_a}\n"
            f"Cluster 2 representative question: {rep_b}\n\n"
            f"Would an agricultural extension officer give the SAME core practical "
            f"advice (same recommendation / treatment) for both questions?\n"
            f"A) YES — same advice, these clusters should be merged\n"
            f"B) NO  — different advice needed, keep them separate\n\n"
            f"Answer (A/B):"
        )
        return self._parse_abc(self._generate_one(prompt), default="B") == "A"


# ══════════════════════════════════════════════════════════════════════════════
# Repair steps B – E
# ══════════════════════════════════════════════════════════════════════════════

def step_b_cross_crop(clusters: dict, judge: RepairJudge, crop: str) -> dict:
    print(f"\n{'─'*60}\nStep B: Cross-crop contamination filter\n{'─'*60}")
    total_removed = 0

    for cid in tqdm(list(clusters.keys()), desc="B: cross-crop"):
        queries = clusters[cid]['queries']
        counts  = clusters[cid]['counts']
        off     = judge.cross_crop_filter(queries, crop)
        if not off:
            continue

        off_set  = set(off)
        off_qs   = [queries[i] for i in off]
        frac     = len(off) / len(queries)
        print(f"  Cluster {cid}: {len(off)}/{len(queries)} off-topic ({frac*100:.0f}%)")
        for q in off_qs[:3]:
            print(f"    - {q[:80]}")

        keep = [i for i in range(len(queries)) if i not in off_set]
        if len(keep) >= 2:
            clusters[cid]['queries'] = [queries[i] for i in keep]
            clusters[cid]['counts']  = [counts[i]  for i in keep]
            clusters[cid]['size']    = sum(clusters[cid]['counts'])
        elif keep == 1:
            # Cluster shrank to 1: mark for deletion
            clusters[cid]['_delete'] = True

        total_removed += len(off)

    # Remove empty/singleton clusters
    to_del = [c for c in clusters if clusters[c].get('_delete')]
    for c in to_del:
        del clusters[c]

    print(f"  Removed {total_removed} off-topic questions total")
    return clusters


def step_c_split(clusters: dict, diverse_reps: dict, result_df: pd.DataFrame,
                 all_embs: np.ndarray, judge: RepairJudge, crop: str,
                 flag_on: str = "C") -> tuple:
    """
    For each cluster rated 'flag_on' by coherence diagnostic:
    run a full split prompt and create sub-clusters.
    Returns (updated_clusters, n_splits_performed).
    """
    print(f"\n{'─'*60}\nStep C: Coherence diagnostic + split  (flag on: {flag_on})\n{'─'*60}")
    q2idx   = {q: i for i, q in enumerate(result_df['query_text'].tolist())}
    max_cid = max(clusters.keys())
    n_splits = 0

    for cid in tqdm(list(clusters.keys()), desc="C: coherence"):
        reps = diverse_reps.get(cid, [])
        if len(reps) < 2:
            continue
        rating = judge.coherence_diagnostic(reps, crop)
        if rating != flag_on:
            continue   # A or B → keep as is

        queries = clusters[cid]['queries']
        counts  = clusters[cid]['counts']
        print(f"\n  Cluster {cid} ({len(queries)} Qs) rated '{rating}': {reps}")

        groups = judge.split_cluster(queries, crop)

        if len(groups) <= 1:
            print("    → LLM returned a single group — no split applied")
            continue

        # Enforce minimum sub-cluster size of 2
        small, large = [], []
        for g in groups:
            (large if len(g['indices']) >= 2 else small).append(g)
        if small:
            if large:
                max(large, key=lambda x: len(x['indices']))['indices'].extend(
                    i for g in small for i in g['indices'])
            else:
                print("    → All sub-groups too small — no split applied")
                continue
            groups = large
            if len(groups) <= 1:
                print("    → After merging small groups: single group — no split")
                continue

        print(f"    → Split into {len(groups)} sub-clusters:")
        sub_ids = []
        for g in groups:
            sub_q = [queries[i] for i in g['indices']]
            sub_c = [counts[i]  for i in g['indices']]
            max_cid += 1
            rep = _elect_rep(sub_q, q2idx, all_embs)
            clusters[max_cid] = {
                'queries':        sub_q,
                'counts':         sub_c,
                'size':           sum(sub_c),
                'unique_queries': len(sub_q),
                'representative': rep,
                'split_label':    g['label'],
                'parent_cluster': cid,
            }
            sub_ids.append(max_cid)
            print(f"      → {max_cid}: '{g['label']}' ({len(sub_q)} Qs, vol={sum(sub_c)})")

        del clusters[cid]
        n_splits += 1

    print(f"\n  Split {n_splits} clusters into sub-clusters")
    return clusters, n_splits


def step_d_merge(clusters: dict, st_model, judge: RepairJudge,
                 crop: str, sim_thresh: float = 0.82, max_pairs: int = 100) -> tuple:
    """
    Find candidate merge pairs via cosine similarity of cluster representatives,
    then LLM-confirm each pair. Absorbs smaller cluster into larger.
    Returns (updated_clusters, n_merges_performed).
    """
    print(f"\n{'─'*60}\nStep D: Merge near-duplicate clusters  (sim ≥ {sim_thresh})\n{'─'*60}")
    active_cids = list(clusters.keys())
    reps        = [clusters[c]['representative'] for c in active_cids]

    print(f"  Encoding {len(reps)} cluster representatives...")
    rep_embs = st_model.encode(
        reps, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    sim = rep_embs @ rep_embs.T  # cosine similarity matrix (normalized embeddings)

    # Collect candidate pairs above threshold, sorted by descending similarity
    cand = sorted(
        [(i, j, float(sim[i, j]))
         for i in range(len(active_cids))
         for j in range(i + 1, len(active_cids))
         if sim[i, j] >= sim_thresh],
        key=lambda x: -x[2],
    )[:max_pairs]

    print(f"  Candidate pairs: {len(cand)}")

    n_merged   = 0
    merged_away = set()

    for i, j, s in tqdm(cand, desc="D: merge check"):
        ca, cb = active_cids[i], active_cids[j]
        if ca in merged_away or cb in merged_away:
            continue
        if ca not in clusters or cb not in clusters:
            continue

        rep_a = clusters[ca]['representative']
        rep_b = clusters[cb]['representative']

        if not judge.should_merge(rep_a, rep_b, crop):
            continue

        # Keep larger; absorb smaller
        survivor = ca if clusters[ca]['size'] >= clusters[cb]['size'] else cb
        absorbed = cb if survivor == ca else ca

        print(f"  MERGE {absorbed} → {survivor}")
        print(f"    '{rep_a[:55]}' ↔ '{rep_b[:55]}'  (sim={s:.3f})")

        clusters[survivor]['queries'].extend(clusters[absorbed]['queries'])
        clusters[survivor]['counts'].extend(clusters[absorbed]['counts'])
        clusters[survivor]['size']           += clusters[absorbed]['size']
        clusters[survivor]['unique_queries'] += clusters[absorbed]['unique_queries']
        clusters[survivor].setdefault('merged_from', []).append(absorbed)
        del clusters[absorbed]
        merged_away.add(absorbed)
        n_merged += 1

    print(f"  Merged {n_merged} cluster pairs")
    return clusters, n_merged


def step_e_raw_mapping(clusters: dict, result_df: pd.DataFrame,
                       raw_csv: Path, crop: str,
                       n_original: int, out_dir: Path) -> None:
    """
    Build query_text → final_cluster_id mapping from repaired clusters,
    then join with the original raw CSV rows.
    Saves repaired_clusters.csv and raw_row_mapping.csv to out_dir.
    """
    print(f"\n{'─'*60}\nStep E: Raw row back-mapping\n{'─'*60}")

    # Build reverse mapping: query_text → final cluster_id
    q2cid = {q: cid for cid, data in clusters.items() for q in data['queries']}

    # Load raw CSV
    print(f"  Loading {raw_csv} ...")
    raw = pd.read_csv(raw_csv, low_memory=False)
    if 'query_text' not in raw.columns:
        raw['query_text'] = raw['QueryText'] if 'QueryText' in raw.columns else raw.iloc[:, 8]
    raw['raw_row_id'] = range(len(raw))

    # Filter to target crop (flexible: strip parentheses for comparison)
    raw_crop_norm = raw['Crop'].str.replace(r'[\(\)]', '', regex=True).str.strip()
    crop_norm     = re.sub(r'[\(\)]', '', crop).strip()
    raw = raw[raw_crop_norm == crop_norm].copy()
    print(f"  Rows after crop filter: {len(raw)}")

    # Map each raw row to its final cluster
    raw['final_cluster_id'] = raw['query_text'].map(q2cid)

    # Build label and representative lookup dicts
    cluster_labels = {
        cid: data.get('split_label', data['representative'])[:120]
        for cid, data in clusters.items()
    }
    rep_set = {data['representative'] for data in clusters.values()}

    raw['cluster_label']     = raw['final_cluster_id'].map(cluster_labels)
    raw['is_representative'] = raw['query_text'].isin(rep_set)

    mapped   = raw['final_cluster_id'].notna().sum()
    unmapped = len(raw) - mapped
    print(f"  Mapped: {mapped}  |  Unmapped (noise): {unmapped}")

    # Save raw mapping
    raw_out = out_dir / 'raw_row_mapping.csv'
    raw.to_csv(raw_out, index=False)
    print(f"  → {raw_out}")

    # Build repaired cluster summary
    total_vol = max(sum(d['size'] for d in clusters.values()), 1)
    summary   = []
    for rank, (cid, data) in enumerate(
        sorted(clusters.items(), key=lambda x: -x[1]['size']), 1
    ):
        cum_vol = sum(
            d['size'] for d in list(clusters.values())[:rank]
        )  # approximate; accurate sort already done above
        merged_from = ','.join(str(c) for c in data.get('merged_from', []))
        summary.append(dict(
            rank                 = rank,
            cluster_id           = cid,
            label                = data.get('split_label', data['representative'])[:120],
            representative       = data['representative'],
            n_unique_questions   = data['unique_queries'],
            query_volume         = data['size'],
            pct_of_total         = round(data['size'] / total_vol * 100, 2),
            was_split            = bool(data.get('parent_cluster')),
            parent_cluster       = data.get('parent_cluster', ''),
            merged_from          = merged_from,
            all_unique_questions = ' | '.join(data['queries']),
        ))

    summ_df  = pd.DataFrame(summary)
    summ_out = out_dir / 'repaired_clusters.csv'
    summ_df.to_csv(summ_out, index=False)
    print(f"  → {summ_out}")

    # Build question-level file: one row per unique question
    rep_set_local = {data['representative'] for data in clusters.values()}
    q_rows = []
    for row in summary:
        questions = row['all_unique_questions'].split(' | ')
        for q in questions:
            q_rows.append(dict(
                cluster_id         = row['cluster_id'],
                question           = q,
                is_representative  = (q == row['representative']),
                rank               = row['rank'],
                label              = row['label'],
                representative     = row['representative'],
                n_unique_questions = row['n_unique_questions'],
                query_volume       = row['query_volume'],
                pct_of_total       = row['pct_of_total'],
                was_split          = row['was_split'],
                parent_cluster     = row['parent_cluster'],
                merged_from        = row['merged_from'],
            ))
    q_df  = pd.DataFrame(q_rows)
    q_out = out_dir / 'cluster_questions.csv'
    q_df.to_csv(q_out, index=False)
    print(f"  → {q_out}")
    print(f"\n  Original clusters: {n_original}  →  Final clusters: {len(clusters)}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 runner (called internally when --raw-file is given)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(raw_csv: Path, crop: str, grid_mode: str,
               max_queries: int, out_dir: Path):
    """
    Run Phase 1 clustering screening on a raw CSV and save results to out_dir.
    Returns list of ClusteringResult objects.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading {raw_csv} ...")
    df = pd.read_csv(raw_csv, low_memory=False)
    if 'query_text' not in df.columns:
        df['query_text'] = df['QueryText'] if 'QueryText' in df.columns else df.iloc[:, 8]

    # Flexible crop filter (strips parentheses)
    raw_crop_norm = df['Crop'].str.replace(r'[\(\)]', '', regex=True).str.strip()
    crop_norm     = re.sub(r'[\(\)]', '', crop).strip()
    df = df[raw_crop_norm == crop_norm].copy()
    print(f"  Rows for '{crop}': {len(df)}")

    # Deduplicate
    df = (df.groupby('query_text', as_index=False)
            .size()
            .rename(columns={'size': 'count'}))
    print(f"  Unique queries after dedup: {len(df)}")

    if len(df) > max_queries:
        df = df.sample(n=max_queries, random_state=42)
        print(f"  Sampled: {max_queries}")

    print(f"\nLoading sentence transformer...")
    model      = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    stop_words = load_stopwords()
    configs    = generate_param_grid(mode=grid_mode)

    print(f"\nPhase 1: {len(configs)} configs  (grid_mode='{grid_mode}')")
    candidates = phase1_fast_screening(df, configs, model, stop_words)

    # Save
    cand_rows = [{'config': str(r.config), **r.config.to_dict(), **r.metrics}
                 for r in candidates]
    pd.DataFrame(cand_rows).to_csv(out_dir / 'phase1_candidates.csv', index=False)

    pkl_path = out_dir / 'phase1_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(candidates, f)
    print(f"  Phase 1 done. Saved {len(candidates)} results to {pkl_path}")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 runner (quick LLM scoring, top-k configs)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(candidates: list, out_dir: Path, model_path: str,
               gpu_id: int, top_k: int = 5, batch_size: int = 8,
               coverage_cap: float = 0.80) -> str:
    """
    Run LLM evaluation on top_k Phase 1 candidates, return best config string.
    """
    from llm_evaluator_hf import evaluate_config_with_hf

    # Sort by composite Phase 1 metric (clusters_for_85pct / n_clusters)
    scored = sorted(candidates,
                    key=lambda r: r.metrics.get('coverage_efficiency', 0),
                    reverse=False)   # lower coverage_efficiency = fewer clusters cover 85%
    top = scored[:top_k]

    print(f"\nPhase 2: evaluating top {len(top)} configs with LLM...")
    judge = RepairJudge(model_path, batch_size=batch_size, gpu_id=gpu_id)

    rows = []
    for result in top:
        scores = evaluate_config_with_hf(result, judge,
                                         batch_size=batch_size,
                                         coverage_cap=coverage_cap)
        rows.append({'config': str(result.config), **scores})
        print(f"  {result.config}: composite={scores.get('composite_score', 0):.4f}")

    p2_df = pd.DataFrame(rows)
    p2_df.to_csv(out_dir / 'phase2_scores.csv', index=False)

    best_cfg = p2_df.sort_values('composite_score', ascending=False).iloc[0]['config']
    print(f"\n  Best config: {best_cfg}")
    return best_cfg


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='LLM-based cluster repair pipeline')

    # Data / model
    ap.add_argument('--raw-file', default=None,
                    help='Input raw CSV. Required for modes: phase1, full.')
    ap.add_argument('--crop', default='Maize Makka',
                    help='Crop name as it appears in the raw CSV Crop column.')
    ap.add_argument('--model', default='/home/kshitij/models/qwen2.5-7b-instruct')
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=8)

    # Pipeline control
    ap.add_argument('--mode', default='full',
                    choices=['phase1', 'phase2', 'repair', 'full'],
                    help='Which part of the pipeline to run.')
    ap.add_argument('--pickle', default=None,
                    help='Path to existing phase1_results.pkl (for modes: phase2, repair).')
    ap.add_argument('--config', default=None,
                    help='Config string to repair. Auto-detects best if omitted.')

    # Phase 1
    ap.add_argument('--grid-mode', default='medium',
                    choices=['quick', 'medium', 'full', 'exhaustive'],
                    help='Phase 1 grid size: quick(18), medium(108), full(240).')
    ap.add_argument('--max-queries', type=int, default=20000)

    # Phase 2
    ap.add_argument('--phase2-top-k', type=int, default=5,
                    help='Number of Phase 1 candidates to evaluate with LLM.')
    ap.add_argument('--coverage-cap', type=float, default=0.80)

    # Repair
    ap.add_argument('--diverse-k', type=int, default=3,
                    help='Number of max-diverse representative questions per cluster.')
    ap.add_argument('--coherence-flag', default='C',
                    choices=['B', 'C'],
                    help='LLM coherence rating that triggers a split (C=strict, B=aggressive).')
    ap.add_argument('--merge-sim', type=float, default=0.82,
                    help='Cosine similarity threshold for merge candidate pairs.')

    args = ap.parse_args()

    # Output directory slug
    slug  = re.sub(r'[^a-z0-9]+', '_', args.crop.lower()).strip('_')
    out_dir = REPAIR_BASE / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'='*60}")
    print(f"Cluster Repair Pipeline — {args.crop}")
    print(f"Mode: {args.mode}  |  Output: {out_dir}")
    print(f"{'='*60}")

    # ─── Phase 1 ──────────────────────────────────────────────────────────────
    candidates = None
    if args.mode in ('phase1', 'full'):
        if not args.raw_file:
            ap.error("--raw-file is required for modes 'phase1' and 'full'.")
        candidates = run_phase1(
            Path(args.raw_file), args.crop, args.grid_mode,
            args.max_queries, out_dir,
        )

    # ─── Load pickle if not built just now ────────────────────────────────────
    if candidates is None:
        pkl_path = Path(args.pickle) if args.pickle else out_dir / 'phase1_results.pkl'
        if not pkl_path.exists():
            ap.error(f"Pickle not found: {pkl_path}. Run with --mode phase1 or --mode full first.")
        print(f"\nLoading {pkl_path} ...")
        with open(pkl_path, 'rb') as f:
            candidates = pickle.load(f)
        print(f"  Loaded {len(candidates)} ClusteringResult objects.")

    # ─── Phase 2 ──────────────────────────────────────────────────────────────
    cfg_str = args.config
    if args.mode in ('phase2', 'full') and cfg_str is None:
        cfg_str = run_phase2(
            candidates, out_dir, args.model, args.gpu_id,
            top_k=args.phase2_top_k, batch_size=args.batch_size,
            coverage_cap=args.coverage_cap,
        )
    elif cfg_str is None:
        # repair mode: try to load from phase2_scores.csv, else pick by Phase 1 metric
        p2_csv = out_dir / 'phase2_scores.csv'
        if p2_csv.exists():
            p2 = pd.read_csv(p2_csv)
            cfg_str = p2.sort_values('composite_score', ascending=False).iloc[0]['config']
            print(f"\nBest config from phase2_scores.csv: {cfg_str}")
        else:
            # Fallback: pick config with lowest coverage_efficiency (tightest clusters)
            cfg_str = str(min(candidates, key=lambda r: r.metrics.get('coverage_efficiency', 1)).config)
            print(f"\nAuto-selected (Phase1 metric): {cfg_str}")

    # ─── Find winning result ───────────────────────────────────────────────────
    if args.mode == 'phase1':
        print("\nPhase 1 complete. Run with --mode phase2 or --mode full for LLM evaluation.")
        return

    result = next((r for r in candidates if str(r.config) == cfg_str), None)
    if result is None:
        sys.exit(f"Config '{cfg_str}' not found. Available: {[str(r.config) for r in candidates[:3]]}")

    n_original = len(result.clusters)
    print(f"\nWinner: {cfg_str}  |  {n_original} clusters  |  {len(result.df)} unique queries")

    if args.mode == 'phase2':
        print("\nPhase 2 complete. Run with --mode repair to apply cluster repair.")
        return

    # ─── Repair Steps A–E ─────────────────────────────────────────────────────
    clusters = copy.deepcopy(result.clusters)

    # Step A: Embeddings + diverse reps
    print(f"\n{'─'*60}\nStep A: Max-diversity representative selection\n{'─'*60}")
    from sentence_transformers import SentenceTransformer
    print("  Loading sentence transformer...")
    st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    texts    = result.df['query_text'].tolist()
    print(f"  Encoding {len(texts)} unique queries...")
    all_embs = st_model.encode(
        texts, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    clusters, diverse_reps = step_a_diverse_reps(
        clusters, result.df, all_embs, k=args.diverse_k)
    print(f"  Done. {len(diverse_reps)} clusters have diverse reps.")

    # Load LLM for B–D
    print(f"\nLoading RepairJudge (Qwen 7B) on GPU {args.gpu_id}...")
    judge = RepairJudge(args.model, batch_size=args.batch_size, gpu_id=args.gpu_id)

    # Step B: Cross-crop filter
    clusters = step_b_cross_crop(clusters, judge, args.crop)

    # Refresh diverse_reps after B (some questions may have been removed)
    for cid in list(diverse_reps.keys()):
        if cid in clusters:
            cluster_q_set     = set(clusters[cid]['queries'])
            diverse_reps[cid] = [q for q in diverse_reps[cid] if q in cluster_q_set]
            if not diverse_reps[cid]:
                diverse_reps[cid] = clusters[cid]['queries'][:1]
        else:
            del diverse_reps[cid]

    # Step C: Coherence + split
    clusters, n_splits = step_c_split(
        clusters, diverse_reps, result.df, all_embs,
        judge, args.crop, flag_on=args.coherence_flag,
    )

    # Step D: Merge
    clusters, n_merges = step_d_merge(
        clusters, st_model, judge, args.crop, sim_thresh=args.merge_sim,
    )

    # Step E: Raw row back-mapping
    raw_csv = Path(args.raw_file) if args.raw_file else REPAIR_BASE / slug / 'input.csv'
    step_e_raw_mapping(clusters, result.df, raw_csv, args.crop, n_original, out_dir)

    # Final summary
    print(f"\n{'='*60}")
    print(f"REPAIR COMPLETE")
    print(f"  Config           : {cfg_str}")
    print(f"  Original clusters: {n_original}")
    print(f"  After repair     : {len(clusters)}")
    print(f"  Splits applied   : {n_splits}")
    print(f"  Merges applied   : {n_merges}")
    print(f"  Outputs in       : {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
