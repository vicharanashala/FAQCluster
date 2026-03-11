#!/usr/bin/env python3
"""
HuggingFace Transformers-based Cluster Quality Evaluation using Qwen2.5-7B-Instruct

Uses standard HuggingFace transformers with a single GPU.  Qwen2.5-7B-Instruct
has a well-behaved chat template, no thinking-mode complications, and fits
entirely on one RTX PRO 6000 Blackwell (102 GiB), so no cross-device sharding.

Scoring method: text generation (greedy, max_new_tokens=4).
  1. Format prompt with Qwen chat template
  2. Call model.generate() — greedy, no sampling
  3. Decode the first generated token(s) and match to A / B / C

Usage:
    python llm_evaluator_hf.py \
        --candidates phase1_candidates.csv \
        --results-pickle phase1_results.pkl \
        --model /home/kshitij/models/qwen2.5-7b-instruct \
        --top-k 15 --batch-size 8
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'hyperparameter_tuning'

# ClusteringResult must be importable when unpickling phase1_results.pkl
import sys as _sys
_sys.path.insert(0, str(SCRIPT_DIR))
from hyperparameter_tuning import ClusteringResult, ClusteringConfig  # noqa: F401


class LocalHFJudge:
    """
    HuggingFace Transformers judge using Qwen2.5-7B-Instruct on a single GPU.

    Uses greedy text generation (max_new_tokens=4).  Qwen2.5-7B-Instruct
    reliably returns a single letter (A / B / C) when prompted correctly,
    so we just decode the first generated token and match it.
    """

    DEFAULT_MODEL = "/home/kshitij/models/qwen2.5-7b-instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 8, gpu_id: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        print(f"Loading model: {model_name}")
        print(f"  device=cuda:{gpu_id}  dtype=bfloat16")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Qwen2.5 uses <|endoftext|> as both EOS and PAD — set pad to a neutral token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for batch generation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": gpu_id},   # single GPU — no cross-device sharding
            trust_remote_code=True,
        )
        self.model.eval()

        self.batch_size = batch_size

        used  = torch.cuda.memory_allocated(gpu_id) / 1e9
        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        print(f"  GPU {gpu_id}: {used:.1f}/{total:.1f} GiB used")
        print("✓ Model loaded")

        # Sanity check
        print("  Running sanity check...")
        ans = self._generate_one(
            "Paris is the capital of:\nA) England\nB) France\nC) Germany\nAnswer (A/B/C):"
        )
        print(f"  Capital of France? → '{ans}'  ({'PASS' if 'B' in ans else 'WARN: expected B'})")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _build_prompt(self, user_text: str) -> str:
        """Wrap user text in Qwen2.5 chat template."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agricultural question clustering expert. "
                    "Answer ONLY with the single letter shown (A, B, or C). "
                    "Do not add any explanation."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_one(self, user_text: str) -> str:
        """Generate a single response (greedy, max 4 new tokens). Returns decoded string."""
        prompt = self._build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_batch(self, user_texts: list[str]) -> list[str]:
        """
        Batch generation.  Left-pads inputs so all sequences align on the right.
        Returns one decoded response string per input.
        """
        prompts = [self._build_prompt(t) for t in user_texts]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,   # Qwen2.5-7B supports 32K; max observed prompt ~1.6K tokens
        ).to(self.device)
        enc.pop("token_type_ids", None)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results = []
        for seq in out:
            new_tokens = seq[input_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results

    @staticmethod
    def _parse_abc(raw: str, default: str = "C") -> str:
        """Extract the first A/B/C letter from a raw model response."""
        for ch in raw.upper():
            if ch in ("A", "B", "C"):
                return ch
        return default

    # ------------------------------------------------------------------
    # Evaluation methods — each uses _generate_batch for throughput
    # ------------------------------------------------------------------

    def evaluate_coherence_batch(self, clusters_queries: list[list[str]]) -> list[float]:
        """
        A=SAME(1.0)  B=MOSTLY(0.7)  C=DIFFERENT(0.2)
        Are all queries in a cluster about the same agricultural topic?
        """
        score_map = {"A": 1.0, "B": 0.7, "C": 0.2}
        prompts = []
        for queries in clusters_queries:
            qstr = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
            prompts.append(
                f"Are these {len(queries)} farmer questions about the SAME specific agricultural topic?\n\n"
                f"Questions:\n{qstr}\n\n"
                f"A) YES — all about the same specific topic\n"
                f"B) MOSTLY — most are related, a few differ slightly\n"
                f"C) NO — questions cover clearly different topics\n\n"
                f"Answer (A/B/C):"
            )
        responses = self._generate_batch(prompts)
        return [score_map.get(self._parse_abc(r), 0.5) for r in responses]

    def evaluate_separation_batch(self, cluster_pairs: list[tuple]) -> list[float]:
        """
        A=DIFFERENT(1.0)  B=SOMEWHAT(0.6)  C=SAME(0.0)
        Should two clusters be kept in separate groups?
        """
        score_map = {"A": 1.0, "B": 0.6, "C": 0.0}
        prompts = []
        for qa, qb in cluster_pairs:
            a_str = "\n".join(f"  A{i+1}. {q}" for i, q in enumerate(qa[:3]))
            b_str = "\n".join(f"  B{i+1}. {q}" for i, q in enumerate(qb[:3]))
            prompts.append(
                f"Should these two groups of farmer questions belong in DIFFERENT clusters?\n\n"
                f"Group A:\n{a_str}\n\nGroup B:\n{b_str}\n\n"
                f"A) YES — clearly different agricultural topics\n"
                f"B) SOMEWHAT — related but distinct enough to stay separate\n"
                f"C) NO — essentially the same topic, should be merged\n\n"
                f"Answer (A/B/C):"
            )
        responses = self._generate_batch(prompts)
        return [score_map.get(self._parse_abc(r), 0.5) for r in responses]

    def evaluate_merge_candidates_batch(self, cluster_pairs: list[tuple]) -> list[float]:
        """
        A=MERGE(1.0)  B=SEPARATE(0.0)
        Should two near-identical clusters be merged?
        """
        score_map = {"A": 1.0, "B": 0.0}
        prompts = []
        for qa, qb in cluster_pairs:
            a_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(qa[:3]))
            b_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(qb[:3]))
            prompts.append(
                f"Should these two farmer question clusters be MERGED or kept SEPARATE?\n\n"
                f"Cluster 1:\n{a_str}\n\nCluster 2:\n{b_str}\n\n"
                f"A) MERGE — same topic, should be one cluster\n"
                f"B) SEPARATE — different enough to stay in separate clusters\n\n"
                f"Answer (A/B):"
            )
        responses = self._generate_batch(prompts)
        return [score_map.get(self._parse_abc(r, default="B"), 0.0) for r in responses]

    def detect_outliers_batch(self, clusters_queries: list[list[str]]) -> list[int]:
        """
        Returns 0-based index of the outlier question, or -1 if none.
        Uses a 1/2/3/4/5/N prompt.
        """
        outlier_indices = []
        for queries in clusters_queries:
            n    = min(len(queries), 5)
            qs   = queries[:n]
            qstr = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
            opts_desc = "  ".join(
                [f"{i+1}) #{i+1}" for i in range(n)] + ["N) None — all belong"]
            )
            prompt = (
                f"Which question (if any) does NOT belong with the others?\n\n"
                f"Questions:\n{qstr}\n\n"
                f"{opts_desc}\n\n"
                f"Answer ({'/'.join([str(i+1) for i in range(n)])}/N):"
            )
            raw = self._generate_one(prompt)
            # Parse: look for a digit 1-n or 'N'
            found = -1
            for ch in raw.upper():
                if ch == "N":
                    found = -1
                    break
                if ch.isdigit() and 1 <= int(ch) <= n:
                    found = int(ch) - 1
                    break
            outlier_indices.append(found)
        return outlier_indices


# ------------------------------------------------------------------
# Shared evaluation helpers
# ------------------------------------------------------------------

def get_unique_queries(cluster, n=None):
    """Return deduplicated queries from a cluster. n=None returns all unique questions."""
    seen = set()
    unique = []
    for q in cluster.get('queries', []):
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q.strip())
        if n is not None and len(unique) == n:
            break
    return unique


def get_merge_candidate_pairs(clusters, top_n_per_cluster=3, max_pairs=30):
    from collections import defaultdict
    kw_index = defaultdict(list)
    for cid, data in clusters.items():
        kws = [k.strip().lower() for k in data.get('keywords', '').split(',') if k.strip()][:top_n_per_cluster]
        for kw in kws:
            kw_index[kw].append(cid)

    pairs = set()
    for cid_list in kw_index.values():
        if len(cid_list) > 1:
            for i in range(len(cid_list)):
                for j in range(i + 1, len(cid_list)):
                    pairs.add((min(cid_list[i], cid_list[j]), max(cid_list[i], cid_list[j])))

    return list(pairs)[:max_pairs]


def all_cluster_ids_sorted(clusters):
    return sorted(
        clusters.keys(),
        key=lambda cid: clusters[cid].get('unique_queries', len(clusters[cid].get('queries', []))),
        reverse=True
    )


def coverage_capped_ids(clusters, all_ids_sorted, coverage_cap: float):
    """
    Return the minimal prefix of all_ids_sorted (sorted by cluster size desc)
    that covers at least `coverage_cap` fraction of total query volume.
    Always returns at least 2 clusters.
    """
    if coverage_cap >= 1.0:
        return all_ids_sorted
    total = sum(clusters[cid]['size'] for cid in all_ids_sorted)
    if total == 0:
        return all_ids_sorted
    cum = 0
    for i, cid in enumerate(all_ids_sorted):
        cum += clusters[cid]['size']
        if cum / total >= coverage_cap:
            return all_ids_sorted[:max(i + 1, 2)]
    return all_ids_sorted


def evaluate_config_with_hf(result, llm_judge, batch_size=8, coverage_cap: float = 1.0):
    """
    Evaluate one clustering configuration using the HuggingFace judge.

    coverage_cap (0.0–1.0): For coherence and outlier passes, only evaluate
    the smallest set of clusters (sorted by size, largest first) that together
    cover this fraction of total query volume.  Scores are weighted by cluster
    query volume so the composite reflects actual farmer impact.
    Set to 1.0 to evaluate all clusters (original behaviour).
    """
    print(f"\nEvaluating config: {result.config}")
    clusters = result.clusters

    # Sort by size descending — used for coverage-capped passes
    all_ids = sorted(clusters.keys(), key=lambda cid: clusters[cid]['size'], reverse=True)
    total_volume = sum(clusters[cid]['size'] for cid in all_ids)

    # IDs to use for coherence / outlier (coverage-capped)
    eval_ids = coverage_capped_ids(clusters, all_ids, coverage_cap)
    cap_pct  = sum(clusters[cid]['size'] for cid in eval_ids) / max(total_volume, 1) * 100
    print(f"  Total clusters: {len(all_ids)}  |  Evaluating: {len(eval_ids)} "
          f"({'all' if len(eval_ids)==len(all_ids) else f'{cap_pct:.0f}% query coverage'})")

    # Pass 1: Coherence (coverage-capped, size-weighted scoring)
    print("  Pass 1: Coherence...")
    coherence_inputs, coherence_weights = [], []
    for cid in eval_ids:
        uq = get_unique_queries(clusters[cid])   # all unique questions — max observed is 91, fits in 1.6K tokens
        if len(uq) >= 2:
            coherence_inputs.append(uq)
            coherence_weights.append(clusters[cid]['size'])

    # Pass 2: Separation — size-biased random pairs from eval_ids
    print("  Pass 2: Separation...")
    n_sep_pairs = min(50, len(eval_ids) // 2)
    rng = np.random.default_rng(42)
    sizes_arr   = np.array([clusters[cid]['size'] for cid in eval_ids], dtype=float)
    probs       = sizes_arr / sizes_arr.sum() if sizes_arr.sum() > 0 else None
    separation_inputs = []
    if n_sep_pairs > 0 and len(eval_ids) >= 2:
        chosen_a = rng.choice(len(eval_ids), size=n_sep_pairs, replace=True, p=probs)
        chosen_b = rng.choice(len(eval_ids), size=n_sep_pairs, replace=True, p=probs)
        for ia, ib in zip(chosen_a, chosen_b):
            if ia == ib:
                ib = (ib + 1) % len(eval_ids)
            qa = get_unique_queries(clusters[eval_ids[ia]], n=3)
            qb = get_unique_queries(clusters[eval_ids[ib]], n=3)
            separation_inputs.append((qa, qb))

    # Pass 3: Merge detection
    print("  Pass 3: Merge detection...")
    merge_pair_ids = get_merge_candidate_pairs(clusters, max_pairs=30)
    merge_inputs = []
    for cid_a, cid_b in merge_pair_ids:
        qa = get_unique_queries(clusters[cid_a], n=3)
        qb = get_unique_queries(clusters[cid_b], n=3)
        merge_inputs.append((qa, qb))
    print(f"  Merge candidate pairs: {len(merge_inputs)}")

    # Pass 4: Outlier detection (coverage-capped)
    print("  Pass 4: Outlier detection...")
    outlier_inputs, outlier_weights = [], []
    for cid in eval_ids:
        # Outlier detection is kept at n=10: asking LLM "which of 91 questions is the odd one?"
        # is cognitively harder and the task prompt uses numbered answer choices (1/2/.../N).
        uq = get_unique_queries(clusters[cid], n=10)
        if len(uq) >= 4:
            outlier_inputs.append(uq)
            outlier_weights.append(clusters[cid]['size'])

    total_calls = len(coherence_inputs) + len(separation_inputs) + len(merge_inputs) + len(outlier_inputs)
    n_batches = total_calls // batch_size + 1
    print(f"\n  Total LLM calls: {total_calls}  (~{n_batches} batches)")

    coherence_scores = []
    for i in tqdm(range(0, len(coherence_inputs), batch_size), desc="  Coherence"):
        coherence_scores.extend(llm_judge.evaluate_coherence_batch(coherence_inputs[i:i+batch_size]))

    separation_scores = []
    for i in tqdm(range(0, len(separation_inputs), batch_size), desc="  Separation"):
        separation_scores.extend(llm_judge.evaluate_separation_batch(separation_inputs[i:i+batch_size]))

    merge_scores = []
    for i in tqdm(range(0, len(merge_inputs), batch_size), desc="  Merge detect"):
        merge_scores.extend(llm_judge.evaluate_merge_candidates_batch(merge_inputs[i:i+batch_size]))

    outlier_raw = []
    for i in tqdm(range(0, len(outlier_inputs), batch_size), desc="  Outliers"):
        outlier_raw.extend(llm_judge.detect_outliers_batch(outlier_inputs[i:i+batch_size]))
    outlier_detections = [1 if idx >= 0 else 0 for idx in outlier_raw]

    # Size-weighted aggregation (coherence and outlier)
    def _weighted_mean(scores, weights):
        if not scores:
            return 0.0
        w = np.array(weights[:len(scores)], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(scores)) / len(scores)
        return float(np.dot(scores, w))

    coherence_mean  = _weighted_mean(coherence_scores, coherence_weights)
    outlier_rate    = _weighted_mean(outlier_detections, outlier_weights)
    separation_mean = float(np.mean(separation_scores)) if separation_scores else 0.0
    merge_rate      = float(np.mean(merge_scores))       if merge_scores       else 0.0

    composite = (
        0.45 * separation_mean
        + 0.30 * coherence_mean
        + 0.25 * (1.0 - outlier_rate)
        - 0.20 * merge_rate
    )

    final_scores = {
        'coherence_mean':      coherence_mean,
        'coherence_std':       float(np.std(coherence_scores)) if coherence_scores else 0.0,
        'separation_mean':     separation_mean,
        'separation_std':      float(np.std(separation_scores)) if separation_scores else 0.0,
        'merge_rate':          merge_rate,
        'merge_pairs_checked': len(merge_inputs),
        'outlier_rate':        outlier_rate,
        'n_coherence_evals':   len(coherence_scores),
        'n_separation_evals':  len(separation_scores),
        'n_outlier_evals':     len(outlier_detections),
        'composite_score':     composite,
    }

    print(f"  coherence={coherence_mean:.3f}  separation={separation_mean:.3f}  "
          f"merge_rate={merge_rate:.3f}  outlier_rate={outlier_rate:.3f}  "
          f"composite={composite:.3f}")

    return final_scores


def select_stratified_candidates(candidates_df: pd.DataFrame, total: int = 15) -> pd.DataFrame:
    """
    Stratified selection of Phase 1 candidates for LLM evaluation.

    Allocation (scales with ``total``):
      - top    (≈8/15): best by Phase 1 composite score
      - middle (≈4/15): median performance range, different hyperparameter region
      - diverse(≈2/15): farthest from selected set in (alpha, n_components) space
      - bottom (remainder): lowest-scoring viable config – calibration baseline

    Returns a copy of the selected rows annotated with a ``_stratum`` column.
    """
    # ----- proportional allocation ------------------------------------------------
    top_n = max(1, round(total * 8 / 15))
    mid_n = max(1, round(total * 4 / 15))
    div_n = max(1, round(total * 2 / 15))
    bot_n = max(1, total - top_n - mid_n - div_n)

    df = candidates_df.copy().reset_index(drop=True)

    # ----- build a Phase 1 composite score ----------------------------------------
    # (needed before dedup so we keep the best row per config string)
    def _safe_norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng + 1e-9)

    ce   = _safe_norm(df['coverage_efficiency']) if 'coverage_efficiency' in df.columns else pd.Series(np.zeros(len(df)))
    c85  = _safe_norm(df['clusters_for_85pct'])  if 'clusters_for_85pct'  in df.columns else pd.Series(np.ones (len(df)))
    nr   = _safe_norm(df['noise_ratio'])          if 'noise_ratio'         in df.columns else pd.Series(np.zeros(len(df)))

    # higher efficiency + fewer clusters for 85 % + lower noise → better
    df['_phase1_score'] = ce - 0.5 * c85 - 0.3 * nr

    # ----- deduplicate: keep best-scoring row per config string --------------------
    # (config strings omit n_components, so multiple rows can share a config name)
    before = len(df)
    df = df.sort_values('_phase1_score', ascending=False)
    df = df.drop_duplicates(subset='config', keep='first').reset_index(drop=True)
    if len(df) < before:
        print(f"  Deduped config strings: {before} → {len(df)} unique configs")

    df = df.sort_values('_phase1_score', ascending=False).reset_index(drop=True)

    selected: dict[int, str] = {}   # index → stratum label

    # ----- top stratum ------------------------------------------------------------
    for i in range(min(top_n, len(df))):
        selected[i] = 'top'

    # ----- bottom stratum ---------------------------------------------------------
    bot_candidates = [i for i in range(len(df) - 1, -1, -1) if i not in selected]
    for i in bot_candidates[:bot_n]:
        selected[i] = 'bottom'

    # ----- middle stratum ---------------------------------------------------------
    mid_center = len(df) // 2
    window     = mid_n * 4
    mid_pool   = [
        i for i in range(max(0, mid_center - window), min(len(df), mid_center + window))
        if i not in selected
    ]
    if len(mid_pool) >= mid_n:
        step    = max(1, len(mid_pool) // mid_n)
        mid_idx = [mid_pool[j * step] for j in range(mid_n)]
    else:
        mid_idx = mid_pool[:mid_n]
    for i in mid_idx:
        selected[i] = 'middle'

    # ----- diverse stratum (farthest in parameter space) --------------------------
    alpha_vals  = df['alpha'].values       if 'alpha'       in df.columns else np.zeros(len(df))
    ncomp_vals  = df['n_components'].values if 'n_components' in df.columns else np.zeros(len(df))
    params      = np.column_stack([_safe_norm(pd.Series(alpha_vals)).values,
                                   _safe_norm(pd.Series(ncomp_vals)).values])

    sel_params  = params[list(selected.keys())]
    remaining   = [i for i in range(len(df)) if i not in selected]

    if remaining:
        # greedy farthest-point: iteratively pick the point with max min-dist to selected set
        div_idx: list[int] = []
        available = list(remaining)
        current_selected_params = sel_params.copy()

        for _ in range(min(div_n, len(available))):
            min_dists = np.array([
                np.linalg.norm(params[i] - current_selected_params, axis=1).min()
                for i in available
            ])
            best     = available[int(np.argmax(min_dists))]
            div_idx.append(best)
            current_selected_params = np.vstack([current_selected_params, params[best]])
            available.remove(best)

        for i in div_idx:
            selected[i] = 'diverse'

    # ----- assemble result --------------------------------------------------------
    sorted_idx = sorted(selected.keys())
    result = df.iloc[sorted_idx].copy()
    result['_stratum'] = [selected[i] for i in sorted_idx]

    counts = result['_stratum'].value_counts().to_dict()
    print(f"  Stratified selection: total={len(result)}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    return result


def merge_shards(output_dir: Path):
    """Merge all phase2_hf_evaluation_shard*.csv files into one sorted final CSV."""
    shard_files = sorted(output_dir.glob('phase2_hf_evaluation_shard*.csv'))
    if not shard_files:
        print("No shard files found in", output_dir)
        return
    dfs = [pd.read_csv(f) for f in shard_files]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('composite_score', ascending=False).reset_index(drop=True)
    out_path = output_dir / 'phase2_hf_evaluation_final.csv'
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(shard_files)} shards → {len(merged)} configs → {out_path}")
    print("\nTop 5:")
    cols = ['config', 'composite_score', 'coherence_mean', 'separation_mean']
    print(merged[cols].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='LLM cluster evaluation via HuggingFace transformers')
    parser.add_argument('--merge-shards', action='store_true',
                        help='Merge phase2_hf_evaluation_shard*.csv files into final CSV and exit.')
    parser.add_argument('--candidates', type=str, required=True,
                        help='CSV file with candidate configs (relative to outputs/hyperparameter_tuning/)')
    parser.add_argument('--results-pickle', type=str, required=True,
                        help='Pickle file with ClusteringResult objects')
    parser.add_argument('--model', type=str,
                        default='/home/kshitij/models/qwen2.5-7b-instruct',
                        help='Local model path or HF model ID')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Evaluate top K candidates from Phase 1 (0 = evaluate ALL unique configs)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Prompts per HF generate() call')
    parser.add_argument('--coverage-cap', type=float, default=1.0,
                        help=('For coherence/outlier passes, only evaluate the '
                              'smallest set of clusters (sorted by size) that '
                              'cover this fraction of query volume. '
                              '0.80 = evaluate clusters covering 80%% of queries '
                              '(~25%% of clusters, 2.7x faster). Default=1.0 (all).'))
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='CUDA device index to run on (default: 0). Use 1 for the second GPU.')
    parser.add_argument('--shard', type=str, default=None,
                        help=('Run only a fraction of configs, e.g. "1/2" runs the first half '
                              'and "2/2" runs the second half. Enables two-terminal dual-GPU '
                              'parallelism: run shard 1/2 on GPU 0 and 2/2 on GPU 1 simultaneously.'))
    parser.add_argument('--compile', action='store_true',
                        help='Apply torch.compile(model, mode="reduce-overhead") for ~20%% speedup '
                             '(adds ~60s one-time compilation cost at startup).')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: just merge existing shard files and exit
    if args.merge_shards:
        merge_shards(OUTPUT_DIR)
        return

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. This will be very slow on CPU.")
    else:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            marker = " ← using" if i == args.gpu_id else ""
            print(f"✓ GPU {i}: {name}  ({mem:.0f} GiB){marker}")

    # Load candidates
    candidates_file = OUTPUT_DIR / args.candidates
    if not candidates_file.exists():
        print(f"Error: {candidates_file} not found. Run Phase 1 first.")
        return
    candidates_df = pd.read_csv(candidates_file)
    print(f"\nLoaded {len(candidates_df)} candidate configurations")

    # Candidate selection: 0 = all unique configs, otherwise stratified top-K
    if args.top_k == 0:
        print("\nMode: ALL unique configs (no stratification)")
        df = candidates_df.copy()
        if 'coverage_efficiency' in df.columns:
            df['_score'] = df['coverage_efficiency']
        else:
            df['_score'] = 0.0
        df = df.sort_values('_score', ascending=False)
        top_candidates = df.drop_duplicates(subset='config', keep='first').reset_index(drop=True)
        print(f"Evaluating ALL {len(top_candidates)} unique configs (deduped from {len(candidates_df)} rows)")
    else:
        print(f"\nSelecting {args.top_k} candidates via stratified sampling...")
        top_candidates = select_stratified_candidates(candidates_df, total=args.top_k)
        print(f"Evaluating {len(top_candidates)} candidates with HuggingFace")

    # Sharding: split configs across multiple processes / GPUs
    if args.shard is not None:
        try:
            shard_idx, n_shards = [int(x) for x in args.shard.split('/')]
            assert 1 <= shard_idx <= n_shards, "shard index must be 1..n_shards"
        except (ValueError, AssertionError):
            print(f"Error: --shard must be in the form 'N/M', e.g. '1/2' or '2/2'. Got: {args.shard}")
            return
        total = len(top_candidates)
        chunk = (total + n_shards - 1) // n_shards
        start = (shard_idx - 1) * chunk
        end   = min(start + chunk, total)
        top_candidates = top_candidates.iloc[start:end].reset_index(drop=True)
        shard_tag = f"shard{shard_idx}of{n_shards}"
        print(f"  Shard {shard_idx}/{n_shards}: evaluating rows {start}–{end-1} "
              f"({len(top_candidates)} configs) on GPU {args.gpu_id}")
    else:
        shard_tag = None

    # Load clustering results
    results_file = OUTPUT_DIR / args.results_pickle
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        return
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)

    # Load model once
    print("\n" + "="*70)
    llm_judge = LocalHFJudge(args.model, batch_size=args.batch_size, gpu_id=args.gpu_id)
    if args.compile:
        print("Applying torch.compile(mode='reduce-overhead') — takes ~60s...")
        import torch as _torch
        llm_judge.model = _torch.compile(llm_judge.model, mode='reduce-overhead')
        # Trigger compilation with one warmup batch
        _enc = llm_judge.tokenizer(
            [llm_judge._build_prompt("warmup")], return_tensors='pt'
        ).to(f'cuda:{args.gpu_id}')
        _enc.pop('token_type_ids', None)
        with _torch.no_grad():
            llm_judge.model.generate(**_enc, max_new_tokens=4, do_sample=False,
                                     pad_token_id=llm_judge.tokenizer.pad_token_id)
        print("✓ torch.compile ready")
    print("="*70)

    evaluation_results = []

    for idx, row in top_candidates.iterrows():
        config_str = row['config']

        matching_result = next(
            (r for r in all_results if str(r.config) == config_str), None
        )
        if matching_result is None:
            print(f"Warning: No ClusteringResult for config {config_str}")
            continue

        llm_scores = evaluate_config_with_hf(
            matching_result, llm_judge,
            batch_size=args.batch_size,
            coverage_cap=args.coverage_cap,
        )

        eval_row = {'config': config_str, **row.to_dict(), **llm_scores}
        evaluation_results.append(eval_row)

        # Save incrementally (shard-aware filename to avoid collisions between parallel runs)
        interim_df = pd.DataFrame(evaluation_results)
        suffix = f'_{shard_tag}' if shard_tag else ''
        interim_file = OUTPUT_DIR / f'phase2_hf_evaluation_interim{suffix}.csv'
        interim_df.to_csv(interim_file, index=False)
        print(f"  ↳ Interim results saved ({len(evaluation_results)} configs so far) → {interim_file.name}")

    # Final results for this shard (or full run)
    final_df = pd.DataFrame(evaluation_results)
    final_df = final_df.sort_values('composite_score', ascending=False)
    final_suffix = f'_{shard_tag}' if shard_tag else '_final'
    final_file = OUTPUT_DIR / f'phase2_hf_evaluation{final_suffix}.csv'
    final_df.to_csv(final_file, index=False)

    print("\n" + "="*70)
    print(f"PHASE 2 (HF) COMPLETE  {'[' + shard_tag + ']' if shard_tag else ''}")
    print(f"Results saved to: {final_file}")
    if shard_tag:
        print(f"  Merge all shards when done: python llm_evaluator_hf.py --merge-shards")
    print("="*70)
    if len(final_df) >= 3:
        print("\nTop 3 configurations by composite score:")
        cols = ['config', 'composite_score', 'coherence_mean', 'separation_mean', 'merge_rate']
        print(final_df[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
