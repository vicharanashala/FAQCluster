#!/usr/bin/env python3
"""
unique_question_finder.py — Hybrid LLM unique-question extractor.

For each cluster in cluster_questions.csv, groups questions that would receive
THE SAME agricultural answer into a single "unique question" entry.

Two modes (--api-provider):
  anthropic  Use Claude Haiku via Batch API (recommended — handles 50+ items)
  local      Use local Qwen 7B with batching (default fallback)

After per-cluster grouping, a cross-cluster dedup pass merges groups from
different clusters that are answer-equivalent (embedding cosine ≥ 0.92).

Outputs (in same directory as --cluster-file):
  unique_questions.csv          — one row per answer-distinct group
  unique_questions_freq.csv     — same, sorted by raw_frequency descending
  unique_question_mapping.csv   — one row per question mapped to its group
  unique_questions_checkpoint.json — per-cluster LLM results (resume support)
  unique_questions_verification.csv — single-row sanity check

Usage (Claude):
  export ANTHROPIC_API_KEY=sk-ant-...
  python analysis/unique_question_finder.py \\
      --cluster-file outputs/repair/maize_makka/cluster_questions.csv \\
      --raw-file data/raw/punjab_maize_raw.csv \\
      --crop "Maize Makka" \\
      --api-provider anthropic

Usage (local):
  python analysis/unique_question_finder.py \\
      --cluster-file outputs/repair/maize_makka/cluster_questions.csv \\
      --raw-file data/raw/punjab_maize_raw.csv \\
      --crop "Maize Makka" \\
      --model /home/kshitij/models/qwen2.5-7b-instruct --gpu-id 0

Add --resume to skip clusters already in the checkpoint.
"""

import re, sys, os, json, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from cluster_repair import RepairJudge  # reuse for local mode + cross-cluster dedup


# ══════════════════════════════════════════════════════════════════════════════
# Shared: prompt builder and JSON parser
# ══════════════════════════════════════════════════════════════════════════════

def build_grouping_prompt(questions: list, label: str, crop: str) -> str:
    n    = len(questions)
    qstr = "\n".join(f"{i+1}. {q.strip()}" for i, q in enumerate(questions))
    return (
        f"You are an agricultural extension officer for {crop}.\n"
        f"Cluster topic: \"{label}\"\n\n"
        f"Below are {n} farmer questions, all about the same general topic.\n"
        f"Group them so that every question in a group would receive the EXACT SAME "
        f"specific agricultural advice — same chemical, same dose, same method, same timing.\n\n"
        f"Rules:\n"
        f"- Same practical recommendation → same group (even if worded differently)\n"
        f"- Different chemical / dose / timing / method / plant part → different group\n"
        f"- Every question index from 1 to {n} MUST appear in exactly one group\n"
        f"- The 'label' field must describe WHAT THE ANSWER IS, not what the question says\n"
        f"  (e.g. 'Chlorpyrifos soil drench for termite', not 'insect control')\n\n"
        f"Questions:\n{qstr}\n\n"
        f"Return ONLY valid JSON — no text before or after:\n"
        f'[{{"group":1,"label":"answer description","indices":[1,2,3]}},'
        f'{{"group":2,"label":"answer description","indices":[4,5]}}]\nJSON:'
    )


def parse_groups(raw: str, n: int, label: str) -> list:
    """Parse LLM JSON output → list of {answer_label, indices} (0-based)."""
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    groups_raw = []
    if m:
        try:
            groups_raw = json.loads(m.group())
        except Exception:
            pass

    result, seen = [], set()
    for g in groups_raw:
        idxs = []
        for i in g.get("indices", []):
            try:
                idx = int(i) - 1
                if 0 <= idx < n and idx not in seen:
                    idxs.append(idx)
                    seen.add(idx)
            except (ValueError, TypeError):
                continue
        if idxs:
            result.append({
                "answer_label": str(g.get("label", ""))[:200].strip(),
                "indices":      idxs,
            })

    for i in range(n):   # fallback: missed indices → singleton
        if i not in seen:
            result.append({"answer_label": "(unclassified)", "indices": [i]})

    return result or [{"answer_label": label, "indices": list(range(n))}]


# ══════════════════════════════════════════════════════════════════════════════
# Claude Batch API judge
# ══════════════════════════════════════════════════════════════════════════════

class ClaudeBatchJudge:
    """Submits all uncached clusters to Claude via the Messages Batch API."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model
        self._anthropic = anthropic
        print(f"  Claude judge: {model}")

    def process_clusters(self, clusters_todo: list,
                         checkpoint: dict, ckpt_file: Path) -> dict:
        """
        clusters_todo = [(cid_key, questions, label, crop), ...]
        Submits one batch, polls until done, parses results into checkpoint.
        Returns updated checkpoint.
        """
        if not clusters_todo:
            return checkpoint

        requests, cid_meta = [], {}
        for cid_key, questions, label, crop in clusters_todo:
            prompt    = build_grouping_prompt(questions, label, crop)
            max_tok   = min(200 + len(questions) * 20, 4096)
            custom_id = f"cid_{cid_key}"
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model":     self.model,
                    "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
            cid_meta[custom_id] = (cid_key, questions, label)

        print(f"\n  Submitting {len(requests)} requests to Claude Batch API...")
        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"  Batch ID: {batch_id}")

        # Poll until done
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            c = batch.request_counts
            print(f"  [{time.strftime('%H:%M:%S')}] processing={c.processing}  "
                  f"succeeded={c.succeeded}  errored={c.errored}")
            if batch.processing_status == "ended":
                break
            time.sleep(30)

        # Collect and parse results
        n_ok = n_err = 0
        for result in self.client.messages.batches.results(batch_id):
            cid_key, questions, label = cid_meta[result.custom_id]
            if result.result.type == "succeeded":
                raw  = result.result.message.content[0].text
                groups = parse_groups(raw, len(questions), label)
                n_ok += 1
            else:
                print(f"  ⚠ {result.custom_id}: {result.result.type}")
                groups = [{"answer_label": "(unclassified)",
                           "indices": [i]} for i in range(len(questions))]
                n_err += 1
            checkpoint[cid_key] = groups

        # Assign group_ids
        for cid_key in checkpoint:
            for gid, g in enumerate(checkpoint[cid_key], 1):
                g["group_id"] = gid

        with open(ckpt_file, 'w') as f:
            json.dump(checkpoint, f)

        print(f"  Batch done: {n_ok} succeeded, {n_err} errored")
        return checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# Local judge (Qwen 7B) with batching for large clusters
# ══════════════════════════════════════════════════════════════════════════════

LOCAL_BATCH = 12


def _run_local_batch(judge, questions, label, crop):
    """Single LLM call for up to LOCAL_BATCH questions."""
    prompt  = build_grouping_prompt(questions, label, crop)
    max_tok = min(150 + len(questions) * 16, 600)
    raw     = judge._gen_long(prompt, max_new_tokens=max_tok)
    return parse_groups(raw, len(questions), label)


def _embed(judge, texts):
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(judge, '_st'):
            judge._st = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                device=judge.device)
        return judge._st.encode(texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        return np.eye(len(texts))


def find_unique_questions_local(judge, questions, label, crop):
    """Batching + cross-batch merge for large clusters with local Qwen 7B."""
    n = len(questions)
    if n <= LOCAL_BATCH:
        raw_groups = _run_local_batch(judge, questions, label, crop)
        groups = raw_groups
    else:
        batches = [questions[i:i+LOCAL_BATCH] for i in range(0, n, LOCAL_BATCH)]
        offset, global_groups = 0, []
        for batch in batches:
            for g in _run_local_batch(judge, batch, label, crop):
                global_groups.append({
                    "answer_label": g["answer_label"],
                    "indices": [offset + i for i in g["indices"]],
                })
            offset += len(batch)
        groups = _merge_cross_batch(judge, global_groups, questions, crop)

    seen, result = set(), []
    for g in groups:
        deduped = [i for i in g["indices"] if i not in seen]
        seen.update(deduped)
        if deduped:
            result.append({"group_id": len(result)+1,
                           "answer_label": g["answer_label"],
                           "indices": deduped})
    for i in range(n):
        if i not in seen:
            result.append({"group_id": len(result)+1,
                           "answer_label": "(unclassified)", "indices": [i]})
    return result


def _merge_cross_batch(judge, groups, questions, crop, sim_thresh=0.88):
    if len(groups) <= 1:
        return groups
    reps  = [questions[g["indices"][0]] for g in groups]
    embs  = _embed(judge, reps)
    sim   = embs @ embs.T
    absorbed = set()
    for i in range(len(groups)):
        if i in absorbed:
            continue
        for j in range(i+1, len(groups)):
            if j in absorbed or sim[i,j] < sim_thresh:
                continue
            prompt = (f"Crop: {crop}.\nQ1: {reps[i]}\nQ2: {reps[j]}\n\n"
                      f"Same core agricultural advice? A) YES  B) NO\nAnswer:")
            if judge._parse_abc(judge._generate_one(prompt), default="B") == "A":
                groups[i]["indices"].extend(groups[j]["indices"])
                if groups[i]["answer_label"] in ("(unclassified)", ""):
                    groups[i]["answer_label"] = groups[j]["answer_label"]
                absorbed.add(j)
    return [g for i, g in enumerate(groups) if i not in absorbed]


# ══════════════════════════════════════════════════════════════════════════════
# Cross-cluster dedup (embedding-only, high threshold for precision)
# ══════════════════════════════════════════════════════════════════════════════

def cross_cluster_dedup(uq_rows: list, freq_map: dict,
                        sim_thresh: float = 0.92) -> list:
    """
    Merge rows from DIFFERENT clusters if their representative questions are
    near-identical in embedding space (cosine >= sim_thresh).
    Keeps the row with higher raw_frequency; sums the frequencies.
    Returns de-duplicated list of uq_rows.
    """
    if len(uq_rows) <= 1:
        return uq_rows

    from sentence_transformers import SentenceTransformer
    print("\nCross-cluster dedup: encoding representative questions...")
    st_model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    reps = [r["representative_question"] for r in uq_rows]
    embs = st_model.encode(reps, batch_size=128, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    sim  = embs @ embs.T

    # Find candidates
    n = len(uq_rows)
    cand = [(i, j, float(sim[i,j]))
            for i in range(n)
            for j in range(i+1, n)
            if sim[i,j] >= sim_thresh
            and uq_rows[i]["cluster_id"] != uq_rows[j]["cluster_id"]]
    cand.sort(key=lambda x: -x[2])
    print(f"  Candidate cross-cluster pairs: {len(cand)}")

    absorbed = set()
    for i, j, s in cand:
        if i in absorbed or j in absorbed:
            continue
        # Keep the one with higher raw_frequency
        keep, drop = (i, j) if uq_rows[i]["raw_frequency"] >= uq_rows[j]["raw_frequency"] else (j, i)
        print(f"  MERGE {uq_rows[drop]['unique_q_id']} → {uq_rows[keep]['unique_q_id']}  "
              f"(sim={s:.3f})")
        uq_rows[keep]["raw_frequency"]        += uq_rows[drop]["raw_frequency"]
        uq_rows[keep]["n_questions_in_group"] += uq_rows[drop]["n_questions_in_group"]
        uq_rows[keep]["merged_cross_cluster"]  = True
        absorbed.add(drop)

    result = [r for i, r in enumerate(uq_rows) if i not in absorbed]
    print(f"  Cross-cluster dedup: {n} → {len(result)} groups "
          f"({len(absorbed)} merged)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Find answer-distinct unique questions within each cluster.")
    ap.add_argument("--cluster-file", required=True)
    ap.add_argument("--raw-file",     required=True)
    ap.add_argument("--crop",         default="Maize Makka")
    ap.add_argument("--api-provider", default="anthropic",
                    choices=["anthropic", "local"])
    ap.add_argument("--api-key",      default=None,
                    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    ap.add_argument("--claude-model", default="claude-haiku-4-5")
    # Local model args (only used when --api-provider local)
    ap.add_argument("--model",      default="/home/kshitij/models/qwen2.5-7b-instruct")
    ap.add_argument("--gpu-id",     type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--resume",     action="store_true",
                    help="Skip clusters already in checkpoint")
    ap.add_argument("--dedup-thresh", type=float, default=0.92,
                    help="Cosine similarity threshold for cross-cluster dedup")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to write outputs (default: same dir as --cluster-file)")
    args = ap.parse_args()

    cluster_file = Path(args.cluster_file)
    out_dir      = Path(args.output_dir) if args.output_dir else cluster_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cluster_questions.csv ────────────────────────────────────────────
    print(f"Loading {cluster_file} ...")
    cq = pd.read_csv(cluster_file, low_memory=False)
    cq['question'] = cq['question'].astype(str)
    print(f"  {len(cq)} rows, {cq['cluster_id'].nunique()} clusters")

    # ── Build per-question raw frequency from the original CSV ───────────────
    print(f"\nLoading {args.raw_file} ...")
    raw = pd.read_csv(args.raw_file, low_memory=False)
    raw['QueryText'] = raw['QueryText'].astype(str)

    # Filter to target crop (strip parentheses for flexible matching)
    raw_crop_norm = raw['Crop'].str.replace(r'[\(\)]', '', regex=True).str.strip()
    crop_norm     = re.sub(r'[\(\)]', '', args.crop).strip()
    raw_crop      = raw[raw_crop_norm == crop_norm]

    freq_map    = raw_crop.groupby('QueryText').size().to_dict()
    total_raw   = len(raw_crop)
    print(f"  {total_raw:,} raw rows for '{args.crop}', "
          f"{len(freq_map):,} unique query texts")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_file  = out_dir / 'unique_questions_checkpoint.json'
    checkpoint = {}
    if args.resume and ckpt_file.exists():
        with open(ckpt_file) as f:
            checkpoint = json.load(f)
        print(f"  Resumed: {len(checkpoint)} clusters already in checkpoint")

    # ── Sort clusters largest-first ───────────────────────────────────────────
    cluster_list = sorted(
        [(cid, grp.reset_index(drop=True)) for cid, grp in cq.groupby('cluster_id')],
        key=lambda x: int(x[1]['query_volume'].iloc[0]), reverse=True,
    )

    # ── LLM grouping ──────────────────────────────────────────────────────────
    if args.api_provider == "anthropic":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            ap.error("Provide --api-key or set ANTHROPIC_API_KEY env var")
        judge_claude = ClaudeBatchJudge(api_key, model=args.claude_model)

        # Build list of clusters not yet cached
        clusters_todo = []
        for cid, grp in cluster_list:
            cid_key   = str(int(cid))
            questions = grp['question'].tolist()
            label     = str(grp['label'].iloc[0]).strip()
            if cid_key not in checkpoint:
                if len(questions) == 1:
                    checkpoint[cid_key] = [{"group_id": 1,
                                            "answer_label": label,
                                            "indices": [0]}]
                else:
                    clusters_todo.append((cid_key, questions, label, args.crop))

        if clusters_todo:
            checkpoint = judge_claude.process_clusters(
                clusters_todo, checkpoint, ckpt_file)
        else:
            print("  All clusters already in checkpoint — nothing to submit")
        judge_local = None

    else:  # local Qwen 7B
        print(f"\nLoading RepairJudge on GPU {args.gpu_id}...")
        judge_local = RepairJudge(args.model,
                                  batch_size=args.batch_size,
                                  gpu_id=args.gpu_id)
        print(f"\nProcessing {len(cluster_list)} clusters...\n{'='*60}")
        for cid, grp in tqdm(cluster_list, desc="Clusters"):
            cid_key   = str(int(cid))
            questions = grp['question'].tolist()
            label     = str(grp['label'].iloc[0]).strip()
            if cid_key in checkpoint:
                continue
            if len(questions) == 1:
                groups = [{"group_id": 1, "answer_label": label, "indices": [0]}]
            else:
                groups = find_unique_questions_local(
                    judge_local, questions, label, args.crop)
            checkpoint[cid_key] = groups
            with open(ckpt_file, 'w') as f:
                json.dump(checkpoint, f)

    # ── Build uq_rows from checkpoint ─────────────────────────────────────────
    print("\nBuilding output rows from checkpoint...")
    uq_rows = []
    q2uq_id = {}

    for cid, grp in cluster_list:
        cid_key   = str(int(cid))
        questions = grp['question'].tolist()
        label     = str(grp['label'].iloc[0]).strip()
        c_rank    = int(grp['rank'].iloc[0])
        c_vol     = int(grp['query_volume'].iloc[0])
        was_split = bool(grp['was_split'].iloc[0])
        parent_cl = grp['parent_cluster'].iloc[0]
        merged_fr = grp['merged_from'].iloc[0]
        groups    = checkpoint.get(cid_key, [])

        for g in groups:
            valid_idxs = [i for i in g['indices'] if i < len(questions)]
            group_qs   = [questions[i] for i in valid_idxs]
            if not group_qs:
                continue

            raw_freq = sum(freq_map.get(q, 0) for q in group_qs)
            rep_q    = max(group_qs, key=lambda q: freq_map.get(q, 0))
            gid      = g.get('group_id', len(uq_rows) + 1)
            uq_id    = f"{int(cid)}_{gid}"

            uq_rows.append(dict(
                unique_q_id              = uq_id,
                cluster_id               = int(cid),
                cluster_rank             = c_rank,
                cluster_label            = label,
                group_id_in_cluster      = gid,
                answer_label             = g['answer_label'],
                representative_question  = rep_q,
                n_questions_in_group     = len(group_qs),
                raw_frequency            = raw_freq,
                pct_of_cluster_volume    = round(raw_freq / c_vol * 100, 2) if c_vol > 0 else 0.0,
                was_cluster_split        = was_split,
                parent_cluster           = parent_cl,
                merged_from              = merged_fr,
                merged_cross_cluster     = False,
            ))
            for i in valid_idxs:
                q2uq_id[(int(cid), questions[i])] = uq_id

    # ── Cross-cluster dedup ───────────────────────────────────────────────────
    uq_rows = cross_cluster_dedup(uq_rows, freq_map,
                                   sim_thresh=args.dedup_thresh)

    # ── Rebuild q2uq_id after dedup (some unique_q_ids may have been absorbed) ─
    # The mapping is still valid since we kept the surviving unique_q_ids.
    # Just update raw_freq in the map for tracing purposes.

    # ── Join unique_q_id back to cluster_questions.csv ────────────────────────
    cq['unique_q_id'] = cq.apply(
        lambda r: q2uq_id.get((int(r['cluster_id']), r['question'])), axis=1)
    cq['raw_freq_individual'] = cq['question'].map(freq_map).fillna(0).astype(int)

    # ── Build final DataFrames ────────────────────────────────────────────────
    uq_df = pd.DataFrame(uq_rows)
    uq_df = uq_df.sort_values(['cluster_rank', 'raw_frequency'],
                               ascending=[True, False]).reset_index(drop=True)

    # unique_questions.csv
    uq_out = out_dir / 'unique_questions.csv'
    uq_df.to_csv(uq_out, index=False)
    print(f"  → {uq_out}  ({len(uq_df)} groups)")

    # unique_questions_freq.csv — key output: rank by frequency, clean columns
    freq_cols = ['rank', 'unique_q_id', 'representative_question', 'raw_frequency',
                 'cluster_id', 'cluster_rank', 'cluster_label', 'answer_label',
                 'n_questions_in_group', 'pct_of_cluster_volume']
    freq_df = uq_df.copy()
    freq_df = freq_df.sort_values('raw_frequency', ascending=False).reset_index(drop=True)
    freq_df.insert(0, 'rank', range(1, len(freq_df)+1))
    freq_df[[c for c in freq_cols if c in freq_df.columns]].to_csv(
        out_dir / 'unique_questions_freq.csv', index=False)
    print(f"  → {out_dir / 'unique_questions_freq.csv'}")

    # unique_question_mapping.csv
    map_out  = out_dir / 'unique_question_mapping.csv'
    map_cols = ['unique_q_id', 'cluster_id', 'question', 'raw_freq_individual',
                'is_representative', 'rank', 'cluster_label', 'representative',
                'n_unique_questions', 'query_volume', 'pct_of_total',
                'was_split', 'parent_cluster', 'merged_from']
    map_cols = [c for c in map_cols if c in cq.columns] + \
               [c for c in cq.columns if c not in map_cols]
    cq[map_cols].to_csv(map_out, index=False)
    print(f"  → {map_out}  ({len(cq)} rows)")

    # ── Verification ──────────────────────────────────────────────────────────
    sum_freq      = int(uq_df['raw_frequency'].sum())
    noise         = total_raw - sum_freq
    n_uq          = len(uq_df)
    n_input_qs    = len(cq)
    n_clusters    = len(cluster_list)

    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")
    print(f"  Total raw rows (crop-filtered)   : {total_raw:,}")
    print(f"  Sum of group raw_frequency values : {sum_freq:,}")
    print(f"  Unaccounted (noise / off-topic)  : {noise:,}")
    print(f"  Frequency check (should be ≤)    : "
          f"{'✓ PASS' if sum_freq <= total_raw else '✗ FAIL'}")
    print()
    print(f"  Input unique question texts      : {n_input_qs:,}  (cluster_questions.csv)")
    print(f"  Output answer-distinct groups    : {n_uq:,}  (unique_questions.csv)")
    print(f"  Reduction ratio                  : {n_input_qs/n_uq:.1f}x")
    print(f"  Avg unique answers per cluster   : {n_uq/n_clusters:.1f}")
    print()
    # Per-cluster distribution
    groups_per_cluster = uq_df.groupby('cluster_id')['group_id_in_cluster'].max()
    print(f"  Clusters with 1 unique answer    : {(groups_per_cluster == 1).sum()}")
    print(f"  Clusters with 2-3 unique answers : {((groups_per_cluster >= 2) & (groups_per_cluster <= 3)).sum()}")
    print(f"  Clusters with 4+ unique answers  : {(groups_per_cluster >= 4).sum()}")
    print(f"{'='*60}")

    # Save a verification row to the summary
    verif = dict(
        total_raw_rows        = total_raw,
        sum_group_frequencies = sum_freq,
        noise_rows            = noise,
        freq_check_pass       = (sum_freq <= total_raw),
        n_input_questions     = n_input_qs,
        n_unique_answer_groups= n_uq,
        reduction_ratio       = round(n_input_qs / n_uq, 2),
        avg_answers_per_cluster = round(n_uq / n_clusters, 2),
    )
    pd.DataFrame([verif]).to_csv(out_dir / 'unique_questions_verification.csv', index=False)
    print(f"\n  → {out_dir / 'unique_questions_verification.csv'}")


if __name__ == '__main__':
    main()
