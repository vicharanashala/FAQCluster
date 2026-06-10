# Pipeline — 7-Stage FAQ Generation

The core pipeline transforms a normalized crop CSV into a final FAQ dataset with Q&A pairs. It runs once per crop, is **resume-safe** (each stage auto-skips if its output already exists), and is invoked by `run_pipeline.py`.

```
Stage 1  →  phase1_results.pkl
Stage 2  →  phase2_scores.csv
Stage 3  →  repaired_clusters.csv, cluster_questions.csv, raw_row_mapping.csv
Stage 4  →  unique_questions_freq.csv, unique_question_mapping.csv
Stage 5  →  unique_questions_freq.csv  (in-place dedup)
Stage 6  →  unique_questions_freq.csv  (in-place filter), corpus_filtered_out.csv
Stage 7  →  unique_questions_freq_qa.csv
```

---

## Stage 1 — Hyperparameter Screening

**Script:** `pipeline/hyperparameter_tuning.py`  
**Key classes:** `ClusteringConfig`, `ClusteringResult`

Runs a grid search over HDBSCAN + UMAP hyperparameter combinations to identify clustering configurations that produce good, stable clusters.

### Input
The normalized CSV, filtered to the target crop, deduplicated to unique `query_text` strings with their call counts.

### Embedding
`paraphrase-multilingual-mpnet-base-v2` (SentenceTransformer) embeds all queries. Handles Hindi/Hinglish/English text natively.

### Hybrid distance matrix
For each alpha value: `hybrid_dist = α × cosine(dense) + (1−α) × jaccard(TF-IDF bool)`. This blends semantic similarity (dense) with lexical overlap (sparse). Results are cached per alpha to avoid recomputation across configs that share the same alpha.

### UMAP + HDBSCAN
UMAP reduces the hybrid distance matrix to `n_components` dimensions. HDBSCAN clusters the reduced representation. Noise points (`label == -1`) with frequency > 50 get their own cluster; remaining noise is reassigned via 1-NN.

### Viability screening
A config is "viable" if:
- `50 ≤ n_clusters ≤ 1500`
- `noise_ratio < 30 %`
- `clusters_for_85pct > 5`

If no config passes, the thresholds are relaxed proportionally to dataset size. As a last resort, the best available config by cluster count + noise is returned.

### Grid modes

| Mode | Configs | Typical use |
|------|---------|-------------|
| `quick` | 18 | CI / quick test |
| `medium` | 108 | Default in API server |
| `full` | 240 | CLI default |
| `exhaustive` | 480 | Rarely needed |

### Caching
`_embed_cache`, `_dist_cache`, and `_umap_cache` are passed between calls. Embeddings are computed once; hybrid distances once per alpha; UMAP projections once per `(alpha, n_neighbors, n_components)` triple.

### Outputs
- `phase1_results.pkl` — serialized list of `ClusteringResult` objects
- `phase1_candidates.csv` — metrics for all viable configs

---

## Stage 2 — LLM Cluster Evaluation

**Script:** `pipeline/llm_evaluator_hf.py`  
**Class:** `LocalHFJudge`

Evaluates the top-K Phase 1 candidates using an LLM to select the best clustering configuration.

### Evaluation passes (4 per config)

**Pass 1 — Coherence**
For each cluster (coverage-capped): asks the LLM whether all questions are about the same agricultural topic. Scores: A=1.0, B=0.7, C=0.2. Weighted by cluster query volume.

**Pass 2 — Separation**
For size-biased random cluster pairs: asks whether an agricultural officer would give *different* practical advice. Scores: A=1.0 (different), B=0.6, C=0.0.

**Pass 3 — Merge detection**
For cluster pairs sharing keywords: asks whether the pair should be merged. Scores: A=1.0 (should merge), B=0.0.

**Pass 4 — Outlier detection**
For each cluster (coverage-capped, up to 10 questions): asks which question (if any) doesn't belong. Counts the fraction with a detected outlier.

### Composite score formula
```
composite = 0.45 × separation_mean
           + 0.30 × coherence_mean
           + 0.25 × (1 − outlier_rate)
           − 0.20 × merge_rate
```

High separation + high coherence + low outliers − low merge rate = best config.

### Coverage cap
`--coverage-cap 0.80` means only evaluate clusters that together cover 80% of query volume. Since larger clusters dominate, this typically evaluates ~25% of clusters and is ~4× faster with comparable accuracy.

### Stratified selection
When `--top-k > 0`, candidates are selected using stratified sampling across top/middle/diverse/bottom strata rather than just taking the top K by Phase 1 metrics.

### Outputs
- `phase2_scores.csv` — composite and component scores per config
- Best config string returned to the pipeline orchestrator

---

## Stage 3 — Cluster Repair (Steps A–E)

**Script:** `pipeline/cluster_repair.py`  
**Class:** `RepairJudge` (extends `LocalHFJudge`)

Re-runs the best clustering config and applies 5 cleanup steps to fix common clustering artifacts.

### Step A — Max-diversity Representative Selection

Pure embedding math, no LLM. For each cluster:
1. Find the **centroid-nearest** question (most typical, becomes the new representative).
2. Use **greedy furthest-point selection** to pick `k` maximally diverse representatives.

The diverse set is used as a diagnostic sample for Steps B and C. The centroid-nearest question becomes the cluster representative.

### Step B — Cross-crop Contamination Filter

LLM prompt: *"Identify queries that clearly ask about a DIFFERENT crop than {crop}."*  
Returns a JSON object `{"off": [1-based indices]}`. Questions at those indices are removed from the cluster. Clusters reduced to < 2 questions are deleted entirely. All clusters are evaluated in parallel (up to 16 threads).

### Step C — Coherence Diagnostic + Split

Two parallel phases:

**Phase 1 (coherence check):** For clusters with ≥ 2 diverse reps: *"Would an agricultural extension officer give the SAME specific practical advice for ALL of these questions?"* Returns A/B/C.

**Phase 2 (split):** Clusters rated at or below `--coherence-flag` (default `C`) are sent to a detailed split prompt: *"Group these queries so ALL queries in each group need the SAME specific practical advice."* Returns a JSON array of `{group, label, indices}`.

Sub-groups with < 2 questions are merged into the largest group. New sub-clusters are assigned new IDs with `split_label` and `parent_cluster` metadata.

### Step D — Merge Near-duplicate Clusters

Encodes all cluster representatives with SentenceTransformer. Finds pairs with cosine similarity ≥ `--merge-sim` (default 0.82). For each candidate pair (up to 100, sorted by descending similarity): LLM confirms whether to merge. The smaller cluster is absorbed into the larger.

### Step E — Raw Row Back-mapping

Builds `query_text → final_cluster_id` reverse index. Joins with the full raw CSV to produce:
- `repaired_clusters.csv` — one row per final cluster with rank, label, volume, and lineage metadata
- `cluster_questions.csv` — one row per unique question per cluster
- `raw_row_mapping.csv` — one row per raw CSV row with its assigned cluster

---

## Stage 4 — Unique Question Extraction

**Script:** `pipeline/unique_question_finder.py`

Within each cluster, not all questions need a different answer. This stage groups questions that would receive the **same agricultural advice** into a single "answer-distinct" group.

### Two provider modes

**Anthropic (Claude Haiku Batch API)** — recommended when `--api-key` is supplied:
- Submits all uncached clusters in a single batch request.
- Polls every 30 seconds until `processing_status == "ended"`.
- No GPU required. Suitable for any machine with internet access.

**Remote LLM (default, same endpoint as other stages)**:
- Processes clusters up to `LOCAL_BATCH` (15) questions per LLM call.
- Large clusters are split into batches; batches are run in parallel then cross-batch merged by embedding similarity.
- Up to `CLUSTER_WORKERS` (8) clusters run concurrently.
- Checkpoint flushed to disk every 20 completions for resume safety.

### Grouping criteria (controlled by `LLM_GROUPING_STRICTNESS`)

**strict** (default): `"every question in a group would receive the EXACT SAME specific agricultural advice — same chemical, same dose, same method, same timing"`

**loose**: `"same general recommendation even if minor details like exact dose differ"`

### Cross-cluster dedup

After per-cluster grouping, any two groups from *different* clusters with embedding cosine ≥ `--dedup-thresh` (default 0.85) are merged: the group with higher `raw_frequency` absorbs the other.

### Representative question selection
The representative question for each group is the member with the highest raw call frequency.

### Checkpoint/resume
`unique_questions_checkpoint.json` stores per-cluster LLM results. Pass `--resume` to skip already-processed clusters.

### Outputs
- `unique_questions.csv` — all groups with full metadata
- `unique_questions_freq.csv` — sorted by `raw_frequency` descending (main FAQ output)
- `unique_question_mapping.csv` — every question → its group ID
- `unique_questions_checkpoint.json` — LLM result cache
- `unique_questions_verification.csv` — sanity check row

---

## Stage 5 — Final Deduplication

**Script:** `pipeline/dedup_freq_csv.py`

Case-insensitive exact-string deduplication of `representative_question`. When duplicates exist (same text after strip + lowercase), the row with higher `raw_frequency` is kept. The `rank` column is dropped from the output.

This is a pure Python/pandas operation — no LLM, no GPU, runs in < 1 second even for 1000 rows.

---

## Stage 6 — Irrelevant Corpus Filter

**Script:** `pipeline/filter_faq_corpus.py`  
**Config:** `config/irrelevant_corpus.yaml`

Removes rows whose text fields contain keywords from the irrelevant corpus.

### Checked columns
`representative_question`, `cluster_label`, `answer_label` (all three must pass for a row to be kept).

### Matching logic
1. Multi-word phrases: substring match against lowercased text.
2. Single words (length ≥ `min_word_length`, default 3): exact match against keyword set.
3. Single words: fuzzy match (`fuzz.ratio ≥ fuzz_thresh`, default 100 = disabled) for keywords with similar length (±2 chars).

### Corpus categories
`market_price`, `weather`, `machinery`, `training`, `incomplete_call`, `subsidy_scheme`, `system_logs`, `irrelevant`, `banking_admin`, `contact_details`

### Cross-crop filter
When `--crops-file crops.yaml --crop <name>` is supplied, keywords from all *other* crops in `crops.yaml` (minus keywords shared with the target crop) are added to the blocklist. This removes questions about the wrong crop that survived Stage 3B.

### Outputs
- `unique_questions_freq.csv` (overwritten with filtered result)
- `corpus_filtered_out.csv` (removed rows — for auditing)

---

## Stage 7 — Q&A Generation

**Script:** `pipeline/vllm_batch_qa_generator.py`

Generates professional English Q&A pairs for each FAQ entry. Uses the same remote OpenAI-compatible LLM endpoint as other stages.

### Per-question prompt
The system prompt includes:
- The target crop name, expert agricultural hints specific to that crop
- A list of other crops to reject (synonym-aware)
- Mandatory requirements: English output, step-by-step technical guidance, chemical dosages, safety notes, KVK referral footer, 200–400 word answer

The user prompt includes the `representative_question` plus the top-5 `sample_questions` to ground the answer in what farmers actually asked.

### Output categories
`Disease`, `Pest`, `Fertilizer and Nutrient`, `Variety`, `Agronomy`, `Other`, `IRRELEVANT_CROP`

Rows with `IRRELEVANT_CROP` or `PARSE_ERROR` categories are removed by the post-pipeline dedup step.

### Output format per row
```
Generated_Question  — polished English question
Generated_Category  — one of the categories above
Generated_Answer    — 200–400 word technical answer
```

---

## Auto-skip Logic

`run_pipeline.py` skips a stage automatically if its output file already exists. This enables resuming after interruption without re-flag arguments:

| Stage | Skip condition |
|-------|---------------|
| 1 | `phase1_results.pkl` exists |
| 2 | `phase2_scores.csv` exists |
| 3 | `cluster_questions.csv` exists |
| 4 | `unique_question_mapping.csv` exists |
| 5 | `corpus_filtered_out.csv` or `unique_questions_freq_qa.csv` exists (downstream done) |
| 6 | `corpus_filtered_out.csv` exists |
| 7 | `unique_questions_freq_qa.csv` exists |

Explicit `--skip-*` flags override stage detection. Per-stage Zoho upload ensures intermediate files survive server restarts.

---

## Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--grid-mode` | `medium` (API), `full` (CLI) | HP grid size |
| `--max-queries` | 20000 | Max unique queries before random sampling |
| `--phase2-top-k` | 5 | Number of Phase 1 candidates to LLM-evaluate |
| `--coverage-cap` | 0.80 | Fraction of query volume to evaluate in Phase 2 |
| `--diverse-k` | 3 | Max-diversity reps per cluster in Step A |
| `--coherence-flag` | `C` | LLM rating that triggers a split (B=aggressive, C=conservative) |
| `--merge-sim` | 0.82 | Cosine similarity threshold for merge candidates |
| `--fuzz-threshold` | 100 | Corpus filter fuzzy match ratio (100 = disabled) |
