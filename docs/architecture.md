# Architecture

## System Overview

FAQCluster is split into two Docker containers that communicate via HTTP over a Tailscale VPN:

```
┌─────────────────────────────────────────────────────────────┐
│  Production VM (GPU server)                                 │
│                                                             │
│  ┌─────────────────────────────┐                           │
│  │  pipeline container :8031   │                           │
│  │  FastAPI (pipeline_server)  │ ←── REST API calls        │
│  │  + all ML pipeline code     │                           │
│  └──────────┬──────────────────┘                           │
│             │ subprocess                                     │
│  ┌──────────▼──────────────────┐                           │
│  │  run_pipeline.py (per crop) │                           │
│  │  Stages 1–7                 │                           │
│  └─────────────────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│  Any machine (no GPU needed)     │
│  frontend container :8030        │
│  React + Nginx                   │
│  → calls pipeline :8031          │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  Zoho WorkDrive (cloud storage)  │
│  Authoritative persistent store  │
│  • cleaned_data.csv (input)      │
│  • outputs/repair/**             │
│  • master.json (state table)     │
└──────────────────────────────────┘
```

The pipeline container runs on a GPU machine. The frontend can run anywhere and talks to the pipeline server over the network (Tailscale in production).

**Zoho WorkDrive is the authoritative persistent store.** `app-data/` inside the container is local scratch space that is populated on demand and cleaned up after each crop's outputs are pushed to Zoho.

---

## Component Map

| Component | File | Role |
|-----------|------|------|
| FastAPI server | `pipeline_server.py` | HTTP entry point, job dispatch, file routes, Zoho sync |
| Job control | `_job_ctl.py` | Per-job cancel events and subprocess tracking |
| Full orchestrator | `run_full.py` | CLI: pre → per-crop pipeline → post |
| Pipeline orchestrator | `run_pipeline.py` | CLI: 7-stage per-crop pipeline with auto-skip and Zoho push |
| Pre-pipeline | `run_pre_pipeline.py` | CLI: state filter + crop normalization |
| Post-pipeline | `run_post_pipeline.py` | CLI: LLM dedup + review file generation |
| Stage 1 | `pipeline/hyperparameter_tuning.py` | HDBSCAN+UMAP grid search |
| Stage 2 | `pipeline/llm_evaluator_hf.py` | LLM cluster-quality evaluator (remote API) |
| Stage 3 | `pipeline/cluster_repair.py` | 5-step LLM cluster repair |
| Stage 4 | `pipeline/unique_question_finder.py` | Answer-distinct question grouper |
| Stage 5 | `pipeline/dedup_freq_csv.py` | Exact-string deduplication |
| Stage 6 | `pipeline/filter_faq_corpus.py` | Keyword corpus filter |
| Stage 7 | `pipeline/vllm_batch_qa_generator.py` | Q&A generation via remote API |
| Post-dedup | `post_pipeline/post_processing_dedup.py` | 2-pass LLM dedup with embedding pre-pass |
| Review gen | `post_pipeline/generate_review_file.py` | Human-review CSV builder |
| State filter | `pre_pipeline/get_state_crop_rows.py` | Filters raw CSV by state/district/domain |
| Crop normalizer | `pre_pipeline/crop_normalizer.py` | Maps raw crop names to canonical forms |
| Crop mapping | `pre_pipeline/mapping.py` | Static {raw variant → canonical} dict (400+ entries) |
| Corpus config | `config/irrelevant_corpus.yaml` | Keyword blocklist for Stage 6 |
| Crop config | `crops.yaml` | Per-crop keyword lists for cross-crop filter |
| Zoho client | `helpers/zoho_workdrive.py` | Zoho WorkDrive CRUD (list, upload, download, delete, rename, move) |

---

## Data Flow

### Full Run (via `/run/full` or `run_full.py`)

```
1. Zoho: stream cleaned_data.csv into /tmp
2. Pre-pipeline
   a. get_state_crop_rows.py   → state_rows.csv   (filter by state/district/domain)
   b. crop_normalizer.py       → normalized.csv   (canonical crop names)
3. Zoho: upload normalized.csv + meta.json
4. For each crop:
   a. Zoho: download any existing intermediate files (enables stage-skipping)
   b. run_pipeline.py          (7 stages, see pipeline.md)
   c. Post-pipeline per crop:
      - post_processing_dedup.py  → {district}_{crop}.csv
      - generate_review_file.py   → {district}_{crop}_review.csv
   d. Zoho: upload final CSVs, clean up local folder
5. Zoho: update master.json (state table)
```

### Per-crop Intermediate Files

```
normalized.csv (input)
      ↓ Stage 1
phase1_results.pkl  phase1_candidates.csv
      ↓ Stage 2
phase2_scores.csv
      ↓ Stage 3
repaired_clusters.csv  cluster_questions.csv  raw_row_mapping.csv
      ↓ Stage 4
unique_questions.csv  unique_question_mapping.csv  unique_questions_checkpoint.json
unique_questions_freq.csv  (sorted by frequency)
      ↓ Stage 5 (in-place)
unique_questions_freq.csv  (deduped)
      ↓ Stage 6 (in-place)
unique_questions_freq.csv  (filtered)
corpus_filtered_out.csv
      ↓ Stage 7
unique_questions_freq_qa.csv
      ↓ Post-pipeline
phase_data_faq.csv  {district}_{crop}.csv  {district}_{crop}_review.csv
```

---

## LLM Integration

All LLM calls go through an **OpenAI-compatible REST endpoint** configured via environment variables. The default model is `google/gemma-4-26B-A4B-it`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_API_URL` | `http://100.100.108.44:8013/v1/chat/completions` | Remote inference endpoint |
| `LLM_MODEL` | `google/gemma-4-26B-A4B-it` | Model name |
| `LLM_API_KEY` | `""` | Bearer token (optional) |
| `LLM_THINKING_ENABLED` | `false` | Enable Gemma thinking tokens |

**Stage 4 (unique question finder)** can optionally use **Claude Haiku** via the Anthropic Batch API when `--api-key` is supplied. This requires no GPU and handles large clusters better through the batch API's polling model.

---

## Zoho WorkDrive Integration

Zoho is the **single source of truth** for all pipeline data. The server never assumes a file is locally present.

### Key patterns

| Operation | Code path |
|-----------|-----------|
| Download one file | `_zoho_sync_down(rel_path)` → `APP_DATA/<rel_path>` |
| Upload one file | `_zoho_sync_up(local_path)` |
| Upload directory | `_zoho_sync_up_dir(local_dir)` |
| Stream large file to /tmp | `_stream_zoho_to_tmp(zoho_path)` (context manager) |
| Walk+download folder | `_zoho_walk_down(zoho_path, local_base)` |

The `ZohoWorkDrive` client (`helpers/zoho_workdrive.py`):
- Refreshes the OAuth2 access token proactively every 45 minutes (tokens expire hourly).
- Retries HTTP 429 / 502 / 503 up to 3 times with exponential back-off.
- Does **not** retry HTTP 500 (these are real API errors).
- Returns HTTP 503 to the API caller on `ConnectionError` or `RetryError`.

### Folder structure in Zoho

```
<root>/
├── cleaned_data.csv              ← raw KCC dataset (not produced here)
├── <state>/
│   ├── <district>_0/
│   │   ├── meta.json             ← {"state", "district", "domains", "crops"}
│   │   └── <district>_0.csv      ← normalized pre-pipeline output
│   └── <district>_1/             ← versioned if domains/state changed
├── outputs/repair/<state>/<district>/<crop>/
│   ├── phase1_results.pkl
│   ├── phase2_scores.csv
│   ├── repaired_clusters.csv
│   ├── cluster_questions.csv
│   ├── unique_questions_freq.csv
│   ├── unique_questions_freq_qa.csv
│   ├── corpus_filtered_out.csv
│   ├── phase_data_faq.csv
│   ├── {district}_{crop}.csv     ← final deduped FAQ
│   └── audit_*.csv               ← human-uploaded audits
└── master.json                   ← state table
```

---

## master.json

A persistent state table built once at startup by walking `outputs/repair/`, then updated incrementally.

```json
{
  "built_at": "2025-06-01T10:00:00Z",
  "data": {
    "<state>": {
      "<district>": {
        "<crop>": {
          "output_file": "outputs/repair/…/….csv",
          "audit_file":  "outputs/repair/…/audit_….csv",
          "downloaded":  false,
          "audited":     false,
          "processed":   true,
          "finished_at": "2025-06-01T10:05:00Z"
        }
      }
    }
  }
}
```

`/app/state-table` serves this from the in-memory `_master_data` cache. Pass `?refresh=true` to trigger a full Zoho walk rebuild.

---

## Output File Schemas

### `unique_questions_freq.csv` (FAQ questions — main output of Stages 1–6)

| Column | Description |
|--------|-------------|
| `rank` | Frequency rank (1 = most asked) |
| `unique_q_id` | `<cluster_id>_<group_id>` |
| `representative_question` | The question shown in the FAQ |
| `sample_questions` | Top-5 phrasings pipe-separated |
| `raw_frequency` | Raw KCC call count |
| `cluster_id` | Source cluster |
| `cluster_rank` | Cluster rank by volume |
| `cluster_label` | Topic label |
| `answer_label` | Answer-group description |
| `n_questions_in_group` | Number of distinct phrasings grouped |
| `pct_of_cluster_volume` | % of cluster's query volume |
| `was_cluster_split` | Whether the cluster was split in Stage 3C |
| `parent_cluster` | Original cluster before split |
| `merged_from` | Source group IDs if cross-cluster merged |
| `merged_cross_cluster` | True if merged across cluster boundaries |

### `unique_questions_freq_qa.csv` (FAQ Q&A pairs — after Stage 7)

All columns above, plus:

| Column | Description |
|--------|-------------|
| `Generated_Question` | Polished English question |
| `Generated_Category` | Disease / Pest / Fertilizer and Nutrient / Variety / Agronomy / Other / IRRELEVANT_CROP |
| `Generated_Answer` | 200–400 word technical answer |
