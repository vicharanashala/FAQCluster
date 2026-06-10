# Entry Points

These four scripts are the top-level CLI interfaces for the pipeline system. They live inside `pipeline_server/`. The pipeline server calls them (directly or as subprocesses); users can also invoke them from the terminal from within the `pipeline_server/` directory.

All four scripts:
- Load `.env` from the repo root automatically (via `python-dotenv`)
- Import `_job_ctl` for cancel-event integration when run inside the server
- Fall back gracefully when `_job_ctl` is not available (standalone CLI mode)

---

## `run_pre_pipeline.py`

State filter + crop normalization. Produces a normalized CSV ready to pass as `--raw-file` to `run_pipeline.py` or `run_full.py`.

### Usage

```bash
python run_pre_pipeline.py \
    --input  zoho_raw.csv \
    --state  Karnataka \
    --crops  Cotton Sugarcane "Sugar Beet" \
    --output karna_norm.csv
```

### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | Yes | — | Raw KCC CSV file |
| `--state` | Yes | — | State name to filter by (e.g. `Karnataka`) |
| `--crops` | Yes | — | One or more primary crop names to keep |
| `--output` | Yes | — | Output path for the normalized CSV |
| `--keep-intermediate` | No | False | Keep the state-filtered intermediate CSV (normally deleted) |

### What it does
1. Calls `pre_pipeline/get_state_crop_rows.py` to filter rows by state.
2. Calls `pre_pipeline/crop_normalizer.py` to map raw crop names to canonical forms, dropping rows not in the crop list.
3. Deletes the intermediate file unless `--keep-intermediate` is set.

---

## `run_pipeline.py`

7-stage per-crop pipeline. Produces FAQ questions and Q&A pairs for one crop.

### Usage

```bash
# Minimal (local LLM, default grid)
python run_pipeline.py \
    --raw-file karna_norm.csv \
    --crop "Cotton"

# Full run with Claude Haiku for Stage 4 and medium grid
python run_pipeline.py \
    --raw-file karna_norm.csv \
    --crop "Cotton" \
    --api-key sk-ant-... \
    --grid-mode medium \
    --output-dir outputs/repair

# Resume a partial run (stages already done are skipped automatically)
python run_pipeline.py \
    --raw-file karna_norm.csv \
    --crop "Cotton" \
    --skip-phase1    # phase1_results.pkl already exists
```

### Arguments

#### I/O (required)
| Flag | Description |
|------|-------------|
| `--raw-file` | Path to normalized KCC CSV |
| `--crop` | Crop name as it appears in the `Crop` column |

#### Model / API
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `google/gemma-4-26B-A4B-it` (from env) | Remote LLM for Stages 2, 3, 7 |
| `--api-key` | — | Anthropic API key for Claude Haiku (Stage 4). If omitted, Stage 4 uses the remote LLM. |
| `--gpu-id` | `0` | CUDA device index |
| `--batch-size` | `8` | LLM batch size |

#### Pipeline control
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `outputs/repair` | Base output directory |
| `--skip-phase1` | False | Skip Stage 1 (load existing `phase1_results.pkl`) |
| `--skip-phase2` | False | Skip Stage 2 (load from `phase2_scores.csv` or auto-detect) |
| `--skip-repair` | False | Skip Stage 3 (use existing `cluster_questions.csv`) |
| `--skip-unique-q` | False | Skip Stage 4 (use existing `unique_questions_freq.csv`) |
| `--skip-corpus-filter` | False | Skip Stage 6 |
| `--skip-qa-gen` | False | Skip Stage 7 |
| `--corpus-file` | `config/irrelevant_corpus.yaml` | Irrelevant keyword blocklist |
| `--crops-file` | `crops.yaml` (if present) | Crops YAML for cross-crop filter |
| `--fuzz-threshold` | `100` | Fuzzy match threshold for corpus filter (100 = disabled) |

#### Tuning
| Flag | Default | Description |
|------|---------|-------------|
| `--max-queries` | `20000` | Max unique queries before random sampling |
| `--grid-mode` | `medium` | HP grid: `quick`(18), `medium`(108), `full`(240), `exhaustive`(480) |
| `--phase2-top-k` | `5` | Number of Phase 1 candidates to LLM-evaluate |
| `--coverage-cap` | `0.80` | Fraction of query volume to evaluate in Phase 2 |

#### Repair
| Flag | Default | Description |
|------|---------|-------------|
| `--diverse-k` | `3` | Max-diverse reps per cluster in Step A |
| `--coherence-flag` | `C` | Rating that triggers a split (`B`=aggressive, `C`=conservative) |
| `--merge-sim` | `0.82` | Cosine similarity threshold for Step D merge candidates |

### Output directory structure
```
outputs/repair/<state>/<district_folder>/<crop_slug>/
├── phase1_results.pkl
├── phase1_candidates.csv
├── phase2_scores.csv
├── repaired_clusters.csv
├── cluster_questions.csv
├── raw_row_mapping.csv
├── unique_questions.csv
├── unique_questions_freq.csv         ← FAQ questions
├── unique_question_mapping.csv
├── corpus_filtered_out.csv
└── unique_questions_freq_qa.csv      ← FAQ Q&A pairs
```

The `<district_folder>` component is the stem of the `--raw-file` path (e.g. `karnataka_0` if the file is `karnataka_0.csv`).

---

## `run_post_pipeline.py`

Post-pipeline: LLM dedup + review file generation. Processes all crops in a repair output directory.

### Usage

```bash
# Process all crops in the output directory
python run_post_pipeline.py \
    --input outputs/repair/karnataka/karnataka_0

# Process specific crops only
python run_post_pipeline.py \
    --input outputs/repair/karnataka/karnataka_0 \
    --crops Cotton Sugarcane

# Skip dedup (generate review files only)
python run_post_pipeline.py \
    --input outputs/repair/karnataka/karnataka_0 \
    --skip-dedup
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Base output directory containing per-crop subdirectories |
| `--crops` | All subdirs | Crop names to process (matches subdirectory names) |
| `--skip-dedup` | False | Skip LLM deduplication (only generate review files) |

### Per crop
Finds subdirectories of `--input` whose names match the slugified crop names. For each:
1. Reads `unique_questions_freq_qa.csv`
2. Runs `deduplicate_and_aggregate` → writes `{district}_{crop}.csv`
3. Runs `generate_review` → writes `{district}_{crop}_review.csv`
4. Uploads both to Zoho

---

## `run_full.py`

Full end-to-end CLI: pre-pipeline → per-crop pipeline → post-pipeline. This is the standalone equivalent of the server's `/run/full` endpoint.

### Usage

```bash
# Full run: raw CSV → FAQ Q&A pairs
python run_full.py \
    --raw-file zoho_raw.csv \
    --state Karnataka \
    --crops Cotton Sugarcane \
    --model google/gemma-4-26B-A4B-it \
    --grid-mode medium

# Skip pre-pipeline (already have normalized CSV)
python run_full.py \
    --raw-file karna_norm.csv \
    --crops Cotton Sugarcane \
    --skip-pre-pipeline

# Load crop list from a text file
python run_full.py \
    --raw-file zoho_raw.csv \
    --state Karnataka \
    --crops-file crops.txt \
    --model google/gemma-4-26B-A4B-it
```

### Arguments

#### I/O (required)
| Flag | Description |
|------|-------------|
| `--raw-file` | Raw KCC CSV (or normalized CSV if `--skip-pre-pipeline`) |
| `--crops` or `--crops-file` | Crop names (mutually exclusive) |

The `--crops-file` format is one crop per line; lines starting with `#` are treated as comments.

#### Model / API
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `google/gemma-4-26B-A4B-it` | Remote LLM model |
| `--api-key` | — | Anthropic API key for Claude Haiku (Stage 4) |
| `--gpu-id` | `0` | CUDA device index |
| `--batch-size` | `8` | LLM batch size |

#### Pipeline control
| Flag | Default | Description |
|------|---------|-------------|
| `--state` | — | State name (required unless `--skip-pre-pipeline`) |
| `--output-dir` | `outputs/repair` | Base output directory |
| `--grid-mode` | `quick` | HP grid size |
| `--skip-pre-pipeline` | False | Use `--raw-file` directly as normalized input |
| `--skip-qa-gen` | False | Skip Stage 7 for all crops |
| `--skip-post-pipeline` | False | Skip `run_post_pipeline.py` after all crops |

### Execution order
1. Pre-pipeline: `run_pre_pipeline.py` → writes `<state>_norm.csv`
2. For each crop (sequentially): `run_pipeline.py` as a subprocess
3. Post-pipeline: `run_post_pipeline.py` on the output directory
