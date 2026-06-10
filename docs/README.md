# FAQCluster Documentation

FAQCluster is an end-to-end FAQ generation system for the Kisan Call Centre (KCC) agricultural dataset. It transforms raw farmer query logs into a clean, ranked FAQ CSV with professionally written English Q&A pairs.

```
Raw KCC CSV
   ↓  Pre-Pipeline (state filter + crop normalization)
Cleaned CSV
   ↓  7-Stage Pipeline (per crop)
   1. HDBSCAN + UMAP hyperparameter screening
   2. LLM evaluation of top clustering configs
   3. Cluster repair (diverse reps, cross-crop filter, coherence split, merge, back-mapping)
   4. Unique question extraction (answer-distinct grouping)
   5. Final deduplication
   6. Irrelevant corpus filter
   7. Q&A generation
Unique FAQ Q&A CSV
   ↓  Post-Pipeline (LLM dedup + review generation)
Final Deduplicated FAQ
```

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [Architecture](architecture.md) | System design, component map, data flow, Zoho integration |
| [Backend / API Server](backend.md) | FastAPI routes, job system, Zoho sync helpers, path safety |
| [Pipeline (Stages 1–7)](pipeline.md) | Every pipeline stage in full detail |
| [Pre-Pipeline](pre_pipeline.md) | State filter, district filter, domain filter, crop normalization |
| [Post-Pipeline](post_pipeline.md) | LLM deduplication, embedding pre-pass, review file generation |
| [Entry Points](entry_points.md) | CLI reference for all four run scripts |
| [Deployment](deployment.md) | Docker Compose, GitHub Actions CI/CD, environment variables |
| [Frontend](frontend.md) | React UI, API contract, component overview |

---

## Project Structure

```
FAQCluster/
├── docker-compose.yml          # Compose file for pipeline + frontend services
├── .env                        # Secrets: Zoho credentials, LLM API key
├── pipeline_server/
│   ├── pipeline_server.py      # FastAPI app — port 8031
│   ├── _job_ctl.py             # Cancellable background-job registry
│   ├── run_pipeline.py         # 7-stage per-crop pipeline orchestrator
│   ├── run_pre_pipeline.py     # Pre-pipeline: state filter + crop normalizer
│   ├── run_post_pipeline.py    # Post-pipeline: LLM dedup + review generation
│   ├── run_full.py             # Full pipeline: pre → per-crop → post
│   ├── crops.yaml              # Crop keyword lists (used by corpus filter)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── requirements.lock
│   ├── pipeline/               # Core ML modules (Stages 1–7)
│   │   ├── hyperparameter_tuning.py     # Stage 1 — HDBSCAN+UMAP grid search
│   │   ├── llm_evaluator_hf.py          # Stage 2 — LLM config evaluator
│   │   ├── cluster_repair.py            # Stage 3 — LLM cluster repair
│   │   ├── unique_question_finder.py    # Stage 4 — answer-distinct grouping
│   │   ├── dedup_freq_csv.py            # Stage 5 — final deduplication
│   │   ├── filter_faq_corpus.py         # Stage 6 — irrelevant corpus filter
│   │   ├── vllm_batch_qa_generator.py   # Stage 7 — Q&A generation
│   │   └── cluster_mapping.py
│   ├── pre_pipeline/
│   │   ├── get_state_crop_rows.py       # State/district/domain CSV filter
│   │   ├── crop_normalizer.py           # Canonical crop-name mapper
│   │   ├── mapping.py                   # Raw-variant → canonical name dict
│   │   └── ...
│   ├── post_pipeline/
│   │   ├── post_processing_dedup.py     # 2-pass LLM dedup with embedding pre-pass
│   │   └── generate_review_file.py      # Farmer-question review CSV builder
│   ├── config/
│   │   └── irrelevant_corpus.yaml       # Keyword blocklist for Stage 6
│   ├── helpers/
│   │   └── zoho_workdrive.py            # Zoho WorkDrive CRUD client
│   └── app-data/                        # Local scratch (Docker volume)
├── helpers/                    # Standalone utility scripts (not used by server)
└── docs/                       # This documentation
```

---

## Quick Start

### Production (Docker)

```bash
# Copy .env to the VM, fill in credentials, then:
docker compose pull
docker compose up -d
# UI at http://<vm-ip>:8030
```

### Local Development

```bash
cd pipeline_server
python -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.lock
uvicorn pipeline_server:app --host 0.0.0.0 --port 8031 --reload
```
