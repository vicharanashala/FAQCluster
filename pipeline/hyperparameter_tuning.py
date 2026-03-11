#!/usr/bin/env python3
"""
Hyperparameter Tuning for Clustering with LLM Judge

Two-phase approach:
1. Fast screening using geometric metrics (noise ratio, cluster count)
2. Deep LLM evaluation on top candidates using Sarvam 30B

Usage:
    python hyperparameter_tuning.py --crop "Paddy (Dhan)" --max-queries 20000 --api-key YOUR_KEY
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from itertools import product
import time
from datetime import datetime
from collections import Counter

# Import clustering components
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'hyperparameter_tuning'


class ClusteringConfig:
    """Configuration for clustering hyperparameters"""
    def __init__(self, alpha, min_cluster_size, min_samples, n_neighbors, n_components=5):
        self.alpha = alpha
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.n_components = n_components
    
    def __str__(self):
        return f"α={self.alpha}_mcs={self.min_cluster_size}_ms={self.min_samples}_nn={self.n_neighbors}"
    
    def to_dict(self):
        return {
            'alpha': self.alpha,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'n_neighbors': self.n_neighbors,
            'n_components': self.n_components
        }


class ClusteringResult:
    """Container for clustering results"""
    def __init__(self, config, df, labels, metrics):
        self.config = config
        self.df = df
        self.labels = labels
        self.metrics = metrics
        self.clusters = self._build_clusters()
    
    def _build_clusters(self):
        """Build cluster dictionary"""
        clusters = {}
        for cluster_id in np.unique(self.labels):
            if cluster_id == -1:
                continue
            mask = self.labels == cluster_id
            cluster_data = self.df[mask].copy()
            clusters[cluster_id] = {
                'size': cluster_data['count'].sum(),
                'unique_queries': len(cluster_data),
                'queries': cluster_data['query_text'].tolist(),
                'counts': cluster_data['count'].tolist(),
                'representative': cluster_data.sort_values('count', ascending=False).iloc[0]['query_text']
            }
        return clusters


def generate_param_grid(mode='full'):
    """
    Generate hyperparameter grid for search
    
    Args:
        mode: 'quick' (18 configs), 'medium' (108 configs), 'full' (240 configs), 'exhaustive' (480 configs)
    """
    if mode == 'quick':
        # Quick test grid - 18 configs
        param_grid = {
            'alpha': [0.3, 0.5, 0.7],
            'min_cluster_size': [2, 3, 5],
            'min_samples': [2],
            'n_neighbors': [15],
            'n_components': [5]
        }
    elif mode == 'medium':
        # Medium grid - 108 configs (original × 2)
        param_grid = {
            'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # 6 values
            'min_cluster_size': [2, 3, 5],
            'min_samples': [2, 3],
            'n_neighbors': [10, 15, 20],
            'n_components': [5]
        }
    elif mode == 'full':
        # Full grid - 240 configs (comprehensive)
        param_grid = {
            'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 7 values
            'min_cluster_size': [2, 3, 4, 5, 7],  # 5 values
            'min_samples': [1, 2, 3],  # 3 values
            'n_neighbors': [8, 10, 15, 20],  # 4 values
            'n_components': [5]  # Fixed
        }
    else:  # exhaustive
        # Exhaustive grid - 480 configs (maximum exploration)
        param_grid = {
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 9 values
            'min_cluster_size': [2, 3, 4, 5, 6, 7, 8],  # 7 values
            'min_samples': [1, 2, 3, 4],  # 4 values
            'n_neighbors': [5, 8, 10, 15, 20],  # 5 values
            'n_components': [3, 5, 7]  # 3 values
        }
    
    configs = []
    for combo in product(*param_grid.values()):
        config = ClusteringConfig(*combo)
        configs.append(config)
    
    print(f"Generated {len(configs)} configurations in '{mode}' mode")
    return configs


def preprocess_text(text, stop_words):
    """Preprocess text for clustering"""
    if not isinstance(text, str):
        return ""
    
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    
    return " ".join(tokens)


def extract_keywords_tfidf(texts, stop_words, max_features=1000, max_df=0.95, min_df=2):
    """Extract TF-IDF keywords"""
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        stop_words=list(stop_words), 
        max_df=max_df, 
        min_df=min_df
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix, vectorizer
    except ValueError:
        return None, None


def calculate_hybrid_distance(dense_embeddings, tfidf_matrix, alpha):
    """Calculate hybrid distance matrix"""
    # Dense distance
    dense_dist = pairwise_distances(dense_embeddings, metric='cosine', n_jobs=-1)
    
    # Sparse distance
    tfidf_bool = (tfidf_matrix > 0).astype(bool)
    try:
        tfidf_dense_bool = tfidf_bool.toarray()
        sparse_dist = pairwise_distances(tfidf_dense_bool, metric='jaccard', n_jobs=-1)
    except MemoryError:
        return dense_dist
    
    # Fusion
    hybrid_dist = alpha * dense_dist + (1 - alpha) * sparse_dist
    return hybrid_dist


def run_clustering(df, config, model, stop_words, verbose=False, use_gpu=True,
                   _embed_cache=None, _dist_cache=None, _umap_cache=None):
    """
    Run clustering with given configuration.

    Optional shared caches (dicts) to avoid recomputing:
      _embed_cache : {} — stores (embeddings, tfidf_matrix, cleaned_df) keyed by True
      _dist_cache  : {} — stores hybrid_dist keyed by alpha (float)
      _umap_cache  : {} — stores reduced_data keyed by (alpha, n_neighbors, n_components)
    Pass the same dict object across calls to enable caching.
    """
    if verbose:
        print(f"\nRunning clustering with config: {config}")

    # ── Preprocessing + embedding (computed once for entire dataset) ──────────
    if _embed_cache is not None and True in _embed_cache:
        embeddings, tfidf_matrix, df = _embed_cache[True]
    else:
        df = df.copy()
        df['cleaned_text'] = df['query_text'].apply(lambda x: preprocess_text(x, stop_words))
        df = df[df['cleaned_text'].str.strip().astype(bool)].copy()
        if len(df) == 0:
            return None

        tfidf_matrix, vectorizer = extract_keywords_tfidf(df['cleaned_text'], stop_words)
        if tfidf_matrix is None:
            return None

        batch_size = 128 if use_gpu else 32
        embeddings = model.encode(
            df['cleaned_text'].tolist(),
            show_progress_bar=verbose,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        if _embed_cache is not None:
            _embed_cache[True] = (embeddings, tfidf_matrix, df)

    # ── Hybrid distance (one per alpha value) ────────────────────────────────
    alpha_key = round(config.alpha, 4)
    if _dist_cache is not None and alpha_key in _dist_cache:
        hybrid_dist = _dist_cache[alpha_key]
    else:
        hybrid_dist = calculate_hybrid_distance(embeddings, tfidf_matrix, config.alpha)
        if _dist_cache is not None:
            _dist_cache[alpha_key] = hybrid_dist

    # ── UMAP (one per alpha × n_neighbors × n_components) ───────────────────
    umap_key = (alpha_key, config.n_neighbors, config.n_components)
    if _umap_cache is not None and umap_key in _umap_cache:
        reduced_data = _umap_cache[umap_key]
    else:
        umap_model = umap.UMAP(
            n_neighbors=config.n_neighbors,
            min_dist=0.0,
            n_components=config.n_components,
            metric='precomputed',
            random_state=42,
            verbose=False
        )
        reduced_data = umap_model.fit_transform(hybrid_dist)
        if _umap_cache is not None:
            _umap_cache[umap_key] = reduced_data

    # ── HDBSCAN (fast — always re-run per config) ─────────────────────────────
    df = df.copy()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    labels = clusterer.fit_predict(reduced_data)

    # Noise handling
    counts = df['count'].values
    max_label = labels.max()
    noise_indices = np.where(labels == -1)[0]

    for idx in noise_indices:
        if counts[idx] > 50:
            max_label += 1
            labels[idx] = max_label

    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 0:
        remaining_noise = np.where(labels == -1)[0]
        if len(remaining_noise) > 0:
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(reduced_data[non_noise_mask], labels[non_noise_mask])
            predicted = knn.predict(reduced_data[remaining_noise])
            labels[remaining_noise] = predicted

    df['cluster_id'] = labels
    metrics = calculate_basic_metrics(df, labels)
    return ClusteringResult(config, df, labels, metrics)


def calculate_basic_metrics(df, labels):
    """Calculate basic clustering metrics"""
    n_clusters = len(np.unique(labels[labels != -1]))
    noise_ratio = (labels == -1).sum() / len(labels)
    
    # Cluster size statistics
    cluster_sizes = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        size = df[mask]['count'].sum()
        cluster_sizes.append(size)
    
    cluster_sizes = sorted(cluster_sizes, reverse=True)
    
    # Coverage statistics
    total_queries = df['count'].sum()
    cumulative = np.cumsum(cluster_sizes)
    coverage_85_idx = np.searchsorted(cumulative, 0.85 * total_queries)
    clusters_for_85pct = coverage_85_idx + 1 if coverage_85_idx < len(cluster_sizes) else len(cluster_sizes)
    
    return {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
        'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        'clusters_for_85pct': clusters_for_85pct,
        'coverage_efficiency': clusters_for_85pct / n_clusters if n_clusters > 0 else 0
    }


def is_viable_config(result):
    """Fast screening: check if config produces viable clustering"""
    metrics = result.metrics
    
    # Criteria for viability
    if metrics['n_clusters'] < 50:  # Too coarse
        return False, "Too few clusters"
    if metrics['n_clusters'] > 1000:  # Too granular
        return False, "Too many clusters"
    if metrics['noise_ratio'] > 0.3:  # Too much noise
        return False, "High noise ratio"
    if metrics['clusters_for_85pct'] < 5:  # Too coarse
        return False, "Poor coverage efficiency"
    
    return True, "Viable"


def phase1_fast_screening(df, configs, model, stop_words):
    """Phase 1: Fast screening with basic metrics — uses shared caches for speed."""
    print("\n" + "="*70)
    print("PHASE 1: FAST SCREENING")
    print("="*70)

    # Shared caches: embeddings computed once; distances once per alpha;
    # UMAP once per (alpha, n_neighbors, n_components)
    embed_cache = {}
    dist_cache  = {}
    umap_cache  = {}

    candidates = []
    results_log = []

    # Count unique UMAP runs for progress info
    unique_umap = len({(round(c.alpha,4), c.n_neighbors, c.n_components) for c in configs})
    print(f"  Unique UMAP runs needed: {unique_umap}  (rest is fast HDBSCAN-only)")

    for config in tqdm(configs, desc="Screening configs"):
        result = run_clustering(
            df, config, model, stop_words, verbose=False,
            _embed_cache=embed_cache,
            _dist_cache=dist_cache,
            _umap_cache=umap_cache,
        )

        if result is None:
            results_log.append({
                'config': str(config),
                'viable': False,
                'reason': 'Clustering failed'
            })
            continue

        viable, reason = is_viable_config(result)

        results_log.append({
            'config': str(config),
            'viable': viable,
            'reason': reason,
            **result.metrics
        })

        if viable:
            candidates.append(result)

    # Save screening results
    screening_df = pd.DataFrame(results_log)
    screening_file = OUTPUT_DIR / 'phase1_screening_results.csv'
    screening_df.to_csv(screening_file, index=False)

    print(f"\n✓ Screened {len(configs)} configurations")
    print(f"✓ Found {len(candidates)} viable candidates")
    print(f"✓ Results saved to: {screening_file}")

    return candidates


def load_stopwords():
    """Load Hinglish stopwords"""
    # Using the same stopwords from improved_clustering.py
    STOP_WORDS = set([
        'a', 'an', 'the', 'in', 'on', 'of', 'for', 'to', 'at', 'by', 'from', 'with', 'and', 'or', 
        'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
        'mein', 'me', 'mai', 'main', 'ko', 'ka', 'ki', 'ke', 'se', 'ne', 'hai', 'hain',
        'kya', 'kyon', 'kab', 'kaise', 'kaha', 'kahan', 'kon', 'kaun',
        'kare', 'karen', 'karna', 'kar', 'lag', 'laga', 'ho', 'hona',
        'crop', 'fasal', 'kheti', 'farm', 'kisan', 'plant', 'field', 'seed',
        'question', 'query', 'problem', 'help', 'info', 'information'
    ])
    return STOP_WORDS


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for clustering')
    parser.add_argument('--crop', type=str, required=True, help='Crop name to analyze')
    parser.add_argument('--max-queries', type=int, default=20000, help='Maximum queries to use')
    parser.add_argument('--api-key', type=str, help='Sarvam API key (legacy, for API-based Phase 2)')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='Which phase to run')
    parser.add_argument('--input-file', type=str, help='Custom input CSV file')
    parser.add_argument('--grid-mode', type=str, default='full',
                       choices=['quick', 'medium', 'full', 'exhaustive'],
                       help='Parameter grid size: quick(18), medium(108), full(240), exhaustive(480)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (-1 = all CPUs)')
    parser.add_argument('--use-hf', action='store_true',
                       help='Phase 2: use HuggingFace transformers (no vLLM required)')
    parser.add_argument('--model', type=str, default='/home/kshitij/models/qwen2.5-7b-instruct',
                       help='Local model path for HF/vLLM Phase 2 evaluation')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Phase 2: evaluate top K candidates from Phase 1 (0 = all unique configs)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Phase 2: LLM batch size')
    parser.add_argument('--coverage-cap', type=float, default=1.0,
                       help='Phase 2: evaluate only clusters covering this fraction of query volume (0.80 recommended)')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='Phase 2: CUDA device index (0 or 1)')
    parser.add_argument('--shard', type=str, default=None,
                       help='Phase 2: config shard to run, e.g. "1/2" or "2/2" for dual-GPU parallelism')
    parser.add_argument('--compile', action='store_true',
                       help='Phase 2: apply torch.compile for ~20%% speedup')

    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 2: skip all data/model loading — just delegate to the LLM evaluator
    if args.phase == 2:
        candidate_file = OUTPUT_DIR / 'phase1_candidates.csv'
        pickle_file    = OUTPUT_DIR / 'phase1_results.pkl'

        if not candidate_file.exists() or not pickle_file.exists():
            print("\n⚠️  Phase 1 outputs not found. Run Phase 1 first.")
            return

        if args.use_hf:
            import subprocess, sys
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / 'llm_evaluator_hf.py'),
                '--candidates',      'phase1_candidates.csv',
                '--results-pickle',  'phase1_results.pkl',
                '--model',           args.model,
                '--top-k',           str(args.top_k),
                '--batch-size',      str(args.batch_size),
                '--coverage-cap',    str(args.coverage_cap),
                '--gpu-id',          str(args.gpu_id),
            ]
            if args.shard:
                cmd += ['--shard', args.shard]
            if args.compile:
                cmd += ['--compile']
            print("\nLaunching HuggingFace evaluator:")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print("\nPhase 2 options:")
            print("  HuggingFace (recommended):")
            print(f"    python llm_evaluator_hf.py --candidates phase1_candidates.csv "
                  f"--results-pickle phase1_results.pkl --model {args.model} "
                  f"--top-k {args.top_k} --batch-size {args.batch_size}")

        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("="*70)
        return

    # Phase 1: load data, model, run screening
    # Load data
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = PROJECT_ROOT / 'data' / 'processed' / 'kcc_master_dataset_remapped.csv'
    
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)

    # Filter by crop
    df = df[df['Crop'] == args.crop].copy()
    print(f"Found {len(df)} rows for crop: {args.crop}")

    # Ensure query_text column exists
    if 'query_text' not in df.columns:
        df['query_text'] = df['QueryText']

    # Deduplicate: aggregate raw rows → unique queries with counts
    df = (df.groupby('query_text', as_index=False)
            .size()
            .rename(columns={'size': 'count'}))
    print(f"Deduplicated to {len(df)} unique queries")

    # Sample if still over limit
    if len(df) > args.max_queries:
        df = df.sample(n=args.max_queries, random_state=42)
        print(f"Sampled {args.max_queries} unique queries")
    
    # Load model and stopwords
    print("\nLoading Sentence Transformer model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    stop_words = load_stopwords()
    
    # Generate parameter grid
    configs = generate_param_grid(mode=args.grid_mode)
    print(f"\nGenerated {len(configs)} hyperparameter configurations")
    
    # Estimate time (after dedup, n≈3K → ~30s/config)
    n = len(df)
    time_per_config_sec = 30 if n <= 5000 else 120 if n <= 10000 else 600
    estimated_hours = (len(configs) * time_per_config_sec) / 3600
    print(f"Dataset size: {n} unique queries")
    print(f"Estimated time: {estimated_hours:.1f} hours (~{time_per_config_sec}s/config)")

    if args.phase == 1:
        candidates = phase1_fast_screening(df, configs, model, stop_words)

        # Save candidate summary CSV
        candidate_summary = []
        for result in candidates:
            candidate_summary.append({
                'config': str(result.config),
                **result.config.to_dict(),
                **result.metrics
            })

        summary_df = pd.DataFrame(candidate_summary)
        summary_file = OUTPUT_DIR / 'phase1_candidates.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Candidate summary saved to: {summary_file}")

        # Save full ClusteringResult objects to pickle (needed by Phase 2 LLM eval)
        import pickle
        pickle_file = OUTPUT_DIR / 'phase1_results.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(candidates, f)
        print(f"✓ ClusteringResult objects saved to: {pickle_file}")

    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
