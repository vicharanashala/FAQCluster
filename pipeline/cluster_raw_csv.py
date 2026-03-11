"""
Clustering script for raw KCC CSV files.
Reads raw QueryText rows, aggregates counts, then runs the same
HDBSCAN+UMAP pipeline used in improved_clustering.py.

Usage:
    python cluster_raw_csv.py --input ../punjab_maize_raw.csv --output-dir ../outputs/maize_clustering
"""

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Default parameters (same as improved_clustering.py)
# ---------------------------------------------------------------------------
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
ALPHA      = 0.5          # Weight for dense embeddings (0=sparse only, 1=dense only)

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.0,
    n_components = 5,
    metric       = 'precomputed',
    random_state = 42,
)

HDBSCAN_PARAMS = dict(
    min_cluster_size       = 3,
    min_samples            = 2,
    metric                 = 'euclidean',
    cluster_selection_method = 'eom',
    prediction_data        = True,
)

NOISE_HIGH_COUNT_THRESHOLD = 50  # Noise points with count > this become their own cluster

# ---------------------------------------------------------------------------
# Stop words (identical to improved_clustering.py)
# ---------------------------------------------------------------------------
STOP_WORDS = set([
    'a','an','the','in','on','of','for','to','at','by','from','with','and','or',
    'is','are','am','was','were','be','been','being','have','has','had','do',
    'does','did','can','could','shoud','would','will','may','might','must',
    'mein','me','mai','main','ko','ka','ki','ke','se','ne','par','per','pe',
    'he','hai','hain','ha','hi','ho','h','tha','thi','the','hu','hun','hoon',
    'kya','kyon','kab','kaise','kaha','kahan','kon','kaun','kis','kisi','kisko','kisse',
    'liye','liya','lie','laye','wale','wali','wala','aur','tatha','evam','bhi',
    'to','agar','magar','lekin','parantu',
    'kare','karen','karein','karna','karne','kar','karo','kiya','kiye',
    'jay','jaye','jana','jane','jata','jati','jate',
    'lag','laga','lagi','lage','lga','lgi','lge','lagna','lagne',
    'aa','aaya','aaye','raha','rahi','rahe','rha','rhi','rhe',
    'ho','hona','hone','honi','pad','pd','padi','pdi','padta','padti','padte',
    'gaya','gyi','gaye','gye','gay','gayi',
    'chahiye','chahie','chahye','sakta','sakti','sakte','sake','saka',
    'bataye','batayen','batain','batao','bataiye','btaye','btao','bata',
    'de','den','dijiye','dijiya','dein',
    'dal','dala','dale','dalen','dalna','dalne','dali','dalo',
    'use','using','used','apply','applied','applying','spray','spraying',
    'crop','phasal','fasal','kheti','farm','farming','farmer','kisan',
    'agriculture','krishi','plant','paudha','podha','paudh','podh','ped',
    'field','khet','seed','beej','seeds','variety','var','varieties','kism',
    'prajati','prajatiyan','type',
    'dhan','paddy','rice','chawal','gehu','wheat','ganna','sugarcane',
    'makka','maize','bajra','millet','aloo','potato','tamatar','tomato',
    'ji','sir','madam','mam','mr','bhai','bhaiya',
    'question','query','problem','samasya','issue','doubt','help','info',
    'information','jankari','detail','details','about','regarding','related',
    'want','know','please','pls','plz','provide','tell','ask','asked','asking',
    'solution','upchar','ilaaj','ilaj','dawa','medicine','control','roktham',
    'nidan','upay','management','niyantran',
    'matra','quantity','dose','amount','rate','price','bhav','muly',
    'number','no','contact','mobile',
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def extract_keywords_tfidf(texts, stop_words, max_features=1000, max_df=0.95, min_df=2):
    print("Extracting TF-IDF keywords...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(stop_words),
        max_df=max_df,
        min_df=min_df,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        print("  Warning: TF-IDF failed (empty vocab?). Skipping sparse features.")
        return [], None, None

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        _, col_indices = row.nonzero()
        if len(col_indices) == 0:
            top_keywords.append(set())
            continue
        scores = row.data
        sorted_idx = np.argsort(scores)[::-1]
        top_k = col_indices[sorted_idx[:5]]
        top_keywords.append(set(feature_names[top_k]))
    return top_keywords, tfidf_matrix, vectorizer


def calculate_hybrid_distance(dense_embeddings, tfidf_matrix, alpha=0.5):
    print("Computing hybrid distance matrix...")
    print("  Dense (cosine)...")
    dense_dist = pairwise_distances(dense_embeddings, metric='cosine', n_jobs=-1)

    print("  Sparse (jaccard)...")
    tfidf_bool = (tfidf_matrix > 0).astype(bool)
    try:
        sparse_dist = pairwise_distances(tfidf_bool.toarray(), metric='jaccard', n_jobs=-1)
    except MemoryError:
        print("  MemoryError: falling back to dense-only.")
        return dense_dist

    print("  Fusing matrices...")
    return alpha * dense_dist + (1 - alpha) * sparse_dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster a raw KCC CSV file.")
    parser.add_argument('--input',      default='../punjab_maize_raw.csv', help='Path to raw CSV')
    parser.add_argument('--output-dir', default='../outputs/maize_clustering', help='Output directory')
    parser.add_argument('--query-col',  default='QueryText', help='Column name for queries')
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Cap on number of unique queries (after aggregation). Default: all.')
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_dir  = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & Aggregate
    # ------------------------------------------------------------------
    print(f"\nLoading {input_path}...")
    raw_df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(raw_df)}")

    if args.query_col not in raw_df.columns:
        # Try case-insensitive match
        match = [c for c in raw_df.columns if c.lower() == args.query_col.lower()]
        if match:
            args.query_col = match[0]
        else:
            raise ValueError(f"Column '{args.query_col}' not found. Available: {raw_df.columns.tolist()}")

    raw_df[args.query_col] = raw_df[args.query_col].fillna("").str.strip()
    raw_df = raw_df[raw_df[args.query_col] != ""]

    agg_df = (
        raw_df.groupby(args.query_col, sort=False)
        .size()
        .reset_index(name='count')
        .rename(columns={args.query_col: 'query_text'})
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )
    print(f"  Unique queries after aggregation: {len(agg_df)}")
    print(f"  Total query volume: {agg_df['count'].sum():,}")
    print(f"  Top queries:\n{agg_df.head(5).to_string(index=False)}\n")

    if args.max_queries:
        agg_df = agg_df.head(args.max_queries)
        print(f"  Using top {len(agg_df)} queries (--max-queries cap)")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("Preprocessing text...")
    agg_df['cleaned_text'] = agg_df['query_text'].apply(preprocess_text)
    agg_df = agg_df[agg_df['cleaned_text'].str.strip().astype(bool)].copy()
    print(f"  Valid queries after preprocessing: {len(agg_df)}")

    # ------------------------------------------------------------------
    # 3. TF-IDF Keywords
    # ------------------------------------------------------------------
    top_keywords_list, tfidf_matrix, _ = extract_keywords_tfidf(agg_df['cleaned_text'], STOP_WORDS)
    agg_df['top_keywords'] = [", ".join(list(k)) for k in top_keywords_list]

    # ------------------------------------------------------------------
    # 4. Sentence Embeddings
    # ------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Generating embeddings...")
    embeddings = model.encode(
        agg_df['cleaned_text'].tolist(),
        show_progress_bar=True,
        batch_size=128,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 5. Hybrid Distance
    # ------------------------------------------------------------------
    hybrid_dist = calculate_hybrid_distance(embeddings, tfidf_matrix, alpha=ALPHA)

    # ------------------------------------------------------------------
    # 6. UMAP Dimensionality Reduction
    # ------------------------------------------------------------------
    print("\nReducing with UMAP...")
    umap_model = umap.UMAP(**UMAP_PARAMS)
    reduced = umap_model.fit_transform(hybrid_dist)
    print(f"  Reduced shape: {reduced.shape}")

    # ------------------------------------------------------------------
    # 7. HDBSCAN Clustering
    # ------------------------------------------------------------------
    print("\nClustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(reduced)

    n_clusters_initial = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_initial    = (labels == -1).sum()
    print(f"  Clusters (initial): {n_clusters_initial}")
    print(f"  Noise points (initial): {n_noise_initial}")

    # ------------------------------------------------------------------
    # 8. Noise Reassignment
    # ------------------------------------------------------------------
    print("\nReassigning noise...")
    counts = agg_df['count'].values
    max_label = labels.max()

    # High-volume noise → own cluster
    for idx in np.where(labels == -1)[0]:
        if counts[idx] > NOISE_HIGH_COUNT_THRESHOLD:
            max_label += 1
            labels[idx] = max_label

    # Remaining noise → nearest neighbour
    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 0:
        remaining_noise = np.where(labels == -1)[0]
        if len(remaining_noise) > 0:
            print(f"  KNN-reassigning {len(remaining_noise)} noise points...")
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(reduced[non_noise_mask], labels[non_noise_mask])
            labels[remaining_noise] = knn.predict(reduced[remaining_noise])

    agg_df['cluster_id'] = labels

    # ------------------------------------------------------------------
    # 9. Summary Statistics (printed for quick analysis)
    # ------------------------------------------------------------------
    cluster_sizes = agg_df.groupby('cluster_id')['count'].sum()
    cluster_uniq  = agg_df.groupby('cluster_id').size()

    n_clusters_final = len(cluster_sizes)
    print(f"\n{'='*55}")
    print(f"  CLUSTERING RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Total unique queries clustered : {len(agg_df):,}")
    print(f"  Total query volume             : {agg_df['count'].sum():,}")
    print(f"  Number of clusters             : {n_clusters_final}")
    print(f"  Avg cluster size (volume)      : {cluster_sizes.mean():.1f}")
    print(f"  Median cluster size (volume)   : {cluster_sizes.median():.1f}")
    print(f"  Max cluster size (volume)      : {cluster_sizes.max()}")
    print(f"  Min cluster size (volume)      : {cluster_sizes.min()}")
    print(f"  Avg unique queries per cluster : {cluster_uniq.mean():.1f}")
    print(f"  Median unique q per cluster    : {cluster_uniq.median():.1f}")
    print(f"  Noise initially                : {n_noise_initial} ({100*n_noise_initial/len(agg_df):.1f}%)")
    print(f"{'='*55}\n")

    # Distribution buckets
    buckets = {
        'singleton (1)':   (cluster_uniq == 1).sum(),
        'small   (2-5)':   ((cluster_uniq >= 2) & (cluster_uniq <= 5)).sum(),
        'medium  (6-20)':  ((cluster_uniq >= 6) & (cluster_uniq <= 20)).sum(),
        'large   (21-50)': ((cluster_uniq >= 21) & (cluster_uniq <= 50)).sum(),
        'xlarge  (51+)':   (cluster_uniq >= 51).sum(),
    }
    print("  Cluster size distribution (by unique queries):")
    for label, cnt in buckets.items():
        bar = '#' * min(cnt, 50)
        print(f"    {label:20s}: {cnt:4d}  {bar}")
    print()

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    # mapping.csv
    mapping_path = output_dir / 'mapping.csv'
    agg_df[['query_text', 'cluster_id', 'cleaned_text', 'top_keywords', 'count']].to_csv(
        mapping_path, index=False
    )
    print(f"Saved mapping → {mapping_path}")

    # summary.csv
    summary_rows = []
    for cid, group in agg_df.groupby('cluster_id'):
        total_vol  = group['count'].sum()
        unique_q   = len(group)
        rep_query  = group.sort_values('count', ascending=False).iloc[0]['query_text']
        all_kws    = []
        for ks in group['top_keywords']:
            if ks:
                all_kws.extend([k.strip() for k in ks.split(',')])
        common_kws = [k for k, _ in Counter(all_kws).most_common(5)] if all_kws else []
        summary_rows.append({
            'cluster_id':     cid,
            'total_volume':   total_vol,
            'unique_queries': unique_q,
            'representative': rep_query,
            'top_keywords':   ", ".join(common_kws),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('total_volume', ascending=False)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary → {summary_path}")

    # Top-10 clusters for quick review
    top10_path = output_dir / 'top10_clusters.csv'
    summary_df.head(10).to_csv(top10_path, index=False)
    print(f"Saved top-10 → {top10_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
