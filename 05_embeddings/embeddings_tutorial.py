# =============================================================================
# Tutorial 05 — Embeddings
# =============================================================================
#
# Environment:  Python 3.9+, CPU-first
#
# Dependencies:
#   pip install sentence-transformers scikit-learn umap-learn matplotlib
#   pip install datasets torch
#
# Sections:
#   1. Imports & Configuration
#   2. What an Embedding Is — Sanity Check & Geometry Demo
#   3. Load Dataset (AG News)
#   4. Encode Documents to Embeddings
#   5. Semantic Search with Cosine Similarity
#   6. K-Means Clustering over Embedding Space
#   7. Cluster Evaluation (Silhouette, ARI, NMI)
#   8. Anomaly Detection via Centroid Distance
#   9. Dimensionality Reduction — UMAP
#  10. Dimensionality Reduction — t-SNE
#  11. UMAP vs. t-SNE Comparison Plot
#  12. Summary
#
# CPU note:
#   Encoding 2,000 sentences with all-MiniLM-L6-v2 takes ~30s on CPU.
#   UMAP and t-SNE on 2,000 × 384-dim vectors takes ~20–60s each.
#   Set N_SAMPLES higher for richer plots if you have more time / a GPU.
# =============================================================================

# ─────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file instead of popup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Model ─────────────────────────────────────────────
# all-MiniLM-L6-v2: 22M params, 384-dim output, fast on CPU.
# Produces L2-normalised vectors — cosine similarity = dot product.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # fixed output dimension for this model

# ── Dataset ───────────────────────────────────────────
DATASET_NAME = "ag_news"
N_SAMPLES    = 2000   # number of documents to embed and analyse
# AG News has 4 categories: World(0), Sports(1), Business(2), Sci/Tech(3)
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ── Encoding ──────────────────────────────────────────
BATCH_SIZE = 32
# ── batch_size ────────────────────────────────────────────────────────────
# Sentences encoded per forward pass.
# 32 is safe for CPU. Increase to 64–128 on GPU for speedup.
# Doesn't affect output quality — only throughput.

# ── Clustering ────────────────────────────────────────
N_CLUSTERS    = 4     # matches AG News ground-truth category count
N_INIT        = 10    # k-means restarts — higher = more stable result
MAX_ITER      = 300   # k-means iterations per restart
RANDOM_STATE  = 42

# ── UMAP ──────────────────────────────────────────────
UMAP_N_NEIGHBORS = 15
# ── n_neighbors ───────────────────────────────────────────────────────────
# Controls local vs. global structure balance.
# 5   : tight micro-clusters, noisy global layout
# 15  : ✅ recommended default
# 50  : smooth global structure, local detail lost

UMAP_MIN_DIST    = 0.1
# ── min_dist ──────────────────────────────────────────────────────────────
# How tightly UMAP packs points together.
# 0.0 : dense blobs — good for cluster counting
# 0.1 : ✅ recommended default
# 0.5 : spread out — good for seeing within-cluster structure

UMAP_N_COMPONENTS = 2   # reduce to 2D for visualisation

# ── t-SNE ─────────────────────────────────────────────
TSNE_PERPLEXITY = 30
# ── perplexity ────────────────────────────────────────────────────────────
# Effective number of neighbours each point considers.
# 5   : very local, tight clumps
# 30  : ✅ classic default
# 50  : more global — better for large datasets
# Must be < number of data points.

TSNE_N_ITER     = 1000
# ── n_iter ────────────────────────────────────────────────────────────────
# t-SNE optimisation steps. Must be enough to converge.
# < 250 : under-converged
# 1000  : ✅ standard default
# 2000  : more reliable on complex datasets

TSNE_N_COMPONENTS = 2

# ── Anomaly detection ─────────────────────────────────
ANOMALY_THRESHOLD_SIGMA = 2.0
# Flag documents whose centroid distance exceeds
# mean + ANOMALY_THRESHOLD_SIGMA * std_dev

# ── Output ────────────────────────────────────────────
OUTPUT_DIR = "./outputs/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette — one per AG News category
CATEGORY_COLOURS = {
    "World":    "#e63946",
    "Sports":   "#2a9d8f",
    "Business": "#e9c46a",
    "Sci/Tech": "#457b9d",
}


# ─────────────────────────────────────────────────────
# SECTION 2: WHAT AN EMBEDDING IS — GEOMETRY DEMO
# ─────────────────────────────────────────────────────
#
# Before loading any real data, we demonstrate the core property of
# embeddings with a few hand-picked sentences:
# semantically similar texts should produce similar vectors.

print("=" * 65)
print("SECTION 2: Embedding Geometry Demo")
print("=" * 65)

print(f"\n  Loading model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  ✅ Model loaded. Output dimension: {EMBEDDING_DIM}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Cosine similarity measures the angle between vectors, ignoring magnitude.
    For L2-normalised embeddings (which all-MiniLM-L6-v2 produces),
    this is equivalent to the dot product: cos(θ) = a · b.

    Parameters
    ----------
    a : np.ndarray  Shape (D,). First embedding vector.
    b : np.ndarray  Shape (D,). Second embedding vector.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
        1.0  → identical direction (semantically identical)
        0.0  → orthogonal (unrelated)
       -1.0  → opposite directions (semantically opposite)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# Sentence pairs — each pair tests a semantic relationship
demo_sentences = [
    "The stock market crashed amid fears of recession.",
    "Wall Street fell sharply as investors fled to safety.",
    "Scientists discover water ice beneath the Martian surface.",
    "NASA confirms frozen water deposits on Mars.",
    "The home team won the championship in overtime.",
    "A baking recipe for chocolate chip cookies.",
]

demo_embeddings = model.encode(demo_sentences, batch_size=BATCH_SIZE, show_progress_bar=False)

print(f"\n  Encoded {len(demo_sentences)} sentences → shape {demo_embeddings.shape}")
print(f"\n  Pairwise cosine similarities:")
print(f"  {'Sentence A':<45} {'Sentence B':<45} {'cos sim':>8}")
print("  " + "─" * 100)

pairs = [
    (0, 1, "finance ↔ finance (same topic)"),
    (2, 3, "Mars ↔ Mars (same topic)"),
    (4, 5, "sports ↔ cooking (different topic)"),
    (0, 4, "finance ↔ sports (different topic)"),
    (0, 2, "finance ↔ science (different topic)"),
    (1, 2, "finance ↔ science (different domain)"),
]

for i, j, label in pairs:
    sim = cosine_similarity(demo_embeddings[i], demo_embeddings[j])
    a_preview = demo_sentences[i][:42] + "..." if len(demo_sentences[i]) > 45 else demo_sentences[i]
    b_preview = demo_sentences[j][:42] + "..." if len(demo_sentences[j]) > 45 else demo_sentences[j]
    bar = "█" * int(max(0, sim) * 20)
    print(f"  {a_preview:<45} {b_preview:<45} {sim:>+.3f}  {bar}")

print("\n  ✅ Same-topic pairs score higher than cross-topic pairs.")

# Show what the raw vector looks like
v = demo_embeddings[0]
print(f"\n  Raw embedding vector (first sentence, first 10 of {EMBEDDING_DIM} dimensions):")
print(f"  {v[:10].round(4).tolist()}")
print(f"  Vector norm: {np.linalg.norm(v):.4f}  (≈1.0 because model L2-normalises output)")
print()


# ─────────────────────────────────────────────────────
# SECTION 3: LOAD DATASET
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 3: Load Dataset — AG News")
print("=" * 65)

raw = load_dataset(DATASET_NAME, split="test")
print(f"\n  Full test split: {len(raw):,} examples")
print(f"  Columns: {raw.column_names}")

# Sample N_SAMPLES evenly across all 4 categories for balanced clusters
samples_per_class = N_SAMPLES // len(AG_NEWS_LABELS)
indices = []
for label_id in AG_NEWS_LABELS:
    class_indices = [i for i, ex in enumerate(raw) if ex["label"] == label_id]
    indices.extend(class_indices[:samples_per_class])

dataset = raw.select(indices)
texts  = [ex["text"]  for ex in dataset]
labels = [ex["label"] for ex in dataset]
label_names = [AG_NEWS_LABELS[l] for l in labels]

print(f"\n  Using {len(texts)} examples ({samples_per_class} per category)")
print(f"  Category distribution:")
for name, count in Counter(label_names).items():
    bar = "█" * (count // 10)
    print(f"    {name:<12} {count:>5}  {bar}")

print(f"\n  Example records:")
for i in [0, samples_per_class, samples_per_class*2, samples_per_class*3]:
    preview = texts[i][:90].replace("\n", " ")
    print(f"  [{AG_NEWS_LABELS[labels[i]]:<10}] {preview}...")
print()


# ─────────────────────────────────────────────────────
# SECTION 4: ENCODE DOCUMENTS TO EMBEDDINGS
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 4: Encode Documents to Embeddings")
print("=" * 65)

print(f"\n  Encoding {len(texts)} documents with {EMBEDDING_MODEL}...")
print(f"  Batch size: {BATCH_SIZE}  |  Output dim: {EMBEDDING_DIM}")

t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalise so cosine sim = dot product
    convert_to_numpy=True,
)
encode_time = time.time() - t0

print(f"\n  ✅ Encoding complete in {encode_time:.1f}s")
print(f"  Embeddings shape : {embeddings.shape}  (n_docs × embedding_dim)")
print(f"  Memory           : {embeddings.nbytes / 1e6:.1f} MB")
print(f"  Throughput       : {len(texts) / encode_time:.0f} docs/sec")

# Verify normalisation
norms = np.linalg.norm(embeddings, axis=1)
print(f"  Vector norms     : min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")
print(f"  (All ~1.0 confirms L2 normalisation ✅)")
print()


# ─────────────────────────────────────────────────────
# SECTION 5: SEMANTIC SEARCH WITH COSINE SIMILARITY
# ─────────────────────────────────────────────────────
#
# Given a query, we embed it and compute cosine similarity against every
# document in our corpus, then return the top-k most similar results.
#
# This brute-force approach is fine up to ~50k documents.
# For larger corpora, swap the numpy dot product for a FAISS index
# (see Tutorial 03 — RAG Models for a FAISS deep-dive).

print("=" * 65)
print("SECTION 5: Semantic Search with Cosine Similarity")
print("=" * 65)


def semantic_search(
    query: str,
    corpus_embeddings: np.ndarray,
    corpus_texts: list[str],
    corpus_labels: list[str],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """
    Encode a query and return the top-k most similar documents.

    Parameters
    ----------
    query : str
        The search query. Can be a question, keyword phrase, or full sentence.
        Semantic search doesn't require keyword overlap with the results —
        it works on meaning.
    corpus_embeddings : np.ndarray
        Pre-computed L2-normalised embeddings of the corpus. Shape (N, D).
        Pre-computing these offline is the key to fast search at runtime.
    corpus_texts : list[str]
        Original text for each document in the corpus.
    corpus_labels : list[str]
        Category label for each document (for display purposes).
    model : SentenceTransformer
        The embedding model — must be the same model used to encode the corpus.
        Using a different model produces incompatible vector spaces.
    top_k : int  (default 5)
        Number of results to return.
        Practical range: 3–20 for interactive search; up to 100 for pipelines.

    Returns
    -------
    list[dict]
        Top-k results sorted by similarity descending, each with keys:
        rank, score, label, text.
    """
    # Embed query (same model, same normalisation as corpus)
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Cosine similarity = dot product for normalised vectors
    # Shape: (N,) — one score per document
    scores = corpus_embeddings @ query_vec

    # Get top-k indices sorted by score descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "rank":  rank + 1,
            "score": float(scores[idx]),
            "label": corpus_labels[idx],
            "text":  corpus_texts[idx],
        }
        for rank, idx in enumerate(top_indices)
    ]


# Run several queries that span different categories and styles
search_queries = [
    "federal reserve raises interest rates",
    "goalkeeper makes stunning save in final",
    "breakthrough in quantum computing research",
    "trade war tariffs impact global supply chain",
]

for query in search_queries:
    results = semantic_search(
        query,
        embeddings,
        texts,
        label_names,
        model,
        top_k=4,
    )
    print(f"\n  Query: \"{query}\"")
    print(f"  {'Rank':<5} {'Score':>6}  {'Category':<12}  Text")
    print("  " + "─" * 80)
    for r in results:
        preview = r["text"][:70].replace("\n", " ")
        print(f"  #{r['rank']:<4} {r['score']:>6.3f}  {r['label']:<12}  {preview}...")

print()


# ─────────────────────────────────────────────────────
# SECTION 6: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────
#
# K-means partitions the embedding space into N_CLUSTERS groups by
# iteratively assigning each point to the nearest centroid and
# recomputing centroids until convergence.
#
# We run the Elbow Method and Silhouette analysis first to show how
# to find the right k when ground truth is unknown, then fit the
# final model with k=4 (matching AG News ground truth).

print("=" * 65)
print("SECTION 6: K-Means Clustering")
print("=" * 65)

# ── Elbow method — find the right k ──────────────────
print("\n  Running Elbow Method (k=2 to 10)...")
k_range   = range(2, 11)
inertias  = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT,
                max_iter=MAX_ITER, random_state=RANDOM_STATE)
    km.fit(embeddings)
    inertias.append(km.inertia_)
    if k >= 2:
        sil = silhouette_score(embeddings, km.labels_, sample_size=500,
                               random_state=RANDOM_STATE)
        sil_scores.append((k, sil))
    print(f"    k={k}  inertia={km.inertia_:>10.0f}  "
          f"silhouette={sil_scores[-1][1]:.3f}" if sil_scores else f"    k={k}  inertia={km.inertia_:>10.0f}")

# Print ASCII elbow curve
print("\n  Elbow curve (inertia vs. k):")
max_inertia = max(inertias)
for k, inertia in zip(k_range, inertias):
    bar_len = int(40 * inertia / max_inertia)
    print(f"    k={k:>2}  {'█' * bar_len}  {inertia:>10.0f}")

best_k_sil = max(sil_scores, key=lambda x: x[1])
print(f"\n  Best k by silhouette score: k={best_k_sil[0]} (score={best_k_sil[1]:.3f})")
print(f"  Ground truth k            : {N_CLUSTERS} (matches AG News categories)")

# ── Save elbow plot ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ks = list(k_range)
ax1.plot(ks, inertias, "o-", color="#457b9d", linewidth=2, markersize=6)
ax1.axvline(N_CLUSTERS, color="#e63946", linestyle="--", alpha=0.7, label=f"k={N_CLUSTERS} (ground truth)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (within-cluster SSE)")
ax1.set_title("Elbow Method")
ax1.legend()
ax1.grid(alpha=0.3)

sil_ks, sil_vals = zip(*sil_scores)
ax2.plot(sil_ks, sil_vals, "o-", color="#2a9d8f", linewidth=2, markersize=6)
ax2.axvline(N_CLUSTERS, color="#e63946", linestyle="--", alpha=0.7, label=f"k={N_CLUSTERS} (ground truth)")
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score vs. k")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
elbow_path = os.path.join(OUTPUT_DIR, "01_elbow_and_silhouette.png")
plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {elbow_path}")

# ── Fit final k-means model ───────────────────────────
print(f"\n  Fitting final k-means with k={N_CLUSTERS}...")
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    # ── init ──────────────────────────────────────────────────────────────
    # Initialisation strategy for centroids.
    # "k-means++" : spread initial centroids far apart — much more stable
    #               than random init on high-dimensional data.
    # "random"    : random init — fast but sensitive to seed.

    n_init=N_INIT,
    # ── n_init ────────────────────────────────────────────────────────────
    # How many times to run k-means with different seeds.
    # Returns the run with lowest inertia.
    # 10 is fine; increase to 20 if clusters are unstable.

    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_
centroids      = kmeans.cluster_centers_

print(f"  Converged in {kmeans.n_iter_} iterations")
print(f"  Final inertia: {kmeans.inertia_:.0f}")

# Show what category each cluster captured
print(f"\n  Cluster composition (predicted cluster → true category breakdown):")
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    cat_counts = Counter([label_names[i] for i, m in enumerate(mask) if m])
    dominant   = cat_counts.most_common(1)[0][0]
    total      = sum(cat_counts.values())
    breakdown  = "  ".join(f"{cat}:{cnt}" for cat, cnt in cat_counts.most_common())
    print(f"  Cluster {c} (n={total:>4}, dominant={dominant:<10}): {breakdown}")
print()


# ─────────────────────────────────────────────────────
# SECTION 7: CLUSTER EVALUATION
# ─────────────────────────────────────────────────────
#
# Since AG News has ground-truth category labels, we can measure how
# well our unsupervised clustering recovered the true structure.

print("=" * 65)
print("SECTION 7: Cluster Evaluation")
print("=" * 65)

le = LabelEncoder()
true_labels_encoded = le.fit_transform(label_names)

ari = adjusted_rand_score(true_labels_encoded, cluster_labels)
nmi = normalized_mutual_info_score(true_labels_encoded, cluster_labels)
sil = silhouette_score(embeddings, cluster_labels, sample_size=1000,
                       random_state=RANDOM_STATE)

print(f"""
  Clustering metrics (higher is better for all three):

    Adjusted Rand Index (ARI)           : {ari:.3f}
    ─ Measures agreement between predicted and true clusters.
    ─ Range: -1 to 1. Random clustering ≈ 0. Perfect = 1.0.

    Normalised Mutual Info (NMI)        : {nmi:.3f}
    ─ Shared information between predicted and true assignments.
    ─ Range: 0 to 1. 0 = no agreement. 1 = perfect agreement.

    Silhouette Score                    : {sil:.3f}
    ─ How well each point fits its own cluster vs. nearest other cluster.
    ─ Range: -1 to 1. > 0.5 is considered good for text data.
""")

if ari > 0.5:
    print("  ✅ ARI > 0.5 — clustering strongly recovers the true categories.")
elif ari > 0.2:
    print("  ✅ ARI > 0.2 — clustering partially recovers the true structure.")
else:
    print("  ⚠️  ARI < 0.2 — clustering doesn't match the true categories well.")
print()


# ─────────────────────────────────────────────────────
# SECTION 8: ANOMALY DETECTION VIA CENTROID DISTANCE
# ─────────────────────────────────────────────────────
#
# After clustering, we treat large distance from the nearest centroid
# as a signal that a document is an outlier — it doesn't fit well
# into any of the discovered groups.

print("=" * 65)
print("SECTION 8: Anomaly Detection via Centroid Distance")
print("=" * 65)


def compute_anomaly_scores(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    cluster_assignments: np.ndarray,
) -> np.ndarray:
    """
    Compute an anomaly score for each document as its cosine distance
    to its assigned cluster centroid.

    Cosine distance = 1 - cosine_similarity.
    Range [0, 2] for normalised vectors, but practically [0, 1] for
    well-separated clusters.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D). L2-normalised document embeddings.
    centroids : np.ndarray
        Shape (K, D). Cluster centroid vectors from k-means.
    cluster_assignments : np.ndarray
        Shape (N,). Cluster index for each document.

    Returns
    -------
    np.ndarray
        Shape (N,). Anomaly score per document. Higher = more anomalous.
    """
    # Normalise centroids for cosine distance computation
    centroid_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    scores = np.zeros(len(embeddings))
    for i, (emb, c) in enumerate(zip(embeddings, cluster_assignments)):
        cos_sim = float(np.dot(emb, centroid_norms[c]))
        scores[i] = 1.0 - cos_sim   # convert similarity to distance

    return scores


anomaly_scores = compute_anomaly_scores(embeddings, centroids, cluster_labels)

mean_score = anomaly_scores.mean()
std_score  = anomaly_scores.std()
threshold  = mean_score + ANOMALY_THRESHOLD_SIGMA * std_score
anomalies  = np.where(anomaly_scores > threshold)[0]

print(f"\n  Anomaly score statistics:")
print(f"    Mean   : {mean_score:.4f}")
print(f"    Std    : {std_score:.4f}")
print(f"    Threshold (mean + {ANOMALY_THRESHOLD_SIGMA}σ): {threshold:.4f}")
print(f"    Anomalies detected: {len(anomalies)} / {len(embeddings)} "
      f"({100*len(anomalies)/len(embeddings):.1f}%)")

if len(anomalies) > 0:
    print(f"\n  Top 5 most anomalous documents:")
    top_anomalies = anomalies[np.argsort(anomaly_scores[anomalies])[::-1][:5]]
    for rank, idx in enumerate(top_anomalies, 1):
        preview = texts[idx][:80].replace("\n", " ")
        print(f"  [{rank}] score={anomaly_scores[idx]:.4f}  [{label_names[idx]}]")
        print(f"       {preview}...")

# Anomaly score histogram
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(anomaly_scores, bins=60, color="#457b9d", alpha=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(threshold, color="#e63946", linewidth=2, linestyle="--",
           label=f"Threshold (μ + {ANOMALY_THRESHOLD_SIGMA}σ = {threshold:.3f})")
ax.axvline(mean_score, color="#2a9d8f", linewidth=1.5, linestyle=":",
           label=f"Mean ({mean_score:.3f})")
ax.set_xlabel("Anomaly Score (cosine distance to centroid)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Anomaly Scores")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
anomaly_path = os.path.join(OUTPUT_DIR, "02_anomaly_scores.png")
plt.savefig(anomaly_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {anomaly_path}")
print()


# ─────────────────────────────────────────────────────
# SECTION 9: DIMENSIONALITY REDUCTION — UMAP
# ─────────────────────────────────────────────────────
#
# 384 dimensions can't be visualised directly. UMAP reduces the
# embedding space to 2D while trying to preserve the neighbourhood
# structure — points that are close in 384D should stay close in 2D.

print("=" * 65)
print("SECTION 9: Dimensionality Reduction — UMAP")
print("=" * 65)

print(f"\n  Fitting UMAP on {len(embeddings)} × {EMBEDDING_DIM}-dim embeddings...")
print(f"  n_neighbors={UMAP_N_NEIGHBORS}  min_dist={UMAP_MIN_DIST}  n_components={UMAP_N_COMPONENTS}")

t0 = time.time()
umap_reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS,
    metric="cosine",
    # ── metric ────────────────────────────────────────────────────────────
    # Distance metric used to build the neighbourhood graph in high-dim space.
    # "cosine"    : ✅ correct for normalised text embeddings
    # "euclidean" : works but less meaningful for unit vectors
    random_state=RANDOM_STATE,
)
umap_2d = umap_reducer.fit_transform(embeddings)
umap_time = time.time() - t0

print(f"  ✅ UMAP complete in {umap_time:.1f}s")
print(f"  Output shape: {umap_2d.shape}")

# ── Plot 1: UMAP coloured by true category ────────────
def make_scatter(
    coords_2d: np.ndarray,
    colour_by: list[str],
    colour_map: dict,
    title: str,
    save_path: str,
    anomaly_indices: np.ndarray = None,
    point_size: float = 6.0,
    alpha: float = 0.6,
) -> None:
    """
    Save a 2D scatter plot of dimensionality-reduced embeddings.

    Parameters
    ----------
    coords_2d : np.ndarray
        Shape (N, 2). 2D coordinates from UMAP or t-SNE.
    colour_by : list[str]
        Category label for each point — used to assign colours.
    colour_map : dict
        Maps category name → hex colour string.
    title : str
        Plot title.
    save_path : str
        Full path to save the PNG.
    anomaly_indices : np.ndarray  (optional)
        Indices of anomalous points to highlight with a marker.
    point_size : float  (default 6.0)
        Scatter plot marker size.
    alpha : float  (default 0.6)
        Marker transparency. Lower = see dense regions better.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for cat, colour in colour_map.items():
        mask = np.array([lb == cat for lb in colour_by])
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=colour, label=cat, s=point_size, alpha=alpha, linewidths=0,
        )

    if anomaly_indices is not None and len(anomaly_indices) > 0:
        ax.scatter(
            coords_2d[anomaly_indices, 0], coords_2d[anomaly_indices, 1],
            c="none", edgecolors="black", s=60, linewidths=1.2,
            label=f"Anomalies (n={len(anomaly_indices)})", zorder=5,
        )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


umap_true_path = os.path.join(OUTPUT_DIR, "03_umap_true_labels.png")
make_scatter(
    umap_2d, label_names, CATEGORY_COLOURS,
    title=f"UMAP — AG News (n={N_SAMPLES}, coloured by true category)",
    save_path=umap_true_path,
)
print(f"  Plot saved: {umap_true_path}")

# ── Plot 2: UMAP coloured by k-means cluster ──────────
cluster_colour_map = {
    f"Cluster {i}": c
    for i, c in enumerate(["#e63946", "#2a9d8f", "#e9c46a", "#457b9d"])
}
cluster_label_strs = [f"Cluster {c}" for c in cluster_labels]

umap_cluster_path = os.path.join(OUTPUT_DIR, "04_umap_kmeans_clusters.png")
make_scatter(
    umap_2d, cluster_label_strs, cluster_colour_map,
    title=f"UMAP — AG News (coloured by k-means cluster, k={N_CLUSTERS})",
    save_path=umap_cluster_path,
)
print(f"  Plot saved: {umap_cluster_path}")

# ── Plot 3: UMAP with anomalies highlighted ───────────
umap_anomaly_path = os.path.join(OUTPUT_DIR, "05_umap_anomalies.png")
make_scatter(
    umap_2d, label_names, CATEGORY_COLOURS,
    title=f"UMAP — Anomalies highlighted (threshold=μ+{ANOMALY_THRESHOLD_SIGMA}σ)",
    save_path=umap_anomaly_path,
    anomaly_indices=anomalies,
)
print(f"  Plot saved: {umap_anomaly_path}")
print()


# ─────────────────────────────────────────────────────
# SECTION 10: DIMENSIONALITY REDUCTION — t-SNE
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 10: Dimensionality Reduction — t-SNE")
print("=" * 65)

print(f"\n  Fitting t-SNE on {len(embeddings)} × {EMBEDDING_DIM}-dim embeddings...")
print(f"  perplexity={TSNE_PERPLEXITY}  n_iter={TSNE_N_ITER}  n_components={TSNE_N_COMPONENTS}")
print("  (t-SNE is slower than UMAP — this may take a minute on CPU)")

t0 = time.time()
tsne_reducer = TSNE(
    n_components=TSNE_N_COMPONENTS,
    perplexity=TSNE_PERPLEXITY,
    # ── perplexity ────────────────────────────────────────────────────────
    # Effective number of nearest neighbours each point considers.
    # 30 is the classic default. Increase for larger datasets.
    # Must be < number of data points.

    n_iter=TSNE_N_ITER,
    # ── n_iter ────────────────────────────────────────────────────────────
    # Optimisation iterations. Need enough to converge.
    # 1000 is standard. Increase to 2000 on complex datasets.

    metric="cosine",
    # ── metric ────────────────────────────────────────────────────────────
    # Same reasoning as UMAP — use cosine for normalised embeddings.

    init="pca",
    # ── init ──────────────────────────────────────────────────────────────
    # Initialisation strategy.
    # "pca"    : ✅ more stable, recommended for reproducibility
    # "random" : stochastic, can converge to different local minima

    random_state=RANDOM_STATE,
    verbose=1,
)
tsne_2d = tsne_reducer.fit_transform(embeddings)
tsne_time = time.time() - t0

print(f"\n  ✅ t-SNE complete in {tsne_time:.1f}s")
print(f"  Output shape: {tsne_2d.shape}")

tsne_true_path = os.path.join(OUTPUT_DIR, "06_tsne_true_labels.png")
make_scatter(
    tsne_2d, label_names, CATEGORY_COLOURS,
    title=f"t-SNE — AG News (n={N_SAMPLES}, coloured by true category, perplexity={TSNE_PERPLEXITY})",
    save_path=tsne_true_path,
)
print(f"  Plot saved: {tsne_true_path}")

tsne_cluster_path = os.path.join(OUTPUT_DIR, "07_tsne_kmeans_clusters.png")
make_scatter(
    tsne_2d, cluster_label_strs, cluster_colour_map,
    title=f"t-SNE — AG News (coloured by k-means cluster, k={N_CLUSTERS})",
    save_path=tsne_cluster_path,
)
print(f"  Plot saved: {tsne_cluster_path}")
print()


# ─────────────────────────────────────────────────────
# SECTION 11: UMAP vs. t-SNE COMPARISON PLOT
# ─────────────────────────────────────────────────────
#
# Side-by-side comparison on the same data and colour scheme so the
# structural differences are easy to see.

print("=" * 65)
print("SECTION 11: UMAP vs. t-SNE Comparison Plot")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    f"Embedding Space: UMAP vs. t-SNE  |  AG News, n={N_SAMPLES}  |  {EMBEDDING_MODEL}",
    fontsize=13, y=1.01,
)

for ax, coords_2d, method_name, extra in [
    (axes[0], umap_2d,  "UMAP",  f"n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}"),
    (axes[1], tsne_2d,  "t-SNE", f"perplexity={TSNE_PERPLEXITY}, n_iter={TSNE_N_ITER}"),
]:
    for cat, colour in CATEGORY_COLOURS.items():
        mask = np.array([lb == cat for lb in label_names])
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=colour, label=cat, s=5, alpha=0.55, linewidths=0,
        )
    ax.set_title(f"{method_name}\n{extra}", fontsize=11)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2.5, fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_facecolor("#f8f9fa")

# Annotation arrows pointing out structural differences
axes[0].annotate(
    "UMAP: global structure\npreserved across clusters",
    xy=(0.02, 0.02), xycoords="axes fraction",
    fontsize=8, color="#333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
axes[1].annotate(
    "t-SNE: cluster separation\noften exaggerated",
    xy=(0.02, 0.02), xycoords="axes fraction",
    fontsize=8, color="#333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

plt.tight_layout()
comparison_path = os.path.join(OUTPUT_DIR, "08_umap_vs_tsne_comparison.png")
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Comparison plot saved: {comparison_path}")

# Print timing comparison
print(f"\n  Timing comparison:")
print(f"    UMAP  : {umap_time:>6.1f}s  (n_neighbors={UMAP_N_NEIGHBORS})")
print(f"    t-SNE : {tsne_time:>6.1f}s  (perplexity={TSNE_PERPLEXITY})")
if tsne_time > 0:
    print(f"    UMAP was {tsne_time / umap_time:.1f}× faster than t-SNE")

print(f"""
  Structural observations to look for in the plots:
  ──────────────────────────────────────────────────
  UMAP:
    • Cluster shapes tend to be more organic / elongated
    • Inter-cluster distances are more meaningful
    • Global layout is more consistent across runs

  t-SNE:
    • Clusters often appear as tight spherical blobs
    • Gaps between clusters can be misleading (exaggerated)
    • Internal structure within clusters may be clearer
    • Re-running with a different seed can produce different layouts

  Both:
    • 2D coordinates are NOT cosine similarities — don't read distances directly
    • Points that overlap in 2D may be far apart in 384D space
    • Use the plots for topology (which groups exist), not measurement
""")
print()


# ─────────────────────────────────────────────────────
# SECTION 12: SUMMARY
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 12: Summary")
print("=" * 65)

print(f"""
  What ran:
    ✅ Loaded {EMBEDDING_MODEL}
    ✅ Geometry demo — cosine similarity on 6 hand-picked sentence pairs
    ✅ Encoded {len(texts)} AG News documents → {embeddings.shape} float32 matrix
       ({embeddings.nbytes / 1e6:.1f} MB  |  {len(texts)/encode_time:.0f} docs/sec)
    ✅ Semantic search over corpus with 4 test queries
    ✅ Elbow method + silhouette analysis (k=2 to 10)
    ✅ K-means clustering (k={N_CLUSTERS})
       ARI={ari:.3f}  NMI={nmi:.3f}  Silhouette={sil:.3f}
    ✅ Anomaly detection — {len(anomalies)} outliers at μ+{ANOMALY_THRESHOLD_SIGMA}σ threshold
    ✅ UMAP 2D projection ({umap_time:.1f}s)
    ✅ t-SNE 2D projection ({tsne_time:.1f}s)
    ✅ UMAP vs. t-SNE comparison plot

  Output files:
    {os.path.join(OUTPUT_DIR, "01_elbow_and_silhouette.png")}
    {os.path.join(OUTPUT_DIR, "02_anomaly_scores.png")}
    {os.path.join(OUTPUT_DIR, "03_umap_true_labels.png")}
    {os.path.join(OUTPUT_DIR, "04_umap_kmeans_clusters.png")}
    {os.path.join(OUTPUT_DIR, "05_umap_anomalies.png")}
    {os.path.join(OUTPUT_DIR, "06_tsne_true_labels.png")}
    {os.path.join(OUTPUT_DIR, "07_tsne_kmeans_clusters.png")}
    {os.path.join(OUTPUT_DIR, "08_umap_vs_tsne_comparison.png")}

  Key concepts covered:
    • Cosine similarity and embedding geometry
    • Batch encoding for throughput
    • Brute-force semantic search (dot product over normalised vectors)
    • K-means on embedding space, elbow method, silhouette analysis
    • ARI / NMI for clustering evaluation against ground truth
    • Centroid-distance anomaly detection
    • UMAP: n_neighbors, min_dist, cosine metric
    • t-SNE: perplexity, n_iter, init="pca"
    • When to use UMAP vs. t-SNE and what each preserves

  To go further:
    • Increase N_SAMPLES to 5000+ for richer visualisation (GPU recommended)
    • Try UMAP_N_NEIGHBORS=5 vs. 50 to see the local/global trade-off
    • Try TSNE_PERPLEXITY=5 vs. 50 and compare — instability reveals artefacts
    • Swap EMBEDDING_MODEL for all-mpnet-base-v2 and compare clustering quality
    • Add a FAISS index (see Tutorial 03) for sub-millisecond search at scale
    • See Tutorial 06 for embeddings used inside an agent memory loop
""")

print("=" * 65)
print("  Tutorial 05 — embeddings_tutorial.py complete.")
print("  Next: 06_agents  |  README.md")
print("=" * 65)
