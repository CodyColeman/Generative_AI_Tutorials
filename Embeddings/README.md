# 🔢 Tutorial 05 — Embeddings

Turn text into geometry. Embeddings are dense vector representations that encode semantic meaning — the foundation of semantic search, clustering, anomaly detection, and retrieval systems.

---

## Overview

| File | What it does | Core libraries |
|---|---|---|
| `embeddings_tutorial.py` | Semantic search, clustering, anomaly detection, and visualisation of embedding space | `sentence-transformers`, `scikit-learn`, `umap-learn`, `matplotlib`, `datasets` |

**Dataset:** [`ag_news`](https://huggingface.co/datasets/ag_news) — 120k news headlines across 4 categories (World, Sports, Business, Sci/Tech). Natural clusters make it ideal for demonstrating embedding geometry.

**Embedding model:** [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22M parameters, 384-dimensional output, fast on CPU.

**Goals:**
- Understand what an embedding vector is and what "semantic similarity" means geometrically
- Perform semantic search with cosine similarity
- Cluster documents with k-means over embedding vectors
- Detect anomalies using distance from cluster centroids
- Visualise embedding space with both UMAP and t-SNE, and understand the differences

---

## Quick Start

```bash
pip install sentence-transformers scikit-learn umap-learn matplotlib datasets torch

python embeddings_tutorial.py
```

> **First run:** Downloads `all-MiniLM-L6-v2` (~90 MB) and a subset of AG News. Subsequent runs use the HuggingFace cache. The full script runs on CPU in a few minutes.

---

## What Is an Embedding?

An embedding model is a neural network trained to map variable-length text to a fixed-length vector such that **semantically similar texts end up close together** in vector space.

```
"The stock market crashed today"    → [0.21, -0.43,  0.87, ..., 0.11]  ← 384 floats
"Wall Street saw heavy losses"      → [0.19, -0.41,  0.85, ..., 0.09]  ← nearby ✅
"Scientists discover new exoplanet" → [-0.54, 0.72, -0.23, ..., 0.63] ← far away ✅
```

The geometry is meaningful:

```
         Sci/Tech cluster          Sports cluster
              ●  ●                    ●   ●
           ●     ●  ●              ●    ●
                                      ●
         Business cluster          World/Politics cluster
           ●   ●                    ●   ●
         ●   ●  ●                ●    ●   ●
```

This structure emerges from training — the model is never told what categories exist. It learns proximity purely from co-occurrence patterns in text.

---

## How Sentence Transformers Work

`all-MiniLM-L6-v2` is a bi-encoder: it encodes each text independently into a fixed vector.

```
Input text
    │
    ▼
Tokenizer  →  token IDs  →  token embeddings
                                  │
                                  ▼
                         Transformer layers (6)
                                  │
                                  ▼
                         Token-level outputs
                         [tok_1, tok_2, ..., tok_n]
                                  │
                         Mean pooling (average all tokens)
                                  │
                                  ▼
                         L2 normalisation
                                  │
                                  ▼
                    384-dimensional unit vector  ✅
```

**Mean pooling** averages all token vectors into one sentence vector. This is the standard approach for sentence embeddings — it outperforms using just the `[CLS]` token for most tasks.

**L2 normalisation** puts every vector on the unit sphere. This makes cosine similarity equivalent to dot product, which is faster to compute and simpler to reason about.

---

## Similarity Metrics

Once you have embedding vectors, you need a way to measure how close two vectors are.

### Cosine Similarity

Measures the angle between two vectors. Ignores magnitude; only direction matters.

```
            A · B
cos(θ) = ─────────
           ‖A‖ ‖B‖
```

| Value | Meaning |
|---|---|
| 1.0 | Identical direction (semantically identical) |
| 0.7–0.9 | Very similar |
| 0.4–0.6 | Somewhat related |
| 0.0–0.3 | Unrelated |
| < 0.0 | Semantically opposite |

> ✅ **Use cosine similarity for sentence embeddings.** Since `all-MiniLM-L6-v2` L2-normalises its outputs, cosine similarity is equivalent to dot product — fast and interpretable.

### Euclidean Distance

Measures straight-line distance through the vector space.

> ⚠️ Works well for raw (non-normalised) vectors. On normalised vectors, Euclidean distance and cosine similarity are monotonically related — pick one and be consistent.

### Dot Product

For normalised vectors, identical to cosine similarity. Preferred in approximate nearest-neighbour (ANN) indexes (FAISS, ScaNN) because it maps to efficient inner-product search.

---

## 🎛️ Parameter Guide

### `batch_size` — Encoding Batch Size

Number of sentences encoded in a single forward pass through the model.

| Value | Effect |
|---|---|
| 16 | Safe for CPU / low-RAM |
| 32 | ✅ Good CPU default |
| 64–128 | Faster on GPU; may OOM on CPU |
| 256+ | GPU with ≥8 GB VRAM |

> ✅ **Recommended default:** `32` on CPU. `sentence-transformers` handles batching automatically via `model.encode(sentences, batch_size=32)`.

> ⚠️ Larger batches don't change the output vectors — only speed. If you get OOM errors, halve the batch size.

---

### `n_clusters` — K-Means Cluster Count

Number of clusters in k-means. For AG News there are 4 ground-truth categories, but you won't always know the right number in advance.

| Value | Effect |
|---|---|
| < true clusters | Over-merging — different topics lumped together |
| = true clusters | ✅ Matches ground truth |
| > true clusters | Over-splitting — coherent topics fragmented |

> ✅ **Start with the Elbow Method or Silhouette Score** (both shown in the script) to find the right `k` when ground truth is unknown.

> ⚠️ K-means assumes spherical clusters of equal size. Embedding clusters are often elongated. If your clusters look poor, try DBSCAN or agglomerative clustering as alternatives.

---

### `n_components` — Dimensionality Reduction Target

The number of dimensions to reduce to before visualisation (typically 2).

| Value | Use case |
|---|---|
| 2 | ✅ 2D scatter plot — the standard |
| 3 | 3D interactive plot (matplotlib 3D or Plotly) |
| 10–50 | Intermediate step — reduce before clustering (can improve k-means on high-dim data) |

> ⚠️ UMAP and t-SNE are for **visualisation only**. Never use the 2D reduced vectors for downstream tasks like search or clustering — information is lost. Always work in the full 384-dimensional space for those.

---

### `n_neighbors` — UMAP Neighbourhood Size

Controls the balance between local and global structure in UMAP.

| Value | Effect |
|---|---|
| 5 | Emphasises local structure — tight micro-clusters, ignores global layout |
| 15 | ✅ Good balance — recommended default |
| 30–50 | Emphasises global structure — cluster separation more reliable, local detail lost |
| 100+ | Very global — approaches PCA-like linear projection |

> ✅ **Recommended default:** `15`. Start here and adjust based on what you're trying to communicate.

> ⚠️ Unlike t-SNE's `perplexity`, UMAP's `n_neighbors` has a consistent interpretation: it's roughly the number of nearby points each point tries to stay connected to.

---

### `min_dist` — UMAP Minimum Distance

Controls how tightly UMAP packs points together.

| Value | Effect |
|---|---|
| 0.0 | Points clump into tight blobs — useful for cluster counting |
| 0.1 | ✅ Good default |
| 0.5 | Points spread out — better for seeing internal cluster structure |
| 0.99 | Very spread — almost no local clustering |

> ✅ **Recommended default:** `0.1`. Use `0.0` when you want to count clusters visually; use `0.5` when you want to see what's inside a cluster.

---

### `perplexity` — t-SNE Perplexity

Loosely: the effective number of neighbours each point considers. The most important t-SNE hyperparameter.

| Value | Effect |
|---|---|
| 5 | Very local — ignores global structure, tight clumps |
| 30 | ✅ Classic default — works for most datasets |
| 50 | More global — better for large datasets (>5k points) |
| 100+ | Very global — blurs local structure |

> ✅ **Recommended default:** `30`. Must be less than the number of points in the dataset.

> ⚠️ t-SNE is highly sensitive to `perplexity`. Run with 2–3 values and compare — the clusters visible across all runs are reliable; clusters that appear in only one run are artefacts.

---

### `n_iter` — t-SNE Iterations

Number of optimisation steps. t-SNE needs enough iterations to converge.

| Value | Risk |
|---|---|
| < 250 | Under-converged — clusters haven't formed properly |
| 1000 | ✅ Standard default |
| 2000–5000 | More reliable on large/complex datasets |

> ⚠️ Always check the final KL divergence (printed by scikit-learn). If it's still decreasing, increase `n_iter`.

---

## UMAP vs. t-SNE — Side by Side

| Property | UMAP | t-SNE |
|---|---|---|
| Speed | ✅ Much faster (minutes vs. hours on large data) | Slow — O(n log n) with Barnes-Hut |
| Global structure | ✅ Preserves better | Poor — distances between clusters are not meaningful |
| Reproducibility | ✅ Deterministic with fixed `random_state` | Stochastic — results vary between runs |
| Scalability | ✅ Scales to millions of points | Struggles beyond ~50k points |
| Hyperparameter sensitivity | Moderate (`n_neighbors`, `min_dist`) | High (`perplexity`, `n_iter`) |
| Cluster separation | Good | Often exaggerated — clusters look more separated than they are |
| Theory | Riemannian geometry / topological data analysis | Student-t distribution in low-dim space |
| When to use | ✅ Default choice for most tasks | When you need a familiar benchmark or the audience knows t-SNE |

> ✅ **Use UMAP by default.** Use t-SNE when you need to compare against published results that used t-SNE, or when explaining to an audience familiar with it.

> ⚠️ **Neither UMAP nor t-SNE preserves distances.** The spatial distance between two points in the 2D plot is *not* their actual cosine similarity in 384D space. Use the plot to see topology (which groups exist), not to read off precise similarities.

---

## Semantic Search Architecture

Embeddings enable sub-millisecond search over millions of documents once the index is built.

```
Offline (build once):                    Online (per query):

Documents                                Query text
    │                                        │
    ▼                                        ▼
Embedding model                         Embedding model
    │                                        │
    ▼                                        ▼
384-dim vectors                         384-dim query vector
    │                                        │
    ▼                                        ▼
Vector index (FAISS)  ←──── ANN search ─────┘
    │
    ▼
Top-k most similar documents
```

**Exact vs. approximate search:**

| Method | Speed | Accuracy | When to use |
|---|---|---|---|
| Brute-force cosine (`numpy`) | Slow at scale | Exact | < 10k documents |
| FAISS `IndexFlatIP` | Fast | Exact | 10k–500k documents |
| FAISS `IndexIVFFlat` | Faster | ~99% | 500k–10M documents |
| FAISS `IndexHNSW` | Fastest | ~97% | > 1M documents, latency-critical |

For a deeper dive into FAISS index types, see [Tutorial 03 — RAG Models](../03_rag_models/README.md).

---

## Clustering Evaluation Metrics

When you have ground-truth labels (like AG News categories), you can evaluate clustering quality objectively.

| Metric | Range | What it measures | Higher is better? |
|---|---|---|---|
| Adjusted Rand Index (ARI) | −1 to 1 | Agreement between predicted and true clusters | ✅ Yes (1.0 = perfect) |
| Normalised Mutual Information (NMI) | 0 to 1 | Shared information between cluster assignments | ✅ Yes (1.0 = perfect) |
| Silhouette Score | −1 to 1 | How well each point fits its own cluster vs. neighbours | ✅ Yes (closer to 1 = better) |
| Inertia (within-cluster SSE) | 0 to ∞ | Sum of squared distances to cluster centroids | ❌ Lower is better |

> ✅ **Silhouette Score** is the most useful metric when you don't have ground-truth labels (the typical real-world case). Use ARI/NMI to validate against known labels during development.

---

## Anomaly Detection with Embeddings

Embeddings give you a natural anomaly score: **distance from the nearest cluster centroid**.

```
For each document d:
  1. Embed d → vector v
  2. Find nearest centroid c* = argmin cosine_distance(v, centroid_k)
  3. anomaly_score = cosine_distance(v, c*)

Documents with high anomaly_score are outliers —
they don't fit well into any of the learned clusters.
```

Practical threshold:
```
threshold = mean(anomaly_scores) + 2 × std(anomaly_scores)
Documents above threshold → flag as anomalous
```

Applications:
- Content moderation: flag off-topic submissions
- Data quality: detect mislabelled or corrupted documents
- Monitoring: alert when incoming text drifts from expected distribution

---

## Common Pitfalls

| Problem | Likely Cause | Fix |
|---|---|---|
| All cosine similarities are ~1.0 | All your texts are very similar, or model is producing near-constant vectors | Check text diversity; verify model is loaded correctly with a sanity-check query |
| K-means produces one giant cluster and several tiny ones | K-means is sensitive to initialisation on high-dim data | Use `init="k-means++"`, increase `n_init` to 20 |
| UMAP plot shows one blob with no structure | `n_neighbors` too high, or texts are genuinely similar | Lower `n_neighbors` to 5–10; verify your data has actual diversity |
| t-SNE gives different clusters every run | t-SNE is stochastic by default | Set `random_state=42` for reproducibility |
| Semantic search returns irrelevant results | Query and documents are from different domains / registers | Try a different model (e.g., `multi-qa-MiniLM-L6-cos-v1` for Q&A style search) |
| Embedding is very slow | Encoding one text at a time (no batching) | Always pass a list to `model.encode()`, never call in a loop |
| Clustering quality is poor despite good embeddings | 384D space is very sparse; k-means struggles | Reduce to 50D with UMAP before clustering |

---

## Choosing an Embedding Model

| Model | Dim | Speed | Best for |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ✅ Fastest | General-purpose, CPU-friendly |
| `all-mpnet-base-v2` | 768 | Medium | Higher quality, same general use |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ✅ Fastest | Question–answer semantic search |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Medium | Multilingual text |
| `text-embedding-3-small` (OpenAI) | 1536 | API call | Highest quality, requires API key |
| `nomic-embed-text-v1` | 768 | Medium | Long documents (up to 8192 tokens) |

> ✅ Start with `all-MiniLM-L6-v2`. If quality is insufficient for your task, move to `all-mpnet-base-v2`. Only consider API-based models if the open-source options genuinely fall short.

---

## What's Next

| Tutorial | What it adds |
|---|---|
| ← [03 RAG Models](../03_rag_models/README.md) | Embeddings applied to retrieval — FAISS indexes, chunking, top-k search |
| ← [04 Fine-Tuning](../04_fine_tuning/README.md) | Fine-tuning the models that produce embeddings |
| → [06 Agents](../06_agents/README.md) | Using embeddings inside an agent's memory and retrieval loop |

---

## Resources

- [`sentence-transformers` documentation](https://www.sbert.net/)
- [SBERT: Sentence-BERT paper](https://arxiv.org/abs/1908.10084)
- [UMAP paper](https://arxiv.org/abs/1802.03426)
- [t-SNE paper (van der Maaten & Hinton)](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [UMAP vs t-SNE — practical guide](https://pair-code.github.io/understanding-umap/)
- [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard) — model leaderboard for choosing the right embedder
- [ANN Benchmarks](https://ann-benchmarks.com/) — speed/accuracy tradeoffs for vector search libraries
