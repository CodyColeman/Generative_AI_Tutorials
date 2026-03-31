# 🔍 Tutorial 03: Retrieval-Augmented Generation (RAG)
### Grounding LLMs in Your Own Data — CPU-Friendly

---

## Overview

RAG is the most practically important GenAI pattern for enterprise use. It lets you connect any LLM to your own documents — internal wikis, PDFs, support tickets, codebases — without fine-tuning or retraining anything.

**Models used:**
- Embedder: `all-MiniLM-L6-v2` (sentence-transformers, 22M params)
- Generator: `distilgpt2` (82M params)

**Dataset used:** [SQuAD v1.1](https://huggingface.co/datasets/squad) — Stanford Q&A dataset with Wikipedia passages and human-written question/answer pairs (gives us ground-truth answers to evaluate against)

---

## Quick Start

```bash
# Install dependencies
pip install transformers torch datasets sentence-transformers faiss-cpu accelerate

# Run the tutorial
python rag_tutorial.py
```

> **First run:** Downloads ~100MB of model weights. Subsequent runs use local cache.  
> **Expected runtime:** ~2–4 min on a modern laptop CPU (embedding step is the bottleneck).

---

## The RAG Pipeline — Visualized

```
INDEXING PHASE (done once, offline)
─────────────────────────────────────────────────────
  Raw Documents
       │
       ▼
  ┌─────────────┐    chunk_size=512
  │   Chunker   │    chunk_overlap=64
  └─────────────┘
       │
       ▼  ["chunk 1 text", "chunk 2 text", ...]
  ┌─────────────────────┐
  │  Embedding Model    │    all-MiniLM-L6-v2
  │ (sentence-transformers) │   → 384-dim vectors
  └─────────────────────┘
       │
       ▼  [[0.12, -0.34, ...], ...]
  ┌─────────────┐
  │ FAISS Index │    IndexFlatIP (exact search)
  └─────────────┘


QUERY PHASE (real-time, per user question)
─────────────────────────────────────────────────────
  User Question: "Who invented the telephone?"
       │
       ▼  [0.08, -0.21, ...]   (same embedding model!)
  ┌──────────────────────────┐
  │  Nearest Neighbor Search │   top_k=3
  └──────────────────────────┘
       │
       ▼  [chunk_17, chunk_42, chunk_91]  ← most relevant passages
  ┌──────────────────────────────────────────┐
  │  Prompt Builder                          │
  │  "Context:\n[chunk 17]\n---\n[chunk 42]  │
  │   \n\nQuestion: ...\nAnswer:"            │
  └──────────────────────────────────────────┘
       │
       ▼
  ┌─────────────┐
  │     LLM     │    distilgpt2 (or any causal LM)
  └─────────────┘
       │
       ▼
  "Alexander Graham Bell invented the telephone in 1876."
```

---

## 🎛️ Parameter Guide — Every Knob Explained

---

### Chunking Parameters

#### `chunk_size` *(int, default: 512 characters)*
How large each text chunk is before embedding. This is the most impactful parameter in the entire RAG system.

| chunk_size | Effect | Best For |
|------------|--------|----------|
| `128–256` | Very granular — precise retrieval, may lose sentence context | FAQs, short facts, structured data |
| `256–512` | **Recommended default** — good balance of precision and context | General knowledge bases, wikis |
| `512–1024` | More context per chunk, but embedding quality degrades | Dense technical docs, papers |
| `> 1024` | Embedding models struggle to represent this much text | Not recommended |

> ⚠️ **Critical rule:** Your embedding model has a max token limit (MiniLM = 256 tokens ≈ ~1000 chars). Chunks longer than this are **silently truncated** during embedding — you'll lose information without knowing it.

#### `chunk_overlap` *(int, default: 64 characters)*
How many characters are shared between adjacent chunks. Prevents losing sentences at chunk boundaries.

```
Without overlap:                    With overlap (overlap=3):
│...sentence A. Sentence B...│     │...sentence A. Sen│
                │Tence B. Sentence C...│               ← "Sen" shared
                                                        "tence B" fully captured
```

| overlap | Effect |
|---------|--------|
| `0` | Risk of cutting relevant sentences; fast indexing |
| `10–15% of chunk_size` | **Recommended** — minimal redundancy, good boundary handling |
| `> 30% of chunk_size` | Excessive duplication; inflates index size |

---

### Embedding Parameters

#### Embedding Model Choice
The embedder is the brain of your retrieval system. Swap `EMBEDDING_MODEL` to upgrade quality:

| Model | Dims | Speed (CPU) | Quality | Notes |
|-------|------|-------------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Best speed/quality ratio |
| `all-MiniLM-L12-v2` | 384 | ⚡⚡ | ⭐⭐⭐ | Slightly better quality |
| `all-mpnet-base-v2` | 768 | ⚡ | ⭐⭐⭐⭐ | Best quality, 3x slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Best for Q&A retrieval specifically** |
| `BAAI/bge-small-en-v1.5` | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Strong general retriever, state-of-art |

> ✅ **Recommendation:** Switch from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` for a free quality upgrade with nearly identical speed.

#### `normalize_embeddings` *(bool, default: True)*
L2-normalizes all vectors to unit length. **Always enable this** when using cosine similarity.
- With normalization: cosine similarity = dot product (FAISS `IndexFlatIP` handles this natively, fast)
- Without normalization: must use `IndexFlatL2` and compute true L2 distance

#### `batch_size` *(int, default: 64)*
How many chunks to embed per forward pass.
| Value | Effect |
|-------|--------|
| `16–32` | Safe for low-RAM machines |
| `64` | Good default for 8GB RAM |
| `128–256` | Faster on machines with 16GB+ RAM |

---

### FAISS Index Types

| Index Type | When to Use | Accuracy | Speed |
|------------|-------------|----------|-------|
| `IndexFlatIP` | < 100K vectors, dev/eval | 100% exact | Moderate |
| `IndexFlatL2` | < 100K vectors, non-normalized | 100% exact | Moderate |
| `IndexIVFFlat` | 100K–10M vectors | ~95–99% | Fast |
| `IndexIVFPQ` | > 10M vectors, RAM constrained | ~90–95% | Very fast, compressed |
| `IndexHNSW` | Any size, best speed/recall | ~99% | Fastest |

**Upgrading to IVF for large corpora:**
```python
# Number of clusters (cells). Rule of thumb: sqrt(num_vectors)
nlist = 100
quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(chunk_embeddings)   # required for IVF
index.add(chunk_embeddings)

# nprobe: how many cells to search at query time
# Higher nprobe → better recall, slower search
index.nprobe = 10  # default 1; use 10–50 for good recall
```

---

### Retrieval Parameters

#### `top_k` *(int, default: 5)*
Number of chunks to retrieve and inject into the prompt.

| top_k | Effect |
|-------|--------|
| `1` | Risky — single point of failure if #1 is wrong |
| `3` | **Recommended default** — good recall, focused prompt |
| `5` | Better recall, larger prompt (watch token limits) |
| `10+` | High recall but noisy; may hurt LLM answer quality |

> ⚠️ **Token budget math:** If each chunk is 400 chars (~100 tokens) and `top_k=5`, that's ~500 tokens of context. Add the query and generation space — you need a model with at least 700+ token context window.

#### `score_threshold` *(float, default: 0.0)*
Minimum similarity score required to include a chunk. Filters out low-relevance noise.

Cosine similarity interpretation:
| Score Range | Meaning |
|-------------|---------|
| `0.8 – 1.0` | Near-identical content |
| `0.6 – 0.8` | Strongly related |
| `0.4 – 0.6` | Moderately related |
| `0.2 – 0.4` | Weakly related |
| `< 0.2` | Probably noise |

**Recommended thresholds by use case:**

| Use Case | Threshold |
|----------|-----------|
| General Q&A | `0.25` |
| Strict factual lookup | `0.45` |
| Broad topic exploration | `0.15` |
| No filtering (debug mode) | `0.0` |

---

### Prompt Engineering for RAG

The prompt template is a critical design decision. Small changes have large effects.

#### Basic template (used in this tutorial):
```
Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
[chunk 1]
---
[chunk 2]

Question: {query}
Answer:
```

#### Anti-hallucination template (stricter):
```
Answer ONLY using the context provided below. 
Do NOT use any prior knowledge. 
If the context does not contain the answer, respond: "Not found in documents."

[CONTEXT START]
{context}
[CONTEXT END]

Question: {query}
Concise answer:
```

#### Source-citing template:
```
Using the numbered sources below, answer the question and cite your source [1], [2], etc.

[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}

Question: {query}
Answer (with citations):
```

---

## 📊 Evaluating Your RAG System

Never ship a RAG system without measuring these:

### Retrieval Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Hit Rate @ K** | Does the correct document appear in top-K? | `sum(answer in top_k_chunks) / total` |
| **MRR** | How highly ranked is the correct chunk? | `mean(1 / rank_of_correct_chunk)` |
| **Precision @ K** | Of top-K retrieved, what fraction are relevant? | `relevant_in_top_k / k` |

### End-to-End Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Exact Match (EM)** | Does the generated answer exactly match ground truth? |
| **F1 Token Overlap** | Word-level overlap between generated and true answer |
| **ROUGE-L** | Longest common subsequence with reference answer |
| **LLM-as-judge** | Use a stronger LLM to rate answer quality 1–5 |

---

## 🔧 Improving Retrieval Quality

### 1. Hybrid Search (Dense + Sparse)
Combine semantic (embedding) search with keyword (BM25) search:
```python
# pip install rank-bm25
from rank_bm25 import BM25Okapi

# BM25 for keyword matching
bm25 = BM25Okapi([chunk.text.split() for chunk in all_chunks])
bm25_scores = bm25.get_scores(query.split())

# Combine: final_score = alpha * bm25_score + (1-alpha) * cosine_score
# alpha = 0.3–0.5 typically works well
```

### 2. Re-ranking
After retrieving top-20 candidates, run a cross-encoder to re-rank them:
```python
# pip install sentence-transformers
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(query, chunk.text) for chunk, _ in candidates]
scores = reranker.predict(pairs)  # more accurate but slower
```

### 3. Query Expansion
Before searching, expand the query with synonyms or decomposed sub-questions:
```python
# Simple: append key terms
expanded_query = f"{query} {extract_keywords(query)}"

# LLM-based: generate 3 alternative phrasings of the query
# then retrieve for all of them and merge results
```

### 4. Metadata Filtering
Pre-filter by metadata (date, department, document type) before embedding search:
```python
# Only search chunks from the "legal" department
relevant_indices = [i for i, c in enumerate(all_chunks)
                    if c.metadata.get("department") == "legal"]
# Create a sub-index or use FAISS IDSelector for filtered search
```

---

## Common Pitfalls

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Low hit rate on retrieval | Chunks too large / embedding mismatch | Reduce `chunk_size`; try `BAAI/bge-small-en-v1.5` |
| LLM ignores retrieved context | Prompt template too weak | Use explicit instructions: "Answer ONLY using the context" |
| Answer is correct but verbose | `max_new_tokens` too high | Reduce to 50–80; add "Concise answer:" to prompt |
| Same chunks retrieved for every query | Embeddings not normalized | Set `normalize_embeddings=True` |
| Out of memory during embedding | `batch_size` too high | Reduce to 16 or 32 |
| Model answers even when context is missing | No fallback instruction | Add "If not in context, say I don't know" to prompt |
| Slow indexing at startup | Embedding thousands of chunks | Pre-compute and save: `faiss.write_index(index, "my.index")` |

---

## Persisting Your Index

Don't re-embed on every run — save and load your FAISS index:

```python
import faiss
import pickle

# Save
faiss.write_index(index, "corpus.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

# Load
index = faiss.read_index("corpus.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)
```

---

## Production-Ready Alternatives

For production RAG, consider these managed vector stores:

| Tool | Type | Notes |
|------|------|-------|
| [ChromaDB](https://www.trychroma.com/) | Local / managed | Easy Python API, great for prototyping |
| [Pinecone](https://www.pinecone.io/) | Cloud managed | Fully managed, scales to billions of vectors |
| [Weaviate](https://weaviate.io/) | Self-hosted / cloud | Built-in hybrid search |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension | Best if you're already on Postgres |
| [Qdrant](https://qdrant.tech/) | Self-hosted / cloud | Fast, built-in filtering |

---

## What's Next

| Tutorial | Topic |
|----------|-------|
| `02_image_generation/` | Stable Diffusion — text-to-image |
| `04_fine_tuning/` | LoRA fine-tuning — adapt a model to your data |
| `05_embeddings/` | Semantic search, clustering, anomaly detection |
| `06_agents/` | Tool-using agents that can call RAG as a tool |

---

## Resources

- [HuggingFace sentence-transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [SQuAD Dataset](https://huggingface.co/datasets/squad)
- [BEIR Benchmark](https://github.com/beir-cellar/beir) — standard RAG retrieval evaluation suite
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/) — higher-level RAG orchestration
- [RAG Paper (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper
