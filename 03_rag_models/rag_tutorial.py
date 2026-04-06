# =============================================================================
# GenAI Tutorial 03: Retrieval-Augmented Generation (RAG)
# =============================================================================
# Environment:  CPU-only (no GPU required)
# Dependencies: pip install transformers torch datasets sentence-transformers
#               faiss-cpu accelerate
#
# This tutorial walks through the complete RAG pipeline:
#   1. What RAG is and why it exists
#   2. Document loading & chunking strategies
#   3. Building an embedding model (sentence-transformers)
#   4. Creating a vector index with FAISS
#   5. Retrieval — finding relevant chunks for a query
#   6. Augmented generation — combining retrieved context with an LLM
#   7. End-to-end RAG pipeline on a real HuggingFace dataset
#   8. Evaluating retrieval quality
# =============================================================================

import torch
import numpy as np
import faiss
import textwrap
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: WHY RAG?
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem: LLMs have a fixed knowledge cutoff. They can't answer questions
# about your internal documents, recent events, or proprietary data.
#
# Naive fix: Stuff all your documents into the prompt.
#   → Context windows are limited (typically 4K–128K tokens)
#   → Expensive — you pay per token in API-based models
#   → Slow — more tokens = slower inference
#
# RAG fix: At query time, RETRIEVE only the relevant document chunks,
# then inject just those chunks into the prompt as context.
#
#  ┌──────────────────────────────────────────────────────────┐
#  │                     RAG PIPELINE                         │
#  │                                                          │
#  │  Documents → Chunk → Embed → Vector Store                │
#  │                                    ↓                     │
#  │  Query → Embed → Similarity Search → Top-K Chunks        │
#  │                                    ↓                     │
#  │              Prompt = "Context: [chunks]\nQ: [query]"    │
#  │                                    ↓                     │
#  │                              LLM → Answer                │
#  └──────────────────────────────────────────────────────────┘

print("=" * 65)
print("GenAI Tutorial 03: Retrieval-Augmented Generation (RAG)")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DOCUMENT LOADING & CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
#
# Real documents (PDFs, wikis, emails) are too large to embed whole.
# We split them into "chunks" — smaller overlapping windows of text.
#
# Key chunking parameters:
#   chunk_size    — how many characters (or tokens) per chunk
#   chunk_overlap — how many characters overlap between adjacent chunks
#                   (overlap prevents cutting a relevant sentence in half)

@dataclass
class Document:
    """A single document with text content and metadata."""
    text: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class Chunk:
    """A chunk of text derived from a document, ready to be embedded."""
    text: str
    doc_id: int
    chunk_id: int
    metadata: Dict = field(default_factory=dict)


def chunk_document(
    doc: Document,
    doc_id: int,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    """
    Split a document into overlapping text chunks.

    Parameters
    ----------
    chunk_size : int  (default 512)
        Number of characters per chunk.
        • Too small (< 100): loses context, fragments sentences
        • Too large (> 1000): embeds too much noise, dilutes relevance
        • Sweet spot: 256–512 chars (~1–2 paragraphs)

    chunk_overlap : int  (default 64)
        Characters shared between adjacent chunks.
        • 0: risk of cutting relevant sentences at boundaries
        • 10–20% of chunk_size: sensible default
        • Too high: redundant retrievals, inflated index

        Example with chunk_size=10, overlap=3:
          Text: "ABCDEFGHIJKLMNOP"
          Chunks: "ABCDEFGHIJ", "HIJKLMNOP"  ← "HIJ" appears in both
    """
    text = doc.text
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at a sentence boundary rather than mid-word
        if end < len(text):
            last_period = chunk_text.rfind(". ")
            if last_period > chunk_size // 2:  # only snap if not too short
                chunk_text = chunk_text[:last_period + 1]

        chunks.append(Chunk(
            text=chunk_text.strip(),
            doc_id=doc_id,
            chunk_id=chunk_id,
            metadata={**doc.metadata, "start_char": start},
        ))
        chunk_id += 1
        start += len(chunk_text) - chunk_overlap

    return chunks


# ── Load real data from HuggingFace: SQuAD (Stanford Question Answering) ──
#
# SQuAD contains Wikipedia passages + human-written Q&A pairs.
# We'll use the passages as our "document corpus" and the questions
# as our test queries — giving us ground-truth answers to evaluate against.

print("\nLoading SQuAD dataset from HuggingFace...")
squad = load_dataset("squad", split="validation[:300]")  # 300 examples to keep it fast
print(f"Loaded {len(squad)} examples\n")

# Build a deduplicated corpus of Wikipedia passages
seen_contexts = set()
documents = []
for i, example in enumerate(squad):
    ctx = example["context"]
    if ctx not in seen_contexts:
        seen_contexts.add(ctx)
        documents.append(Document(
            text=ctx,
            metadata={
                "title": example["title"],
                "doc_idx": len(documents),
            }
        ))

print(f"Unique document passages: {len(documents)}")
print(f"Sample document (truncated):\n  {documents[0].text[:200]}...\n")

# Chunk all documents
all_chunks: List[Chunk] = []
for doc_id, doc in enumerate(documents):
    chunks = chunk_document(doc, doc_id, chunk_size=400, chunk_overlap=50)
    all_chunks.extend(chunks)

print(f"Total chunks after splitting: {len(all_chunks)}")
avg_len = np.mean([len(c.text) for c in all_chunks])
print(f"Average chunk length: {avg_len:.0f} chars\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
#
# Embeddings convert text into dense vectors (lists of floats) that capture
# semantic meaning. Similar meanings → similar vectors → close in vector space.
#
# We use sentence-transformers, which are fine-tuned specifically to produce
# good *sentence-level* semantic similarity — unlike general LLM embeddings.
#
# CPU-friendly embedding models (ordered by size / speed):
#   "all-MiniLM-L6-v2"    — 22M params, 384-dim, very fast  ✅ (this tutorial)
#   "all-MiniLM-L12-v2"   — 33M params, 384-dim, slightly better quality
#   "all-mpnet-base-v2"   — 110M params, 768-dim, best quality, slower
#   "multi-qa-MiniLM-L6-cos-v1" — optimized specifically for Q&A retrieval

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Embed all chunks — this is the "indexing" phase (done once, offline)
print(f"Embedding {len(all_chunks)} chunks (this takes ~30–60s on CPU)...")
chunk_texts = [c.text for c in all_chunks]

start = time.time()
chunk_embeddings = embedder.encode(
    chunk_texts,
    batch_size=64,          # process 64 chunks at a time
    show_progress_bar=True,
    normalize_embeddings=True,  # L2-normalize → cosine sim = dot product (faster)
    convert_to_numpy=True,
)
elapsed = time.time() - start

print(f"\nEmbedding complete in {elapsed:.1f}s")
print(f"Embedding matrix shape: {chunk_embeddings.shape}")
print(f"  → {chunk_embeddings.shape[0]} chunks × {chunk_embeddings.shape[1]} dimensions\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: VECTOR INDEX WITH FAISS
# ─────────────────────────────────────────────────────────────────────────────
#
# FAISS (Facebook AI Similarity Search) is a library for fast nearest-neighbor
# search over dense vectors. It's the most common choice for local RAG systems.
#
# Index types (from simplest to most scalable):
#
#   IndexFlatIP   — Exact inner product search. No compression. 100% accurate.
#                   Great for < 100K vectors. Use for development / evaluation.
#
#   IndexFlatL2   — Exact L2 (Euclidean) distance. Same performance as IP
#                   on normalized vectors. (IP and cosine are equivalent
#                   when vectors are L2-normalized — which we did above.)
#
#   IndexIVFFlat  — Inverted file index. Clusters vectors into buckets (cells),
#                   then only searches nearby cells at query time.
#                   Much faster, slight accuracy loss. Good for 100K–10M vectors.
#                   Requires training step: index.train(embeddings)
#
#   IndexHNSW     — Hierarchical Navigable Small World graph. Very fast,
#                   good recall, no training needed. Best for > 1M vectors.

EMBEDDING_DIM = chunk_embeddings.shape[1]

# For our tutorial size (~few hundred chunks), IndexFlatIP is perfect
index = faiss.IndexFlatIP(EMBEDDING_DIM)  # IP = inner product (≡ cosine on normalized vecs)
index.add(chunk_embeddings.astype(np.float32))

print(f"FAISS index built: {index.ntotal} vectors indexed\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
#
# At query time:
#   1. Embed the query using the SAME embedding model used for indexing
#   2. Search the FAISS index for the top-K nearest vectors
#   3. Return the corresponding text chunks

def retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[Tuple[Chunk, float]]:
    """
    Retrieve the top-K most relevant chunks for a query.

    Parameters
    ----------
    top_k : int  (default 5)
        Number of chunks to retrieve.
        • Too few (1–2): may miss relevant context
        • Too many (10+): dilutes the prompt, may confuse the LLM
        • Sweet spot: 3–5 for most tasks

    score_threshold : float  (default 0.0)
        Minimum cosine similarity score to include a chunk.
        Scores range from -1.0 to 1.0 (on normalized vectors).
        • 0.0: include everything (no filter)
        • 0.3: weak relevance filter
        • 0.5: moderate filter — only clearly relevant chunks
        • 0.7: strict filter — only highly similar chunks
        Use a threshold to avoid hallucination from irrelevant chunks.
    """
    # Embed the query (must use same model + normalization as corpus)
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Search the index
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        if score >= score_threshold:
            results.append((all_chunks[idx], float(score)))

    return results


# ── Test retrieval directly ──────────────────────────────────────────────
print("=" * 65)
print("RETRIEVAL TEST — Similarity Search")
print("=" * 65)

test_queries = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "How does photosynthesis work?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = retrieve(query, top_k=3, score_threshold=0.2)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  [{i}] Score: {score:.3f} | Title: {chunk.metadata.get('title', 'N/A')}")
        print(f"       {chunk.text[:120].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: AUGMENTED GENERATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Now we combine retrieval with a text-generation LLM.
# The key is prompt construction: we inject retrieved chunks as "context"
# before asking the question.
#
# Prompt template (simple but effective):
#
#   Use the following context to answer the question.
#   If the answer is not in the context, say "I don't know."
#
#   Context:
#   [chunk 1 text]
#   ---
#   [chunk 2 text]
#
#   Question: [user's question]
#   Answer:

print("\n\nLoading LLM for generation (distilgpt2)...")
GEN_MODEL = "distilgpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
gen_model.eval()
print("LLM loaded.\n")


def build_rag_prompt(query: str, chunks: List[Tuple[Chunk, float]]) -> str:
    """
    Construct a prompt by prepending retrieved context before the question.

    Prompt design tips:
    - Be explicit: "Use only the context below" reduces hallucination
    - Add a fallback: "If not in context, say I don't know" prevents confabulation
    - Separator lines (---) help the model distinguish between chunks
    - Keep the question at the END — models attend more to recent tokens
    """
    context_parts = []
    for chunk, score in chunks:
        context_parts.append(chunk.text.strip())

    context_str = "\n---\n".join(context_parts)

    prompt = (
        f"Use the following context to answer the question.\n"
        f"If the answer is not in the context, say \"I don't know.\"\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt


def rag_generate(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.25,
    max_new_tokens: int = 100,
    temperature: float = 0.4,
    top_p: float = 0.9,
) -> Dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.

    Returns a dict with the answer, retrieved chunks, and prompt used.
    """
    # Step 1: Retrieve relevant chunks
    retrieved = retrieve(query, top_k=top_k, score_threshold=score_threshold)

    if not retrieved:
        return {
            "query": query,
            "answer": "No relevant context found.",
            "chunks": [],
            "prompt": "",
        }

    # Step 2: Build the prompt
    prompt = build_rag_prompt(query, retrieved)

    # Step 3: Tokenize — truncate if prompt is too long for the model
    max_prompt_tokens = 900  # leave room for generation within 1024 token limit
    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    prompt_len = inputs["input_ids"].shape[1]

    # Step 4: Generate
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    answer = gen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up — stop at the first newline (the model sometimes rambles)
    answer = answer.split("\n")[0].strip()

    return {
        "query": query,
        "answer": answer,
        "chunks": retrieved,
        "prompt_tokens": prompt_len,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: END-TO-END RAG ON SQUAD QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Now we run the full pipeline: use real SQuAD questions, retrieve context
# from our indexed corpus, generate answers, and compare with ground truth.

print("=" * 65)
print("END-TO-END RAG PIPELINE — SQuAD Questions")
print("=" * 65)

# Pick a diverse sample of questions from SQuAD
sample_questions = []
seen_titles = set()
for example in squad:
    title = example["title"]
    if title not in seen_titles and len(sample_questions) < 6:
        sample_questions.append({
            "question": example["question"],
            "ground_truth": example["answers"]["text"][0] if example["answers"]["text"] else "N/A",
            "title": title,
        })
        seen_titles.add(title)

print(f"\nRunning RAG on {len(sample_questions)} questions...\n")

for i, item in enumerate(sample_questions, 1):
    print(f"[{i}] Question: {item['question']}")
    print(f"     Ground truth: {item['ground_truth']}")

    result = rag_generate(
        item["question"],
        top_k=3,
        score_threshold=0.2,
        max_new_tokens=60,
        temperature=0.3,
    )

    print(f"     RAG answer:    {result['answer']}")
    print(f"     Prompt tokens: {result['prompt_tokens']} | "
          f"Retrieved chunks: {len(result['chunks'])}")

    if result["chunks"]:
        top_chunk, top_score = result["chunks"][0]
        print(f"     Top chunk score: {top_score:.3f} "
              f"(title: {top_chunk.metadata.get('title', 'N/A')})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ABLATION — RAG vs. NO-RAG
# ─────────────────────────────────────────────────────────────────────────────
#
# This shows the core value proposition: the same LLM, same question,
# but with vs. without retrieved context.

print("=" * 65)
print("ABLATION: RAG vs. VANILLA LLM (no retrieval)")
print("=" * 65)

gen_pipeline = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer,
    device=-1,
)

ablation_q = sample_questions[0]["question"]
print(f"\nQuestion: {ablation_q}")
print(f"Ground truth: {sample_questions[0]['ground_truth']}\n")

# Without RAG
vanilla_result = gen_pipeline(
    f"Question: {ablation_q}\nAnswer:",
    max_new_tokens=60,
    do_sample=True,
    temperature=0.4,
    top_p=0.9,
    pad_token_id=gen_tokenizer.eos_token_id,
)[0]["generated_text"]
vanilla_answer = vanilla_result.split("Answer:")[-1].strip().split("\n")[0]
print(f"[Without RAG]\n  {vanilla_answer}\n")

# With RAG
rag_result = rag_generate(ablation_q, top_k=3, score_threshold=0.2, temperature=0.3)
print(f"[With RAG]\n  {rag_result['answer']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: RETRIEVAL QUALITY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Before trusting your RAG system, evaluate the RETRIEVER separately.
# If retrieval is broken, the LLM can't save you.
#
# Key metric: Hit Rate @ K
#   For each query, did the ground-truth answer appear in the top-K chunks?
#
# Other metrics:
#   MRR (Mean Reciprocal Rank) — rewards putting the right chunk at rank 1
#   NDCG — normalized discounted cumulative gain (used in IR research)

print("=" * 65)
print("RETRIEVAL EVALUATION — Hit Rate @ K")
print("=" * 65)

def evaluate_retrieval(
    examples: List[Dict],
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Compute Hit Rate @ K: fraction of queries where the ground-truth
    answer text appears in at least one of the top-K retrieved chunks.
    """
    hits = {k: 0 for k in k_values}
    total = 0

    for item in examples:
        query = item["question"]
        answer = item["ground_truth"].lower()

        if not answer or answer == "n/a":
            continue

        results = retrieve(query, top_k=max(k_values))
        total += 1

        for k in k_values:
            top_k_chunks = [chunk.text.lower() for chunk, _ in results[:k]]
            if any(answer in chunk_text for chunk_text in top_k_chunks):
                hits[k] += 1

    return {k: hits[k] / total if total > 0 else 0.0 for k in k_values}

# Build a larger eval set from SQuAD
eval_examples = [
    {
        "question": ex["question"],
        "ground_truth": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
    }
    for ex in squad[:50]
]

print(f"\nEvaluating retrieval on {len(eval_examples)} SQuAD questions...\n")
hit_rates = evaluate_retrieval(eval_examples, k_values=[1, 3, 5])

for k, rate in hit_rates.items():
    bar = "█" * int(rate * 30)
    print(f"  Hit Rate @ {k}: {rate:.1%}  {bar}")

print()
print("Interpretation:")
print("  Hit Rate @ 1: Does the #1 chunk contain the answer?")
print("  Hit Rate @ 3: Does any of the top 3 chunks contain the answer?")
print("  Hit Rate @ 5: Does any of the top 5 chunks contain the answer?")
print()
print("  A high Hit Rate @ 5 with low Hit Rate @ 1 suggests the answer")
print("  is being retrieved but not ranked at the top. Consider re-ranking.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: INTERACTIVE RAG — ASK YOUR OWN QUESTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("INTERACTIVE RAG — Query the SQuAD Corpus")
print("=" * 65)
print("Type a question to search the indexed Wikipedia passages.")
print("Type 'quit' to exit.\n")

while True:
    try:
        user_query = input("Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not user_query or user_query.lower() == "quit":
        break

    result = rag_generate(
        user_query,
        top_k=3,
        score_threshold=0.15,
        max_new_tokens=80,
        temperature=0.4,
    )

    print(f"\nAnswer: {result['answer']}")
    print("\nRetrieved context snippets:")
    for j, (chunk, score) in enumerate(result["chunks"], 1):
        print(f"  [{j}] (score={score:.3f}) {chunk.text[:150].strip()}...")
    print()

print("\n" + "=" * 65)
print("Tutorial complete! Check README.md for the full parameter guide.")
print("=" * 65)
