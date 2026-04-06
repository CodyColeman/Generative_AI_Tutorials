# =============================================================================
# GenAI Tutorial 01: Large Language Models (LLMs) — Text Generation & Prompting
# =============================================================================
# Environment:  CPU-only (no GPU required)
# Dependencies: pip install transformers torch datasets accelerate
#
# This tutorial walks through:
#   1. Loading a pre-trained LLM from HuggingFace
#   2. Basic text generation with greedy decoding
#   3. Sampling strategies (temperature, top-k, top-p / nucleus)
#   4. Prompt engineering techniques
#   5. Batch generation
#   6. Working with a real HuggingFace dataset as prompts
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import textwrap
import time

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL SELECTION & LOADING
# ─────────────────────────────────────────────────────────────────────────────
#
# For CPU-friendly experimentation we use DistilGPT-2 (~330MB).
# It is a distilled (smaller, faster) version of GPT-2.
#
# Other CPU-viable model options (swap MODEL_NAME to try them):
#   "gpt2"               — 124M params, the classic baseline
#   "distilgpt2"         — 82M params,  ~2x faster than gpt2, slightly lower quality
#   "microsoft/phi-1_5"  — 1.3B params, surprisingly capable; slow on CPU
#   "facebook/opt-125m"  — 125M params, Meta's open pre-trained transformer
#
# Rule of thumb: model size ↑ → quality ↑, but speed ↓ and RAM ↑

MODEL_NAME = "distilgpt2"

print(f"Loading tokenizer and model: {MODEL_NAME}")
print("(First run will download ~350MB — subsequent runs use local cache)\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set pad token to eos token — required for batch generation
# (GPT-2 family models don't define a pad token by default)
tokenizer.pad_token = tokenizer.eos_token
model.eval()  # Switch to inference mode — disables dropout layers

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GREEDY DECODING — THE BASELINE
# ─────────────────────────────────────────────────────────────────────────────
#
# Greedy decoding picks the single most probable next token at every step.
# It is deterministic (same input → same output every time) and fast,
# but tends to produce repetitive, bland text.

def greedy_generate(prompt: str, max_new_tokens: int = 80) -> str:
    """Generate text using greedy (argmax) decoding."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # ← greedy when False
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

prompt = "Artificial intelligence is transforming the way we"
print("=" * 60)
print("GREEDY DECODING")
print("=" * 60)
print(f"Prompt: {prompt}\n")
print(textwrap.fill(greedy_generate(prompt), width=70))
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SAMPLING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
#
# Instead of always picking the top token, sampling draws from the probability
# distribution — introducing controlled randomness and diversity.

def sampled_generate(
    prompt: str,
    max_new_tokens: int  = 100,
    temperature: float   = 1.0,
    top_k: int           = 0,
    top_p: float         = 1.0,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1,
) -> list[str]:
    """
    Generate text with configurable sampling parameters.

    Parameters
    ----------
    temperature : float  (default 1.0)
        Controls randomness by scaling the logits before softmax.
        • < 1.0 → sharper distribution → more focused / conservative output
        • = 1.0 → unchanged distribution (raw model probabilities)
        • > 1.0 → flatter distribution → more creative / chaotic output
        Practical range: 0.2 (near-deterministic) to 1.5 (very creative)
        Sweet spot for most tasks: 0.7 – 1.0

    top_k : int  (default 0 = disabled)
        At each step, keep only the top-K most probable tokens and
        redistribute probability mass over just those K options.
        • Low k (10-40)  → safe, coherent, less varied
        • High k (100+)  → more diverse, risk of odd word choices
        • 0              → disabled (all tokens considered)

    top_p : float  (default 1.0 = disabled)
        Nucleus sampling. Keep the smallest set of tokens whose cumulative
        probability exceeds p, then sample from that nucleus.
        • 0.9 is a very common default — keeps ~90% of probability mass
        • Lower p → smaller nucleus → more focused
        • 1.0     → disabled (full vocabulary sampled)
        top_p is generally preferred over top_k in modern usage.

    repetition_penalty : float  (default 1.0 = no penalty)
        Penalizes tokens that have already appeared in the output.
        • 1.0  → no penalty
        • 1.2  → mild penalty (good default to reduce repetition)
        • 1.5+ → strong penalty (may degrade coherence)

    num_return_sequences : int
        How many independent completions to generate in one call.
    """
    inputs = tokenizer(
        [prompt] * num_return_sequences,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in output_ids:
        new_tokens = seq[prompt_len:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return results


# ── 3a. Temperature sweep ──────────────────────────────────────────────────
print("=" * 60)
print("TEMPERATURE SWEEP  (same prompt, different temperatures)")
print("=" * 60)
prompt = "The future of renewable energy depends on"

for temp in [0.3, 0.7, 1.2]:
    print(f"\n── temperature={temp} ──")
    result = sampled_generate(prompt, temperature=temp, top_p=0.9, max_new_tokens=60)
    print(textwrap.fill(result[0], width=70))

print()

# ── 3b. Top-p (nucleus) sampling ──────────────────────────────────────────
print("=" * 60)
print("TOP-P (NUCLEUS) SAMPLING")
print("=" * 60)
prompt = "Once upon a time in a world where machines could think,"

for top_p in [0.5, 0.9, 0.99]:
    print(f"\n── top_p={top_p} ──")
    result = sampled_generate(prompt, temperature=0.9, top_p=top_p, max_new_tokens=60)
    print(textwrap.fill(result[0], width=70))

print()

# ── 3c. Repetition penalty ────────────────────────────────────────────────
print("=" * 60)
print("REPETITION PENALTY")
print("=" * 60)
prompt = "Machine learning models learn patterns from data and"

for penalty in [1.0, 1.3]:
    print(f"\n── repetition_penalty={penalty} ──")
    result = sampled_generate(
        prompt, temperature=0.9, top_p=0.9,
        repetition_penalty=penalty, max_new_tokens=80
    )
    print(textwrap.fill(result[0], width=70))

print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PROMPT ENGINEERING TECHNIQUES
# ─────────────────────────────────────────────────────────────────────────────
#
# The way you phrase a prompt dramatically changes the output.
# These patterns work on any text-generation model.

print("=" * 60)
print("PROMPT ENGINEERING TECHNIQUES")
print("=" * 60)

# ── 4a. Zero-shot prompting ────────────────────────────────────────────────
# Give the model a task with no examples. Works well for instruction-tuned
# models; smaller base models (like GPT-2) may need more guidance.
zero_shot = "Summarize the main benefit of solar energy in one sentence:"
print(f"\n[Zero-shot]\n{zero_shot}")
print(textwrap.fill(sampled_generate(zero_shot, temperature=0.7, top_p=0.9, max_new_tokens=50)[0], 70))

# ── 4b. Few-shot prompting ─────────────────────────────────────────────────
# Provide 2–3 examples of input→output before your actual query.
# This "shows" the model the format and style you want.
few_shot = """Convert the sentence to a professional tone.

Casual: "hey can u send me that report asap"
Professional: "Could you please send me the report at your earliest convenience?"

Casual: "this meeting is gonna take forever"
Professional: "This meeting may run longer than anticipated."

Casual: "i cant figure out this bug its driving me crazy"
Professional:"""

print(f"\n[Few-shot]\n{few_shot}")
print(textwrap.fill(sampled_generate(few_shot, temperature=0.5, top_p=0.9, max_new_tokens=30)[0], 70))

# ── 4c. Role / persona prompting ──────────────────────────────────────────
# Prepend a role description to steer the model's voice and expertise.
role_prompt = "As an expert data scientist explaining to a business executive: What is a neural network?"
print(f"\n[Role prompting]\n{role_prompt}")
print(textwrap.fill(sampled_generate(role_prompt, temperature=0.7, top_p=0.9, max_new_tokens=80)[0], 70))

# ── 4d. Chain-of-thought style ────────────────────────────────────────────
# Ask the model to reason step by step before giving an answer.
cot_prompt = "Question: If a company's revenue grew 20% each year for 3 years starting at $1M, what is the final revenue? Let's think step by step:"
print(f"\n[Chain-of-thought]\n{cot_prompt}")
print(textwrap.fill(sampled_generate(cot_prompt, temperature=0.3, top_p=0.9, max_new_tokens=100)[0], 70))

print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: BATCH GENERATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Processing multiple prompts in a single forward pass is much faster
# than looping one-by-one on a CPU. Use padding + attention masks.

print("=" * 60)
print("BATCH GENERATION")
print("=" * 60)

batch_prompts = [
    "The key advantage of cloud computing is",
    "Cybersecurity best practices include",
    "The most important skill for a data analyst is",
]

tokenizer.padding_side = "left"   # GPT-style models should left-pad for batching
inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - start
print(f"\nGenerated {len(batch_prompts)} completions in {elapsed:.2f}s\n")

for i, (prompt, output) in enumerate(zip(batch_prompts, output_ids)):
    prompt_len = inputs["input_ids"].shape[1]
    new_text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
    print(f"[{i+1}] Prompt: {prompt}")
    print(f"     Output: {textwrap.fill(new_text, width=60, subsequent_indent='             ')}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: REAL DATASET PROMPTS (HuggingFace Datasets)
# ─────────────────────────────────────────────────────────────────────────────
#
# We load the "ag_news" dataset — a news topic classification dataset with
# 120,000 training articles. We'll use the headlines as generation prompts.

print("=" * 60)
print("REAL-WORLD PROMPTS FROM HUGGINGFACE DATASET (AG News)")
print("=" * 60)

print("\nLoading AG News dataset (test split, ~7,600 articles)...")
dataset = load_dataset("ag_news", split="test")

# AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Pick a few diverse examples — one from each category
samples = {}
for example in dataset:
    label = example["label"]
    if label not in samples:
        samples[label] = example
    if len(samples) == 4:
        break

print("\nUsing real news headlines as prompts:\n")
for label, example in sorted(samples.items()):
    # Use just the first sentence of the article as a prompt
    headline = example["text"].split(".")[0] + "."
    category = label_names[label]

    print(f"[{category}] Original headline:")
    print(f"  {headline}")

    # Generate a continuation of the news article
    continuation = sampled_generate(
        headline,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=60,
    )[0]

    print(f"  Model continuation:")
    print(f"  {textwrap.fill(continuation, width=65, subsequent_indent='  ')}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: THE PIPELINE API — QUICK & EASY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
#
# HuggingFace pipelines wrap tokenization + model + decoding into one call.
# Great for quick experiments; less control than manual generation.

print("=" * 60)
print("HUGGINGFACE PIPELINE API")
print("=" * 60)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,          # -1 = CPU; 0 = first GPU if available
)

pipeline_prompt = "The three most important trends in artificial intelligence are"
print(f"\nPrompt: {pipeline_prompt}\n")

results = generator(
    pipeline_prompt,
    max_new_tokens=80,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.9,
    top_p=0.92,
    repetition_penalty=1.1,
)

for i, r in enumerate(results, 1):
    # The pipeline returns the full text (prompt + generation)
    full_text = r["generated_text"]
    generated = full_text[len(pipeline_prompt):]
    print(f"[Sequence {i}]")
    print(textwrap.fill(generated.strip(), width=70))
    print()

print("=" * 60)
print("Tutorial complete! Check README.md for deep-dive parameter notes.")
print("=" * 60)
