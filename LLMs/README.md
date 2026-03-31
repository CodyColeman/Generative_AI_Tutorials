# 🤖 Tutorial 01: Large Language Models (LLMs)
### Text Generation & Prompting — CPU-Friendly

---

## Overview

This tutorial introduces **Large Language Models (LLMs)** — the backbone of tools like ChatGPT, Copilot, and Gemini. You'll learn how they generate text, what every generation parameter does, and how to write prompts that get consistent, high-quality output.

**Model used:** `distilgpt2` (82M parameters, runs on any modern laptop CPU)  
**Dataset used:** [AG News](https://huggingface.co/datasets/ag_news) — 120K news articles from HuggingFace

---

## Quick Start

```bash
# 1. Install dependencies
pip install transformers torch datasets accelerate

# 2. Run the tutorial
python llm_tutorial.py
```

> First run will download ~350MB of model weights to `~/.cache/huggingface/`.  
> All subsequent runs load from local cache (instant).

---

## How LLMs Work — The 30-Second Version

An LLM is a **next-token predictor**. Given the sequence of tokens so far, it outputs a probability distribution over the entire vocabulary (~50,000 words/subwords). "Generation" is just repeatedly sampling from that distribution and appending each new token until a stop condition is met.

```
Input:  "The sky is"
Model:  → {"blue": 42%, "clear": 18%, "dark": 9%, "falling": 0.01%, ...}
Sample: → "blue"
Input:  "The sky is blue"
Model:  → {"and": 31%, "today": 22%, ...}
... and so on
```

Everything else — temperature, top-k, top-p — is about **how we sample** from that distribution.

---

## 🎛️ Generation Parameters — The Full Knob Guide

### `max_new_tokens` *(int, default varies)*
Maximum number of tokens to generate. Note: **tokens ≠ words**. On average, 1 word ≈ 1.3 tokens (English). The word "unfortunately" = 3 tokens.

| Value | Use Case |
|-------|----------|
| 20–50 | Short answers, classifications, headlines |
| 50–150 | Paragraphs, summaries, explanations |
| 150–500 | Articles, reports, code blocks |
| 500+ | Long-form content (slow on CPU) |

---

### `temperature` *(float, default: 1.0)*
The single most important dial. Controls **randomness** by scaling logit scores before the softmax.

```
Low temp  → peaked distribution  → model picks "safe" high-prob tokens
High temp → flat distribution    → model explores low-prob tokens more
```

| Temperature | Behavior | Best For |
|-------------|----------|----------|
| `0.1 – 0.3` | Near-deterministic, very focused | Factual Q&A, classification, structured output |
| `0.4 – 0.7` | Balanced — coherent but not boring | Summaries, business writing, explanations |
| `0.8 – 1.0` | Creative, some unpredictability | Storytelling, brainstorming, dialogue |
| `1.1 – 1.5` | Wild, experimental | Ideation, poetry, when coherence is optional |
| `> 1.5` | Usually incoherent | Rarely useful |

> ⚠️ **Tip:** `temperature=0` is equivalent to greedy decoding (always picks the top token). Use `do_sample=False` instead for true greedy behavior.

---

### `top_k` *(int, default: 0 = disabled)*
At each step, restrict sampling to only the **K most probable tokens**, then renormalize.

```
full vocab: ["the"=40%, "a"=15%, "my"=8%, "her"=6%, ..., "zygote"=0.0001%]
top_k=3:    ["the"=59%, "a"=22%, "my"=12%]  ← only these 3 are considered
```

| top_k | Effect |
|-------|--------|
| `10` | Very conservative — "safe" outputs |
| `40` | Good balance (common default) |
| `100` | More varied, occasionally surprising |
| `0` | Disabled — full vocab sampled |

> ⚠️ **Tip:** `top_k` has a weakness: for a given step, the "right" number of reasonable candidates varies. `top_p` handles this more elegantly.

---

### `top_p` *(float, default: 1.0 = disabled)* — **Nucleus Sampling**
Instead of a fixed K, keep the **smallest set of tokens whose cumulative probability ≥ p**.

```
Step with high certainty:  ["the"=85%, "a"=10%] → nucleus at p=0.9 = just 2 tokens
Step with high uncertainty: many tokens needed  → nucleus at p=0.9 = maybe 200 tokens
```

This adapts dynamically — it's selective when the model is confident, open when it's uncertain.

| top_p | Effect |
|-------|--------|
| `0.7` | Tight nucleus — safe, coherent |
| `0.9` | **Most popular default** — great balance |
| `0.95` | Slightly more diverse |
| `1.0` | Disabled |

> ✅ **Best practice:** Use `top_p=0.9` with `temperature=0.8` as your starting point for most creative tasks. These two work together — `top_p` narrows candidates, `temperature` controls how you sample among them.

---

### `repetition_penalty` *(float, default: 1.0)*
Reduces the probability of tokens that have already appeared in the output.

```
penalty=1.0 → no change
penalty=1.2 → repeated tokens are 1.2x less likely to appear again
penalty=1.5 → repeated tokens are 1.5x less likely
```

| Value | Effect |
|-------|--------|
| `1.0` | No penalty (model may loop) |
| `1.1 – 1.2` | Gentle discouragement — **recommended default** |
| `1.3 – 1.5` | Strong — use for longer generations prone to looping |
| `> 1.5` | Can degrade quality; model avoids useful repeated words |

> ⚠️ **Watch out:** Over-penalizing can prevent the model from using useful conjunctions ("the", "and", "is") appropriately in long outputs.

---

### `do_sample` *(bool, default: False)*
Master switch for sampling. When `False`, ignores temperature/top_k/top_p entirely.

| Value | Behavior |
|-------|----------|
| `False` | Greedy / beam search — deterministic |
| `True` | Sampling — stochastic, enables temp/top_k/top_p |

---

### `num_beams` *(int, default: 1)*
Enables **beam search** when > 1. Instead of committing to one token at a time, maintains `num_beams` candidate sequences in parallel and picks the globally best one.

```
num_beams=1  → greedy (fastest)
num_beams=4  → explore 4 paths simultaneously (4x slower, often better quality)
num_beams=8  → high quality, slow
```

> ⚠️ **Note:** Beam search is incompatible with `do_sample=True`. Use one or the other.

| Setting | Use Case |
|---------|----------|
| `num_beams=1, do_sample=True` | Creative, diverse outputs |
| `num_beams=4, do_sample=False` | High-quality deterministic outputs (translations, summaries) |

---

### `num_return_sequences` *(int, default: 1)*
Generate multiple independent completions in one call. Useful for:
- Showing users options to choose from
- Picking the best output programmatically (e.g., highest perplexity score)
- Diversity sampling for downstream tasks

---

## 💬 Prompt Engineering Guide

### Zero-Shot
Give the task directly with no examples. Works best with instruction-tuned models.
```
Summarize the following in one sentence: [text]
```

### Few-Shot
Show 2–5 examples before your actual query. Dramatically improves format consistency.
```
Input: "happy"  → Output: "sad"
Input: "fast"   → Output: "slow"
Input: "bright" → Output:
```

### Role Prompting
Prepend a persona to steer voice, expertise level, and tone.
```
You are a senior software engineer doing a code review. Evaluate this function: [code]
```

### Chain-of-Thought (CoT)
Ask the model to reason before answering. Append "Let's think step by step:" to difficult questions.
```
Q: [complex question] Let's think step by step:
```

### Instruction Framing
For base models (not instruction-tuned), frame output format explicitly:
```
Write a Python function that [task]. The function should:
- Accept [input]
- Return [output]
- Handle [edge case]

def [function_name](
```

---

## 🔧 Common Recipes

### Creative writing (stories, dialogue)
```python
temperature=1.0, top_p=0.92, repetition_penalty=1.1
```

### Factual / technical answers
```python
temperature=0.3, top_p=0.85, do_sample=True
```

### Summarization
```python
temperature=0.5, top_p=0.9, num_beams=4, do_sample=False
```

### Brainstorming (diverse ideas)
```python
temperature=1.1, top_p=0.95, num_return_sequences=5
```

---

## 📊 Model Comparison (CPU)

| Model | Params | RAM | Speed (CPU) | Quality |
|-------|--------|-----|-------------|---------|
| `distilgpt2` | 82M | ~0.5GB | ⚡⚡⚡ | ⭐⭐ |
| `gpt2` | 124M | ~0.7GB | ⚡⚡⚡ | ⭐⭐ |
| `gpt2-medium` | 355M | ~1.5GB | ⚡⚡ | ⭐⭐⭐ |
| `facebook/opt-125m` | 125M | ~0.6GB | ⚡⚡⚡ | ⭐⭐ |
| `microsoft/phi-1_5` | 1.3B | ~3GB | ⚡ | ⭐⭐⭐⭐ |

> For production or quality-sensitive use cases, use API-based models (OpenAI, Anthropic, etc.) or GPU-hosted open models.

---

## 🧪 Experiments to Try

1. **Temperature sensitivity test:** Run the same creative prompt 5 times at temp=0.3 vs 1.2. Notice how output diversity changes.

2. **Top-p vs Top-k:** Use an identical prompt with `top_k=40` vs `top_p=0.9`. Which produces more natural outputs?

3. **Repetition penalty audit:** Generate 200 tokens without a penalty, then with `repetition_penalty=1.3`. Look for looping patterns.

4. **Prompt format matters:** Try the same question as (a) a raw statement, (b) a question, (c) a few-shot example. Compare quality.

5. **Batch vs loop speed:** Time `N=10` prompts as a batch vs a `for` loop. Measure the speedup.

---

## Common Pitfalls

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Model repeats itself endlessly | No repetition penalty | Add `repetition_penalty=1.2` |
| Output is incoherent | Temperature too high | Lower to `0.7–0.9` |
| Output is generic/boring | Temperature too low | Raise to `0.9–1.1` |
| Output cuts off mid-sentence | `max_new_tokens` too low | Increase it |
| Batch generation errors | Missing pad token | Set `tokenizer.pad_token = tokenizer.eos_token` |
| Out of memory on CPU | Model too large | Use `distilgpt2` or `facebook/opt-125m` |

---

## What's Next

| Tutorial | Topic |
|----------|-------|
| `02_image_generation/` | Stable Diffusion — generating images from text |
| `03_rag_models/` | Retrieval-Augmented Generation — grounding LLMs in your own data |
| `04_fine_tuning/` | Fine-tuning a model on a custom dataset with LoRA |

---

## Resources

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — excellent visual explainer
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
