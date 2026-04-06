# 🎯 Tutorial 04 — Fine-Tuning Generative Models

Fine-tune language models with LoRA/PEFT and image diffusion models with Dreambooth — without needing a GPU farm.

---

## Overview

| File | What it does | Core libraries |
|---|---|---|
| `finetune_llm.py` | LoRA fine-tuning of a causal LM on instruction data | `transformers`, `peft`, `trl`, `datasets` |
| `finetune_image.py` | Dreambooth + LoRA for personalising a diffusion model | `diffusers`, `peft`, `accelerate`, `torch` |

**Dataset (LLM):** [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — 15k human-written instruction/response pairs across 8 task categories.

**Dataset (Image):** A small set of sample concept images pulled from HuggingFace, used as the "subject" for Dreambooth training.

**Goals:**
- Understand *why* you would fine-tune vs. prompt-engineer vs. use RAG
- Learn LoRA mechanics: what low-rank decomposition actually does to a weight matrix
- Fine-tune a language model on instruction data with `SFTTrainer`
- Fine-tune a diffusion model on a custom concept with Dreambooth
- Save, load, and merge LoRA adapters for both modalities

---

## Quick Start

```bash
# LLM fine-tuning
pip install transformers peft trl datasets accelerate bitsandbytes torch

python finetune_llm.py

# Image fine-tuning
pip install diffusers peft accelerate transformers torch torchvision

python finetune_image.py
```

> **First run:** `finetune_llm.py` will download `facebook/opt-125m` (~250 MB) and a subset of dolly-15k. `finetune_image.py` will download `runwayml/stable-diffusion-v1-5` (~4 GB) on first run — subsequent runs use the cache. Both scripts are CPU-runnable but will be slow; see the GPU note in each script header.

---

## When to Fine-Tune (vs. the Alternatives)

This is the most important question to answer before you write a single line of training code.

```
Given a task you want an LLM to perform well...

                        ┌─────────────────────────────────────┐
                        │  Can a good prompt already do this?  │
                        └──────────────┬──────────────────────┘
                                       │
                          Yes ─────────┴──────── No
                           │                      │
                    Use prompt                     │
                    engineering              Does the task require
                    (zero/few-shot,          grounding in specific,
                     CoT, etc.)              up-to-date documents?
                                                   │
                                      Yes ─────────┴──────── No
                                       │                      │
                                   Use RAG             Does the task require
                                (Tutorial 03)          a specific *style*,
                                                       *format*, or *behaviour*
                                                       the base model lacks?
                                                               │
                                                  Yes ─────────┴──────── No
                                                   │                      │
                                            Fine-tune ✅           You probably
                                                              don't need fine-tuning
```

| Approach | Best for | Weakness |
|---|---|---|
| Prompt engineering | Quick iteration, general tasks | Token cost per call, limited control |
| RAG | Tasks requiring factual grounding, live data | Retrieval quality ceiling, latency |
| Fine-tuning | Style transfer, consistent format, domain behaviour | Cost, data requirements, overfitting risk |
| All three combined | Production systems | Complexity |

> **Rule of thumb:** exhaust prompting and RAG first. Fine-tuning is the right tool when you need the model to *be* something different, not just *know* something different.

---

## How LoRA Works

LoRA (Low-Rank Adaptation) is the dominant parameter-efficient fine-tuning technique. Instead of updating all model weights, it freezes them and inserts small trainable matrices alongside the existing ones.

### The Core Idea

A standard weight matrix `W` has shape `(d_out, d_in)`. Updating it fully during training means updating `d_out × d_in` parameters — millions for a transformer.

LoRA instead learns a **low-rank decomposition** of the update:

```
W' = W + ΔW
       ↑
   ΔW = B × A
        ↑   ↑
   (d_out × r) (r × d_in)

   where r << min(d_out, d_in)
```

For a weight matrix of shape `(4096, 4096)` and `r=8`:
- Full update: **16,777,216** parameters
- LoRA update: `(4096×8) + (8×4096)` = **65,536** parameters → **256× fewer**

During inference you can either:
1. **Keep adapters separate** — load the base model and apply the adapter on top (fast switching between adapters)
2. **Merge** — fold `ΔW` into `W` permanently, so inference is identical to a fully fine-tuned model with zero overhead

```
Training:                          Inference (merged):

  Input                              Input
    │                                  │
    ▼                                  ▼
  W (frozen)  ←── LoRA: B×A        W + B×A (single matrix)
    │                                  │
    ▼                                  ▼
  Output                             Output

  Only A and B are updated.         No extra cost.
```

---

## 🎛️ Parameter Guide — LLM Fine-Tuning

### `r` — LoRA Rank

Controls the rank of the decomposition matrices `A` and `B`. This is the most important LoRA hyperparameter.

| Value | Trainable params (per layer) | Use case |
|---|---|---|
| 4 | Very few | Quick style transfer, minimal compute |
| 8 | Few | ✅ Standard starting point for most tasks |
| 16 | Moderate | Complex behavioural adaptation |
| 32–64 | Many | Approaching full fine-tune territory |
| 128+ | Very many | Rarely beneficial; risk of defeating LoRA's purpose |

> ✅ **Recommended default:** `r=8`. Start here. If the model underfits (doesn't learn the task), try `r=16` before touching anything else.

> ⚠️ Higher `r` is not always better. The point of LoRA is that the *update* to a pre-trained weight is intrinsically low-rank. If you push `r` too high, you start memorising noise rather than capturing the true update direction.

---

### `lora_alpha` — LoRA Scaling Factor

Scales the LoRA output before it's added to the frozen weight: `ΔW = (alpha/r) × B×A`. Controls the *magnitude* of the adapter's influence.

| `lora_alpha` | Effect |
|---|---|
| `= r` | Scale factor of 1.0 — neutral |
| `= 2×r` | Scale factor of 2.0 — ✅ common convention |
| Very high | Adapter dominates; can destabilise training |
| Very low | Adapter barely contributes |

> ✅ **Recommended default:** set `lora_alpha = 2 * r`. If `r=8`, use `lora_alpha=16`. This is the community convention and rarely needs changing.

> ⚠️ `lora_alpha` and learning rate interact. If you increase `lora_alpha`, reduce your LR proportionally — you're effectively scaling the gradients.

---

### `lora_dropout` — Dropout on LoRA Layers

Regularisation applied to the LoRA adapter layers during training. Prevents the adapter from overfitting to small datasets.

| Value | Effect |
|---|---|
| 0.0 | No dropout — fine for large datasets |
| 0.05 | ✅ Light regularisation — good default |
| 0.1 | Moderate — useful for small datasets (<1k examples) |
| 0.2+ | Aggressive — use only if severe overfitting |

> ✅ **Recommended default:** `0.05` for most tasks.

> ⚠️ Dropout is disabled at inference automatically by PyTorch when you call `model.eval()`. You don't need to remove it manually.

---

### `target_modules` — Which Layers to Adapt

Specifies which weight matrices in the transformer to attach LoRA adapters to. Different architectures use different names.

| Value | What it targets | Trainable params |
|---|---|---|
| `["q_proj", "v_proj"]` | Attention query + value only | ✅ Fewest — good starting point |
| `["q_proj", "k_proj", "v_proj", "o_proj"]` | All attention projections | Moderate |
| `["q_proj", "v_proj", "gate_proj", "up_proj"]` | Attention + MLP layers | More |
| `"all-linear"` | Every linear layer | Most — approaching full fine-tune |

> ✅ **Recommended default:** `["q_proj", "v_proj"]`. Attention layers capture style and task format; adding MLP layers helps if the task requires more factual/domain adaptation.

> ⚠️ Module names vary by architecture. `q_proj`/`v_proj` works for LLaMA, Mistral, OPT. GPT-2 style models use `c_attn`. If you get a key error, call `print(model)` and inspect the layer names.

---

### `num_train_epochs` — Training Epochs

How many full passes through the training dataset.

| Value | Risk |
|---|---|
| 1 | Underfitting on small datasets |
| 2–3 | ✅ Sweet spot for instruction tuning |
| 5+ | Overfitting risk — model starts memorising examples |
| 10+ | Almost certainly overfitting |

> ✅ **Recommended default:** `3` epochs. Watch your validation loss; if it stops falling before epoch 3, stop early.

---

### `per_device_train_batch_size` — Batch Size

Number of examples processed per gradient step, per device.

| Value | Effect |
|---|---|
| 1 | ✅ CPU / low-VRAM safe; slow convergence |
| 4 | Good GPU default |
| 8–16 | Fast training; needs ≥16 GB VRAM |

> ⚠️ Use `gradient_accumulation_steps` to simulate larger batches without more memory. `batch_size=1, accumulation=8` is equivalent to `batch_size=8` for gradient computation.

---

### `max_seq_length` — Maximum Token Length

Sequences longer than this are truncated. Affects memory use quadratically in attention.

| Value | Trade-off |
|---|---|
| 256 | Fast, low memory — fine for short instructions |
| 512 | ✅ Good balance for instruction tuning |
| 1024 | Needed for longer contexts; 4× the memory of 256 |
| 2048+ | Requires significant VRAM; use gradient checkpointing |

> ✅ **Recommended default:** `512` for dolly-style instruction data.

---

## 🎛️ Parameter Guide — Image Fine-Tuning (Dreambooth + LoRA)

### Dreambooth Concept

Dreambooth teaches the diffusion model a new **concept** (a specific person, object, or style) using only 5–20 images. It binds the concept to a rare **trigger word** (e.g. `"sks"`) so the model can generate that concept on demand.

```
Training images: 15 photos of your subject
Trigger word:    "sks"

After training:
  "a photo of sks dog"        → generates your specific dog
  "sks dog in a space suit"   → your specific dog, in a space suit
  "oil painting of sks dog"   → your specific dog, in oil paint style
```

---

### `instance_prompt` — The Trigger Phrase

The text prompt used for all training images. Format: `"a [trigger_word] [class_noun]"`.

| Example | When to use |
|---|---|
| `"a sks dog"` | Subject is a specific animal |
| `"a sks person"` | Subject is a person (use for face/portrait work) |
| `"a sks [style] painting"` | Teaching an art style |

> ✅ Use `"sks"` as your trigger token — it's rare in the training data so it doesn't carry pre-existing meaning.

> ⚠️ The trigger word must appear **exactly** the same way at inference. `"sks dog"` and `"a sks dog"` can behave differently.

---

### `prior_preservation_loss` — Class Preservation

Dreambooth can catastrophically forget what the general class (e.g. "dog") looks like while learning your specific instance. Prior preservation prevents this by co-training on generated examples of the general class.

| Setting | Effect |
|---|---|
| Disabled | Faster training; risk of language drift |
| Enabled ✅ | Stable training; model retains general class knowledge |

> ✅ **Always enable for face/person fine-tuning.** For style LoRAs, it's less critical.

> ⚠️ Enabling prior preservation doubles your effective training data — it generates `num_class_images` (default 200) class images before training begins. This adds upfront time.

---

### `num_train_epochs` (Image) — Epochs for Dreambooth

| Value | Risk |
|---|---|
| 50–100 | Underfitting — concept not learned |
| 200–400 | ✅ Sweet spot for most subjects |
| 800+ | Overfitting — outputs look like your training photos, not diverse generations |

> ⚠️ Image fine-tuning overfitting looks different from LLM overfitting: the model loses the ability to generalise the concept to new prompts. Test with varied prompts during training to catch this early.

---

### `learning_rate` (Image) — Diffusion LoRA LR

Much lower than LLM fine-tuning because diffusion models are more sensitive.

| Value | Effect |
|---|---|
| `1e-4` | Aggressive — fast learning, risk of destroying prior |
| `1e-5` | ✅ Standard for LoRA on diffusion models |
| `5e-6` | Conservative — use if prior preservation fails |

---

### `lora_rank` (Image) — Rank for Diffusion LoRA

Same mechanics as `r` in PEFT, but conventions differ slightly for diffusion models.

| Value | Use case |
|---|---|
| 4 | Style LoRAs, subtle character shifts |
| 8 | ✅ Good general default |
| 16–32 | Detailed subject learning (faces, specific objects) |

---

## LoRA Across Both Modalities — Side-by-Side

| Concept | LLM LoRA | Diffusion LoRA |
|---|---|---|
| What's being adapted | Transformer attention/MLP weights | UNet attention layers |
| Typical rank | 8–16 | 4–16 |
| Training data size | 1k–100k examples | 5–20 images |
| Training time (CPU) | Hours | Days (use GPU) |
| Adapter file size | 10–100 MB | 2–50 MB |
| Merge back to base | ✅ Supported via `merge_and_unload()` | ✅ Supported |
| Share adapters | HuggingFace Hub | CivitAI or HuggingFace Hub |
| Trigger mechanism | Task format / system prompt | Trigger word in prompt |

---

## Dataset Notes

### Dolly-15k (LLM)
`databricks/databricks-dolly-15k` contains 15,011 records, each with:
- `instruction` — the task prompt
- `context` — optional supporting text (empty string if not applicable)
- `response` — the expected output
- `category` — one of: `open_qa`, `closed_qa`, `classification`, `summarization`, `information_extraction`, `creative_writing`, `brainstorming`, `general_qa`

The script formats these into a standard instruction template before training:
```
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

> ⚠️ For CPU runs, the script trains on a 500-example subset. Set `MAX_TRAIN_SAMPLES = None` in the script to use the full dataset (requires a GPU for practical training time).

---

## Training Data Requirements for Dreambooth

| Subject type | Min images | Recommended | Notes |
|---|---|---|---|
| Specific person / face | 15 | 20–30 | Varied lighting and angles critical |
| Specific object | 10 | 15–20 | Clean background helps |
| Art style | 5 | 10–20 | Consistency of style matters more than count |
| Pet / animal | 10 | 15 | Varied poses, consistent subject |

**Image quality checklist:**
- No heavy compression artifacts
- Subject clearly visible and not cropped
- Varied angles, lighting, and backgrounds (avoids overfitting to a specific setting)
- Consistent subject (all images are of the *same* subject)
- Square crop or near-square (512×512 is the training resolution)

---

## Saving and Loading Adapters

### LLM Adapter

```python
# Save adapter only (small — just the LoRA weights)
model.save_pretrained("./outputs/lora-adapter")
tokenizer.save_pretrained("./outputs/lora-adapter")

# Load later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./outputs/lora-adapter")

# Merge adapter into base weights (no inference overhead)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./outputs/merged-model")
```

### Diffusion LoRA

```python
# Save as .safetensors (standard format, compatible with ComfyUI / A1111)
pipeline.save_lora_weights("./outputs/image-lora")

# Load at inference
pipeline.load_lora_weights("./outputs/image-lora")
pipeline("a sks dog on the moon", num_inference_steps=30)

# Adjust adapter strength at inference (0.0 = no effect, 1.0 = full)
pipeline.set_adapters(["default"], adapter_weights=[0.8])
```

> ✅ Keep your adapter files separate from the base model. This lets you swap adapters or update the base model independently.

---

## Common Pitfalls

| Problem | Likely Cause | Fix |
|---|---|---|
| Loss doesn't decrease at all | Learning rate too low, or `target_modules` missed all trainable layers | Print `model.print_trainable_parameters()` — if 0, your target_modules names are wrong |
| Model repeats or collapses after fine-tuning | Too many epochs, LR too high | Reduce epochs to 2–3, halve the LR |
| `KeyError` on `target_modules` | Module names don't match architecture | Run `print(model)` and copy the correct layer names |
| OOM error during training | Batch size or seq length too large | Set `batch_size=1`, enable `gradient_checkpointing=True` |
| Dreambooth outputs ignore trigger word | Trigger word appears in base training data | Switch to a rarer token (`sks`, `xjy`, `ohwx`) |
| Dreambooth forgets general class | Prior preservation disabled | Enable `with_prior_preservation=True` |
| Image LoRA too strong / burns concept | Rank too high or LR too high | Reduce `lora_rank` to 4, halve LR |
| Adapter loads but generates garbage | Base model mismatch | Adapter must be loaded onto the same base model it was trained on |

---

## What's Next

| Tutorial | What it adds |
|---|---|
| ← [03 RAG Models](../03_rag_models/README.md) | Retrieval-augmented generation — grounding without fine-tuning |
| → [05 Embeddings](../05_embeddings/README.md) | Semantic search, clustering, and visualising embedding space |
| → [06 Agents](../06_agents/README.md) | Tool-using LLMs, function calling, building an agent loop |

---

## Resources

**LoRA (Language Models)**
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — original paper
- [PEFT documentation](https://huggingface.co/docs/peft) — HuggingFace PEFT library
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [Dolly-15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Dreambooth / Diffusion LoRA**
- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2208.12242) — original paper
- [diffusers Dreambooth training script](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
- [LoRA for diffusion models — diffusers docs](https://huggingface.co/docs/diffusers/training/lora)
- [CivitAI](https://civitai.com) — community LoRA models (⚠️ contains NSFW content — use with discretion)

**General**
- [Practical Tips for Fine-Tuning LLMs](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka's guide
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — quantised LoRA for even lower memory footprint
