# =============================================================================
# Tutorial 04 — Fine-Tuning Language Models with LoRA / PEFT
# =============================================================================
#
# Environment:  Python 3.9+, CPU-first (GPU strongly recommended for full runs)
#
# Dependencies:
#   pip install transformers peft trl datasets accelerate bitsandbytes torch
#
# Sections:
#   1. Imports & Configuration
#   2. What LoRA Does — Live Parameter Count Demo
#   3. Load Dataset (Databricks Dolly-15k)
#   4. Format Data into Instruction Template
#   5. Load Base Model & Tokenizer
#   6. Apply LoRA with PEFT
#   7. Fine-Tune with SFTTrainer
#   8. Inference — Before vs. After Comparison
#   9. Save Adapter & Merge into Base Model
#  10. Summary
#
# CPU note:
#   This script runs end-to-end on CPU using a 500-example subset of Dolly
#   and facebook/opt-125m (~250 MB). Full training on the complete dataset
#   requires a GPU. Set MAX_TRAIN_SAMPLES = None to use the full dataset.
# =============================================================================

# ─────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────

import os
import json
import time
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

# ── Model ─────────────────────────────────────────────
# facebook/opt-125m: small causal LM, fast to download, CPU-runnable.
# For a more capable fine-tune, swap in:
#   "facebook/opt-350m"          (~700 MB)
#   "facebook/opt-1.3b"          (~2.5 GB, GPU recommended)
#   "mistralai/Mistral-7B-v0.1"  (GPU required)
BASE_MODEL = "facebook/opt-125m"

# ── Dataset ───────────────────────────────────────────
DATASET_NAME = "databricks/databricks-dolly-15k"

# Number of training examples to use. Set to None for the full 15k dataset
# (GPU strongly recommended for the full set).
MAX_TRAIN_SAMPLES = 500

# ── LoRA Hyperparameters ──────────────────────────────
LORA_R = 8            # Rank of the decomposition matrices. See README §r.
LORA_ALPHA = 16       # Scaling factor. Convention: 2 * r. See README §lora_alpha.
LORA_DROPOUT = 0.05   # Dropout on adapter layers. See README §lora_dropout.

# Weight matrix names to attach adapters to.
# OPT uses "q_proj" / "v_proj" for its attention projections.
# Print the model to find names for other architectures.
TARGET_MODULES = ["q_proj", "v_proj"]

# ── Training Hyperparameters ──────────────────────────
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1     # Keep at 1 for CPU. Increase on GPU.
GRADIENT_ACCUMULATION = 4     # Effective batch size = 1 * 4 = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
LOGGING_STEPS = 10

# ── Output ────────────────────────────────────────────
OUTPUT_DIR = "./outputs/finetune_llm"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora-adapter")
MERGED_DIR  = os.path.join(OUTPUT_DIR, "merged-model")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# SECTION 2: WHAT LoRA DOES — LIVE PARAMETER COUNT DEMO
# ─────────────────────────────────────────────────────
#
# Before touching any data, let's make the LoRA math concrete.
# This section demonstrates — numerically — how LoRA reduces the number
# of trainable parameters for a single weight matrix.

print("=" * 65)
print("SECTION 2: LoRA Parameter Count Demo")
print("=" * 65)


def lora_param_demo(d_out: int, d_in: int, r: int) -> None:
    """
    Print a comparison of full fine-tune vs. LoRA trainable parameters
    for a single weight matrix of shape (d_out, d_in).

    Parameters
    ----------
    d_out : int
        Output dimension of the weight matrix.
    d_in : int
        Input dimension of the weight matrix.
    r : int
        LoRA rank. The LoRA update ΔW = B × A where
        B has shape (d_out, r) and A has shape (r, d_in).
        Lower r → fewer parameters → more regularisation.
    """
    full_params = d_out * d_in
    lora_params = (d_out * r) + (r * d_in)
    reduction   = full_params / lora_params

    print(f"\n  Weight matrix shape : ({d_out}, {d_in})")
    print(f"  LoRA rank (r)       : {r}")
    print(f"  Full fine-tune params : {full_params:>12,}")
    print(f"  LoRA params (A + B)   : {lora_params:>12,}   (B:{d_out}×{r}  +  A:{r}×{d_in})")
    print(f"  Reduction factor      : {reduction:>11.0f}×")


# Demonstrate on sizes typical of small, medium, and large transformers
lora_param_demo(d_out=768,  d_in=768,  r=LORA_R)   # OPT-125m attention
lora_param_demo(d_out=2048, d_in=2048, r=LORA_R)   # OPT-1.3b attention
lora_param_demo(d_out=4096, d_in=4096, r=LORA_R)   # LLaMA-7B / Mistral-7B

print()


# ─────────────────────────────────────────────────────
# SECTION 3: LOAD DATASET
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 3: Load Dataset — Databricks Dolly-15k")
print("=" * 65)

raw_dataset = load_dataset(DATASET_NAME, split="train")

print(f"\n  Full dataset size : {len(raw_dataset):,} examples")
print(f"  Columns           : {raw_dataset.column_names}")

# Inspect the task category breakdown
from collections import Counter
category_counts = Counter(raw_dataset["category"])
print("\n  Category breakdown:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 200)
    print(f"    {cat:<30} {count:>5}  {bar}")

# Take a subset for CPU-friendly training
if MAX_TRAIN_SAMPLES is not None:
    dataset = raw_dataset.select(range(MAX_TRAIN_SAMPLES))
    print(f"\n  Using subset: {MAX_TRAIN_SAMPLES} examples (set MAX_TRAIN_SAMPLES=None for full dataset)")
else:
    dataset = raw_dataset
    print(f"\n  Using full dataset: {len(dataset):,} examples")

# Show one raw example
print("\n  Example record:")
sample = dataset[0]
for key, val in sample.items():
    preview = str(val)[:120].replace("\n", " ↵ ")
    print(f"    {key:<15}: {preview}")

print()


# ─────────────────────────────────────────────────────
# SECTION 4: FORMAT DATA INTO INSTRUCTION TEMPLATE
# ─────────────────────────────────────────────────────
#
# Raw Dolly records have separate 'instruction', 'context', and 'response'
# fields. We need to concatenate them into a single string that teaches the
# model the desired input→output format.
#
# SFTTrainer expects a dataset with a single text column (named "text" by
# default, configurable via dataset_text_field).

print("=" * 65)
print("SECTION 4: Format Data into Instruction Template")
print("=" * 65)


def format_instruction(example: dict) -> dict:
    """
    Convert a Dolly record into a single instruction-following string.

    The template structure follows the Alpaca instruction format, which has
    become a community standard for instruction fine-tuning. The model learns
    to map everything between ### Instruction / ### Context into the text
    that follows ### Response.

    Parameters
    ----------
    example : dict
        A single dataset record with keys:
        - instruction : str  The task description or question.
        - context     : str  Optional supporting text (empty string if absent).
        - response    : str  The expected model output.

    Returns
    -------
    dict
        A dict with a single key "text" containing the formatted string.
        SFTTrainer reads this column during training.
    """
    instruction = example["instruction"].strip()
    context     = example["context"].strip()
    response    = example["response"].strip()

    if context:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return {"text": text}


formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

print(f"\n  Formatted {len(formatted_dataset):,} examples.")
print("\n  Example formatted record:")
print("  " + "─" * 60)
for line in formatted_dataset[0]["text"].split("\n"):
    print(f"  {line}")
print("  " + "─" * 60)
print()


# ─────────────────────────────────────────────────────
# SECTION 5: LOAD BASE MODEL & TOKENIZER
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 5: Load Base Model & Tokenizer")
print("=" * 65)

print(f"\n  Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# OPT uses a right-side padding token by default, but for causal LM training
# we pad on the right with the EOS token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"  Vocabulary size    : {tokenizer.vocab_size:,}")
print(f"  Pad token          : '{tokenizer.pad_token}' (id {tokenizer.pad_token_id})")
print(f"  EOS token          : '{tokenizer.eos_token}' (id {tokenizer.eos_token_id})")

print(f"\n  Loading model: {BASE_MODEL}")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # float32 for CPU; use float16/bfloat16 on GPU
)
print(f"  Loaded in {time.time() - t0:.1f}s")

# Count total parameters in the base model
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters   : {total_params:,}  ({total_params / 1e6:.1f}M)")

# Show model architecture so we can see the layer names
print("\n  Model architecture (abbreviated):")
for name, module in list(model.named_modules())[:20]:
    print(f"    {name:<50} {type(module).__name__}")
print("    ...")
print()


# ─────────────────────────────────────────────────────
# SECTION 6: APPLY LoRA WITH PEFT
# ─────────────────────────────────────────────────────
#
# get_peft_model wraps the base model, freezes all original weights, and
# inserts trainable LoRA matrices alongside the target layers.
# Nothing about the model's forward pass changes — LoRA is additive.

print("=" * 65)
print("SECTION 6: Apply LoRA Adapters with PEFT")
print("=" * 65)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    r=LORA_R,
    # ── r (rank) ──────────────────────────────────────────────────────────
    # Rank of matrices A and B. ΔW = B × A where B ∈ R^(d×r), A ∈ R^(r×d).
    # Lower r → fewer params, more regularisation.
    # Higher r → more expressiveness, more compute.
    # Start at 8. If underfitting, try 16.

    lora_alpha=LORA_ALPHA,
    # ── lora_alpha ────────────────────────────────────────────────────────
    # Scaling factor. The adapter output is scaled by (alpha / r) before
    # being added to the frozen weight output.
    # Convention: alpha = 2 * r. Rarely needs tuning independently of r.

    lora_dropout=LORA_DROPOUT,
    # ── lora_dropout ──────────────────────────────────────────────────────
    # Dropout probability applied to LoRA layers during training.
    # Disabled automatically at inference (model.eval()).
    # Use 0.05–0.1 to regularise on small datasets.

    target_modules=TARGET_MODULES,
    # ── target_modules ────────────────────────────────────────────────────
    # Which weight matrices to attach LoRA adapters to.
    # OPT attention uses "q_proj" and "v_proj".
    # Run print(model) to find names for other architectures.

    bias="none",
    # ── bias ──────────────────────────────────────────────────────────────
    # Whether to train bias parameters alongside LoRA.
    # "none" → only LoRA matrices are trained (standard practice).
    # "all"  → also trains biases (rarely beneficial, more params).
)

model = get_peft_model(model, lora_config)

print("\n  LoRA config applied. Trainable parameter breakdown:")
model.print_trainable_parameters()

# Show what's frozen vs. trainable
print("\n  Sample parameter status (first 15 named params):")
print(f"  {'Parameter':<55} {'Shape':<20} Trainable?")
print("  " + "─" * 85)
for name, param in list(model.named_parameters())[:15]:
    shape_str = str(tuple(param.shape))
    trainable = "✅ YES" if param.requires_grad else "❌ frozen"
    print(f"  {name:<55} {shape_str:<20} {trainable}")
print("  ...")
print()


# ─────────────────────────────────────────────────────
# SECTION 7: FINE-TUNE WITH SFTTrainer
# ─────────────────────────────────────────────────────
#
# SFTTrainer (Supervised Fine-tuning Trainer) from the trl library extends
# HuggingFace Trainer with quality-of-life features for instruction tuning:
#   - Handles packing (concatenates short examples to fill max_seq_length)
#   - Integrates with PEFT transparently
#   - Masks the instruction/context tokens from the loss so the model only
#     learns to predict the response (optional, via response_template)

print("=" * 65)
print("SECTION 7: Fine-Tune with SFTTrainer")
print("=" * 65)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_TRAIN_EPOCHS,
    # ── num_train_epochs ──────────────────────────────────────────────────
    # Full passes through the training dataset.
    # 2–3 is the sweet spot for instruction tuning.
    # Beyond 5: overfitting risk (model memorises examples).

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    # ── per_device_train_batch_size ───────────────────────────────────────
    # Examples processed per gradient step per device.
    # Keep at 1 for CPU. On a 16 GB GPU, try 4–8.

    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    # ── gradient_accumulation_steps ───────────────────────────────────────
    # Accumulate gradients over N steps before updating weights.
    # Effective batch size = per_device_batch_size × accumulation_steps.
    # Use this to simulate larger batches without more memory.

    learning_rate=LEARNING_RATE,
    # ── learning_rate ─────────────────────────────────────────────────────
    # Step size for the AdamW optimiser.
    # 2e-4 is a good default for LoRA. If loss spikes, halve it.
    # If convergence is very slow, try 5e-4.

    max_seq_length=MAX_SEQ_LENGTH,
    # ── max_seq_length ────────────────────────────────────────────────────
    # Sequences longer than this are truncated; shorter ones are padded.
    # Memory scales quadratically with this value in attention.
    # 512 covers most Dolly examples. Use 1024+ for longer-form tasks.

    logging_steps=LOGGING_STEPS,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_strategy="epoch",
    report_to="none",          # Set to "wandb" or "tensorboard" for tracking

    warmup_ratio=0.03,
    # ── warmup_ratio ──────────────────────────────────────────────────────
    # Fraction of total training steps for the LR warmup phase.
    # Ramps LR from 0 to learning_rate over the first 3% of steps.
    # Prevents the model from taking destructive steps early in training.

    lr_scheduler_type="cosine",
    # ── lr_scheduler_type ─────────────────────────────────────────────────
    # LR schedule after warmup. "cosine" decays smoothly to near-zero.
    # Alternatives: "linear" (simpler), "constant" (no decay).

    fp16=False,                # Set True on CUDA GPU for 2× speedup
    bf16=False,                # Set True on Ampere+ GPU (A100, RTX 3090+)
    dataloader_pin_memory=False,
    dataset_text_field="text", # Column name from our formatted_dataset
)

print(f"\n  Training configuration:")
print(f"    Base model          : {BASE_MODEL}")
print(f"    Dataset             : {DATASET_NAME} ({len(formatted_dataset)} examples)")
print(f"    Epochs              : {NUM_TRAIN_EPOCHS}")
print(f"    Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"    Learning rate       : {LEARNING_RATE}")
print(f"    Max seq length      : {MAX_SEQ_LENGTH}")
print(f"    LoRA rank           : {LORA_R}")
print(f"    Device              : {'CUDA' if torch.cuda.is_available() else 'CPU'}")

total_steps = (len(formatted_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_TRAIN_EPOCHS
print(f"    Estimated steps     : ~{total_steps}")

print("\n  Starting training...\n")
t_train_start = time.time()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
)

train_result = trainer.train()

t_train_end = time.time()
train_duration = t_train_end - t_train_start

print(f"\n  Training complete in {train_duration:.0f}s ({train_duration/60:.1f} min)")
print(f"  Final training loss : {train_result.training_loss:.4f}")
print(f"  Total steps         : {train_result.global_step}")
print()


# ─────────────────────────────────────────────────────
# SECTION 8: INFERENCE — BEFORE vs. AFTER COMPARISON
# ─────────────────────────────────────────────────────
#
# We run the same prompts through the base model and the fine-tuned model to
# see whether instruction-following behaviour has changed.

print("=" * 65)
print("SECTION 8: Inference — Before vs. After Comparison")
print("=" * 65)


def generate_response(
    model,
    tokenizer,
    instruction: str,
    context: str = "",
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate a response from the model for a given instruction.

    Parameters
    ----------
    model : PreTrainedModel or PeftModel
        The language model to generate from.
    tokenizer : PreTrainedTokenizer
        Tokenizer matching the model.
    instruction : str
        The task instruction / question to send to the model.
    context : str  (default "")
        Optional supporting context. If empty, the context block is omitted.
    max_new_tokens : int  (default 150)
        Maximum number of tokens to generate.
        Higher values allow longer responses but increase latency.
        Practical range: 50–512.
    temperature : float  (default 0.7)
        Sampling temperature.
        • < 1.0 → more focused, less random
        • > 1.0 → more creative, less coherent
        • 0.0   → greedy decoding (always picks the highest-probability token)
        See Tutorial 01 for a detailed exploration of this parameter.
    do_sample : bool  (default True)
        If True, sample from the probability distribution.
        If False, use greedy decoding (ignores temperature).

    Returns
    -------
    str
        The model's generated response text (decoded, stripped).
    """
    if context:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# Load the base model (without LoRA) for comparison
print("\n  Loading base model for comparison...")
base_model_compare = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

# Test prompts — pick examples from categories the model should handle
test_prompts = [
    {
        "instruction": "Explain what a transformer neural network is in simple terms.",
        "context": "",
    },
    {
        "instruction": "What are three benefits of eating a plant-based diet?",
        "context": "",
    },
    {
        "instruction": "Classify the sentiment of the following sentence as positive, negative, or neutral.",
        "context": "The movie had some interesting ideas but the pacing was painfully slow.",
    },
]

print("\n" + "=" * 65)
print("  COMPARISON RESULTS")
print("=" * 65)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n  ── Prompt {i} ──────────────────────────────────────────")
    print(f"  Instruction : {prompt['instruction']}")
    if prompt["context"]:
        print(f"  Context     : {prompt['context']}")

    # Base model response
    base_response = generate_response(base_model_compare, tokenizer, **prompt)
    print(f"\n  BASE MODEL:")
    for line in textwrap.wrap(base_response or "(empty / no response)", width=60):
        print(f"    {line}")

    # Fine-tuned model response
    ft_response = generate_response(model, tokenizer, **prompt)
    print(f"\n  FINE-TUNED (LoRA):")
    for line in textwrap.wrap(ft_response or "(empty / no response)", width=60):
        print(f"    {line}")

del base_model_compare
print()


# ─────────────────────────────────────────────────────
# SECTION 9: SAVE ADAPTER & MERGE INTO BASE MODEL
# ─────────────────────────────────────────────────────
#
# Two save strategies:
#
#   A) Save adapter only (~10–50 MB)
#      Keeps LoRA weights separate from the base model.
#      Load by attaching the adapter back to the base model at runtime.
#      Useful when: you want to swap adapters, share without the base, or
#      fine-tune multiple adapters on one base.
#
#   B) Merge adapter into base model
#      Folds ΔW = B×A back into W permanently.
#      Produces a standard model with zero inference overhead.
#      Useful when: you're deploying and don't need adapter flexibility.

print("=" * 65)
print("SECTION 9: Save Adapter & Merge into Base Model")
print("=" * 65)

# ── Strategy A: Save adapter only ────────────────────
print(f"\n  Saving LoRA adapter to: {ADAPTER_DIR}")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

adapter_files = os.listdir(ADAPTER_DIR)
print(f"  Saved files: {adapter_files}")

# Show adapter file sizes
total_adapter_size = 0
for f in adapter_files:
    path = os.path.join(ADAPTER_DIR, f)
    size_mb = os.path.getsize(path) / 1e6
    total_adapter_size += size_mb
    print(f"    {f:<40} {size_mb:.2f} MB")
print(f"  Total adapter size: {total_adapter_size:.2f} MB")

# ── Demonstrate loading the adapter back ─────────────
print(f"\n  Demo: Loading adapter from disk onto fresh base model...")
base_for_reload = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
reloaded_model = PeftModel.from_pretrained(base_for_reload, ADAPTER_DIR)
print(f"  ✅ Adapter loaded. Type: {type(reloaded_model).__name__}")
del base_for_reload, reloaded_model

# ── Strategy B: Merge and save ────────────────────────
print(f"\n  Merging adapter into base weights...")
print(f"  (This folds ΔW = B×A into W for zero-overhead inference)")

merged_model = model.merge_and_unload()
# merge_and_unload():
#   1. Computes ΔW = B × A for every LoRA layer
#   2. Adds ΔW to the frozen W in-place
#   3. Removes the LoRA modules from the model
#   Returns a standard AutoModelForCausalLM with no PEFT wrapper.

merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

merged_size = sum(
    os.path.getsize(os.path.join(MERGED_DIR, f)) / 1e6
    for f in os.listdir(MERGED_DIR)
    if os.path.isfile(os.path.join(MERGED_DIR, f))
)
print(f"  ✅ Merged model saved to: {MERGED_DIR}")
print(f"  Merged model size: {merged_size:.0f} MB")
print(f"  (Compare to adapter: {total_adapter_size:.2f} MB — the adapter alone is much smaller)")

# Save a training manifest for reproducibility
manifest = {
    "base_model":           BASE_MODEL,
    "dataset":              DATASET_NAME,
    "max_train_samples":    MAX_TRAIN_SAMPLES,
    "lora_r":               LORA_R,
    "lora_alpha":           LORA_ALPHA,
    "lora_dropout":         LORA_DROPOUT,
    "target_modules":       TARGET_MODULES,
    "num_train_epochs":     NUM_TRAIN_EPOCHS,
    "batch_size":           PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation":GRADIENT_ACCUMULATION,
    "learning_rate":        LEARNING_RATE,
    "max_seq_length":       MAX_SEQ_LENGTH,
    "final_loss":           round(train_result.training_loss, 4),
    "total_steps":          train_result.global_step,
    "train_duration_sec":   round(train_duration, 1),
    "adapter_dir":          ADAPTER_DIR,
    "merged_dir":           MERGED_DIR,
}

manifest_path = os.path.join(OUTPUT_DIR, "training_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n  Training manifest saved to: {manifest_path}")
print()


# ─────────────────────────────────────────────────────
# SECTION 10: SUMMARY
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 10: Summary")
print("=" * 65)

print(f"""
  What ran:
    ✅ Loaded {BASE_MODEL} ({total_params/1e6:.0f}M parameters)
    ✅ Applied LoRA adapters to {TARGET_MODULES}
       → rank={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}
    ✅ Fine-tuned on {len(formatted_dataset)} Dolly-15k instruction examples
       → {NUM_TRAIN_EPOCHS} epochs, LR={LEARNING_RATE}
       → Final loss: {train_result.training_loss:.4f}
       → Duration: {train_duration:.0f}s ({train_duration/60:.1f} min)
    ✅ Ran before/after inference comparison
    ✅ Saved LoRA adapter to:   {ADAPTER_DIR}
    ✅ Saved merged model to:   {MERGED_DIR}
    ✅ Training manifest:       {manifest_path}

  Key concepts covered:
    • LoRA low-rank decomposition (ΔW = B × A)
    • PEFT get_peft_model — freezing base weights, injecting adapters
    • SFTTrainer — instruction dataset formatting, training loop
    • Adapter-only save vs. merge_and_unload
    • Before/after inference comparison

  To go further:
    • Swap BASE_MODEL for a larger model (opt-350m, Mistral-7B with GPU)
    • Set MAX_TRAIN_SAMPLES = None to train on the full 15k dataset
    • Try r=16 or target_modules += ["k_proj", "o_proj"] for more capacity
    • Enable fp16=True on a CUDA GPU for ~2× training speedup
    • See finetune_image.py for LoRA applied to diffusion models
    • See README.md for the full parameter guide and decision framework
""")

print("=" * 65)
print("  Tutorial 04 — finetune_llm.py complete.")
print("  Next: finetune_image.py  |  README.md")
print("=" * 65)
