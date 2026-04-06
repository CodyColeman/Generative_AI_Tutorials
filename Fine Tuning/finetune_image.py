# =============================================================================
# Tutorial 04 — Fine-Tuning Image Generation with Dreambooth + LoRA
# =============================================================================
#
# Environment:  Python 3.9+, CPU-possible but GPU strongly recommended
#
# Dependencies:
#   pip install diffusers peft accelerate transformers torch torchvision
#   pip install Pillow requests datasets
#
# Sections:
#   1. Imports & Configuration
#   2. Dreambooth Concept — What We're Actually Doing
#   3. Prepare Training Images (download sample concept images)
#   4. Generate Prior Preservation Images
#   5. Load Base Pipeline & Extract UNet / Text Encoder
#   6. Apply LoRA Adapters to UNet
#   7. Build Training Dataset & DataLoader
#   8. Training Loop
#   9. Inference — Trigger Word Exploration
#  10. Save LoRA Weights & Reload Demo
#  11. Summary
#
# GPU note:
#   CPU training is possible for the tiny demo pipeline used here but will
#   take a long time. For real Dreambooth training set USE_TINY_PIPELINE=False
#   and run on a machine with ≥16 GB VRAM (A100, RTX 3090, etc.).
#
# CPU fallback:
#   USE_TINY_PIPELINE=True uses "hf-internal-testing/tiny-stable-diffusion-pipe"
#   (same architecture as SD 1.5, only a few MB) so the full code path runs
#   on CPU in minutes. Outputs will look like noise — that's expected.
#   Set USE_TINY_PIPELINE=False to use runwayml/stable-diffusion-v1-5 (~4 GB).
# =============================================================================

# ─────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────

import os
import json
import math
import time
import shutil
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import requests
from io import BytesIO

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ── Pipeline choice ───────────────────────────────────
USE_TINY_PIPELINE = True
# True  → hf-internal-testing/tiny-stable-diffusion-pipe (CPU demo, tiny model)
# False → runwayml/stable-diffusion-v1-5 (real training, GPU required)

if USE_TINY_PIPELINE:
    MODEL_NAME = "hf-internal-testing/tiny-stable-diffusion-pipe"
else:
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# ── Concept / Dreambooth settings ─────────────────────
INSTANCE_PROMPT = "a sks dog"
# ── instance_prompt ───────────────────────────────────────────────────────
# The text prompt used for ALL training images.
# Format: "a [trigger_word] [class_noun]"
# "sks" is used as the trigger word — it's rare in pre-training data so
# it doesn't carry pre-existing meaning that would interfere with learning.
# At inference, include "sks" in prompts to invoke the learned concept.

CLASS_PROMPT = "a dog"
# ── class_prompt ──────────────────────────────────────────────────────────
# Describes the general class of your subject (without the trigger word).
# Used only when WITH_PRIOR_PRESERVATION=True to generate class images
# that prevent the model from forgetting what a generic dog looks like.

TRIGGER_WORD = "sks"

WITH_PRIOR_PRESERVATION = True
# ── with_prior_preservation ───────────────────────────────────────────────
# If True, co-trains on generated "class images" alongside your instance
# images. This prevents catastrophic forgetting of the general class.
# The prior preservation loss is:
#   total_loss = instance_loss + prior_loss_weight * class_loss
# Always enable for face/person training. Optional for styles/objects.

PRIOR_LOSS_WEIGHT = 1.0
# ── prior_loss_weight ─────────────────────────────────────────────────────
# Relative weight of the class preservation loss.
# 1.0 = equal weight to instance loss (standard).
# Increase if the model forgets the general class; decrease if it resists
# learning the specific instance.

NUM_CLASS_IMAGES = 10
# Number of class images to generate for prior preservation.
# Production runs use 100–200. We use 10 here for CPU speed.

# ── LoRA settings ─────────────────────────────────────
LORA_RANK = 8
# ── lora_rank ─────────────────────────────────────────────────────────────
# Rank of the LoRA decomposition applied to UNet attention layers.
# Same mechanics as finetune_llm.py but conventions differ for diffusion:
#   4  → subtle style shifts, low-detail subjects
#   8  → ✅ good general default
#   16 → detailed subjects (faces, pets with fine features)
#   32 → approaching full fine-tune; rarely necessary

LORA_ALPHA = 16   # Scaling: alpha / rank applied to adapter output

# ── Training settings ─────────────────────────────────
RESOLUTION = 64 if USE_TINY_PIPELINE else 512
# ── resolution ────────────────────────────────────────────────────────────
# Training and inference image resolution in pixels (square).
# SD 1.5 was trained at 512×512. Always match your training images to this.
# Tiny pipeline uses 64×64 to keep CPU runtime manageable.

NUM_TRAIN_STEPS = 50 if USE_TINY_PIPELINE else 400
# ── num_train_steps ───────────────────────────────────────────────────────
# Total gradient update steps (not epochs — image fine-tuning counts steps).
# Typical ranges:
#   50–100  : underfitting (concept barely learned)
#   200–400 : ✅ sweet spot for most subjects
#   800+    : overfitting (loses ability to generalise concept to new prompts)
# We use 50 for the tiny CPU demo.

TRAIN_BATCH_SIZE = 1
LEARNING_RATE = 1e-5 if not USE_TINY_PIPELINE else 1e-4
# ── learning_rate (image) ─────────────────────────────────────────────────
# Diffusion models are more sensitive to LR than LLMs.
# 1e-5  : ✅ standard for real LoRA training on SD 1.5
# 1e-4  : acceptable for tiny demo (model weights are random-scale anyway)
# 5e-6  : conservative — use if prior preservation fails

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1e-2
ADAM_EPSILON = 1e-8

# ── Paths ─────────────────────────────────────────────
OUTPUT_DIR        = "./outputs/finetune_image"
INSTANCE_DATA_DIR = os.path.join(OUTPUT_DIR, "instance_images")
CLASS_DATA_DIR    = os.path.join(OUTPUT_DIR, "class_images")
LORA_SAVE_DIR     = os.path.join(OUTPUT_DIR, "lora-weights")
SAMPLES_DIR       = os.path.join(OUTPUT_DIR, "inference_samples")

for d in [OUTPUT_DIR, INSTANCE_DATA_DIR, CLASS_DATA_DIR, LORA_SAVE_DIR, SAMPLES_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────
# SECTION 2: DREAMBOOTH CONCEPT — WHAT WE'RE ACTUALLY DOING
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 2: Dreambooth Concept")
print("=" * 65)

print("""
  Stable Diffusion learns to generate images by reversing a noise
  process. During training, it learns: given a noisy image and a
  text prompt, predict what noise was added.

  The UNet sits at the heart of this: it takes a noisy latent + text
  embeddings and predicts the noise. Fine-tuning the UNet teaches it
  to associate our trigger word with our specific subject.

  Standard SD pipeline:
  ┌────────────────────────────────────────────────────┐
  │  Text prompt                                       │
  │      │                                             │
  │      ▼                                             │
  │  CLIP Text Encoder → text embeddings               │
  │                            │                       │
  │  Random noise latent ──────┤                       │
  │                            ▼                       │
  │                        UNet  ← we fine-tune this   │
  │                            │                       │
  │                            ▼                       │
  │                    Denoised latent                 │
  │                            │                       │
  │                            ▼                       │
  │                    VAE Decoder → image             │
  └────────────────────────────────────────────────────┘

  Dreambooth training objective:
    For each training image x:
      1. Encode x to latent z via VAE encoder
      2. Add random noise ε at timestep t  →  z_t = z + ε
      3. Run UNet(z_t, t, text_embeddings("a sks dog"))
      4. Loss = MSE(predicted_noise, actual_noise_ε)
      5. Backprop only through LoRA adapter matrices

  With prior preservation:
    total_loss = instance_loss
               + prior_loss_weight × MSE(UNet(z_t, t, "a dog"),  ε)
                                                            ↑
                                              class images keep the
                                              model from forgetting
                                              what generic dogs look like
""")


# ─────────────────────────────────────────────────────
# SECTION 3: PREPARE TRAINING IMAGES
# ─────────────────────────────────────────────────────
#
# For a real Dreambooth run you'd supply 10–20 photos of your subject.
# Here we download a small set of dog images from a public HuggingFace
# dataset to serve as our "instance images".

print("=" * 65)
print("SECTION 3: Prepare Training Images")
print("=" * 65)


@dataclass
class ConceptImage:
    """
    Represents a single training image for Dreambooth.

    Attributes
    ----------
    path   : str   Absolute path to the image file on disk.
    prompt : str   The text prompt associated with this image.
                   All instance images share INSTANCE_PROMPT.
                   Class images share CLASS_PROMPT.
    """
    path: str
    prompt: str


def download_instance_images(save_dir: str, num_images: int = 8) -> list[ConceptImage]:
    """
    Download sample dog images from the HuggingFace `beans` dataset as
    stand-in instance images.

    In a real Dreambooth workflow you would replace this function with
    your own images of the subject you want to teach the model.

    Parameters
    ----------
    save_dir   : str   Directory to save downloaded images.
    num_images : int   How many images to download (default 8).
                       Production runs use 10–20 images.
                       More is not always better — diversity matters more
                       than quantity. 5 varied images > 20 near-identical.

    Returns
    -------
    list[ConceptImage]
        List of ConceptImage objects pointing to the saved files.
    """
    print(f"\n  Downloading {num_images} sample images from HuggingFace...")

    # We use the Stanford Dogs subset of Oxford Pets via HuggingFace
    # as a convenient source of real dog photos.
    try:
        ds = load_dataset("carmonalazaro/wikiart", split="train", streaming=True)
        # Fallback: if wikiart isn't available, use CIFAR-10 dog class
    except Exception:
        ds = None

    concept_images = []

    if ds is None:
        # Hard fallback: generate solid-colour placeholder images
        print("  (Using placeholder images — install datasets for real photos)")
        colours = [
            (180, 120, 80), (160, 100, 60), (200, 140, 90),
            (170, 110, 70), (190, 130, 85), (155, 95, 55),
            (175, 115, 75), (185, 125, 80),
        ]
        for i, colour in enumerate(colours[:num_images]):
            img = Image.new("RGB", (RESOLUTION, RESOLUTION), colour)
            path = os.path.join(save_dir, f"instance_{i:03d}.png")
            img.save(path)
            concept_images.append(ConceptImage(path=path, prompt=INSTANCE_PROMPT))
        print(f"  Created {len(concept_images)} placeholder images.")
        return concept_images

    # Stream images from dataset until we have enough
    count = 0
    for item in ds:
        if count >= num_images:
            break
        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            path = os.path.join(save_dir, f"instance_{count:03d}.png")
            img.save(path)
            concept_images.append(ConceptImage(path=path, prompt=INSTANCE_PROMPT))
            count += 1
        except Exception:
            continue

    if not concept_images:
        # Final fallback if streaming fails
        for i in range(num_images):
            img = Image.new("RGB", (RESOLUTION, RESOLUTION), (160 + i * 5, 100, 70))
            path = os.path.join(save_dir, f"instance_{i:03d}.png")
            img.save(path)
            concept_images.append(ConceptImage(path=path, prompt=INSTANCE_PROMPT))

    print(f"  ✅ {len(concept_images)} instance images ready in: {save_dir}")
    for ci in concept_images:
        size_kb = os.path.getsize(ci.path) / 1024
        print(f"    {os.path.basename(ci.path)}  ({size_kb:.1f} KB)  prompt: '{ci.prompt}'")
    return concept_images


instance_images = download_instance_images(INSTANCE_DATA_DIR, num_images=8)
print()


# ─────────────────────────────────────────────────────
# SECTION 4: GENERATE PRIOR PRESERVATION IMAGES
# ─────────────────────────────────────────────────────
#
# Prior preservation images are generated by the *base* model (before
# any fine-tuning) using the class prompt ("a dog"). These are used as
# a regularisation signal during training to prevent the model from
# overwriting its general knowledge of the class.
#
# In production: generate 100–200 class images.
# Here: NUM_CLASS_IMAGES (default 10) to keep CPU time manageable.

print("=" * 65)
print("SECTION 4: Generate Prior Preservation Images")
print("=" * 65)

class_images: list[ConceptImage] = []

if WITH_PRIOR_PRESERVATION:
    existing_class = list(Path(CLASS_DATA_DIR).glob("*.png"))

    if len(existing_class) >= NUM_CLASS_IMAGES:
        print(f"\n  Found {len(existing_class)} existing class images — skipping generation.")
        class_images = [
            ConceptImage(path=str(p), prompt=CLASS_PROMPT)
            for p in existing_class[:NUM_CLASS_IMAGES]
        ]
    else:
        print(f"\n  Generating {NUM_CLASS_IMAGES} class images with prompt: '{CLASS_PROMPT}'")
        print(f"  Using base model: {MODEL_NAME}")
        print("  (These preserve the model's knowledge of the general class)\n")

        print("  Loading base pipeline for class image generation...")
        prior_pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        prior_pipeline = prior_pipeline.to(DEVICE)
        prior_pipeline.set_progress_bar_config(disable=True)

        for i in range(NUM_CLASS_IMAGES):
            result = prior_pipeline(
                CLASS_PROMPT,
                num_inference_steps=20 if USE_TINY_PIPELINE else 30,
                guidance_scale=7.5,
                height=RESOLUTION,
                width=RESOLUTION,
            )
            img = result.images[0]
            path = os.path.join(CLASS_DATA_DIR, f"class_{i:03d}.png")
            img.save(path)
            class_images.append(ConceptImage(path=path, prompt=CLASS_PROMPT))
            print(f"  Generated class image {i+1}/{NUM_CLASS_IMAGES}: {os.path.basename(path)}")

        # Free memory before training
        del prior_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  ✅ {len(class_images)} class images saved to: {CLASS_DATA_DIR}")
else:
    print("\n  Prior preservation disabled (WITH_PRIOR_PRESERVATION=False).")
    print("  ⚠️  Risk: model may forget general class knowledge during training.")

print()


# ─────────────────────────────────────────────────────
# SECTION 5: LOAD BASE PIPELINE COMPONENTS
# ─────────────────────────────────────────────────────
#
# For training we need direct access to the individual components of the
# diffusion pipeline rather than the high-level StableDiffusionPipeline.
# We load each component separately so we can:
#   - Freeze the VAE (we never want to train it)
#   - Freeze or fine-tune the text encoder
#   - Apply LoRA only to the UNet

print("=" * 65)
print("SECTION 5: Load Base Pipeline Components")
print("=" * 65)

print(f"\n  Loading components from: {MODEL_NAME}")

# Noise scheduler — defines the forward (noise-adding) process.
# DDPMScheduler is the training scheduler; at inference a faster
# scheduler (DDIM, DPM++) is used instead.
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
print(f"  ✅ Noise scheduler: {type(noise_scheduler).__name__}")
print(f"     Timesteps: {noise_scheduler.config.num_train_timesteps}")
print(f"     Beta schedule: {noise_scheduler.config.beta_schedule}")

# CLIP tokenizer + text encoder — converts prompt text to embeddings
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
text_encoder = text_encoder.to(DEVICE)
print(f"  ✅ Text encoder: {type(text_encoder).__name__}")

# VAE — encodes images to latent space and decodes latents to images.
# We freeze the VAE entirely — no LoRA here, just use it for encoding.
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
vae = vae.to(DEVICE)
vae.requires_grad_(False)   # ← VAE is always frozen
print(f"  ✅ VAE: {type(vae).__name__} (frozen)")

# UNet — this is what we fine-tune with LoRA
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
unet = unet.to(DEVICE)
print(f"  ✅ UNet: {type(unet).__name__} — this gets the LoRA adapters")

# Freeze text encoder — we're only training the UNet LoRA
# (For style LoRAs it can help to also fine-tune the text encoder,
#  but UNet-only is the standard Dreambooth approach)
text_encoder.requires_grad_(False)
print(f"\n  Parameter freeze status:")
print(f"    VAE          : fully frozen ✅")
print(f"    Text encoder : fully frozen ✅")
print(f"    UNet         : will receive LoRA adapters →")
print()


# ─────────────────────────────────────────────────────
# SECTION 6: APPLY LoRA ADAPTERS TO UNET
# ─────────────────────────────────────────────────────
#
# We attach LoRA to the UNet's attention projection layers.
# The UNet attention uses the same q/k/v/out_proj structure as a
# language model transformer, so the same LoRA logic applies.

print("=" * 65)
print("SECTION 6: Apply LoRA Adapters to UNet")
print("=" * 65)

unet_lora_config = LoraConfig(
    r=LORA_RANK,
    # ── r (rank) ───────────────────────────────────────────────────────
    # Rank of LoRA decomposition for UNet attention matrices.
    # 8 is a good general default. Use 16 for detailed subjects (faces).

    lora_alpha=LORA_ALPHA,
    # ── lora_alpha ─────────────────────────────────────────────────────
    # Scale = alpha / r. Convention: set alpha = 2 × r.

    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    # ── target_modules (UNet) ──────────────────────────────────────────
    # UNet attention projections. Diffusers names them differently to
    # transformer LLMs:
    #   to_q    : query projection
    #   to_k    : key projection
    #   to_v    : value projection
    #   to_out.0: output projection
    # Targeting all four is standard for Dreambooth LoRA.
    # For lighter-weight adapters, drop to_k.

    lora_dropout=0.0,
    # ── lora_dropout (image) ───────────────────────────────────────────
    # Typically 0 for image fine-tuning — the dataset is already small
    # (10–20 images) and dropout can destabilise diffusion training.
    # Use 0.05 only if you have 50+ training images.

    bias="none",
    init_lora_weights="gaussian",
    # ── init_lora_weights ──────────────────────────────────────────────
    # How to initialise the A matrix.
    # "gaussian"  : A ~ N(0, 1/r). Slightly better for diffusion models.
    # True (default): A ~ N(0, 1). Standard.
    # Both initialise B = 0, so ΔW = B×A = 0 at the start of training —
    # training begins from an exact copy of the base model.
)

unet = get_peft_model(unet, unet_lora_config)

print("\n  UNet LoRA applied:")
unet.print_trainable_parameters()

# Collect only the trainable LoRA parameters for the optimiser
lora_params = [p for p in unet.parameters() if p.requires_grad]
total_lora = sum(p.numel() for p in lora_params)
print(f"\n  LoRA parameter tensors to optimise: {len(lora_params)}")
print(f"  Total trainable floats             : {total_lora:,}")
print()


# ─────────────────────────────────────────────────────
# SECTION 7: BUILD TRAINING DATASET & DATALOADER
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 7: Build Training Dataset & DataLoader")
print("=" * 65)


class DreamboothDataset(Dataset):
    """
    PyTorch Dataset for Dreambooth training.

    Yields batches containing both instance images (the subject we're
    teaching) and class images (prior preservation regularisation).
    If prior preservation is disabled, only instance images are returned.

    Parameters
    ----------
    instance_images : list[ConceptImage]
        Training images of the subject to learn, each tagged with
        the instance prompt ("a sks dog").
    class_images : list[ConceptImage]
        Images of the general class, generated by the base model,
        each tagged with the class prompt ("a dog").
        May be an empty list if prior preservation is disabled.
    tokenizer : CLIPTokenizer
        Tokenizer used to convert prompts to token IDs.
    resolution : int
        Target image resolution. Images are resized and centre-cropped
        to (resolution × resolution) squares.
    """

    def __init__(
        self,
        instance_images: list,
        class_images: list,
        tokenizer,
        resolution: int = 512,
    ):
        self.instance_images = instance_images
        self.class_images = class_images
        self.tokenizer = tokenizer
        self.resolution = resolution

        # Image preprocessing pipeline
        # 1. Resize to resolution (keeps aspect ratio, then crops)
        # 2. CenterCrop to exact square
        # 3. Convert to tensor in [0, 1]
        # 4. Normalise to [-1, 1] (SD 1.5 expects this range)
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # [0,1] → [-1,1]
        ])

    def __len__(self) -> int:
        # Dataset length = number of instance images.
        # Class images are accessed cyclically (via index % len).
        return len(self.instance_images)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt string to a fixed-length token id tensor."""
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, idx: int) -> dict:
        instance = self.instance_images[idx]
        instance_img = Image.open(instance.path).convert("RGB")

        batch = {
            "instance_pixel_values": self.image_transforms(instance_img),
            "instance_prompt_ids":   self._tokenize(instance.prompt),
        }

        if self.class_images:
            # Cycle through class images regardless of instance count
            class_idx = idx % len(self.class_images)
            cls = self.class_images[class_idx]
            class_img = Image.open(cls.path).convert("RGB")
            batch["class_pixel_values"] = self.image_transforms(class_img)
            batch["class_prompt_ids"]   = self._tokenize(cls.prompt)

        return batch


def collate_fn(examples: list[dict]) -> dict:
    """
    Custom collate function that stacks instance and class samples into
    a single batch tensor.

    When prior preservation is active, we concatenate instance and class
    images along the batch dimension before passing to the UNet. The loss
    is then computed over the full concatenated batch and split at the midpoint.
    """
    has_class = "class_pixel_values" in examples[0]

    pixel_values = torch.stack([e["instance_pixel_values"] for e in examples])
    prompt_ids   = torch.stack([e["instance_prompt_ids"]   for e in examples])

    if has_class:
        class_pixel_values = torch.stack([e["class_pixel_values"] for e in examples])
        class_prompt_ids   = torch.stack([e["class_prompt_ids"]   for e in examples])
        # Concatenate: [instance_batch | class_batch]
        pixel_values = torch.cat([pixel_values, class_pixel_values], dim=0)
        prompt_ids   = torch.cat([prompt_ids,   class_prompt_ids],   dim=0)

    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "input_ids":    prompt_ids,
    }


train_dataset = DreamboothDataset(
    instance_images=instance_images,
    class_images=class_images,
    tokenizer=tokenizer,
    resolution=RESOLUTION,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,   # 0 for Windows / CPU compatibility
)

n_instance = len(instance_images)
n_class    = len(class_images)
print(f"\n  Dataset stats:")
print(f"    Instance images  : {n_instance}")
print(f"    Class images     : {n_class}  ({'active' if WITH_PRIOR_PRESERVATION else 'disabled'})")
print(f"    Batch size       : {TRAIN_BATCH_SIZE}")
print(f"    Effective batch  : {TRAIN_BATCH_SIZE * (2 if class_images else 1)} "
      f"({'instance + class' if class_images else 'instance only'})")
print(f"    Training steps   : {NUM_TRAIN_STEPS}")
print()


# ─────────────────────────────────────────────────────
# SECTION 8: TRAINING LOOP
# ─────────────────────────────────────────────────────
#
# The diffusion training loop:
#   For each batch:
#     1.  Encode pixel values to latents via VAE encoder
#     2.  Sample random timesteps t ~ Uniform(0, T)
#     3.  Add Gaussian noise to latents at timestep t → noisy_latents
#     4.  Encode text prompts to embeddings via CLIP text encoder
#     5.  Predict the noise: UNet(noisy_latents, t, text_embeddings)
#     6.  Compute MSE loss between predicted noise and actual noise
#     7.  If prior preservation: split batch, add weighted class loss
#     8.  Backprop & update LoRA parameters

print("=" * 65)
print("SECTION 8: Training Loop")
print("=" * 65)

optimiser = torch.optim.AdamW(
    lora_params,
    lr=LEARNING_RATE,
    betas=(ADAM_BETA1, ADAM_BETA2),
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=ADAM_EPSILON,
)

# LR scheduler: linear warmup for first 5% of steps, then constant
num_warmup = max(1, int(0.05 * NUM_TRAIN_STEPS))
scheduler  = torch.optim.lr_scheduler.LinearLR(
    optimiser,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=num_warmup,
)

unet.train()
text_encoder.eval()
vae.eval()

loss_history = []
t_train_start = time.time()

print(f"\n  Starting training for {NUM_TRAIN_STEPS} steps...")
print(f"  Instance prompt : '{INSTANCE_PROMPT}'")
print(f"  Class prompt    : '{CLASS_PROMPT}' (prior preservation: {WITH_PRIOR_PRESERVATION})")
print(f"  Resolution      : {RESOLUTION}×{RESOLUTION}")
print(f"  Learning rate   : {LEARNING_RATE}")
print(f"  LoRA rank       : {LORA_RANK}\n")

data_iter    = itertools.cycle(train_dataloader)
global_step  = 0
LOG_EVERY    = max(1, NUM_TRAIN_STEPS // 10)

while global_step < NUM_TRAIN_STEPS:
    batch = next(data_iter)

    pixel_values = batch["pixel_values"].to(DEVICE)
    input_ids    = batch["input_ids"].to(DEVICE)

    # ── Step 1: Encode images to latent space ──────────
    with torch.no_grad():
        # VAE encoder compresses images from pixel space (B, 3, H, W)
        # to latent space (B, 4, H/8, W/8). The 0.18215 scaling factor
        # is the SD 1.5 latent normalisation constant.
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # ── Step 2: Sample random timesteps ────────────────
    # Each image gets a different random timestep. The model learns
    # to denoise at all noise levels simultaneously.
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=DEVICE,
    ).long()

    # ── Step 3: Add noise to latents ───────────────────
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # ── Step 4: Encode text prompts ────────────────────
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]

    # ── Step 5: Predict noise with UNet ────────────────
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # ── Step 6: Compute loss ───────────────────────────
    if WITH_PRIOR_PRESERVATION and class_images:
        # Batch is [instance | class] concatenated. Split at midpoint.
        mid = bsz // 2
        model_pred_instance, model_pred_class = model_pred[:mid], model_pred[mid:]
        noise_instance, noise_class           = noise[:mid], noise[mid:]

        instance_loss = F.mse_loss(model_pred_instance.float(), noise_instance.float())
        class_loss    = F.mse_loss(model_pred_class.float(), noise_class.float())
        loss          = instance_loss + PRIOR_LOSS_WEIGHT * class_loss
    else:
        loss = F.mse_loss(model_pred.float(), noise.float())

    # ── Step 7: Backprop & update ──────────────────────
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
    optimiser.step()
    scheduler.step()

    loss_history.append(loss.item())
    global_step += 1

    if global_step % LOG_EVERY == 0 or global_step == 1:
        avg_loss  = sum(loss_history[-LOG_EVERY:]) / len(loss_history[-LOG_EVERY:])
        elapsed   = time.time() - t_train_start
        steps_rem = NUM_TRAIN_STEPS - global_step
        eta       = (elapsed / global_step) * steps_rem if global_step > 0 else 0
        lr_now    = optimiser.param_groups[0]["lr"]
        print(f"  Step {global_step:>4}/{NUM_TRAIN_STEPS}  "
              f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

t_train_end = time.time()
train_duration = t_train_end - t_train_start
final_loss = sum(loss_history[-10:]) / min(10, len(loss_history))

print(f"\n  ✅ Training complete!")
print(f"  Duration      : {train_duration:.0f}s ({train_duration/60:.1f} min)")
print(f"  Final loss    : {final_loss:.4f}")
print(f"  Total steps   : {global_step}")
print()


# ─────────────────────────────────────────────────────
# SECTION 9: INFERENCE — TRIGGER WORD EXPLORATION
# ─────────────────────────────────────────────────────
#
# We reassemble the fine-tuned UNet back into a full pipeline and
# run several prompts to demonstrate the trigger word effect.
# Each prompt variation tests a different aspect of what Dreambooth learned:
#   - Does the trigger word invoke the specific subject?
#   - Does the model generalise to novel contexts?
#   - Does removing the trigger word revert to generic generation?

print("=" * 65)
print("SECTION 9: Inference — Trigger Word Exploration")
print("=" * 65)

print("\n  Assembling fine-tuned pipeline...")

# Load the full pipeline and swap in our fine-tuned UNet
inference_pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    safety_checker=None,
)
inference_pipeline.unet = unet   # Replace with our LoRA-adapted UNet
inference_pipeline = inference_pipeline.to(DEVICE)
inference_pipeline.set_progress_bar_config(disable=True)

# Test prompts — each pair shows with vs. without trigger word
test_prompts = [
    # (prompt, description)
    (f"a photo of {TRIGGER_WORD} dog",            "Baseline — trigger word, no context"),
    (f"a photo of a dog",                          "Baseline — no trigger word (should be generic)"),
    (f"a {TRIGGER_WORD} dog on the moon",          "Trigger word + novel context"),
    (f"a {TRIGGER_WORD} dog wearing a party hat",  "Trigger word + accessory"),
    (f"oil painting of a {TRIGGER_WORD} dog",      "Trigger word + style transfer"),
]

NUM_INFERENCE_STEPS = 20 if USE_TINY_PIPELINE else 30
GUIDANCE_SCALE = 7.5

print(f"\n  Generating inference samples...")
print(f"  Steps: {NUM_INFERENCE_STEPS}  |  CFG scale: {GUIDANCE_SCALE}")
print(f"  Output directory: {SAMPLES_DIR}\n")

inference_log = []

for i, (prompt, description) in enumerate(test_prompts):
    print(f"  [{i+1}/{len(test_prompts)}] {description}")
    print(f"         Prompt: '{prompt}'")

    result = inference_pipeline(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=RESOLUTION,
        width=RESOLUTION,
        generator=torch.manual_seed(42 + i),   # fixed seed for reproducibility
    )

    img = result.images[0]
    filename = f"sample_{i:02d}_{TRIGGER_WORD}_{'with' if TRIGGER_WORD in prompt else 'without'}.png"
    path = os.path.join(SAMPLES_DIR, filename)
    img.save(path)

    inference_log.append({
        "prompt":      prompt,
        "description": description,
        "filename":    filename,
        "trigger_word_present": TRIGGER_WORD in prompt,
        "steps":       NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed":        42 + i,
    })
    print(f"         Saved: {filename}\n")

print(f"  ✅ {len(test_prompts)} inference images saved to: {SAMPLES_DIR}")

# Notes on what to look for in the outputs
print("""
  What to look for in the outputs:
  ─────────────────────────────────
  ✅ Good signs:
     • Prompts WITH "sks" generate images that resemble your training subject
     • Prompts WITHOUT "sks" generate a different, generic dog
     • Novel-context prompts (moon, party hat) still capture the subject identity

  ⚠️ Signs of problems:
     • All prompts look identical → trigger word bled into base class weights
       → Fix: enable prior preservation, reduce learning rate
     • Trigger prompts look identical to non-trigger → concept not learned
       → Fix: more training steps, higher LoRA rank
     • Highly distorted / incoherent outputs → overfitting
       → Fix: fewer training steps, lower LR
""")


# ─────────────────────────────────────────────────────
# SECTION 10: SAVE LoRA WEIGHTS & RELOAD DEMO
# ─────────────────────────────────────────────────────
#
# Diffusion LoRA adapters are conventionally saved as .safetensors files.
# The standard format is compatible with:
#   • ComfyUI  — drag-and-drop into the LoRA loader node
#   • Automatic1111 (A1111) — place in models/Lora/
#   • Kohya_ss — native format
#   • HuggingFace Hub — uploadable directly

print("=" * 65)
print("SECTION 10: Save LoRA Weights & Reload Demo")
print("=" * 65)

# ── Save ─────────────────────────────────────────────
print(f"\n  Saving LoRA weights to: {LORA_SAVE_DIR}")
unet.save_pretrained(LORA_SAVE_DIR)

lora_files = [f for f in os.listdir(LORA_SAVE_DIR) if os.path.isfile(os.path.join(LORA_SAVE_DIR, f))]
total_lora_size = sum(
    os.path.getsize(os.path.join(LORA_SAVE_DIR, f)) / 1e6
    for f in lora_files
)
print(f"  Saved files:")
for f in lora_files:
    size_mb = os.path.getsize(os.path.join(LORA_SAVE_DIR, f)) / 1e6
    print(f"    {f:<40} {size_mb:.2f} MB")
print(f"  Total size: {total_lora_size:.2f} MB")

# ── Demonstrate loading ───────────────────────────────
print(f"\n  Demo: Loading LoRA adapter onto a fresh pipeline...")
print(f"  (Shows that the adapter file is fully self-contained)")

fresh_pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    safety_checker=None,
)
fresh_pipeline = fresh_pipeline.to(DEVICE)

# Load LoRA weights into the fresh pipeline's UNet
# In a production workflow with diffusers >= 0.21 you can also use:
#   fresh_pipeline.load_lora_weights(LORA_SAVE_DIR)
#   fresh_pipeline.set_adapters(["default"], adapter_weights=[0.8])
fresh_unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
from peft import PeftModel
fresh_unet_peft = PeftModel.from_pretrained(fresh_unet, LORA_SAVE_DIR)
fresh_pipeline.unet = fresh_unet_peft.to(DEVICE)

print(f"  ✅ LoRA adapter loaded successfully.")
print(f"     Pipeline type: {type(fresh_pipeline).__name__}")
print(f"     UNet type    : {type(fresh_pipeline.unet).__name__}")

# ── Adapter strength control ──────────────────────────
print(f"""
  Adapter strength at inference:
  ───────────────────────────────
  You can scale the LoRA contribution without retraining:

    # Load with PEFT then scale manually
    # (diffusers >= 0.21 native API):
    pipeline.load_lora_weights("{LORA_SAVE_DIR}")
    pipeline.set_adapters(["default"], adapter_weights=[0.8])
    # 0.0 = base model, 1.0 = full LoRA, 0.5–0.8 = blended

  This is useful for:
    • Dialling in the right concept strength without retraining
    • Blending multiple LoRA adapters (style + subject)
    • A/B testing at different LoRA intensities
""")

del fresh_pipeline, fresh_unet_peft

# ── Save training manifest ────────────────────────────
manifest = {
    "model_name":           MODEL_NAME,
    "use_tiny_pipeline":    USE_TINY_PIPELINE,
    "instance_prompt":      INSTANCE_PROMPT,
    "class_prompt":         CLASS_PROMPT,
    "trigger_word":         TRIGGER_WORD,
    "with_prior_preservation": WITH_PRIOR_PRESERVATION,
    "prior_loss_weight":    PRIOR_LOSS_WEIGHT,
    "num_class_images":     NUM_CLASS_IMAGES,
    "lora_rank":            LORA_RANK,
    "lora_alpha":           LORA_ALPHA,
    "resolution":           RESOLUTION,
    "num_train_steps":      NUM_TRAIN_STEPS,
    "learning_rate":        LEARNING_RATE,
    "batch_size":           TRAIN_BATCH_SIZE,
    "num_instance_images":  len(instance_images),
    "final_loss":           round(final_loss, 4),
    "train_duration_sec":   round(train_duration, 1),
    "lora_save_dir":        LORA_SAVE_DIR,
    "samples_dir":          SAMPLES_DIR,
    "inference_log":        inference_log,
}

manifest_path = os.path.join(OUTPUT_DIR, "training_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"  Training manifest saved: {manifest_path}")
print()


# ─────────────────────────────────────────────────────
# SECTION 11: SUMMARY
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 11: Summary")
print("=" * 65)

print(f"""
  What ran:
    ✅ Downloaded {len(instance_images)} instance images (training subject)
    ✅ Generated {len(class_images)} class images for prior preservation
    ✅ Loaded {MODEL_NAME} pipeline components:
       VAE (frozen) + Text encoder (frozen) + UNet (LoRA)
    ✅ Applied LoRA to UNet attention layers
       → rank={LORA_RANK}, alpha={LORA_ALPHA}, target: to_q/to_k/to_v/to_out
    ✅ Trained for {NUM_TRAIN_STEPS} steps
       → prior_loss_weight={PRIOR_LOSS_WEIGHT if WITH_PRIOR_PRESERVATION else "N/A"}
       → Final loss: {final_loss:.4f}
       → Duration: {train_duration:.0f}s ({train_duration/60:.1f} min)
    ✅ Generated {len(test_prompts)} inference samples to: {SAMPLES_DIR}
    ✅ Saved LoRA adapter to: {LORA_SAVE_DIR}
    ✅ Training manifest: {manifest_path}

  Key concepts covered:
    • Diffusion model training objective (noise prediction / MSE)
    • Dreambooth trigger word binding
    • Prior preservation loss (catastrophic forgetting prevention)
    • UNet LoRA target modules (to_q / to_k / to_v / to_out)
    • VAE latent encoding in the training loop
    • Adapter strength control at inference

  To go further:
    • Set USE_TINY_PIPELINE=False for real SD 1.5 training (GPU required)
    • Replace download_instance_images() with your own subject photos
    • Increase NUM_TRAIN_STEPS to 400 for a fuller concept bake-in
    • Try LORA_RANK=16 for subjects with fine details (faces, pets)
    • Experiment with adapter_weights at inference (0.5–0.9 range)
    • See finetune_llm.py for LoRA applied to language models
    • See README.md for the full parameter guide and decision framework
""")

print("=" * 65)
print("  Tutorial 04 — finetune_image.py complete.")
print("  Next: 05_embeddings  |  README.md")
print("=" * 65)
