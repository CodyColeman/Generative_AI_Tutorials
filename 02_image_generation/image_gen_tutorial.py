# =============================================================================
# GenAI Tutorial 02: Image Generation — Text-to-Image with Diffusion Models
# =============================================================================
# Environment:  CPU-friendly (GPU strongly recommended for real use)
# Dependencies: pip install diffusers transformers torch accelerate Pillow
#               pip install invisible-watermark  # optional, for SD XL
#
# This tutorial walks through:
#   1. How diffusion models work (concept)
#   2. Loading a pipeline with HuggingFace Diffusers
#   3. Core generation parameters (steps, guidance, seed, size)
#   4. Prompt engineering for images
#   5. Negative prompts — what to EXCLUDE from the image
#   6. Image-to-image generation (img2img)
#   7. Inpainting — editing a specific region of an image
#   8. Saving outputs and building a prompt experiment log
#   9. Exploring different models & art styles (guide)
# =============================================================================

import torch
import os
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: HOW DIFFUSION MODELS WORK
# ─────────────────────────────────────────────────────────────────────────────
#
# Diffusion models learn to REVERSE a noise process.
#
# Training phase:
#   Take a real image → gradually add Gaussian noise over T steps
#   until it's pure noise → train the model to predict what noise
#   was added at each step (so it can undo it).
#
# Generation phase (what we use):
#   Start with pure random noise → run the model T times in reverse,
#   each step removing a little noise → eventually a clean image emerges.
#
#   Step 0:  ████████ (pure noise)
#   Step 10: ▓▒░░▓▓░▒ (rough shapes forming)
#   Step 20: ▒░  ░▒▒░ (structure emerging)
#   Step 50: 🖼️        (final image)
#
# The TEXT PROMPT is injected at every denoising step via cross-attention,
# steering the noise removal toward the described concept.
#
# Key papers: DDPM (Ho et al. 2020), Stable Diffusion (Rombach et al. 2022)

print("=" * 65)
print("GenAI Tutorial 02: Image Generation with Diffusion Models")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MODEL SELECTION & PIPELINE LOADING
# ─────────────────────────────────────────────────────────────────────────────
#
# We use runwayml/stable-diffusion-v1-5 — the most widely used open SD model.
# At ~4GB it's large but runs on CPU (slowly) and GPU (fast).
#
# For a faster CPU demo, we use a highly compressed/distilled variant.
# See Section 9 and README.md for a full model catalogue.
#
# CPU vs GPU note:
#   CPU:  ~5–15 min per image at 512x512, 20 steps — great for learning
#   GPU:  ~3–8 sec per image — required for real workflows
#
# To use GPU (if available), change device to "cuda":
#   pipe = pipe.to("cuda")
#
# The diffusers library abstracts away the full pipeline:
#   Text Encoder (CLIP) → UNet (denoiser) → VAE (decoder) → PIL Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
)

# Output directory for generated images
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_DIR / "generation_log.json"

# ── Detect available device ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"\nDevice detected: {device.upper()}")
if device == "cpu":
    print("⚠️  Running on CPU — generation will be slow (~5-15 min/image).")
    print("   For real workflows, a GPU is strongly recommended.\n")
else:
    print(f"   GPU available — fast generation enabled.\n")

# ── Model choice ──────────────────────────────────────────────────────────
#
# For CPU: use a lightweight model
# For GPU: use full SD 1.5 or SD XL for higher quality
#
# CPU-friendly option (much smaller/faster):
CPU_MODEL  = "hf-internal-testing/tiny-stable-diffusion-pipe"  # ~50MB, for testing
# Full quality options (swap these in when you have GPU or patience):
SD15_MODEL = "runwayml/stable-diffusion-v1-5"                  # ~4GB, great quality
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"        # ~7GB, best quality

# We default to the tiny model for CPU demo — swap to SD15_MODEL for real results
USE_MODEL = CPU_MODEL if device == "cpu" else SD15_MODEL

print(f"Loading model: {USE_MODEL}")
print("(First run downloads model weights — may take a few minutes)\n")


def load_txt2img_pipeline(model_id: str = USE_MODEL) -> StableDiffusionPipeline:
    """
    Load a text-to-image Stable Diffusion pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.
        See README.md Section "Model Catalogue" for options.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,    # disable NSFW filter for tutorial (re-enable in production)
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Memory optimization — enable if running out of VRAM on GPU
    # pipe.enable_attention_slicing()       # reduces VRAM ~30%, slight speed cost
    # pipe.enable_xformers_memory_efficient_attention()  # requires xformers package

    return pipe


pipe = load_txt2img_pipeline()
print("Pipeline loaded.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CORE GENERATION — TEXT TO IMAGE
# ─────────────────────────────────────────────────────────────────────────────

# Experiment log — records every generation for reproducibility
generation_log = []

def save_log():
    with open(LOG_FILE, "w") as f:
        json.dump(generation_log, f, indent=2)


def generate_image(
    prompt: str,
    negative_prompt: str       = "",
    num_inference_steps: int   = 20,
    guidance_scale: float      = 7.5,
    width: int                 = 512,
    height: int                = 512,
    seed: int                  = None,
    num_images: int            = 1,
    filename_prefix: str       = "gen",
) -> list[Image.Image]:
    """
    Generate images from a text prompt.

    Parameters
    ----------
    prompt : str
        The text description of the image you want.
        See Section 4 for prompt engineering tips.

    negative_prompt : str  (default "")
        Text describing what you DON'T want in the image.
        This is one of the most powerful quality levers.
        See Section 5 for negative prompt recipes.

    num_inference_steps : int  (default 20)
        Number of denoising steps. More steps = more refined image, slower.

        | Steps  | Quality          | Time (GPU) | Use For              |
        |--------|------------------|------------|----------------------|
        | 10–15  | Draft/sketch     | ~1–2s      | Fast iteration       |
        | 20–30  | Good quality     | ~3–5s      | General use ✅       |
        | 40–50  | High quality     | ~7–10s     | Final renders        |
        | 75–100 | Diminishing gains| ~15–25s    | Rarely worth it      |

        With DPMSolverMultistep scheduler: good quality at just 20–25 steps.
        With DDIM scheduler: typically needs 50 steps for same quality.

    guidance_scale : float  (default 7.5)
        Classifier-Free Guidance (CFG) scale. Controls how strongly the
        image adheres to the prompt vs. exploring freely.

        Think of it as "prompt obedience":
        • Low  (1–4):   Dreamy, abstract, prompt loosely followed
        • Mid  (5–8):   Balanced — creative but recognizable  ✅
        • High (9–15):  Strict prompt adherence, vivid, may oversaturate
        • Very high (>15): Colors burn out, artifacts appear

        Sweet spot: 7.0–8.5 for photorealistic; 5.0–7.0 for artistic styles.

    width / height : int  (default 512)
        Output image dimensions in pixels.
        ⚠️  SD 1.x models were trained at 512x512. Deviating causes artifacts.
        ⚠️  SD XL models were trained at 1024x1024.
        Always use multiples of 64 (VRAM alignment).

        | Model     | Recommended Size        |
        |-----------|------------------------|
        | SD 1.x    | 512×512 (default)       |
        | SD 1.x    | 512×768 (portrait)      |
        | SD 2.x    | 768×768                 |
        | SD XL     | 1024×1024               |

    seed : int  (default None = random)
        Random seed for the initial noise. Same seed + same prompt = same image.
        Crucial for reproducibility and A/B comparisons.

        To reproduce an image: note the seed from the log file, pass it back in.
        To explore variations: keep seed fixed, change guidance_scale or prompt.

    num_images : int  (default 1)
        Images to generate in one pipeline call. Batching is more efficient
        than calling generate() N times separately.
    """
    # Set seed for reproducibility
    generator = None
    actual_seed = seed
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        actual_seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(actual_seed)

    print(f"Generating: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
    print(f"  steps={num_inference_steps}, guidance={guidance_scale}, "
          f"size={width}x{height}, seed={actual_seed}")

    start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        num_images_per_prompt=num_images,
    )
    elapsed = time.time() - start
    print(f"  Generated in {elapsed:.1f}s\n")

    # Save images and log
    images = result.images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    for i, img in enumerate(images):
        fname = OUTPUT_DIR / f"{filename_prefix}_{timestamp}_{i}.png"
        img.save(fname)
        saved_paths.append(str(fname))

    generation_log.append({
        "timestamp": timestamp,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": actual_seed,
        "model": USE_MODEL,
        "time_seconds": round(elapsed, 2),
        "saved_to": saved_paths,
    })
    save_log()

    return images


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PROMPT ENGINEERING FOR IMAGES
# ─────────────────────────────────────────────────────────────────────────────
#
# Image prompts have a different grammar from text prompts.
# The model was trained on image captions, so structure matters.
#
# General anatomy of a strong image prompt:
#
#   [Subject] [Action/Pose] [Setting/Background] [Style] [Quality tags] [Lighting]
#
# Example:
#   "A lone astronaut sitting on a red sand dune, staring at two moons on the horizon,
#    cinematic photography, hyperrealistic, volumetric lighting, 8k, award winning"

print("=" * 65)
print("SECTION 4: Prompt Engineering Experiments")
print("=" * 65)

prompts_to_test = [
    # Basic prompt
    {
        "label": "basic",
        "prompt": "a cat sitting on a windowsill",
        "negative_prompt": "",
    },
    # Adding style and quality boosters
    {
        "label": "styled",
        "prompt": (
            "a cat sitting on a windowsill, golden hour lighting, "
            "soft bokeh background, professional photography, sharp focus, "
            "8k resolution, award winning photo"
        ),
        "negative_prompt": "blurry, low quality, cartoon, illustration",
    },
    # Artistic style direction
    {
        "label": "artistic",
        "prompt": (
            "a cat sitting on a windowsill, oil painting, impressionist style, "
            "Monet inspired, loose brushstrokes, warm palette, museum quality"
        ),
        "negative_prompt": "photograph, realistic, sharp, digital art",
    },
    # Cinematic / concept art style
    {
        "label": "cinematic",
        "prompt": (
            "a cat sitting on a windowsill at night, cyberpunk cityscape outside, "
            "neon reflections, rain on glass, cinematic composition, "
            "dramatic lighting, concept art, artstation trending"
        ),
        "negative_prompt": "daytime, blurry, amateur, ugly",
    },
]

for item in prompts_to_test:
    images = generate_image(
        prompt=item["prompt"],
        negative_prompt=item["negative_prompt"],
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=42,                # fixed seed so you can compare style changes
        filename_prefix=f"prompt_{item['label']}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: NEGATIVE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
#
# Negative prompts tell the model what to push AWAY from during generation.
# They're applied using the same CFG mechanism as positive prompts but in
# the opposite direction.
#
# Think of it as: the model moves toward the positive prompt AND
#                 away from the negative prompt simultaneously.
#
# Universal quality negative prompt (works for almost everything):
UNIVERSAL_NEGATIVE = (
    "ugly, deformed, blurry, low quality, low resolution, "
    "jpeg artifacts, watermark, signature, text, logo, "
    "extra limbs, extra fingers, missing fingers, mutation, "
    "cropped, out of frame, worst quality, bad anatomy"
)

# Portrait-specific negative prompt additions:
PORTRAIT_NEGATIVE = UNIVERSAL_NEGATIVE + (
    ", bad proportions, asymmetrical face, cross-eyed, "
    "bad teeth, open mouth, double chin, overexposed skin"
)

# Architecture / landscape negative prompt:
LANDSCAPE_NEGATIVE = UNIVERSAL_NEGATIVE + (
    ", people, persons, humans, cars, modern elements"
)

print("=" * 65)
print("SECTION 5: Negative Prompt Impact Demo")
print("=" * 65)

portrait_prompt = (
    "professional headshot portrait of a business executive, "
    "neutral background, studio lighting, sharp focus, 4k"
)

for label, neg in [("no_negative", ""), ("with_negative", PORTRAIT_NEGATIVE)]:
    generate_image(
        prompt=portrait_prompt,
        negative_prompt=neg,
        num_inference_steps=25,
        guidance_scale=8.0,
        seed=123,
        filename_prefix=f"negative_{label}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: GUIDANCE SCALE SWEEP
# ─────────────────────────────────────────────────────────────────────────────
#
# Same seed, same prompt — only guidance_scale changes.
# This lets you visualize the creative vs. literal tradeoff.

print("=" * 65)
print("SECTION 6: Guidance Scale Sweep")
print("=" * 65)

cfg_prompt = (
    "a majestic dragon perched on a mountain peak at sunrise, "
    "fantasy art, dramatic lighting, epic composition"
)

for cfg in [3.0, 7.5, 12.0, 15.0]:
    generate_image(
        prompt=cfg_prompt,
        negative_prompt=UNIVERSAL_NEGATIVE,
        num_inference_steps=20,
        guidance_scale=cfg,
        seed=777,
        filename_prefix=f"cfg_{cfg}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SCHEDULERS — DIFFERENT NOISE REMOVAL STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
#
# The scheduler controls HOW noise is removed at each denoising step.
# Different schedulers produce different aesthetics and require different
# step counts for optimal quality.
#
# Available schedulers (swap into the pipeline):
#
#   DPMSolverMultistepScheduler  — Fast, high quality at 20-25 steps  ✅
#   DDIM                         — Deterministic, good at 50+ steps
#   EulerAncestralDiscrete       — "Euler A" — popular for artistic styles
#   UniPCMultistepScheduler      — Very fast convergence, 10-15 steps possible
#   PNDMScheduler                — Original SD default, 50 steps
#
# To swap schedulers:
#   from diffusers import DPMSolverMultistepScheduler
#   pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("=" * 65)
print("SECTION 7: Scheduler Comparison")
print("=" * 65)

scheduler_prompt = (
    "enchanted forest with glowing mushrooms, fairies, "
    "magical atmosphere, concept art, detailed illustration"
)

schedulers = {
    "DPMSolver": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
    "EulerA": EulerAncestralDiscreteScheduler,
}

original_scheduler_config = pipe.scheduler.config

for sched_name, SchedulerClass in schedulers.items():
    print(f"Swapping scheduler to: {sched_name}")
    pipe.scheduler = SchedulerClass.from_config(original_scheduler_config)

    # DPMSolver is efficient at 20 steps; others may need more
    steps = 20 if sched_name == "DPMSolver" else 30

    generate_image(
        prompt=scheduler_prompt,
        negative_prompt=UNIVERSAL_NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=7.5,
        seed=999,
        filename_prefix=f"scheduler_{sched_name}",
    )

# Restore default scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(original_scheduler_config)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: IMAGE-TO-IMAGE (img2img)
# ─────────────────────────────────────────────────────────────────────────────
#
# img2img starts from an existing image + adds noise + denoises toward a prompt.
# Use cases:
#   - Style transfer (photo → oil painting)
#   - Upscaling / enhancing an existing image
#   - Iterating on a generated image
#   - Turning a sketch into a finished illustration
#
# Key parameter: strength
#   strength=0.0 → image unchanged
#   strength=0.5 → half the steps start from noise (50% deviation from original)
#   strength=1.0 → full generation ignoring the original (same as txt2img)
#   Sweet spot: 0.5–0.75 for style transfer; 0.3–0.5 for subtle enhancement

print("=" * 65)
print("SECTION 8: Image-to-Image Generation")
print("=" * 65)


def create_test_input_image(size: tuple = (512, 512)) -> Image.Image:
    """
    Create a simple procedural test image to use as img2img input.
    In real use, load any image: Image.open("your_photo.jpg")
    """
    img = Image.new("RGB", size, color=(135, 180, 200))
    draw = ImageDraw.Draw(img)
    # Draw a simple landscape: sky, ground, sun, trees
    draw.rectangle([0, 0, size[0], size[1]//2], fill=(100, 150, 220))  # sky
    draw.rectangle([0, size[1]//2, size[0], size[1]], fill=(80, 130, 60))  # ground
    draw.ellipse([size[0]//2-40, 50, size[0]//2+40, 130], fill=(255, 230, 50))  # sun
    # Trees
    for x in [80, 180, 350, 420]:
        draw.polygon([(x, size[1]//2), (x-30, size[1]//2+80), (x+30, size[1]//2+80)],
                     fill=(40, 100, 40))
    return img


# Load img2img pipeline (shares weights with txt2img, minimal extra memory)
print("Loading img2img pipeline...")
img2img_pipe = StableDiffusionImg2ImgPipeline(
    **{k: getattr(pipe, k) for k in
       ["vae", "text_encoder", "tokenizer", "unet", "scheduler",
        "safety_checker", "feature_extractor", "requires_safety_checker"]}
)
img2img_pipe.safety_checker = None

input_image = create_test_input_image()
input_image.save(OUTPUT_DIR / "img2img_input.png")
print("  Input image saved to outputs/img2img_input.png\n")

img2img_styles = [
    {
        "label": "watercolor",
        "prompt": "landscape, watercolor painting, soft washes, artistic, beautiful",
        "strength": 0.65,
    },
    {
        "label": "cinematic",
        "prompt": "dramatic landscape, cinematic photography, golden hour, epic, 8k",
        "strength": 0.60,
    },
    {
        "label": "anime",
        "prompt": "landscape, anime style, Studio Ghibli inspired, vibrant colors, detailed",
        "strength": 0.70,
    },
]

for item in img2img_styles:
    print(f"img2img style: {item['label']} (strength={item['strength']})")
    generator = torch.Generator(device=device).manual_seed(42)

    result = img2img_pipe(
        prompt=item["prompt"],
        negative_prompt=UNIVERSAL_NEGATIVE,
        image=input_image,
        strength=item["strength"],   # ← key img2img parameter
        guidance_scale=7.5,
        num_inference_steps=20,
        generator=generator,
    )
    out_path = OUTPUT_DIR / f"img2img_{item['label']}.png"
    result.images[0].save(out_path)
    print(f"  Saved: {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: INPAINTING — EDIT A REGION OF AN IMAGE
# ─────────────────────────────────────────────────────────────────────────────
#
# Inpainting replaces a masked region with new content generated from a prompt.
# Use cases:
#   - Remove unwanted objects from photos
#   - Replace backgrounds
#   - Add/change elements in a specific area
#   - Product image editing

print("=" * 65)
print("SECTION 9: Inpainting")
print("=" * 65)


def create_inpaint_inputs(size=(512, 512)):
    """Create a base image and a mask for the inpainting demo."""
    # Base image: simple scene
    base = Image.new("RGB", size, (100, 150, 200))
    draw = ImageDraw.Draw(base)
    draw.rectangle([0, size[1]//2, size[0], size[1]], fill=(80, 120, 60))
    draw.ellipse([50, 50, 150, 150], fill=(255, 220, 50))  # sun in top-left

    # Mask: white = region to REPLACE, black = region to KEEP
    # We'll replace the center of the image
    mask = Image.new("RGB", size, "black")
    mask_draw = ImageDraw.Draw(mask)
    cx, cy = size[0]//2, size[1]//2
    mask_draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill="white")

    return base, mask


print("Loading inpainting pipeline...")
inpaint_pipe = StableDiffusionInpaintPipeline(
    **{k: getattr(pipe, k) for k in
       ["vae", "text_encoder", "tokenizer", "unet", "scheduler",
        "safety_checker", "feature_extractor", "requires_safety_checker"]}
)
inpaint_pipe.safety_checker = None

base_img, mask_img = create_inpaint_inputs()
base_img.save(OUTPUT_DIR / "inpaint_base.png")
mask_img.save(OUTPUT_DIR / "inpaint_mask.png")
print("  Base and mask saved to outputs/\n")

inpaint_subjects = [
    "a glowing crystal ball, magical, detailed, fantasy art",
    "a roaring campfire with sparks, warm light, photorealistic",
    "a blooming sunflower, vibrant yellow, macro photography",
]

for i, subject_prompt in enumerate(inpaint_subjects):
    print(f"Inpainting subject: {subject_prompt[:50]}...")
    generator = torch.Generator(device=device).manual_seed(i * 100)

    result = inpaint_pipe(
        prompt=subject_prompt,
        negative_prompt=UNIVERSAL_NEGATIVE,
        image=base_img,
        mask_image=mask_img,
        num_inference_steps=20,
        guidance_scale=8.0,
        generator=generator,
    )
    out_path = OUTPUT_DIR / f"inpaint_result_{i}.png"
    result.images[0].save(out_path)
    print(f"  Saved: {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: GENERATION LOG & REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
#
# Every generation was logged to outputs/generation_log.json.
# To reproduce any image, copy its entry's parameters back into generate_image().

print("=" * 65)
print("SECTION 10: Generation Log Summary")
print("=" * 65)

print(f"\nAll generations logged to: {LOG_FILE}")
print(f"Total generations this session: {len(generation_log)}")
print("\nTo reproduce any image:")
print("  1. Open outputs/generation_log.json")
print("  2. Find the entry with the image you want")
print("  3. Call generate_image() with the same prompt, seed, steps, guidance_scale")
print("  Same seed + same params = byte-identical image\n")

# Print a summary table
print(f"{'#':<4} {'Prefix':<25} {'Steps':<7} {'CFG':<6} {'Seed':<12} {'Time(s)'}")
print("-" * 60)
for i, entry in enumerate(generation_log):
    prefix = Path(entry["saved_to"][0]).stem[:24] if entry.get("saved_to") else "N/A"
    print(f"{i+1:<4} {prefix:<25} {entry['steps']:<7} {entry['guidance_scale']:<6} "
          f"{entry['seed']:<12} {entry['time_seconds']}")

print("\n" + "=" * 65)
print("Tutorial complete!")
print(f"  → All images saved to: {OUTPUT_DIR.resolve()}")
print(f"  → Full log: {LOG_FILE.resolve()}")
print("  → Check README.md for model catalogue, art style guide, and LoRA tips")
print("=" * 65)
