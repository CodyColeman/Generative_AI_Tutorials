# 🎨 Tutorial 02: Image Generation

### Text-to-Image with Stable Diffusion — GPU Recommended, CPU Viable

------------------------------------------------------------------------

## Overview

This tutorial covers the full spectrum of image generation with diffusion models — from basic text-to-image, to style control, to editing existing images. You'll understand what every knob does and how to get consistent, high-quality results.

**Library:** HuggingFace [Diffusers](https://huggingface.co/docs/diffusers)\
**Default model:** `runwayml/stable-diffusion-v1-5` (GPU) / `hf-internal-testing/tiny-stable-diffusion-pipe` (CPU demo)

> 👉 **Coming up in [Tutorial 04: Fine-Tuning](../04_fine_tuning/):** Learn how to use **LoRAs** (Low-Rank Adapters) to fine-tune image models for a specific person, art style, or product — and how LoRAs work alongside language model fine-tuning.

------------------------------------------------------------------------

## Quick Start

``` bash
# Install dependencies
pip install diffusers transformers torch accelerate Pillow

# Optional but recommended
pip install xformers          # faster GPU attention
pip install invisible-watermark  # required for SD XL safety

# Run the tutorial
python image_gen_tutorial.py
```

> **CPU warning:** Each image takes 5–15 minutes on CPU at 512×512 with 20 steps. The tutorial defaults to a tiny demo model on CPU — swap `USE_MODEL = SD15_MODEL` in the script for real quality results on a GPU machine.

------------------------------------------------------------------------

## How Diffusion Works — The Visual Intuition

```         
TRAINING (what happened before we got here):
  Real photo ──► add noise ──► add more noise ──► pure static
  Model learns: "given this noisy image, what noise should I subtract?"

GENERATION (what we do):
  Pure random noise
       │  Step 1: model removes a little noise (guided by your prompt)
       ▼
  Slightly less noisy
       │  Step 2: remove more noise
       ▼
  ...20–50 steps later...
       ▼
  🖼️  Your image

  At EVERY step, cross-attention layers inject your text prompt,
  steering the noise removal toward your description.
```

------------------------------------------------------------------------

## 🎛️ Parameter Guide — Every Knob Explained

------------------------------------------------------------------------

### `num_inference_steps` *(int, default: 20)*

Number of denoising iterations. The model runs forward once per step.

| Steps | Quality       | GPU Time | CPU Time | Notes                          |
|-------|---------------|----------|----------|--------------------------------|
| `10`  | Draft         | \~1s     | \~3 min  | Fast iteration, visible noise  |
| `20`  | Good          | \~3s     | \~8 min  | **Default — great balance** ✅ |
| `30`  | Better        | \~5s     | \~12 min | Noticeably sharper details     |
| `50`  | High          | \~8s     | \~20 min | Use for final renders          |
| `75+` | Marginal gain | \~15s+   | \~35 min | Rarely worth the cost          |

> ✅ **With `DPMSolverMultistepScheduler`:** 20 steps gives quality comparable to 50 steps with older schedulers. This is the recommended default scheduler.

------------------------------------------------------------------------

### `guidance_scale` *(float, default: 7.5)*

**Classifier-Free Guidance (CFG)** — controls how literally the model follows your prompt. Under the hood, the model runs twice per step: once with the prompt, once without. CFG scale is how much it amplifies the difference.

```         
Low guidance  →  model is "inspired by" your prompt, more organic
High guidance →  model is "obedient to" your prompt, more vivid/sharp
```

| guidance_scale | Behavior | Best For |
|--------------------------------|--------------------|--------------------|
| `1.0 – 3.0` | Ignores prompt, free generation | Abstract art exploration |
| `4.0 – 6.0` | Loose adherence, painterly | Artistic styles, loose compositions |
| `7.0 – 8.5` | **Balanced** — creative + accurate | General use, most prompts ✅ |
| `9.0 – 12.0` | Strict, vivid, saturated | Concept art, strong color intent |
| `13.0 – 20.0` | Over-saturated, may burn colors | Rarely useful; test first |

> ⚠️ **Watch for:** Guidance \> 12 often causes color clipping, harsh edges, and unnaturally vivid outputs. If details look "fried", lower your CFG.

------------------------------------------------------------------------

### `seed` *(int, default: None = random)*

The random seed for initial noise. This is your **reproducibility lever**.

``` python
# Reproduce an exact image:
generate_image(prompt="...", seed=42)   # same image every time

# Explore variations of a good composition:
for seed in [42, 43, 44, 45]:
    generate_image(prompt="...", seed=seed)   # same prompt, different noise

# A/B test a parameter change:
generate_image(prompt="...", seed=42, guidance_scale=7.5)
generate_image(prompt="...", seed=42, guidance_scale=10.0)  # same base noise
```

> ✅ **Best practice:** Always log your seeds (the tutorial does this automatically). When you get a result you love, note the seed so you can reproduce it or make small variations.

------------------------------------------------------------------------

### `width` / `height` *(int, default: 512)*

Output image resolution. **Must be a multiple of 64** (VAE alignment requirement).

| Model  | Native Res | Notes                        |
|--------|------------|------------------------------|
| SD 1.x | 512×512    | Training resolution — safest |
| SD 1.x | 512×768    | Portrait, works well         |
| SD 1.x | 768×512    | Landscape, works well        |
| SD 2.x | 768×768    | Training resolution          |
| SD XL  | 1024×1024  | Training resolution          |
| SD XL  | 1024×1536  | Portrait                     |

> ⚠️ **Don't exceed native res by too much with SD 1.x** — you get "extra heads," repeated patterns, and distorted anatomy. For higher-res output, generate at 512×512 then upscale with a dedicated upscaler (Real-ESRGAN, Stable Diffusion upscale).

------------------------------------------------------------------------

### `strength` *(float, img2img only, default: 0.8)*

In image-to-image mode, `strength` controls how much the starting image is destroyed before re-generation.

```         
strength=0.0  ─────  Original image unchanged
strength=0.3  ─────  Subtle style/color shifts, composition preserved
strength=0.6  ─────  Clear style change, main shapes recognizable  ✅
strength=0.8  ─────  Heavy transformation, loose reference to original
strength=1.0  ─────  Ignore original entirely (same as txt2img)
```

| Use Case                            | Recommended Strength |
|-------------------------------------|----------------------|
| Style transfer (photo → painting)   | `0.55 – 0.70`        |
| Minor enhancement / upscaling       | `0.25 – 0.45`        |
| Using a sketch as a loose reference | `0.70 – 0.85`        |
| Starting from a rough composition   | `0.80 – 0.95`        |

------------------------------------------------------------------------

### Schedulers — Noise Removal Algorithms

The scheduler is swappable without reloading the model. Each has a different "feel":

| Scheduler | Steps Needed | Character | Best For |
|------------------|--------------------|------------------|------------------|
| `DPMSolverMultistepScheduler` | 20–25 | Sharp, efficient | **General use default** ✅ |
| `EulerAncestralDiscreteScheduler` | 20–30 | Painterly, creative | Artistic styles, illustration |
| `DDIMScheduler` | 40–60 | Deterministic, clean | When you need exact reproducibility |
| `UniPCMultistepScheduler` | 10–15 | Very fast | Quick iteration drafts |
| `HeunDiscreteScheduler` | 20–30 | High detail | Fine textures, portraits |

``` python
from diffusers import EulerAncestralDiscreteScheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

------------------------------------------------------------------------

## 🖊️ Prompt Engineering for Images

### The Anatomy of a Strong Prompt

```         
[Subject] + [Action/Pose] + [Environment] + [Style] + [Quality Tags] + [Lighting]
```

**Example:**

```         
"a medieval knight standing at the edge of a cliff overlooking a stormy sea,
 dramatic cinematic composition, digital painting, artstation trending,
 volumetric fog, rim lighting, 8k resolution"
```

### Style Keywords That Work

| Style | Keywords |
|--------------------------------|----------------------------------------|
| Photorealistic | `hyperrealistic, DSLR photo, 85mm lens, shallow depth of field, photojournalism` |
| Oil Painting | `oil painting, impasto, canvas texture, old masters, chiaroscuro` |
| Watercolor | `watercolor, loose washes, wet on wet, translucent, soft edges` |
| Anime / Manga | `anime style, manga, Studio Ghibli, Makoto Shinkai, cel shading` |
| Concept Art | `concept art, artstation, cinematic, matte painting, epic composition` |
| Pixel Art | `pixel art, 16-bit, retro game, isometric, sprite` |
| 3D Render | `3D render, octane render, cinema 4D, subsurface scattering, ray tracing` |
| Ink Drawing | `ink illustration, pen and ink, cross-hatching, graphic novel` |

### Quality Boosters

These tokens reliably push quality up regardless of subject:

```         
4k, 8k, highly detailed, sharp focus, award winning, masterpiece,
professional, intricate details, high resolution
```

### Lighting Keywords

```         
golden hour, volumetric lighting, rim lighting, studio lighting,
dramatic shadows, soft diffused light, neon lights, bioluminescent,
candlelight, moonlight, fog, god rays
```

------------------------------------------------------------------------

## ❌ Negative Prompts — What to Push Away

Negative prompts are **one of the highest-ROI levers** in image generation. Always use them.

### Universal Negative Prompt (use for everything):

```         
ugly, deformed, blurry, low quality, low resolution, jpeg artifacts,
watermark, signature, text, logo, extra limbs, extra fingers, missing fingers,
mutation, cropped, out of frame, worst quality, bad anatomy
```

### Portrait-Specific:

```         
[universal] + bad proportions, asymmetrical face, cross-eyed, bad teeth,
overexposed skin, plastic skin, bad hands, six fingers
```

### Landscape/Architecture:

```         
[universal] + people, cars, modern elements, power lines, litter
```

### Photorealistic Work:

```         
[universal] + cartoon, illustration, painting, drawing, anime, 3D render
```

------------------------------------------------------------------------

## 🌐 Exploring Models on HuggingFace

HuggingFace hosts thousands of image generation models. You can swap any of them into this tutorial by changing `USE_MODEL`.

``` python
# Just change this one line to try any HuggingFace image model:
USE_MODEL = "dreamshaper-8"   # or any model ID from huggingface.co/models
```

### Curated Model Recommendations by Style

| Model | Style | HuggingFace ID |
|------------------|------------------|-------------------------------------|
| **Stable Diffusion 1.5** | Versatile baseline | `runwayml/stable-diffusion-v1-5` |
| **Stable Diffusion XL** | Highest quality, 1024px | `stabilityai/stable-diffusion-xl-base-1.0` |
| **DreamShaper** | Fantasy, painterly | `Lykon/dreamshaper-8` |
| **Realistic Vision** | Photorealism | `SG161222/Realistic_Vision_V6.0_B1_noVAE` |
| **Deliberate** | Semi-realistic, detailed | `XpucT/Deliberate` |
| **AbsoluteReality** | Photographic | `Lykon/absolute-reality-1.81` |
| **Anything V5** | Anime / manga | `stablediffusionapi/anything-v5` |
| **OpenJourney** | Midjourney-inspired | `prompthero/openjourney-v4` |
| **Protogen** | Sci-fi, futuristic | `darkstorm2150/Protogen_x5.8_Official_Release` |
| **Inkpunk Diffusion** | Punk illustration | `Envvi/Inkpunk-Diffusion` |

### How to Browse HuggingFace for Models

1.  Go to [**huggingface.co/models**](https://huggingface.co/models)
2.  Filter by **Task → Text-to-Image**
3.  Sort by **Most Downloads** or **Trending**
4.  Look for the model card — check sample images before downloading

### ⚠️ IMPORTANT: NSFW Models on HuggingFace

> HuggingFace does **not** apply content filtering to hosted models. Many image generation models — especially fine-tuned variants — are trained on explicit content and will generate NSFW output **without any warning**.
>
> **How to stay safe:** - Read the model title carefully. Don't click on any models with obvious NSFW content. Stay in the first page or two of the search - this *generally* has clean models. Read the model card fully before using any community model - Look for models with `_SFW`, `safe`, or `no-NSFW` in the name/description - Be especially cautious with models in categories like "people," "portraits," or "realistic" - The base SD 1.5 / SDXL models from Stability AI have NSFW filters built in — community fine-tunes often **remove** these filters - In production systems: always re-enable the safety checker (`safety_checker=True`)
>
> **Re-enabling the safety checker:**
>
> ``` python
> pipe = StableDiffusionPipeline.from_pretrained(
>     model_id,
>     # Simply remove the safety_checker=None lines
>     # The default will load the NSFW classifier automatically
> )
> ```

------------------------------------------------------------------------

## 🔁 LoRAs — Lightweight Style Add-ons

LoRA (Low-Rank Adaptation) files let you inject a specific style, person, or concept into *any* base model without retraining it. They're typically tiny (5–150MB) and stack on top of the base model at inference time.

> 📖 **For a deep dive into how LoRAs work — including training your own — see [Tutorial 04: Fine-Tuning](../04_fine_tuning/).** That tutorial covers both image LoRAs AND language model fine-tuning with LoRA/PEFT.

### Using a LoRA in Diffusers (preview):

``` python
# Load a base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Load a LoRA on top (from HuggingFace or a local .safetensors file)
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors")

# Use the LoRA trigger word in your prompt
image = pipe("a toy face portrait of a CEO, professional lighting").images[0]
```

### Where to Find LoRAs

| Platform | URL | Notes |
|--------------------------------|------------------|-----------------------|
| **CivitAI** | [civitai.com](https://civitai.com) | Largest LoRA library — ⚠️ also has NSFW content, filter carefully |
| **HuggingFace** | Search `lora` on models page | More curated, safer |
| **LoRA the Explorer** | [huggingface.co/spaces/multimodalart/LoraTheExplorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer) | Interactive LoRA browser |

------------------------------------------------------------------------

## 📂 Output Structure

```         
outputs/
├── generation_log.json          ← Full record of every generation
├── prompt_basic_[timestamp].png
├── prompt_styled_[timestamp].png
├── prompt_artistic_[timestamp].png
├── cfg_3.0_[timestamp].png
├── cfg_7.5_[timestamp].png
├── scheduler_DPMSolver_[timestamp].png
├── img2img_input.png
├── img2img_watercolor_[timestamp].png
├── inpaint_base.png
└── inpaint_result_0.png
```

The `generation_log.json` records every parameter used so you can reproduce any image exactly.

------------------------------------------------------------------------

## Common Pitfalls

| Problem | Likely Cause | Fix |
|-----------------------|---------------------------------|-----------------|
| Blurry, soft images | Steps too low / wrong scheduler | Use 25–30 steps with DPMSolver |
| Washed out colors | Guidance scale too low | Try `guidance_scale=8.0–10.0` |
| Burnt, oversaturated colors | Guidance scale too high | Lower to `7.0–8.5` |
| Extra limbs / bad anatomy | No negative prompt | Add the universal negative prompt |
| Out of memory (GPU) | Resolution too high | Lower to 512×512; enable `attention_slicing` |
| img2img ignores input | Strength too high | Lower to `0.5–0.65` |
| Repeated patterns / tiling | Exceeding training resolution | Stick to 512×512 for SD 1.x |
| NSFW content from community model | Model fine-tuned without safety filter | Re-enable `safety_checker` or use a safe base model |

------------------------------------------------------------------------

## What's Next

| Tutorial | Topic |
|------------------------------------------|------------------------------|
| [**`04_fine_tuning/`**](../04_fine_tuning/) | **Train your own LoRA** — teach the model a new art style, person, or product with just 10–50 images. Covers both image LoRAs and language model fine-tuning. |
| [`03_rag_models/`](../03_rag_models/) | RAG — connect an LLM to your documents |
| [`05_embeddings/`](../05_embeddings/) | Semantic search and clustering |

------------------------------------------------------------------------

## Resources

-   [HuggingFace Diffusers Docs](https://huggingface.co/docs/diffusers)
-   [Stable Diffusion Prompt Guide](https://stable-diffusion-art.com/prompt-guide/)
-   [HuggingFace Model Hub — Text to Image](https://huggingface.co/models?pipeline_tag=text-to-image)
-   [CivitAI — Community Models & LoRAs](https://civitai.com) *(⚠️ contains NSFW — filter by SFW)*
-   [Scheduler Comparison Guide](https://huggingface.co/docs/diffusers/api/schedulers/overview)
-   [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — excellent visual explainer
