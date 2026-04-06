# 🧠 GenAI Tutorials

A practical, runnable repository of **Generative AI tutorials in Python** — built for intermediate Python developers who are new to GenAI. Each tutorial covers a distinct area of the GenAI landscape with working code, real datasets, and deep-dive parameter guides.

---

## Repository Structure

```
genai-tutorials/
│
├── 01_llms/                   ✅ AVAILABLE
│   ├── llm_tutorial.py        Text generation & prompting (CPU-friendly)
│   └── README.md              Full parameter guide + prompt engineering tips
│
├── 02_image_generation/       ✅ AVAILABLE
│   ├── image_gen_tutorial.py  Stable Diffusion / text-to-image generation
│   └── README.md
│
├── 03_rag_models/             ✅ AVAILABLE
│   ├── rag_tutorial.py        Retrieval-Augmented Generation
│   └── README.md
│
├── 04_fine_tuning/            ✅ AVAILABLE
│   ├── finetune_tutorial.py   Fine-tuning with LoRA / PEFT
│   └── README.md
│
├── 05_embeddings/             ✅ AVAILABLE
│   ├── embeddings_tutorial.py Semantic search & vector similarity
│   └── README.md
│
└── 06_agents/                 ✅ AVAILABLE
    ├── agents_tutorial.py     Tool-using LLM agents
    └── README.md
```

---

## Tutorials at a Glance

| # | Topic | Model(s) | Dataset | Hardware |
|---|-------|----------|---------|----------|
| 01 | **LLMs & Prompting** | DistilGPT-2 | AG News | CPU |
| 02 | **Image Generation** | Stable Diffusion | COCO Captions | GPU recommended |
| 03 | **RAG Models** | sentence-transformers + GPT-2 | SQuAD / custom docs | CPU |
| 04 | **Fine-Tuning** | GPT-2 + LoRA | Custom / HuggingFace | CPU / GPU |
| 05 | **Embeddings** | all-MiniLM-L6-v2 | GLUE / custom | CPU |
| 06 | **Agents** | Tool-augmented LLM | Live APIs | CPU |

---

## Philosophy

Each tutorial is built around three goals:

1. **Runnable** — clone and run, minimal setup, real output
2. **Explainable** — every parameter is documented; no black boxes
3. **Tunable** — each README explains what happens when you turn each knob

---

## Getting Started

```bash
git clone <your-repo-url>
cd genai-tutorials

# Install base dependencies (each tutorial may add more)
pip install transformers torch datasets accelerate

# Run the first tutorial
cd 01_llms
python llm_tutorial.py
```

---

## Contributing / Extending

Each tutorial folder is self-contained. To add a new one:
1. Create a new numbered folder (`07_your_topic/`)
2. Add a `tutorial.py` with sections following the established pattern
3. Add a `README.md` using the existing READMEs as a template
4. Update this root README table

---

*Built with [HuggingFace Transformers](https://huggingface.co/docs/transformers), [Datasets](https://huggingface.co/docs/datasets), and PyTorch.*
