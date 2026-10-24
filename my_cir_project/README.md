# Training-Free Text-to-Image Retrieval (Compositional Image Retrieval)

A minimal, from-scratch, CPU-only implementation of the idea behind
**CIReVL** (Karthik, Roth, Mancini & Akata, *"Vision-by-Language for
Training-Free Compositional Image Retrieval"*, ICLR 2024 — see
[../README.md](../README.md) for the original authors' repo, which this
project takes as inspiration, not as code).

## Idea

Given a reference image and a text instruction describing how it should
change (e.g. "make it black and white"), retrieve the best-matching image
from a gallery — without training any task-specific model:

1. **Caption** the reference image with BLIP → plain-text description.
2. **Rewrite** that caption with a small local LLM, conditioned on the
   instruction → a new target description, entirely in text space.
3. **Retrieve** with CLIP: embed the rewritten caption as text, embed every
   gallery image, rank by cosine similarity.

No part of the pipeline is trained on this task — all three models are
used exactly as pretrained.

## Models used (all small enough to run on CPU)

| Step       | Model                                   |
|------------|------------------------------------------|
| Captioning | `Salesforce/blip-image-captioning-base`  |
| Rewriting  | `Qwen/Qwen2.5-0.5B-Instruct`              |
| Retrieval  | `openai/clip-vit-base-patch32`            |

## Setup

```sh
pip install -r requirements.txt
python download_images.py   # fetches ~24 sample photos into data/images/
```

## Usage

Build the gallery index once (captions + CLIP embeddings are cached):

```sh
python main.py build-index
```

Query with a reference image + an edit instruction:

```sh
python main.py query --ref-image data/images/img_1.jpg --instruction "make it black and white" --topk 5
```

This prints the BLIP caption, the LLM-rewritten target description, and the
top-K closest gallery images by CLIP similarity.

### Web frontend ("PixelPhrase")

```sh
python app.py
```

Then open http://localhost:5000 — upload a reference image, type an edit
instruction, and see the rewritten caption plus the top matching gallery
images with scores. Requires `cache/index.pkl` to already exist
(`build-index` above).

## What's intentionally simplified vs. the original paper

- Tiny CPU-friendly models (BLIP-base, CLIP ViT-B/32, Qwen2.5-0.5B) instead
  of BLIP-2-FlanT5-XXL / GPT-4 / larger OpenCLIP backbones.
- A small custom photo gallery instead of the official FashionIQ/CIRR/CIRCO
  benchmarks, so no formal recall@K evaluation — this is a qualitative demo.
- `rewrite_caption` always greedy-decodes (no sampling), which keeps results
  deterministic for a class demo.

## Known limits

- shortcut: gallery is only ~24 images, so retrieval quality is not
  meaningful as a benchmark result — swap in more images under
  `data/images/` and rerun `build-index` to scale it up.
