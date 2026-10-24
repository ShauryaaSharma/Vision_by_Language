"""Core steps: caption a reference image, rewrite the caption per an
instruction using a small local LLM, and retrieve matching images with CLIP.

This is a from-scratch, CPU-friendly reimplementation of the idea behind
CIReVL (Karthik & Roth et al., ICLR 2024): compose image edits in text
space (caption -> LLM rewrite) instead of training a dedicated model, then
retrieve with off-the-shelf CLIP.
"""
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPModel, CLIPProcessor,
    AutoTokenizer, AutoModelForCausalLM,
)

DEVICE = "cpu"

_blip = {}
_clip = {}
_llm = {}


def _load_blip():
    if not _blip:
        _blip["processor"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip["model"] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(DEVICE).eval()
    return _blip["processor"], _blip["model"]


def _load_clip():
    if not _clip:
        _clip["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    return _clip["processor"], _clip["model"]


def _load_llm():
    if not _llm:
        name = "Qwen/Qwen2.5-0.5B-Instruct"
        _llm["tokenizer"] = AutoTokenizer.from_pretrained(name)
        _llm["model"] = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
    return _llm["tokenizer"], _llm["model"]


@torch.no_grad()
def caption_image(image: Image.Image) -> str:
    processor, model = _load_blip()
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def rewrite_caption(base_caption: str, instruction: str) -> str:
    tokenizer, model = _load_llm()
    prompt = (
        "You rewrite an image description to match a requested change. "
        "Reply with ONLY the new description, one sentence, no preamble.\n\n"
        f"Original description: {base_caption}\n"
        f"Requested change: {instruction}\n"
        "New description:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    reply = reply.strip().split("\n")[0].strip()
    return reply or base_caption


@torch.no_grad()
def embed_images(images: list[Image.Image]) -> torch.Tensor:
    # shortcut: get_image_features() returns the raw vision tower output (not the
    # projected embedding) on this transformers build, so project manually.
    processor, model = _load_clip()
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    feats = model.visual_projection(pooled)
    return torch.nn.functional.normalize(feats, dim=-1)


@torch.no_grad()
def embed_text(texts: list[str]) -> torch.Tensor:
    processor, model = _load_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    pooled = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output
    feats = model.text_projection(pooled)
    return torch.nn.functional.normalize(feats, dim=-1)
