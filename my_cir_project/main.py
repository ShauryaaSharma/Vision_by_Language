import argparse
import os
import pickle
from pathlib import Path

import torch
from PIL import Image

import pipeline


def build_index(images_dir: str, out_path: str) -> None:
    paths = sorted(Path(images_dir).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No .jpg images found in {images_dir}")
    images = [Image.open(p).convert("RGB") for p in paths]

    print(f"Captioning {len(paths)} images with BLIP...")
    captions = [pipeline.caption_image(img) for img in images]

    print("Embedding images with CLIP...")
    features = pipeline.embed_images(images)

    data = {"paths": [str(p) for p in paths], "captions": captions, "features": features}
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved index with {len(paths)} images to {out_path}")


def query(index_path: str, ref_image_path: str, instruction: str, topk: int) -> None:
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    ref_img = Image.open(ref_image_path).convert("RGB")
    base_caption = pipeline.caption_image(ref_img)
    new_caption = pipeline.rewrite_caption(base_caption, instruction)
    print(f"Base caption:     {base_caption}")
    print(f"Rewritten target: {new_caption}")

    query_feat = pipeline.embed_text([new_caption])[0]
    sims = data["features"] @ query_feat
    ranked = torch.argsort(sims, descending=True)

    print(f"\nTop {topk} matches:")
    for rank in ranked[:topk]:
        i = rank.item()
        print(f"  {sims[i]:.4f}  {data['paths'][i]}   (caption: {data['captions'][i]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal training-free text-to-image CIR demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_parser = sub.add_parser("build-index", help="Caption + embed the image gallery")
    build_parser.add_argument("--images-dir", default="static/gallery")
    build_parser.add_argument("--out", default="cache/index.pkl")

    query_parser = sub.add_parser("query", help="Run a reference image + text edit against the gallery")
    query_parser.add_argument("--ref-image", required=True)
    query_parser.add_argument("--instruction", required=True)
    query_parser.add_argument("--index", default="cache/index.pkl")
    query_parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "build-index":
        build_index(args.images_dir, args.out)
    elif args.cmd == "query":
        query(args.index, args.ref_image, args.instruction, args.topk)


if __name__ == "__main__":
    main()
