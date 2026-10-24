"""PixelPhrase web frontend: caption -> LLM rewrite -> CLIP retrieval, in the browser."""
import pickle
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import pipeline

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "cache" / "index.pkl"
IMAGES_DIR = BASE_DIR / "data" / "images"

app = Flask(__name__, static_folder="static")

with open(INDEX_PATH, "rb") as f:
    INDEX = pickle.load(f)


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/images/<name>")
def serve_image(name):
    return send_from_directory(IMAGES_DIR, name)


@app.get("/api/gallery")
def gallery():
    return jsonify([
        {"name": Path(p).name, "caption": c}
        for p, c in zip(INDEX["paths"], INDEX["captions"])
    ])


@app.post("/api/query")
def query():
    instruction = request.form.get("instruction", "").strip()
    image_file = request.files.get("image")
    if not instruction or image_file is None:
        return jsonify({"error": "image and instruction are both required"}), 400

    ref_img = Image.open(image_file.stream).convert("RGB")
    base_caption = pipeline.caption_image(ref_img)
    new_caption = pipeline.rewrite_caption(base_caption, instruction)

    query_feat = pipeline.embed_text([new_caption])[0]
    sims = INDEX["features"] @ query_feat
    ranked = sims.argsort(descending=True)[:8]

    results = [
        {
            "name": Path(INDEX["paths"][i]).name,
            "caption": INDEX["captions"][i],
            "score": round(sims[i].item(), 4),
        }
        for i in ranked.tolist()
    ]
    return jsonify({
        "base_caption": base_caption,
        "rewritten_caption": new_caption,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
