"""PixelPhrase web frontend: caption -> LLM rewrite -> CLIP retrieval, in the browser."""
import pickle
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

import pipeline

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "cache" / "index.pkl"

# static_url_path="" serves static/* at the root (e.g. static/gallery/x.jpg -> /gallery/x.jpg),
# which matches the design's relative asset links (style.css, app.js) in static/index.html.
app = Flask(__name__, static_folder="static", static_url_path="")

with open(INDEX_PATH, "rb") as f:
    INDEX = pickle.load(f)


def _gallery_item(i: int) -> dict:
    name = Path(INDEX["paths"][i]).name
    return {"id": Path(name).stem, "src": f"/gallery/{name}", "caption": INDEX["captions"][i]}


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/gallery")
def gallery():
    return jsonify([_gallery_item(i) for i in range(len(INDEX["paths"]))])


@app.post("/api/query")
def query():
    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "An edit instruction is required."}), 400

    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "A reference image is required."}), 400

    try:
        k = int(request.form.get("k", 6))
    except ValueError:
        k = 6

    try:
        ref_img = Image.open(image_file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "That file isn't a valid image."}), 400

    try:
        base_caption = pipeline.caption_image(ref_img)
        target_caption = pipeline.rewrite_caption(base_caption, instruction)
        query_feat = pipeline.embed_text([target_caption])[0]
    except Exception:
        return jsonify({"error": "The model pipeline failed to process that request."}), 500

    sims = INDEX["features"] @ query_feat
    ranked = sims.argsort(descending=True)[:k]

    results = [
        {**_gallery_item(i), "score": round(sims[i].item(), 4)}
        for i in ranked.tolist()
    ]
    return jsonify({
        "base_caption": base_caption,
        "target_caption": target_caption,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
