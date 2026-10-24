"""Fetch a small set of sample photos to use as the retrieval gallery."""
from pathlib import Path
import requests

OUT_DIR = Path(__file__).parent / "static" / "gallery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 25
CANDIDATE_IDS = range(1, 300)

saved = len(list(OUT_DIR.glob("*.jpg")))
for i in CANDIDATE_IDS:
    if saved >= TARGET_COUNT:
        break
    out_path = OUT_DIR / f"img_{i}.jpg"
    if out_path.exists():
        saved += 1
        continue
    url = f"https://picsum.photos/id/{i}/640/480"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        continue  # some ids are gaps in picsum's catalog
    out_path.write_bytes(resp.content)
    saved += 1
    print(f"Saved {out_path}")

print(f"Done. {len(list(OUT_DIR.glob('*.jpg')))} images in {OUT_DIR}")
