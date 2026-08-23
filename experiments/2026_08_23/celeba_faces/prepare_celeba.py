"""Download and prepare CelebA (with attributes) as MNIST-style arrays.

Source: huggingface.co/datasets/huggan/CelebA-faces-with-attributes
(132 parquet shards; we take the first N_SHARDS ~ 1500 faces each).
Output in data/:
    celeba/images_48.npy   (N, 48, 48) float32 in [0, 1], grayscale
    celeba/attrs_48.npy    (N, 40) int8 in {0, 1}
    celeba/attr_names.json

Aligned CelebA frames are 178x218; we center-crop to 178x178 and resize.

Run:  .venv/bin/python experiments/2026_08_23/celeba_faces/prepare_celeba.py
"""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data"
CACHE = ROOT / "data" / "celeba" / "shards"
SIZE = 48
# Shards 11 and 20 are corrupt AT SOURCE on the HF CDN (no PAR1 footer,
# verified via ranged reads) — skipped, replaced by 26 and 27.
SHARD_IDS = [i for i in range(28) if i not in (11, 20)]
BASE = ("https://huggingface.co/datasets/huggan/CelebA-faces-with-attributes"
        "/resolve/main/data/train-{i:05d}-of-00132.parquet")

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def fetch(i):
    dst = CACHE / f"shard{i:05d}.parquet"
    if dst.exists() and dst.stat().st_size > 1e6:
        return dst
    url = BASE.format(i=i)
    print(f"downloading shard {i}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "zenith-research"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    images, attrs = [], []
    for i in SHARD_IDS:
        t = pq.read_table(fetch(i))
        cols = {n: t.column(n).to_pylist() for n in ATTR_NAMES}
        imgs = t.column("image").to_pylist()
        for k in range(len(imgs)):
            img = Image.open(io.BytesIO(imgs[k]["bytes"])).convert("L")
            w, h = img.size
            top = (h - w) // 2
            img = img.crop((0, top, w, top + w)).resize((SIZE, SIZE), Image.LANCZOS)
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            attrs.append([int(bool(cols[n][k])) for n in ATTR_NAMES])
        print(f"shard {i}: total {len(images)} faces", flush=True)

    X = np.stack(images)
    A = np.asarray(attrs, dtype=np.int8)
    np.save(OUT / "celeba/images_48.npy", X)
    np.save(OUT / "celeba/attrs_48.npy", A)
    (OUT / "celeba/attr_names.json").write_text(json.dumps(ATTR_NAMES))

    male = A[:, ATTR_NAMES.index("Male")]
    must = A[:, ATTR_NAMES.index("Mustache")]
    hat = A[:, ATTR_NAMES.index("Wearing_Hat")]
    print(f"DONE n={len(X)} shape={X.shape} "
          f"male={male.mean():.2%} mustache={must.mean():.2%} hat={hat.mean():.2%} "
          f"woman_with_mustache={int(((1 - male) * must).sum())}")


if __name__ == "__main__":
    main()
