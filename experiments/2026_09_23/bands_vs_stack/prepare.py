"""Decode the CelebA shards once: full-size grayscale images plus the 40 attributes.

Writes to data/celeba/:
    gray_218x178.npy   uint8 (N, 218, 178)   the original aligned images, grayscale
    attrs_40.npy       uint8 (N, 40)         1 = attribute present
    attr_names_40.json list of the 40 names, in column order
    image_ids.json     the CelebA file names, in row order
"""
import io, json, glob, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/celeba/shards"
DST = ROOT / "data/celeba"

t0 = time.time()
files = sorted(glob.glob(str(SRC / "shard*.parquet")))
names = None
imgs, attrs, ids = [], [], []
for f in files:
    t = pq.read_table(f)
    if names is None:
        names = [c for c in t.column_names if c not in ("image", "label", "image_id")]
        assert len(names) == 40, names
    cols = {c: t.column(c).to_pylist() for c in names}
    for i, (im, iid) in enumerate(zip(t.column("image").to_pylist(), t.column("image_id").to_pylist())):
        g = Image.open(io.BytesIO(im["bytes"])).convert("L")
        assert g.size == (178, 218), g.size
        imgs.append(np.asarray(g, dtype=np.uint8))
        attrs.append([1 if cols[c][i] == 1 else 0 for c in names])
        ids.append(iid)
    print(f"{Path(f).name}: {len(imgs)} images so far, {time.time()-t0:.0f}s", flush=True)

imgs = np.stack(imgs); attrs = np.asarray(attrs, dtype=np.uint8)
np.save(DST / "gray_218x178.npy", imgs)
np.save(DST / "attrs_40.npy", attrs)
(DST / "attr_names_40.json").write_text(json.dumps(names, indent=1))
(DST / "image_ids.json").write_text(json.dumps(ids))
print("images", imgs.shape, imgs.dtype, "attrs", attrs.shape)
print("positive rate per attribute:")
for n, r in zip(names, attrs.mean(0)):
    print(f"  {n:22s} {r:.3f}")
