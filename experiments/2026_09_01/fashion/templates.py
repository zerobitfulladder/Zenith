"""What the Fashion experts learned to look at, and which garments they saw."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts")); sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E, settle as S
import run as RUN

OUT = HERE / "results"
W = np.load(OUT / "weights.npz")["W"].astype(np.float64)
_, _, Xte, yte = E.load(RUN.DS)
idx = RUN.winners(W, Xte)
h = len(W)
cnt = np.zeros(h); ld = np.zeros((h, 10))
r, c = np.nonzero(idx >= 0)
np.add.at(cnt, idx[r, c], 1.0); np.add.at(ld, (idx[r, c], yte[r]), 1.0)
d = ld / np.maximum(ld.sum(1, keepdims=True), 1)
print("expert class purity %.3f  (chance 0.10)  live %d/%d" %
      (d.max(1)[cnt > 0].mean(), int((cnt > 0).sum()), h))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = np.argsort(-cnt)
k = W.shape[1]
tile = np.full((h * (E.PS + 1) - 1, k * (E.PS + 1) - 1), np.nan)
for a, j in enumerate(order):
    for b in range(k):
        tile[a * 6:a * 6 + 5, b * 6:b * 6 + 5] = W[j, b, :25].reshape(5, 5)
fig, ax = plt.subplots(1, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [k, 14]})
ax[0].imshow(tile, cmap="RdBu_r", interpolation="nearest")
ax[0].set_title(f"{h} experts x {k} templates (5x5 ink part)", fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].imshow(d[order], cmap="magma", aspect="auto", interpolation="nearest")
ax[1].set_title("garments each expert won (row-normalised)", fontsize=9)
ax[1].set_xticks(range(10)); ax[1].set_xticklabels(RUN.NAMES, rotation=90, fontsize=7)
ax[1].set_yticks(range(h))
ax[1].set_yticklabels([f"{j}  n={int(cnt[j]):,}" for j in order], fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "templates.png", dpi=130)
print("-> results/templates.png")
