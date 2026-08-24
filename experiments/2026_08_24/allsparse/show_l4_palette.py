"""Render all 200 L4 memories of the 8x8 dense rig, grouped by owner label.
Hardened read per memory (same recipe as generation figures)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_4layer_topk import expand  # noqa: E402
from run_rich_palette_8x8 import G1, G2, G3, S2, S3, W2, W3, render8  # noqa: E402

DIR = ROOT / "experiments" / "2026_08_24" / "allsparse" / "results"
z = np.load(DIR / "weights_K64_100_dense.npz")
W1n, W2n, W3n, Wt = z["W1"], z["W2"], z["W3"], z["Wtop"]
K1, K2, K3 = len(W1n), len(W2n), len(W3n)
CODE3_DIM = G3 * G3 * K3
owner = np.argmax(Wt[:, CODE3_DIM:], axis=1)
order = np.argsort(owner, kind="stable")


class Bank:
    def __init__(self, Wb):
        self.W = Wb
        self.k = Wb.shape[0]


b2, b3 = Bank(W2n), Bank(W3n)


def _harden_map(m, k=1):
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b_ in range(m.shape[1]):
            seg = np.maximum(m[a, b_], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b_, idx] = seg[idx]
    return o


def render_memory(row):
    c3 = np.maximum(row[:CODE3_DIM], 0.0).reshape(G3, G3, K3)
    m2 = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
    m1 = _harden_map(expand(m2, b2, W2, S2, (G1, G1, K1)))
    return render8(m1, W1n)


cols, rows_n = 20, 10
fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 0.85, rows_n * 0.98))
for ax in np.ravel(axes):
    ax.axis("off")
for i, t in enumerate(order):
    ax = np.ravel(axes)[i]
    ax.imshow(render_memory(Wt[t]), cmap="gray")
    ax.set_title(str(owner[t]), fontsize=6)
fig.suptitle("All 200 L4 memories, grouped by label (8x8 dense rig, hardened read)",
             fontsize=12)
fig.tight_layout()
fig.savefig(DIR / "palette_L4.png", dpi=130)
print("written:", DIR / "palette_L4.png")
