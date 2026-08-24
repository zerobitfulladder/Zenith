"""Render L2 and L3 templates as pixel images (8x8 rig, all-dense arm).

Each unit: place a lone activation at the center of its level's grid,
expand down through the dictionaries below, feathered pixel render —
same recipe as the l3_parts figures on the 4x4 rigs.
"""

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
W1n, W2n, W3n = z["W1"], z["W2"], z["W3"]
K1, K2, K3 = len(W1n), len(W2n), len(W3n)


class Bank:
    def __init__(self, Wb):
        self.W = Wb
        self.k = Wb.shape[0]


b2, b3 = Bank(W2n), Bank(W3n)


def render_l2(u):
    c2 = np.zeros((G2, G2, K2))
    c2[G2 // 2, G2 // 2, u] = 1.0
    m1 = expand(c2, b2, W2, S2, (G1, G1, K1))
    return render8(m1, W1n)


def render_l3(u):
    c3 = np.zeros((G3, G3, K3))
    c3[G3 // 2, G3 // 2, u] = 1.0
    m2 = expand(c3, b3, W3, S3, (G2, G2, K2))
    m1 = expand(m2, b2, W2, S2, (G1, G1, K1))
    return render8(m1, W1n)


for name, n, cols, fn in [("L2", K2, 8, render_l2), ("L3", K3, 10, render_l3)]:
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.05))
    for ax in np.ravel(axes):
        ax.axis("off")
    for u in range(n):
        np.ravel(axes)[u].imshow(fn(u), cmap="gray")
    fig.suptitle(f"{name} templates rendered to pixels — all {n} "
                 f"(8x8 rig, dense arm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(DIR / f"parts_{name}.png", dpi=130)
    plt.close(fig)
    print("written:", DIR / f"parts_{name}.png")
