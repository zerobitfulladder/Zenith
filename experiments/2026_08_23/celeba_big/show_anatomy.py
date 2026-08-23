"""Anatomy gallery of the big faces rig: every dictionary, rendered.

l1_strokes.png   - the 48 raw 4x4 stroke templates
l2_motifs.png    - all 96 motif units, expanded to pixels, cropped to RF
l3_parts.png     - all 128 part units, expanded to pixels, cropped to RF
archive_sample.png - the 24 most-rehearsed stored faces, rendered

Run:  .venv/bin/python experiments/2026_08_23/celeba_big/show_anatomy.py
"""

import os
from pathlib import Path

os.environ["GF_REPORT"] = "sparselearn"
os.environ["GF_SIDE"] = "48"
os.environ["GF_W1_STR"] = "1"
os.environ["GF_K1"] = "48"
os.environ["GF_K2L"] = "96"
os.environ["GF_K3"] = "128"
os.environ["GF_KTOP"] = "1000"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "celeba_faces"))  # noqa: E402

from run_4layer_topk import (  # noqa: E402
    CODE3_DIM, G1, G2, G3, K1, K2, K3, W2_STR, W2_WIN, W3_STR, W3_WIN,
    expand, render_pixels,
)
from run_celeba_faces import harden_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "2026_08_23" / "celeba_big" / "results"


class Bank:
    def __init__(self, W):
        self.W = W
        self.k = W.shape[0]


def crop_active(img, pad=3):
    a = np.abs(img)
    if a.max() <= 0:
        return img
    ys, xs = np.where(a > 0.15 * a.max())
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def main():
    w = np.load(OUT / "weights.npz")
    b1, b2, b3 = Bank(w["W1"]), Bank(w["W2"]), Bank(w["W3"])
    Wtop, wins = w["Wtop"], w["top_wins"]

    fig, axes = plt.subplots(6, 8, figsize=(8, 6.4))
    for t, ax in enumerate(axes.flat):
        ax.imshow(b1.W[t].reshape(4, 4), cmap="gray")
        ax.axis("off")
    fig.suptitle("L1 — the 48 stroke templates (4x4)")
    fig.tight_layout()
    fig.savefig(OUT / "l1_strokes.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(8, 12, figsize=(13, 9))
    for u, ax in enumerate(axes.flat):
        if u < K2:
            c2 = np.zeros((G2, G2, K2))
            c2[G2 // 2, G2 // 2, u] = 1.0
            m1 = harden_map(expand(c2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
            ax.imshow(crop_active(render_pixels(m1, b1)), cmap="gray")
        ax.axis("off")
    fig.suptitle("L2 — all 96 motif units (rendered, cropped to RF)")
    fig.tight_layout()
    fig.savefig(OUT / "l2_motifs.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(8, 16, figsize=(16, 8.5))
    for u, ax in enumerate(axes.flat):
        if u < K3:
            c3 = np.zeros((G3, G3, K3))
            c3[G3 // 2, G3 // 2, u] = 1.0
            m2 = harden_map(expand(c3, b3, W3_WIN, W3_STR, (G2, G2, K2)))
            m1 = harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
            ax.imshow(crop_active(render_pixels(m1, b1)), cmap="gray")
        ax.axis("off")
    fig.suptitle("L3 — all 128 part units (rendered, cropped to RF)")
    fig.tight_layout()
    fig.savefig(OUT / "l3_parts.png", dpi=110)
    plt.close(fig)

    order = np.argsort(-wins)[:24]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax, m in zip(axes.flat, order):
        code3 = np.maximum(Wtop[m, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = harden_map(expand(harden_map(code3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
        m1 = harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
        ax.imshow(render_pixels(m1, b1), cmap="gray")
        ax.set_title(f"m{m} ({int(wins[m])} wins)", fontsize=7)
        ax.axis("off")
    fig.suptitle("The archive — 24 most-rehearsed stored faces")
    fig.tight_layout()
    fig.savefig(OUT / "archive_sample.png", dpi=110)
    plt.close(fig)
    print(f"anatomy figures written to {OUT}")


if __name__ == "__main__":
    main()
