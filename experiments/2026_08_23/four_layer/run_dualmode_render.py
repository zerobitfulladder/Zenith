"""Dual-mode test: same weights, recognition reads graded, generation reads
hardened (top-1 per position at the memory read and at every expansion).

No training — loads four_layer/results/reluall/weights.npz (the best
recognizer, whose squelch-mode generation is mush) and renders generation and
round-trip reconstruction in three read modes: squelch (baseline mush),
top-3-hardened, top-1-hardened.

Run:  .venv/bin/python experiments/2026_08_23/four_layer/run_dualmode_render.py
"""

import os
from pathlib import Path

# Which trained rig to render: "reluall" (default) or "topk".
SRC = os.environ.get("GF_SRC", "reluall")
os.environ["GF_REPORT"] = SRC   # must be set before the import below

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import center_norm
from run_4layer_topk import (
    CODE3_DIM,
    G1,
    G2,
    G3,
    K1,
    K2,
    K3,
    K_OUT1,
    K_OUT2,
    K_OUT3,
    LAM,
    W2_STR,
    W2_WIN,
    W3_STR,
    W3_WIN,
    encode_over_batch,
    encode_pixels_batch,
    expand,
    render_pixels,
    load_data,
)

ROOT = Path(__file__).resolve().parents[3]
WDIR = ROOT / "experiments" / "2026_08_23" / "four_layer" / "results" / SRC


class Bank:
    def __init__(self, W):
        self.W = W
        self.k = W.shape[0]


def harden(map3d, k_keep):
    """Per position keep only the k_keep strongest entries (magnitudes kept)."""
    out = np.zeros_like(map3d)
    g = map3d.shape[0]
    for gi in range(g):
        for gj in range(g):
            seg = np.maximum(map3d[gi, gj], 0.0)
            if seg.max() <= 0:
                continue
            order = np.argsort(seg)[::-1][:k_keep]
            out[gi, gj, order] = seg[order]
    return out


def render_mode(code3, d1, d2, d3, mode):
    """mode: 'squelch' (as trained/rendered today), or an int k -> harden to
    top-k per position at the memory read and after every expansion."""
    if mode == "squelch":
        m3 = code3
    else:
        m3 = harden(code3, mode)
    m2 = expand(m3, d3, W3_WIN, W3_STR, (G2, G2, K2))
    if mode != "squelch":
        m2 = harden(m2, mode)
    m1 = expand(m2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    if mode != "squelch":
        m1 = harden(m1, mode)
    return render_pixels(m1, d1)


def main():
    w = np.load(WDIR / "weights.npz")
    d1, d2, d3 = Bank(w["W1"]), Bank(w["W2"]), Bank(w["W3"])
    Wtop = w["Wtop"]
    label_half = Wtop[:, CODE3_DIM:]
    owner = label_half.argmax(axis=1)

    modes = ["squelch", 3, 1]
    names = {"squelch": "graded read (as before)", 3: "hardened top-3", 1: "hardened top-1"}

    # Generation in all modes.
    fig, axes = plt.subplots(len(modes), 10, figsize=(16, 2.1 * len(modes) + 0.8))
    for mi, mode in enumerate(modes):
        for j in range(10):
            label = np.zeros(10)
            label[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), label]))
            winner = int(np.argmax(Wtop @ z_hat))
            code3 = np.maximum(Wtop[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
            ax = axes[mi, j]
            ax.imshow(render_mode(code3, d1, d2, d3, mode), cmap="gray")
            if mi == 0:
                ax.set_title(str(j), fontsize=9)
            if j == 0:
                ax.set_ylabel(names[mode], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Same weights, different read modes — label-only generation")
    fig.tight_layout()
    fig.savefig(WDIR / "dualmode_generation.png", dpi=110)
    plt.close(fig)

    # Round-trip reconstruction: encode graded (matched to training), render per mode.
    _, _, Xte, _ = load_data()
    M1 = encode_pixels_batch(Xte[:8], d1, K_OUT1)
    M2 = encode_over_batch(M1, d2, W2_WIN, W2_STR, G2, K_OUT2)
    M3 = encode_over_batch(M2, d3, W3_WIN, W3_STR, G3, K_OUT3)

    fig, axes = plt.subplots(1 + len(modes), 8, figsize=(14, 2.0 * (1 + len(modes)) + 0.6))
    for k in range(8):
        axes[0, k].imshow(Xte[k], cmap="gray")
        axes[0, k].axis("off")
        for mi, mode in enumerate(modes):
            ax = axes[1 + mi, k]
            ax.imshow(render_mode(M3[k], d1, d2, d3, mode), cmap="gray")
            if k == 0:
                ax.set_ylabel(names[mode], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Round-trip pixels->L3->pixels: input row, then each read mode")
    fig.tight_layout()
    fig.savefig(WDIR / "dualmode_roundtrip.png", dpi=110)
    plt.close(fig)

    print(f"Figures written to {WDIR}")


if __name__ == "__main__":
    main()
