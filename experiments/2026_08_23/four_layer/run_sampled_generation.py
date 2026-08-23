"""Sampled generation: third read mode on the trained reluall weights.

Per position, the stored profile is treated as an (unnormalized)
distribution and ONE template is sampled with probability proportional to
coefficient^(1/T); sampling repeats after each downward expansion. Fresh
randomness per ask -> different complete digits for the same label.
T -> 0 recovers the deterministic argmax prototype. No training happens.

Run:  .venv/bin/python experiments/2026_08_23/four_layer/run_sampled_generation.py
"""

import os
from pathlib import Path

os.environ["GF_REPORT"] = "reluall"

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
    LAM,
    W2_STR,
    W2_WIN,
    W3_STR,
    W3_WIN,
    expand,
    render_pixels,
)

ROOT = Path(__file__).resolve().parents[3]
WDIR = ROOT / "experiments" / "2026_08_23" / "four_layer" / "results" / "reluall"

N_SAMPLES = 5
TEMP = 0.5
TEMP_ROWS = [1.0, 0.5, 0.33]


class Bank:
    def __init__(self, W):
        self.W = W
        self.k = W.shape[0]


def sample_map(map3d, rng, temp):
    """Per position, sample one entry with p ~ coefficient^(1/temp).
    temp <= 0 means argmax. Chosen entry keeps its stored magnitude."""
    out = np.zeros_like(map3d)
    g = map3d.shape[0]
    for gi in range(g):
        for gj in range(g):
            seg = np.maximum(map3d[gi, gj], 0.0)
            m = seg.max()
            if m <= 0:
                continue
            if temp <= 0:
                w = int(np.argmax(seg))
            else:
                p = (seg / m) ** (1.0 / temp)
                p /= p.sum()
                w = int(rng.choice(len(seg), p=p))
            out[gi, gj, w] = seg[w] if seg[w] > 0 else m
    return out


def generate(code3, d1, d2, d3, rng, temp, sample_below=True):
    """sample_below=False: stochastic choice only at the top (structural
    variation among stored parts), argmax below (coherent rendering)."""
    m3 = sample_map(code3, rng, temp)
    m2 = expand(m3, d3, W3_WIN, W3_STR, (G2, G2, K2))
    m2 = sample_map(m2, rng, temp if sample_below else 0.0)
    m1 = expand(m2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    m1 = sample_map(m1, rng, temp if sample_below else 0.0)
    return render_pixels(m1, d1)


def class_code(Wtop, j):
    label = np.zeros(10)
    label[j] = LAM
    z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), label]))
    winner = int(np.argmax(Wtop @ z_hat))
    return np.maximum(Wtop[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)


def main():
    w = np.load(WDIR / "weights.npz")
    d1, d2, d3 = Bank(w["W1"]), Bank(w["W2"]), Bank(w["W3"])
    Wtop = w["Wtop"]

    # Figure 1: per class — argmax prototype, then N samples at TEMP.
    fig, axes = plt.subplots(10, 1 + N_SAMPLES, figsize=(1.7 * (1 + N_SAMPLES), 16.5))
    for j in range(10):
        code3 = class_code(Wtop, j)
        rng = np.random.default_rng(1000 + j)
        axes[j, 0].imshow(generate(code3, d1, d2, d3, rng, temp=0.0), cmap="gray")
        for s in range(N_SAMPLES):
            axes[j, 1 + s].imshow(
                generate(code3, d1, d2, d3, rng, temp=TEMP, sample_below=False), cmap="gray")
        for ax in axes[j]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[j, 0].set_ylabel(str(j), rotation=0, fontsize=11, labelpad=12)
    axes[0, 0].set_title("argmax", fontsize=9)
    for s in range(N_SAMPLES):
        axes[0, 1 + s].set_title(f"sample {s + 1}", fontsize=9)
    fig.suptitle(f"Sample-at-top generation, T={TEMP} — structural variety, hardened rendering")
    fig.tight_layout()
    fig.savefig(WDIR / "sampled_variety_top.png", dpi=110)
    plt.close(fig)

    # Figure 2: temperature sweep for one class (7).
    code3 = class_code(Wtop, 7)
    fig, axes = plt.subplots(len(TEMP_ROWS), 8, figsize=(13, 1.8 * len(TEMP_ROWS) + 0.7))
    for ti, t in enumerate(TEMP_ROWS):
        rng = np.random.default_rng(7)
        for s in range(8):
            axes[ti, s].imshow(generate(code3, d1, d2, d3, rng, temp=t), cmap="gray")
            axes[ti, s].set_xticks([])
            axes[ti, s].set_yticks([])
        axes[ti, 0].set_ylabel(f"T={t}", fontsize=9)
    fig.suptitle("Temperature as a style dial — eight asks for a 7 at each T")
    fig.tight_layout()
    fig.savefig(WDIR / "sampled_temperature.png", dpi=110)
    plt.close(fig)

    print(f"Figures written to {WDIR}")


if __name__ == "__main__":
    main()
