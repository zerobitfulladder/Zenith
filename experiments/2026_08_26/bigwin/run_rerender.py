"""Diagnose the big-window generation mush: too many overlapping voices?

Hypothesis: each L1 position sums ~16 overlapping 15x15 L2 hypotheses;
their disagreement muddies the vote before hardening. If true, rendering
from only the most confident L2 positions should sharpen the digits.

Re-renders label-only generation from bigwin/results/weights.npz with
the top {49 (all), 25, 12, 6} L2 positions by stored confidence, zeroing
the rest. CPU only, no retraining.

Run:  .venv/bin/python experiments/2026_08_26/bigwin/run_rerender.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ.setdefault("GF_KTOP", "400")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_4layer_topk import LAM, expand  # noqa: E402
from run_rich_palette_8x8 import G1, W1, render8  # noqa: E402
from gain_feedback import center_norm  # noqa: E402

RESULTS = ROOT / "experiments" / "2026_08_26" / "bigwin" / "results"
K1, K2 = 1024, 2048
W2, S2 = 8, 2
G2 = 7
CODE_DIM = G2 * G2 * K2
KEEPS = [49, 25, 12, 6]


def _harden_map(m, k=1):
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b_ in range(m.shape[1]):
            seg = np.maximum(m[a, b_], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b_, idx] = seg[idx]
    return o


def main():
    wz = np.load(RESULTS / "weights.npz")
    W1n, W2n, Wtn = wz["W1"], wz["W2"], wz["Wtop"]

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2 = Bank(W2n)
    fig, axes = plt.subplots(len(KEEPS), 10,
                             figsize=(10.5, 1.15 * len(KEEPS) + 0.6))
    for ai, keep in enumerate(KEEPS):
        for j in range(10):
            lab = np.zeros(10)
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
            row = Wtn[int(np.argmax(Wtn @ z_hat))]
            c2 = _harden_map(np.maximum(row[:CODE_DIM], 0.0).reshape(G2, G2, K2))
            conf = c2.max(axis=2)
            thresh = np.sort(conf.ravel())[::-1][keep - 1]
            c2[conf < thresh] = 0.0
            m1 = _harden_map(expand(c2, b2, W2, S2, (G1, G1, K1)))
            ax = axes[ai, j]
            ax.imshow(render8(m1, W1n), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[ai, 0].set_ylabel(f"top {keep}\npos", fontsize=8, rotation=0,
                               ha="right", va="center")
    fig.suptitle("Big-window generation vs number of contributing L2 positions")
    fig.tight_layout()
    fig.savefig(RESULTS / "generation_sparse_positions.png", dpi=110)
    plt.close(fig)
    print("Wrote", RESULTS / "generation_sparse_positions.png")


if __name__ == "__main__":
    main()
