"""Render L1 template palettes: 4x4 flagship (36) vs 8x8 K=64 and K=512.

L1 learning is independent of output mode and upper layers, so training
L1 alone on the same seed/stream reproduces the experiments' palettes
exactly. Also renders nearest-neighbor family panels (anchor + its 7
closest templates by cosine) — the visual form of the near-tie story.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_gpu_minibatch import B, DTYPE, Dict, XP_NAME, level_pass, xp  # noqa: E402
from run_4layer_topk import ETA1, SIDE, TRAIN_N, load_data  # noqa: E402

OUT = ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8" / "results"


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def train_l1(k1, win):
    g = SIDE - win + 1
    pos = [(r, c) for r in range(g) for c in range(g)]
    d1 = Dict(k1, win * win, ETA1)
    Xtr, _, _, _ = load_data()
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    for _ in range(2):
        for s in range(0, TRAIN_N, B):
            level_pass(Xtr_x[s:s + B][..., None], d1, pos, win, 1, g, False, True)
    return to_np(d1.W), to_np(d1.win_counts)


def gallery(W, wins, win, title, fname, max_n=120, cols=12):
    order = np.argsort(wins)[::-1][:max_n]
    n = len(order)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.85, rows * 0.95))
    for ax in np.ravel(axes):
        ax.axis("off")
    for i, t in enumerate(order):
        ax = np.ravel(axes)[i]
        p = W[t].reshape(win, win)
        m = np.abs(p).max() + 1e-9
        ax.imshow(p, cmap="gray", vmin=-m, vmax=m)
        ax.set_title(f"w{wins[t]}", fontsize=5)
    shown = f"top {n} of {len(W)} by wins" if n < len(W) else f"all {len(W)}"
    fig.suptitle(f"{title} ({shown})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def families(W, wins, win, title, fname, n_anchors=4):
    C = W @ W.T
    anchors = np.argsort(wins)[::-1][:n_anchors]
    fig, axes = plt.subplots(n_anchors, 8, figsize=(8 * 1.0, n_anchors * 1.15))
    for r, a in enumerate(anchors):
        nb = np.argsort(C[a])[::-1][:8]          # self first, then 7 nearest
        for c, t in enumerate(nb):
            ax = axes[r, c]
            p = W[t].reshape(win, win)
            m = np.abs(p).max() + 1e-9
            ax.imshow(p, cmap="gray", vmin=-m, vmax=m)
            ax.set_title("anchor" if t == a else f"{C[a, t]:.2f}", fontsize=6)
            ax.axis("off")
    fig.suptitle(f"{title} — anchor + 7 nearest by cosine", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(ROOT / "experiments/2026_08_23/gpu_minibatch/"
                "results/stride1/weights.npz")
    W36 = z["W1"]
    wins36 = np.arange(len(W36))[::-1]  # no saved counts; keep given order
    gallery(W36, wins36, 4, "4x4 flagship palette, K=36", "templates_4x4_K36.png",
            max_n=36, cols=9)
    families(W36, wins36, 4, "4x4 K=36", "families_4x4_K36.png", n_anchors=3)

    for k1 in (64, 512):
        W, wins = train_l1(k1, 8)
        gallery(W, wins, 8, f"8x8 palette, K={k1}", f"templates_8x8_K{k1}.png",
                max_n=(64 if k1 == 64 else 120), cols=(8 if k1 == 64 else 12))
        families(W, wins, 8, f"8x8 K={k1}", f"families_8x8_K{k1}.png", n_anchors=4)
        C = W @ W.T
        off = C[~np.eye(len(C), dtype=bool)]
        print(f"K={k1}: mean|cos|={np.abs(off).mean():.3f} "
              f"max cos={off.max():.3f}")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
