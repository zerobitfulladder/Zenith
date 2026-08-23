"""Chunked evaluation of the trained 8-level pyramid (weights from disk).

Probes (Male) at L4-L7, unit galleries at L5/L6, roundtrip with luminance,
archive sample. Encoding runs on GPU in chunks of 128 to fit 6 GB VRAM.

Run:  .venv/bin/python experiments/2026_08_23/pyramid/run_pyramid_eval.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from run_pyramid import (  # noqa: E402  (its main is guarded)
    CH, CODE_DIM, GRIDS, LEVELS, SIDE, Dict, ETAS, ETATOP, KTOP,
    encode_stack, harden_np, render_down, xp, XP_NAME,
)

import os

ROOT = Path(__file__).resolve().parents[3]
OUT = (ROOT / "experiments" / "2026_08_23"
       / os.environ.get("GF_OUT", "pyramid/results"))
EV = 3000
CHUNK = 64

ATTR_NAMES = json.loads((ROOT / "data/celeba/attr_names.json").read_text())
A_IDX = {n: i for i, n in enumerate(ATTR_NAMES)}


def main():
    w = np.load(OUT / "weights.npz")
    dims = [16] + [LEVELS[li][0] ** 2 * CH[li - 1] for li in range(1, len(LEVELS))]
    dics = [Dict(LEVELS[i][2], dims[i], ETAS[i]) for i in range(len(LEVELS))]
    for i, d in enumerate(dics):
        d.W = xp.asarray(w[f"W{i+1}"], dtype=xp.float32)
        d.n_boot = d.k
    Wtn, wins = w["Wtop"], w["top_wins"]

    X = np.load(ROOT / "data/celeba/images_48.npy")
    A = np.load(ROOT / "data/celeba/attrs_48.npy").astype(np.float64)

    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    codes = {li: [] for li in range(max(1, len(LEVELS) - 3), len(LEVELS))}
    top_maps = []
    for s in range(0, EV, CHUNK):
        e = min(s + CHUNK, EV)
        maps = encode_stack(xp.asarray(X[s:e].astype(np.float64)), dics, learning=False)
        for li in codes:
            codes[li].append(to_np(maps[li]).reshape(len(maps[li]), -1).astype(np.float32))
        if s < 8:
            top_maps = to_np(maps[-1])[:8]
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
    codes = {li: np.concatenate(v) for li, v in codes.items()}

    male = A[:EV, A_IDX["Male"]]
    probes = {}
    for li, C in codes.items():
        clf = LogisticRegression(max_iter=500)
        clf.fit(C[:2000], male[:2000])
        probes[li] = float(clf.score(C[2000:], male[2000:]))
    print("male-probes:", {f"L{li+1}": round(p, 4) for li, p in probes.items()}, flush=True)

    class Bank:
        pass

    dics_np = []
    for i in range(len(LEVELS)):
        b = Bank()
        b.W = w[f"W{i+1}"]
        b.k = LEVELS[i][2]
        dics_np.append(b)

    from run_4layer_topk import expand  # generic

    def render_from_level(li, unit):
        g = GRIDS[li]
        m = np.zeros((g, g, LEVELS[li][2]))
        m[g // 2, g // 2, unit] = 1.0
        for lj in range(li, 0, -1):
            win, stride, _ = LEVELS[lj]
            gp = GRIDS[lj - 1]
            m = expand(harden_np(m), dics_np[lj], win, stride, (gp, gp, CH[lj - 1]))
            m = harden_np(m, lum_ch=(lj - 1 == 0))
        k1 = LEVELS[0][2]
        num = np.zeros((SIDE, SIDE))
        den = np.zeros((SIDE, SIDE))
        F = np.outer([0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5])
        for gi in range(GRIDS[0]):
            for gj in range(GRIDS[0]):
                seg = np.maximum(m[gi, gj, :k1], 0.0)
                if seg.max() <= 0:
                    continue
                idx = int(np.argmax(seg))
                patch = dics_np[0].W[idx].reshape(4, 4) * seg[idx]
                num[gi:gi + 4, gj:gj + 4] += patch * F * seg[idx]
                den[gi:gi + 4, gj:gj + 4] += F * seg[idx]
        img = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        a2 = np.abs(img)
        if a2.max() > 0:
            ys, xs = np.where(a2 > 0.15 * a2.max())
            img = img[max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
        return img

    gallery_levels = [(len(LEVELS) - 2, 48), (len(LEVELS) - 1, 60)]
    for li, count in gallery_levels:
        cols = 12
        rows = int(np.ceil(count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
        for u, ax in enumerate(np.asarray(axes).flat):
            if u < count:
                ax.imshow(render_from_level(li, u % LEVELS[li][2]), cmap="gray")
            ax.axis("off")
        fig.suptitle(f"L{li+1} units")
        fig.tight_layout()
        fig.savefig(OUT / f"units_L{li+1}.png", dpi=110)
        plt.close(fig)
        print(f"units_L{li+1} done", flush=True)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        axes[0, i].imshow(X[i], cmap="gray")
        axes[1, i].imshow(render_down(top_maps[i], dics_np), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Roundtrip through 7 levels (with luminance)")
    fig.tight_layout()
    fig.savefig(OUT / "roundtrip.png", dpi=110)
    plt.close(fig)

    g = GRIDS[-1]
    order = np.argsort(-wins)[:12]
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for ax, mi in zip(axes.flat, order):
        code = np.maximum(Wtn[mi, :CODE_DIM], 0.0).reshape(g, g, LEVELS[-1][2])
        ax.imshow(render_down(code, dics_np), cmap="gray")
        ax.set_title(f"m{mi} ({int(wins[mi])})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Archive sample (8-level, luminance)")
    fig.tight_layout()
    fig.savefig(OUT / "archive_sample.png", dpi=110)
    plt.close(fig)

    (OUT / "report.md").write_text(
        f"# 8-level pyramid eval\n\nMale-probes: "
        f"{ {f'L{li+1}': round(p, 4) for li, p in probes.items()} }\n"
        "Training: 12 epochs, no early stop — settling cascade (L1 frozen by "
        "ep3, L5-L7 still drifting 0.2-0.4 rad/epoch at ep12).\n"
        "Figures: units_L5.png, units_L6.png, roundtrip.png, archive_sample.png\n")
    print("RESULT eval done", flush=True)


if __name__ == "__main__":
    main()
