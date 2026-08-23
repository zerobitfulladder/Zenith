"""Inpainting via constellation completion: masked digit + label -> filled digit.

Uses the saved stride-1 MNIST flagship (no training). The masked image is
encoded (masked windows fall below the contrast floor -> silent); the top
memory is queried with [partial code ; label]; the winner's stored
constellation supplies the code at masked grid positions, the input keeps
its own at visible ones; the merged constellation renders through the
hardened dual-mode chain.

Run:  .venv/bin/python experiments/2026_08_23/inpaint/run_inpaint.py
"""

import os
from pathlib import Path

os.environ["GF_REPORT"] = "sparselearn"
os.environ["GF_W1_STR"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from run_4layer_topk import (  # noqa: E402
    CODE3_DIM, G1, G2, G3, K1, K2, K3, LAM,
    K_OUT1, K_OUT2, K_OUT3, W2_STR, W2_WIN, W3_STR, W3_WIN,
    encode_over_batch, encode_pixels_batch, expand, render_pixels, load_data,
)
from gain_feedback import center_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
WDIR = ROOT / "experiments" / "2026_08_23" / "gpu_minibatch" / "results" / "stride1"
OUT = ROOT / "experiments" / "2026_08_23" / "inpaint" / "results"


class Bank:
    def __init__(self, W):
        self.W = W
        self.k = W.shape[0]


def harden(m, k=1):
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            seg = np.maximum(m[a, b], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b, idx] = seg[idx]
    return o


def render_code3(code3, b1, b2, b3):
    m2 = harden(expand(harden(code3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
    m1 = harden(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
    return render_pixels(m1, b1)


def l3_center_pixels():
    """Pixel-center of each L3 grid position (stride-1 L1 geometry)."""
    centers = np.zeros((G3, G3, 2))
    for r in range(G3):
        for c in range(G3):
            centers[r, c] = (2 * r + 5, 2 * c + 5)
    return centers


MASKS = {
    "bottom": lambda: (slice(14, 28), slice(0, 28)),
    "top": lambda: (slice(0, 14), slice(0, 28)),
    "left": lambda: (slice(0, 28), slice(0, 14)),
    "right": lambda: (slice(0, 28), slice(14, 28)),
    "center": lambda: (slice(8, 20), slice(8, 20)),
}
PLAN = ["bottom", "top", "left", "right", "bottom", "center", "top", "right"]


def main():
    w = np.load(WDIR / "weights.npz")
    b1, b2, b3 = Bank(w["W1"]), Bank(w["W2"]), Bank(w["W3"])
    Wtop = w["Wtop"]

    _, _, Xte, yte = load_data()
    idx = [int(np.where(yte == d)[0][0]) for d in range(8)]
    centers = l3_center_pixels()

    fig, axes = plt.subplots(4, 8, figsize=(14, 7.6))
    for col, (ti, mname) in enumerate(zip(idx, PLAN)):
        x = Xte[ti].copy()
        label = int(yte[ti])
        ms = MASKS[mname]()
        xm = x.copy()
        xm[ms] = 0.0

        # Encode the masked image, get its L3 constellation.
        M1 = encode_pixels_batch(xm[None], b1, K_OUT1)
        M2 = encode_over_batch(M1, b2, W2_WIN, W2_STR, G2, K_OUT2)
        M3 = encode_over_batch(M2, b3, W3_WIN, W3_STR, G3, K_OUT3)
        code = M3[0]

        # Retrieve with partial code + label.
        h_hat, _ = center_norm(code.reshape(-1))
        lab = np.zeros(10)
        if not os.environ.get("GF_NOLABEL"):
            lab[label] = LAM
        z_hat, _ = center_norm(np.concatenate([h_hat, lab]))
        winner = int(np.argmax(Wtop @ z_hat))
        mem = np.maximum(Wtop[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)

        # Merge: masked L3 positions take the memory's entries — rescaled to
        # the visible code's loudness (memory rows are unit-normalized, so
        # their raw entries are ~2% of encoded-correlation scale and would
        # be squelched at render).
        vis_scale = code[code > 0].mean() if (code > 0).any() else 1.0
        mem_scale = mem[mem > 0].mean() if (mem > 0).any() else 1.0
        mem_scaled = mem * (vis_scale / max(mem_scale, 1e-9))
        mask_img = np.zeros((28, 28), bool)
        mask_img[ms] = True
        merged = code.copy()
        for r in range(G3):
            for c in range(G3):
                py, px = centers[r, c].astype(int)
                if mask_img[py, px]:
                    merged[r, c] = mem_scaled[r, c]

        axes[0, col].imshow(x, cmap="gray")
        axes[1, col].imshow(xm, cmap="gray")
        axes[2, col].imshow(render_code3(mem, b1, b2, b3), cmap="gray")
        axes[3, col].imshow(render_code3(merged, b1, b2, b3), cmap="gray")
        axes[0, col].set_title(f"{label} ({mname})", fontsize=8)
        for r in range(4):
            axes[r, col].axis("off")
    for r, name in enumerate(["original", "masked input", "memory's guess", "FILLED (merged)"]):
        axes[r, 0].set_ylabel(name, fontsize=8)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    tag = "_nolabel" if os.environ.get("GF_NOLABEL") else ""
    title = ("Inpainting from the masked image ALONE (no label)"
             if tag else "Inpainting: masked digit + label -> constellation-level fill")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / f"inpaint{tag}.png", dpi=110)
    print(f"figure written to {OUT / f'inpaint{tag}.png'}")


if __name__ == "__main__":
    main()
