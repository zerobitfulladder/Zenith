"""CelebA finale: the flagship rig on faces, with the composition test.

Consolidated unit, stride-1, GPU mini-batch, 48x48 grayscale faces,
top layer over [L3 code ; 0.5 * 40-attribute multi-hot]. Deliverables:
attribute-conditioned portraits, attribute-delta heatmaps
(mustache-delta must localize on the upper lip — the falsifiable
prediction), and woman+mustache composition — a combination that occurs
ZERO times in the training data, so retrieval cannot fake it.

Run:  .venv/bin/python experiments/2026_08_23/celeba_faces/run_celeba_faces.py
"""

import json
import os
import time
from pathlib import Path

os.environ["GF_REPORT"] = "sparselearn"
os.environ["GF_SIDE"] = "48"
os.environ["GF_W1_STR"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from run_gpu_minibatch import (  # noqa: E402
    B,
    Dict,
    XP_NAME,
    _center_norm_rows,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    CODE3_DIM,
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    G1,
    G2,
    G3,
    K1,
    K2,
    K3,
    KTOP,
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
)
from gain_feedback import center_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "2026_08_23" / "celeba_faces" / "results"
EPOCHS = 2
N_ATTR = 10   # attributes appended to the label half? no — full 40 below
DELTA_N = 1000   # faces encoded (CPU) for delta computation

ATTR_NAMES = json.loads(
    (ROOT / "data/celeba/attr_names.json").read_text())
A_IDX = {n: i for i, n in enumerate(ATTR_NAMES)}
LABEL_DIM = len(ATTR_NAMES)
TOP_DIM = CODE3_DIM + LABEL_DIM

PORTRAITS = [
    ["Male", "No_Beard"], ["Male", "Mustache"], ["Male", "Eyeglasses"],
    ["Male", "Wearing_Hat"], ["Wearing_Lipstick", "Smiling"],
    ["Heavy_Makeup", "Blond_Hair"], ["Wearing_Lipstick", "Wavy_Hair"],
    ["Bald", "Male"], ["Smiling", "Male"], ["Heavy_Makeup", "Young"],
]


def attr_query(names):
    v = np.zeros(LABEL_DIM)
    for n in names:
        v[A_IDX[n]] = LAM
    return v


def harden_map(m, k=1):
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            seg = np.maximum(m[a, b], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b, idx] = seg[idx]
    return o


def render_code3(code3, b1, b2, b3):
    m2 = harden_map(expand(harden_map(code3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
    m1 = harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
    return render_pixels(m1, b1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / "data/celeba/images_48.npy")
    A = np.load(ROOT / "data/celeba/attrs_48.npy").astype(np.float64)
    N = len(X)
    print(f"faces={N} side={X.shape[1]} grids G1={G1} G2={G2} G3={G3} "
          f"code3={CODE3_DIM} top={TOP_DIM}", flush=True)

    d1 = Dict(K1, 16, ETA1)
    d2 = Dict(K2, W2_WIN * W2_WIN * K1, ETA2)
    d3 = Dict(K3, W3_WIN * W3_WIN * K2, ETA3)
    top = Dict(KTOP, TOP_DIM, ETATOP)

    Xx = xp.asarray(X.astype(np.float64))
    Lx = xp.asarray(A * LAM)

    t0 = time.time()
    for ep in range(EPOCHS):
        for s in range(0, N, B):
            xb = Xx[s:s + B]
            lb = Lx[s:s + B]
            m1 = level_pass(xb[..., None], d1, POS1(), 4, 1, G1, False, True)
            m2 = level_pass(m1, d2, POS2(), W2_WIN, K1, G2, True, True)
            m3 = level_pass(m2, d3, POS3(), W3_WIN, K2, G3, True, True)
            H = m3.reshape(len(xb), -1)
            H, okh = _center_norm_rows(H)
            Z = xp.concatenate([H, lb], axis=1)
            Z, okz = _center_norm_rows(Z)
            okz = okz & okh
            if top.n_boot < top.k:
                top.bootstrap(Z[okz])
            if top.n_boot >= top.k:
                Ct = Z @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cvals = xp.max(Ct, axis=1) * okz
                keep = cvals > 0
                top.update(Z[keep], winners[keep], cvals[keep])
        print(f"epoch {ep + 1} done {time.time() - t0:.0f}s", flush=True)
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    W1n, W2n, W3n, Wtn = map(to_np, (d1.W, d2.W, d3.W, top.W))
    np.savez(OUT / "weights.npz", W1=W1n, W2=W2n, W3=W3n, Wtop=Wtn,
             top_wins=to_np(top.win_counts))

    class Bank:
        def __init__(self, W):
            self.W = W
            self.k = W.shape[0]

    b1, b2, b3 = Bank(W1n), Bank(W2n), Bank(W3n)

    # ---- Portraits: attribute-conditioned retrieval + hardened render -----
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    for ax, names in zip(axes.flat, PORTRAITS):
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), attr_query(names)]))
        winner = int(np.argmax(Wtn @ z_hat))
        code3 = np.maximum(Wtn[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        ax.imshow(render_code3(code3, b1, b2, b3), cmap="gray")
        ax.set_title("+".join(n.replace("Wearing_", "") for n in names), fontsize=7)
        ax.axis("off")
    fig.suptitle("Attribute-conditioned portraits (hardened read)")
    fig.tight_layout()
    fig.savefig(OUT / "portraits.png", dpi=110)
    plt.close(fig)

    # ---- Encode a subset (CPU) for attribute deltas -----------------------
    sub = X[:DELTA_N].astype(np.float64)
    M1 = encode_pixels_batch(sub, b1, K_OUT1)
    M2 = encode_over_batch(M1, b2, W2_WIN, W2_STR, G2, K_OUT2)
    M3 = encode_over_batch(M2, b3, W3_WIN, W3_STR, G3, K_OUT3)
    C3 = M3.reshape(DELTA_N, -1)
    Asub = A[:DELTA_N]

    def delta(attr):
        i = A_IDX[attr]
        pos = C3[Asub[:, i] > 0.5]
        neg = C3[Asub[:, i] < 0.5]
        return pos.mean(axis=0) - neg.mean(axis=0), len(pos)

    d_must, n_must = delta("Mustache")
    d_glass, n_glass = delta("Eyeglasses")
    print(f"delta support: mustache n={n_must}, glasses n={n_glass}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    for ax, (dv, name) in zip(axes, [(d_must, "Mustache delta"), (d_glass, "Eyeglasses delta")]):
        img = render_code3(np.maximum(dv, 0.0).reshape(G3, G3, K3), b1, b2, b3)
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Attribute deltas rendered — do they localize?")
    fig.tight_layout()
    fig.savefig(OUT / "delta_heatmaps.png", dpi=110)
    plt.close(fig)

    # ---- Composition: female base + mustache delta ------------------------
    fem = (Asub[:, A_IDX["Male"]] < 0.5)
    base_fem = C3[fem].mean(axis=0)
    z_hat, _ = center_norm(np.concatenate(
        [np.zeros(CODE3_DIM), attr_query(["Wearing_Lipstick", "Smiling"])]))
    tmpl_fem = np.maximum(Wtn[int(np.argmax(Wtn @ z_hat)), :CODE3_DIM], 0.0)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for row, (base, bname) in enumerate([(base_fem, "female mean code"),
                                         (tmpl_fem, "female memory")]):
        bnorm = np.linalg.norm(base)
        dm = d_must / (np.linalg.norm(d_must) + 1e-9)
        for col, beta in enumerate([0.0, 0.3, 0.6, 1.0]):
            comp = np.maximum(base + beta * bnorm * dm, 0.0).reshape(G3, G3, K3)
            axes[row, col].imshow(render_code3(comp, b1, b2, b3), cmap="gray")
            if row == 0:
                axes[row, col].set_title(f"+{beta} mustache", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(bname, fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Composition: woman + mustache (0 such faces exist in the data)")
    fig.tight_layout()
    fig.savefig(OUT / "composition.png", dpi=110)
    plt.close(fig)

    print(f"RESULT train_s={train_s:.0f} figures in {OUT}", flush=True)
    (OUT / "report.md").write_text(
        f"# CelebA faces on the flagship rig\n\ntrain {train_s:.0f}s, "
        f"{N} faces, grids {G1}/{G2}/{G3}, code3 {CODE3_DIM}.\n"
        f"delta support: mustache {n_must}, glasses {n_glass} (of {DELTA_N}).\n"
        "Figures: portraits.png, delta_heatmaps.png, composition.png\n")


def POS1():
    return [(r, c) for r in range(0, 48 - 4 + 1, 1) for c in range(0, 48 - 4 + 1, 1)]


def POS2():
    return [(r, c) for r in range(0, G1 - W2_WIN + 1, W2_STR) for c in range(0, G1 - W2_WIN + 1, W2_STR)]


def POS3():
    return [(r, c) for r in range(0, G2 - W3_WIN + 1, W3_STR) for c in range(0, G2 - W3_WIN + 1, W3_STR)]


if __name__ == "__main__":
    main()
