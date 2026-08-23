"""Batch 2: shared stroke dictionary, overlapping windows, feathered rendering.

One K1-template Zenith dictionary is swept across all window positions
(convergent-vocabulary argument: patch statistics repeat across space, so
independent hypercolumns converge to the same vocabulary anyway — sharing is
the shortcut, and every template sees 169x the data).

Windows are 4x4 at stride 2 -> 13x13 = 169 overlapping positions. The code is
per-position top-1 (winner index + correlation), concatenated (dim 6084).
L2 is the concat free-allocation Zenith (K2=200, lam=0.5) from batch 1.

Rendering (generation and reconstruction) is confidence-weighted overlap-add
with a feather window: every pixel is the consensus of the windows covering
it, each weighted by its confidence and by how central the pixel is to that
window. Hard codes, soft canvas — blending happens only at render time and is
never written back to weights.

Run:  .venv/bin/python experiments/2026_08_23/batch2/run_batch2.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import EPS, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000
SIDE = 28
PATCH = 4
STRIDE = 2
K1 = 36                      # shared stroke vocabulary size
K2 = 200                     # top-layer templates (from the K2 sweep verdict)
ETA1 = 0.02                  # shared dictionary gets 169 updates/image — keep steps small
ETA2 = 0.04
EPOCHS = 2
LAM = 0.5
SEED = 42
NORM_FLOOR = 0.15
RENDER_SQUELCH = 0.10   # at render time, a position's coefficients below this
                        # fraction of its strongest are silenced — draw the
                        # memory, not its birth-noise residue
DATASET = os.environ.get("GF_DATASET", "mnist")   # "mnist" or "fashion"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "batch2" / "results" / DATASET
# ---------------------------------------------------------------------------

POS = [(r, c) for r in range(0, SIDE - PATCH + 1, STRIDE)
       for c in range(0, SIDE - PATCH + 1, STRIDE)]          # 13x13 = 169
N_POS = len(POS)
CODE_DIM = N_POS * K1
FEATHER = np.outer([0.5, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 0.5])


def load_data():
    prefix = "fashion_mnist" if DATASET == "fashion" else "mnist"
    X = np.load(ROOT / f"data/mnist/{'fashion' if 'fashion' in prefix else 'digits'}/train_images.npy").astype(np.float64)
    y = np.load(ROOT / f"data/mnist/{'fashion' if 'fashion' in prefix else 'digits'}/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def encode_image(x2d, dic, learning):
    """Sparse per-position top-1 code for one image; optionally trains the
    shared dictionary. Correlations are computed once against the dictionary
    as it stood at the start of the image (within-image staleness is
    negligible at ETA1=0.02)."""
    P = np.stack([x2d[r:r + PATCH, c:c + PATCH].ravel() for r, c in POS])
    Pc = P - P.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Pc, axis=1)
    valid = norms > NORM_FLOOR
    Phat = np.zeros_like(Pc)
    Phat[valid] = Pc[valid] / norms[valid, None]
    C = Phat @ dic.W.T
    code = np.zeros(CODE_DIM)
    for i in np.where(valid)[0]:
        if learning:
            w = dic.learn(Phat[i], C[i])
        else:
            w = int(np.argmax(C[i]))
        code[i * K1 + w] = max(float(C[i, w]), 0.0)
    return code


def encode_batch(X2d, dic):
    N = len(X2d)
    code = np.zeros((N, CODE_DIM))
    for pi, (r, c) in enumerate(POS):
        P = X2d[:, r:r + PATCH, c:c + PATCH].reshape(N, -1)
        P = P - P.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(P, axis=1)
        ok = norms > NORM_FLOOR
        P[ok] /= norms[ok, None]
        C = P @ dic.W.T
        wins = C.argmax(axis=1)
        vals = np.maximum(C[np.arange(N), wins], 0.0) * ok
        code[np.arange(N), pi * K1 + wins] = vals
    return code


def render(code_half, dic):
    """Feathered confidence-weighted overlap-add: pixel = consensus of the
    windows covering it."""
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = float(np.max(code_half)) if np.max(code_half) > 0 else 0.0
    for pi, (r, c) in enumerate(POS):
        seg = np.maximum(code_half[pi * K1:(pi + 1) * K1], 0.0)
        conf = float(seg.max())
        if conf <= 0.0 or conf < RENDER_SQUELCH * peak:
            continue
        seg[seg < RENDER_SQUELCH * conf] = 0.0
        patch = (seg @ dic.W).reshape(PATCH, PATCH)
        num[r:r + PATCH, c:c + PATCH] += patch * FEATHER * conf
        den[r:r + PATCH, c:c + PATCH] += FEATHER * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    rng = np.random.default_rng(SEED)
    dic = ZenithLayer(K1, PATCH * PATCH, ETA1, rng)
    l2 = ZenithLayer(K2, CODE_DIM + 10, ETA2, rng)

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc="batch2", ncols=80):
            code = encode_image(x, dic, learning=True)
            h_hat, h_norm = center_norm(code)
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = LAM
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                l2.learn(z_hat, l2.forward(z_hat))

    # ---- Evaluation -------------------------------------------------------
    label_half = l2.W[:, CODE_DIM:]
    owner = label_half.argmax(axis=1)
    alloc = np.bincount(owner, minlength=10)

    gen_imgs, consistent = [], 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), label]))
        winner = int(np.argmax(l2.forward(z_hat)))
        consistent += int(owner[winner] == j)
        gen_imgs.append(render(l2.W[winner, :CODE_DIM], dic))

    Cte = encode_batch(Xte, dic)
    Ctr = encode_batch(Xtr[:PROBE_N], dic)

    H = Cte - Cte.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    Z = np.concatenate([H, np.zeros((len(H), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ l2.W.T
    winners = C2.argmax(axis=1)
    acc_hard = float((owner[winners] == yte).mean())
    votes = np.maximum(C2, 0.0) @ label_half
    acc_soft = float((votes.argmax(axis=1) == yte).mean())
    counts = np.maximum(np.bincount(owner, minlength=10), 1)
    acc_norm = float(((votes / counts).argmax(axis=1) == yte).mean())
    kv = 10
    part = np.argpartition(-C2, kv, axis=1)[:, :kv]
    mask = np.zeros_like(C2)
    mask[np.arange(len(C2))[:, None], part] = 1.0
    acc_topk = float((((np.maximum(C2, 0.0) * mask) @ label_half).argmax(axis=1) == yte).mean())

    probe = LogisticRegression(max_iter=1000)
    probe.fit(Ctr, ytr[:PROBE_N])
    probe_acc = float(probe.score(Cte, yte))

    p = dic.win_counts / dic.win_counts.sum()
    dic_entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(dic.k))
    dead = int((dic.win_counts == 0).sum())

    print(f"RESULT hard={acc_hard:.4f} soft={acc_soft:.4f} soft_norm={acc_norm:.4f} "
          f"soft_top10={acc_topk:.4f} probe={probe_acc:.4f} consistent={consistent}/10 "
          f"dic_entropy={dic_entropy:.3f} dead={dead}")

    # ---- Figures ----------------------------------------------------------
    fig, axes = plt.subplots(6, 6, figsize=(7, 7.4))
    for t, ax in enumerate(axes.flat):
        ax.imshow(dic.W[t].reshape(PATCH, PATCH), cmap="bwr")
        ax.axis("off")
    fig.suptitle("Shared stroke vocabulary (36 templates, all positions)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shared_dictionary.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[j], cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Label-only generation, feathered overlap-add (lam={LAM})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_labelonly.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for k in range(8):
        code = encode_batch(Xte[k:k + 1], dic)[0]
        axes[0, k].imshow(Xte[k], cmap="gray")
        axes[1, k].imshow(render(code, dic), cmap="gray")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("Test images (top) vs feathered overlap-add reconstruction (bottom)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recon_test.png", dpi=110)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    alloc_s = " ".join(str(a) for a in alloc)
    (OUTPUT_DIR / "report.md").write_text("\n".join([
        "# Batch 2 — shared dictionary, stride-2 overlap, feathered rendering (MNIST)",
        "",
        f"Config: {PATCH}x{PATCH} windows, stride {STRIDE} ({N_POS} positions), "
        f"shared K1={K1}, K2={K2}, lam={LAM}, ETA1={ETA1}, ETA2={ETA2}, "
        f"EPOCHS={EPOCHS}, TRAIN_N={TRAIN_N}, SEED={SEED}.",
        "Baseline (batch 1, independent nodes, no overlap, K2=200): "
        "hard 0.648, soft 0.705, soft_norm 0.725, soft_top10 0.737, probe 0.916.",
        "",
        "| hard | soft | soft_norm | soft_top10 | probe | label-only consistent | dic entropy | dead | templates per class |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {acc_hard:.4f} | {acc_soft:.4f} | {acc_norm:.4f} | {acc_topk:.4f} | "
        f"{probe_acc:.4f} | {consistent}/10 | {dic_entropy:.3f} | {dead} | {alloc_s} |",
        "",
        "Figures: shared_dictionary.png, generation_labelonly.png, recon_test.png",
    ]) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
