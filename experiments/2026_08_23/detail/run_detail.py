"""Detail build: batch-2 architecture with a stride-1 dense code.

One change from run_batch2.py: the shared dictionary REPORTS at stride 1
(625 window positions, code dim 22500) while it LEARNS only at the stride-2
subgrid — training dynamics identical to the 8-seed-validated rig, but the
top layer's constellations carry ~4x the spatial detail and every rendered
pixel is the consensus of up to 16 overlapping windows.

Run:  .venv/bin/python experiments/2026_08_23/detail/run_detail.py
"""

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
PROBE_N = 4000
SIDE = 28
PATCH = 4
K1 = 36
K2 = 200
ETA1 = 0.02
ETA2 = 0.04
EPOCHS = 2
LAM = 0.5
SEED = 42
NORM_FLOOR = 0.15
RENDER_SQUELCH = 0.10
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "detail" / "results"
# ---------------------------------------------------------------------------

POS = [(r, c) for r in range(0, SIDE - PATCH + 1) for c in range(0, SIDE - PATCH + 1)]  # 625
LEARN_AT = np.array([(r % 2 == 0 and c % 2 == 0) for r, c in POS])   # the validated stride-2 subgrid
N_POS = len(POS)
CODE_DIM = N_POS * K1
FEATHER = np.outer([0.5, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 0.5])


def load_data():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def encode_image(x2d, dic, learning):
    P = np.stack([x2d[r:r + PATCH, c:c + PATCH].ravel() for r, c in POS])
    Pc = P - P.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Pc, axis=1)
    valid = norms > NORM_FLOOR
    Phat = np.zeros_like(Pc)
    Phat[valid] = Pc[valid] / norms[valid, None]
    C = Phat @ dic.W.T
    code = np.zeros(CODE_DIM)
    for i in np.where(valid)[0]:
        if learning and LEARN_AT[i]:
            w = dic.learn(Phat[i], C[i])
        else:
            w = int(np.argmax(C[i]))
        code[i * K1 + w] = max(float(C[i, w]), 0.0)
    return code


def encode_batch(X2d, dic):
    N = len(X2d)
    code = np.zeros((N, CODE_DIM), dtype=np.float32)
    for pi, (r, c) in enumerate(POS):
        P = X2d[:, r:r + PATCH, c:c + PATCH].reshape(N, -1)
        P = P - P.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(P, axis=1)
        ok = norms > NORM_FLOOR
        P[ok] /= norms[ok, None]
        C = P @ dic.W.T
        wins = C.argmax(axis=1)
        vals = np.maximum(C[np.arange(N), wins], 0.0) * ok
        code[np.arange(N), pi * K1 + wins] = vals.astype(np.float32)
    return code


def render(code_half, dic):
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = float(np.max(code_half)) if np.max(code_half) > 0 else 0.0
    for pi, (r, c) in enumerate(POS):
        seg = np.maximum(code_half[pi * K1:(pi + 1) * K1], 0.0)
        conf = float(seg.max())
        if conf <= 0.0 or conf < RENDER_SQUELCH * peak:
            continue
        seg = seg.copy()
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
        for x, y in tqdm(list(zip(Xtr, ytr)), desc="detail", ncols=80):
            code = encode_image(x, dic, learning=True)
            h_hat, h_norm = center_norm(code)
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = LAM
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                l2.learn(z_hat, l2.forward(z_hat))

    np.savez(OUTPUT_DIR / "weights.npz", W1=dic.W, W2=l2.W)

    label_half = l2.W[:, CODE_DIM:]
    owner = label_half.argmax(axis=1)

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
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=np.float32)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ l2.W.T.astype(np.float32)
    winners = C2.argmax(axis=1)
    acc_hard = float((owner[winners] == yte).mean())
    votes = np.maximum(C2, 0.0) @ label_half.astype(np.float32)
    kv = 10
    part = np.argpartition(-C2, kv, axis=1)[:, :kv]
    mask = np.zeros_like(C2)
    mask[np.arange(len(C2))[:, None], part] = 1.0
    acc_topk = float((((np.maximum(C2, 0.0) * mask) @ label_half.astype(np.float32)).argmax(axis=1) == yte).mean())

    probe = LogisticRegression(max_iter=1000)
    probe.fit(Ctr, ytr[:PROBE_N])
    probe_acc = float(probe.score(Cte, yte))

    print(f"RESULT hard={acc_hard:.4f} top10={acc_topk:.4f} probe={probe_acc:.4f} "
          f"consistent={consistent}/10")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[j], cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Label-only generation, stride-1 dense constellation (lam={LAM})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_labelonly.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for k in range(8):
        code = encode_batch(Xte[k:k + 1], dic)[0].astype(np.float64)
        axes[0, k].imshow(Xte[k], cmap="gray")
        axes[1, k].imshow(render(code, dic), cmap="gray")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("Test images (top) vs stride-1 dense reconstruction (bottom)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recon_test.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join([
        "# Detail build — stride-1 dense code (MNIST)",
        "",
        f"Config: {PATCH}x{PATCH} windows at stride 1 ({N_POS} positions, "
        f"code dim {CODE_DIM}), learning at the stride-2 subgrid only, "
        f"K1={K1}, K2={K2}, lam={LAM}, EPOCHS={EPOCHS}, SEED={SEED}.",
        "Batch-2 baselines: hard 0.743, top10 0.782, probe 0.9146 (seed 42).",
        "",
        "| hard | top10 | probe | label-only consistent |",
        "|---|---|---|---|",
        f"| {acc_hard:.4f} | {acc_topk:.4f} | {probe_acc:.4f} | {consistent}/10 |",
        "",
        "Figures: generation_labelonly.png, recon_test.png. Weights: weights.npz",
    ]) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
