"""Encoder/decoder with a unification layer — the perception->action bridge.

Two independent 2-layer stacks (E1,E2 and D1,D2 — different seeds, so
genuinely different template vocabularies) both see the SAME image and
project up; the unification layer learns cn([enc_code ; dec_code])
jointly with the standard rule — cross-modal co-occurrence, the
label-concat pattern generalized to a full second modality.

Inference: image -> encoder only -> query [enc ; empty] -> winner's
stored DECODER half reprojects down the decoder stack (top-1 hardened,
feathered overlap-add) -> reconstructed image at the other end.
(Later: the decoder stack becomes temporal motor commands.)

Run:  .venv/bin/python experiments/2026_08_26/bridge/run_enc_dec.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from run_4layer_topk import load_data  # noqa: E402
from run_temporal_digits import SeqDict, cn  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "bridge" / "results" / "enc_dec"
H = 28
W1_, S1_ = 8, 2
POS1 = [(r, c) for r in range(0, H - W1_ + 1, S1_) for c in range(0, H - W1_ + 1, S1_)]
G1 = 11
W2_, S2_ = 3, 2
POS2 = [(r, c) for r in range(0, G1 - W2_ + 1, S2_) for c in range(0, G1 - W2_ + 1, S2_)]
G2 = 5
K1, K2 = 64, 128
CODE = G2 * G2 * K2                  # 3200
KU = 400
ETA = 0.05
L1_N, L2_N, UNI_N = 1500, 1500, 15000
EPOCHS_U = 2
NORM_FLOOR = 1e-3
_f = np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5])
FEATHER = np.outer(_f, _f)


def cn_rows(V):
    V = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(V, axis=1)
    return V / np.maximum(n, 1e-9)[:, None], n > NORM_FLOOR


def l1_map(img, B1):
    V = np.stack([img[r:r + W1_, c:c + W1_].ravel() for r, c in POS1])
    Vh, ok = cn_rows(V)
    return (np.maximum(Vh @ B1.T, 0.0) * ok[:, None]).reshape(G1, G1, K1)


def skeleton(block):
    """(3,3,K1) window -> per-position top-1 skeleton, flattened."""
    sk = np.zeros_like(block)
    for a in range(block.shape[0]):
        for b in range(block.shape[1]):
            seg = block[a, b]
            if seg.max() > 0:
                u = int(np.argmax(seg))
                sk[a, b, u] = seg[u]
    return sk.ravel()


def l2_code(m1, B2):
    V = np.stack([m1[r:r + W2_, c:c + W2_, :].ravel() for r, c in POS2])
    Vh, ok = cn_rows(V)
    c = np.maximum(Vh @ B2.T, 0.0) * ok[:, None]
    f = c.ravel()
    return (f / (np.linalg.norm(f) + 1e-9)).astype(np.float32)


def train_stack(X, seed):
    b1 = SeqDict(K1, W1_ * W1_, ETA, seed=seed)
    for i in range(L1_N):
        V = np.stack([X[i][r:r + W1_, c:c + W1_].ravel() for r, c in POS1])
        Vh, ok = cn_rows(V)
        for v in Vh[ok]:
            b1.step(v.astype(np.float32))
    b2 = SeqDict(K2, W2_ * W2_ * K1, ETA, seed=seed + 1)
    for i in range(L2_N):
        m1 = l1_map(X[i], b1.W)
        for r, c in POS2:
            t = skeleton(m1[r:r + W2_, c:c + W2_, :])
            if t.max() <= 0:
                continue
            b2.step(cn(t).astype(np.float32))
    return b1.W, b2.W


def render_decoder(dec_code, D1, D2):
    c2 = np.maximum(dec_code, 0.0).reshape(G2, G2, K2)
    m1 = np.zeros((G1, G1, K1))
    flat = c2.reshape(-1, K2)
    for i, (r, c) in enumerate(POS2):
        seg = flat[i]
        if seg.max() <= 0:
            continue
        u = int(np.argmax(seg))
        m1[r:r + W2_, c:c + W2_, :] += float(seg.max()) * D2[u].reshape(
            W2_, W2_, K1)
    num = np.zeros((H, H))
    den = np.zeros((H, H))
    for i, (r, c) in enumerate(POS1):
        gi, gj = r // S1_, c // S1_
        seg = np.maximum(m1[gi, gj], 0.0)
        if seg.max() <= 0:
            continue
        u = int(np.argmax(seg))
        patch = D1[u].reshape(W1_, W1_)
        conf = float(seg.max())
        num[r:r + W1_, c:c + W1_] += patch * FEATHER * conf
        den[r:r + W1_, c:c + W1_] += FEATHER * conf
    img = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    v = np.maximum(img, 0.0)
    return v / (v.max() + 1e-9)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    E1, E2 = train_stack(Xtr, seed=100)
    D1, D2 = train_stack(Xtr, seed=200)
    print("stacks trained", flush=True)

    uni = SeqDict(KU, 2 * CODE, ETA, seed=300)
    for _ in range(EPOCHS_U):
        for i in range(UNI_N):
            enc = l2_code(l1_map(Xtr[i], E1), E2)
            dec = l2_code(l1_map(Xtr[i], D1), D2)
            uni.step(cn(np.concatenate([enc, dec])).astype(np.float32))
    print("unification trained", flush=True)

    judge = LogisticRegression(max_iter=1000)
    judge.fit(Xtr[:10000].reshape(10000, -1), ytr[:10000])
    class_means = [Xtr[:UNI_N][ytr[:UNI_N] == c].mean(axis=0) for c in range(10)]

    picks = [int(np.where(yte == c)[0][0]) for c in range(10)]
    recons, rows = [], []
    for c, idx in enumerate(picks):
        img = Xte[idx]
        enc = l2_code(l1_map(img, E1), E2)
        q = cn(np.concatenate([enc, np.zeros(CODE, np.float32)])
               ).astype(np.float32)
        winner = int(np.argmax(uni.W @ q))
        rec = render_decoder(uni.W[winner, CODE:], D1, D2)
        recons.append((img, rec))
        rv = cn(rec.ravel())
        corr_in = float(rv @ cn(img.ravel()))
        corr_mean = float(rv @ cn(class_means[c].ravel()))
        pred = int(judge.predict(rec.reshape(1, -1))[0])
        rows.append((c, pred, corr_in, corr_mean))
        print(f"class {c}: judge says {pred}, corr(input)={corr_in:.3f}, "
              f"corr(class mean)={corr_mean:.3f}", flush=True)

    ok = sum(int(p == c) for c, p, _, _ in rows)
    print(f"identity preserved: {ok}/10", flush=True)

    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.9))
    for c, (img, rec) in enumerate(recons):
        axes[0, c].imshow(img, cmap="gray")
        axes[1, c].imshow(rec, cmap="gray")
        for r in range(2):
            axes[r, c].axis("off")
    axes[0, 0].set_title("input (test set)", fontsize=7, loc="left")
    axes[1, 0].set_title("reconstruction via decoder stack", fontsize=7,
                         loc="left")
    fig.suptitle("Encoder -> unification -> decoder reprojection "
                 "(unseen test digits)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reconstructions.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Encoder/decoder bridge (unification layer)",
         "",
         f"E/D stacks: 8x8s2/{K1} -> 3x3s2/{K2}; unification KU={KU} over "
         f"[enc ; dec], {UNI_N} images x {EPOCHS_U} epochs.",
         f"Identity preserved (pixel-LR judge): {ok}/10 on unseen test digits.",
         "",
         "| class | judge | corr(input) | corr(class mean) |",
         "|---|---|---|---|"]
        + [f"| {c} | {p} | {ci:.3f} | {cm:.3f} |" for c, p, ci, cm in rows]
        + ["", "Figure: reconstructions.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
