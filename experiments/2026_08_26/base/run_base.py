"""Base rig for 2026-08-26: the best 3-layer network, widened, standalone.

L1 8x8 s1, K1=1024 (dense pixel-window learning)
L2 3x3 s2 over L1's code, K2=1024 (skeleton learning targets, standard)
L3 top over [whole L2 code ; lam*onehot label], KTOP units (dense targets)

Dense relu communication at every interface; top-1 (argmax) plasticity in
every bank; generation = label-only query -> winning L3 unit -> hardened
(top-1) reverse reconstruction through L2 -> L1 -> pixels.

Rebuild of run_wide3level.py's dense arm (validated at K=512: probe .9660,
hard .8972), widened 512 -> 1024 at both banks. Deliverables:
generation.png + template galleries for every layer.

Run:  .venv/bin/python experiments/2026_08_26/base/run_base.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    Dict,
    XP_NAME,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    EPOCHS,
    ETA1,
    ETA2,
    ETATOP,
    KTOP,
    LAM,
    PROBE_N,
    TRAIN_N,
    expand,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    POS1,
    POS2,
    S2,
    W1,
    W2,
    render8,
    top_view,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "base" / "results"
K1 = 1024
K2 = 1024
CODE_DIM = G2 * G2 * K2
EVAL_B = 125


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def stack(xb, d1, d2, learning):
    m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
    return level_pass(m1, d2, POS2, W2, K1, G2, True, learning)


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    top = Dict(KTOP, CODE_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m2 = stack(xb, d1, d2, True)
            Zd, okd = top_view(m2.reshape(len(xb), -1), lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
            if top.n_boot >= top.k:
                Ct = Zd @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cvals = xp.max(Ct, axis=1) * okd
                keep = cvals > 0
                top.update(Zd[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    W1n, W2n, Wtn = to_np(d1.W), to_np(d2.W), to_np(top.W)
    np.savez(OUTPUT_DIR / "weights.npz", W1=W1n, W2=W2n, Wtop=Wtn)

    # ---- Reference metrics (probe on L2 code, hard readout, consistency) --
    def encode(X):
        C2s = []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            C2s.append(to_np(stack(xb, d1, d2, False).reshape(-1, CODE_DIM)))
        return np.concatenate(C2s)

    C2tr, C2te = encode(Xtr[:PROBE_N]), encode(Xte)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(C2tr, ytr[:PROBE_N])
    probe = float(clf.score(C2te, yte))

    owner = np.argmax(Wtn[:, CODE_DIM:], axis=1)
    H = C2te - C2te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    hard = float((owner[(Z @ Wtn.T).argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    # ---- Generation: label in, image out (hardened reverse read) ----------
    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2 = Bank(W2n)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
        row = Wtn[int(np.argmax(Wtn @ z_hat))]
        c2 = np.maximum(row[:CODE_DIM], 0.0).reshape(G2, G2, K2)
        m1 = _harden_map(expand(_harden_map(c2), b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1, W1n), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle("Base rig — label-only generation (3-layer, 8x8 L1, dense)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation.png", dpi=110)
    plt.close(fig)

    # ---- Template galleries, most-rehearsed first -------------------------
    wins1, wins2, winst = (to_np(d.win_counts) for d in (d1, d2, top))

    order1 = np.argsort(wins1)[::-1][:256]
    fig, axes = plt.subplots(16, 16, figsize=(12, 12.4))
    for ax, u in zip(axes.flat, order1):
        ax.imshow(W1n[u].reshape(W1, W1), cmap="gray")
        ax.axis("off")
    fig.suptitle(f"L1 templates — top 256 of {K1} by wins (8x8 pixel patches)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L1.png", dpi=110)
    plt.close(fig)

    order2 = np.argsort(wins2)[::-1][:100]
    fig, axes = plt.subplots(10, 10, figsize=(11, 11.4))
    for ax, u in zip(axes.flat, order2):
        c2 = np.zeros((G2, G2, K2))
        c2[G2 // 2, G2 // 2, u] = 1.0
        m1 = expand(c2, b2, W2, S2, (G1, G1, K1))
        ax.imshow(render8(m1, W1n), cmap="gray")
        ax.axis("off")
    fig.suptitle(f"L2 templates — top 100 of {K2} by wins (rendered via L1)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L2.png", dpi=110)
    plt.close(fig)

    ordert = np.argsort(winst)[::-1][:100]
    fig, axes = plt.subplots(10, 10, figsize=(11, 11.8))
    for ax, u in zip(axes.flat, ordert):
        c2 = np.maximum(Wtn[u, :CODE_DIM], 0.0).reshape(G2, G2, K2)
        m1 = _harden_map(expand(_harden_map(c2), b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1, W1n), cmap="gray")
        ax.set_title(str(owner[u]), fontsize=7)
        ax.axis("off")
    fig.suptitle(f"L3 units — top 100 of {KTOP} by wins (hardened reads, "
                 "title = owning label)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L3.png", dpi=110)
    plt.close(fig)

    line = (f"RESULT train_s={train_s:.1f} probeL2={probe:.4f} "
            f"hard={hard:.4f} consistent={consistent}/10")
    print(line)
    (OUTPUT_DIR / "report.md").write_text("\n".join([
        f"# Base rig 2026-08-26 (3-layer, L1 8x8/{K1}, L2 {K2}, dense, top-1)",
        "",
        f"Train wall-time: {train_s:.1f}s ({XP_NAME}, B={B}).",
        "wide3level K=512 dense reference: probe .9660, hard .8972.",
        "",
        "| probe L2 | hard | consistent | train_s |",
        "|---|---|---|---|",
        f"| {probe:.4f} | {hard:.4f} | {consistent}/10 | {train_s:.0f} |",
        "",
        "Generation: generation.png. Templates: templates_L{1,2,3}.png. "
        "Weights: weights.npz.",
    ]) + "\n")


if __name__ == "__main__":
    main()
