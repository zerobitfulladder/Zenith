"""Residual LEARNING at L1 (tuned inhibition in plasticity) + dense/resid speech.

Speaker 1 rotates toward the window (standard); its contribution is
subtracted; speaker 2 rotates toward the remainder; cap 3, contrast floor
stops. Arms: (a) resid learning + dense speech (guard), (b) resid learning
+ resid speech (the bet). L2/L3/L4 fully standard. Predictions in README.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/resid_learning/run_resid_learning.py
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
    _center_norm_rows,
    _windows,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    EPOCHS,
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    K2,
    K3,
    KTOP,
    LAM,
    NORM_FLOOR,
    PROBE_N,
    TRAIN_N,
    expand,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    G3,
    POS1,
    POS2,
    POS3,
    S2,
    S3,
    W1,
    W2,
    W3,
    render8,
    top_view,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "resid_learning" / "results"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
KMAX = 3


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def l1_pass(xb, d1, learning, speech):
    """Residual learning cascade (frozen-W within batch), dense or resid speech."""
    x0 = xb[..., None]
    Vf = _windows(x0, POS1, W1)
    Vh, ok = _center_norm_rows(Vf.reshape(-1, W1 * W1))
    if learning and d1.n_boot < d1.k:
        d1.bootstrap(Vh[ok])
    learn_ready = learning and d1.n_boot >= d1.k
    rows = xp.arange(len(Vh))
    r = Vh.copy()
    active = ok.copy()
    out = xp.zeros((len(Vh), d1.k), dtype=DTYPE)
    triplets = []
    for _ in range(KMAX):
        if not bool(active.any()):
            break
        Cc = r @ d1.W.T
        am = xp.argmax(Cc, axis=1)
        v = Cc[rows, am]
        good = active & (v > 0)
        vv = xp.where(good, v, xp.zeros_like(v))
        out[rows, am] += vv
        if learn_ready:
            rn = xp.linalg.norm(r, axis=1, keepdims=True)
            T = r / xp.maximum(rn, 1e-9)
            triplets.append((T[good], am[good], v[good]))
        r = r - vv[:, None] * d1.W[am]
        active = good & (xp.linalg.norm(r, axis=1) > NORM_FLOOR)
    for T, w_, v_ in triplets:
        d1.update(T, w_, v_)
    if speech == "resid":
        m1 = out * 1.0
    else:
        m1 = xp.maximum(Vh @ d1.W.T, 0.0) * ok[:, None]
    return m1.reshape(len(xb), G1, G1, d1.k)


def run_arm(speech, Xtr, ytr, Xte, yte, labels_x):
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m1 = l1_pass(xb, d1, True, speech)
            m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, True)
            m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, True)
            Zd, okd = top_view(m3.reshape(len(xb), -1), lb)
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

    def encode(X):
        C2s, C3s = [], []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            m1 = l1_pass(xb, d1, False, speech)
            m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, False)
            m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, False)
            C2s.append(to_np(m2.reshape(len(xb), -1)))
            C3s.append(to_np(m3.reshape(len(xb), -1)))
        return np.concatenate(C2s), np.concatenate(C3s)

    C2tr, C3tr = encode(Xtr[:PROBE_N])
    C2te, C3te = encode(Xte)

    probes = {}
    for name, tr, te in [("L2", C2tr, C2te), ("L3", C3tr, C3te)]:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(tr, ytr[:PROBE_N])
        probes[name] = float(clf.score(te, yte))

    Wtn = to_np(top.W)
    owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)
    H = C3te - C3te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    C = Z @ Wtn.T
    hard = float((owner[C.argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    # palette diagnostics
    W1n = to_np(d1.W)
    Cc = W1n @ W1n.T
    off = Cc[~np.eye(K1, dtype=bool)]
    clone08 = float((off > 0.8).mean())
    clone09 = float((off > 0.9).mean())
    xb = xp.asarray(Xte[:EVAL_B], dtype=DTYPE)
    Vf = _windows(xb[..., None], POS1, W1)
    Vh, ok = _center_norm_rows(Vf.reshape(-1, W1 * W1))
    Cd = to_np(Vh @ d1.W.T)[to_np(ok)]
    Cd.sort(axis=1)
    margin = float(np.median(Cd[:, -1] - Cd[:, -2]))

    np.savez(OUTPUT_DIR / f"weights_{speech}.npz",
             W1=W1n, W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2, b3 = Bank(to_np(d2.W)), Bank(to_np(d3.W))

    def _harden_map(m, k=1):
        o = np.zeros_like(m)
        for a in range(m.shape[0]):
            for b_ in range(m.shape[1]):
                seg = np.maximum(m[a, b_], 0.0)
                if seg.max() > 0:
                    idx = np.argsort(seg)[::-1][:k]
                    o[a, b_, idx] = seg[idx]
        return o

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        row = Wtn[int(np.argmax(Wtn @ z_hat))]
        c3 = np.maximum(row[:CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2r = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
        m1r = _harden_map(expand(m2r, b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1r, W1n), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — residual learning, {speech} speech")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_{speech}.png", dpi=110)
    plt.close(fig)

    # sorted palette gallery
    order = [int(np.abs(Cc).sum(axis=1).argmax())]
    used = np.zeros(K1, dtype=bool)
    used[order[0]] = True
    for _ in range(K1 - 1):
        sims = Cc[order[-1]].copy()
        sims[used] = -np.inf
        nxt = int(sims.argmax())
        order.append(nxt)
        used[nxt] = True
    cols, rows_n = 22, 24
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 0.55, rows_n * 0.62))
    for ax in np.ravel(axes):
        ax.axis("off")
    for i, t in enumerate(order):
        ax = np.ravel(axes)[i]
        p = W1n[t].reshape(W1, W1)
        m = np.abs(p).max() + 1e-9
        ax.imshow(p, cmap="gray", vmin=-m, vmax=m)
    fig.suptitle(f"L1 palette after residual learning ({speech}-speech arm), "
                 "similarity-sorted", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"palette_sorted_{speech}.png", dpi=140)
    plt.close(fig)

    return dict(speech=speech, train_s=train_s, probeL2=probes["L2"],
                probeL3=probes["L3"], hard=hard, consistent=consistent,
                clone08=clone08, clone09=clone09, margin=margin)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for speech in ["dense", "resid"]:
        r = run_arm(speech, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {speech}: probeL2={r['probeL2']:.4f} probeL3={r['probeL3']:.4f} "
              f"hard={r['hard']:.4f} consistent={r['consistent']}/10 "
              f"clones>0.8={r['clone08']:.4f} >0.9={r['clone09']:.4f} "
              f"margin={r['margin']:.3f} train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# Residual learning at L1 (8x8, K1=512; L2/L3/L4 standard)",
        "",
        "| speech | probe L2 | probe L3 | hard | consistent | pairs cos>0.8 | >0.9 | margin | train_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['speech']} | {r['probeL2']:.4f} | {r['probeL3']:.4f} | "
                     f"{r['hard']:.4f} | {r['consistent']}/10 | {r['clone08']:.4f} | "
                     f"{r['clone09']:.4f} | {r['margin']:.3f} | {r['train_s']:.0f} |")
    lines += ["", "Std-learning references: dense .9094 (margin 0.038) / "
              "top3 .8642 / resid speech .8502. Std palette: adjacent-pair "
              "median cos 0.816, max 0.968."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
