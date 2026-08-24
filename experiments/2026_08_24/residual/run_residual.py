"""Residual competition speech: winner subtracts what it explained,
survivors compete over the remainder; stop at the contrast floor, cap 3.

8x8 geometry, K1=512, K2=64, K3=100. Arm A: residual speech at L1 only.
Arm B: residual speech at all three levels. Learning untouched (standard
level_pass calls). Predictions in README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/residual/run_residual.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "residual" / "results"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
KMAX = 3


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def resid_out(prev_maps, dic, positions, win, g_out):
    """Residual-competition output: up to KMAX complementary speakers."""
    N = len(prev_maps)
    Vf = _windows(prev_maps, positions, win)
    Vh, ok = _center_norm_rows(Vf.reshape(N * len(positions), -1))
    r = Vh.copy()
    out = xp.zeros((len(Vh), dic.k), dtype=DTYPE)
    active = ok.copy()
    rows = xp.arange(len(Vh))
    for _ in range(KMAX):
        if not bool(active.any()):
            break
        Cc = r @ dic.W.T
        am = xp.argmax(Cc, axis=1)
        v = Cc[rows, am]
        good = active & (v > 0)
        vv = xp.where(good, v, xp.zeros_like(v))
        out[rows, am] += vv
        r = r - vv[:, None] * dic.W[am]
        active = good & (xp.linalg.norm(r, axis=1) > NORM_FLOOR)
    speakers = float(((out > 0).sum(axis=1))[ok].mean())
    return out.reshape(N, g_out, g_out, dic.k), speakers


def run_arm(arm, Xtr, ytr, Xte, yte, labels_x):
    """arm: 'L1resid' (resid at L1 only) or 'allresid' (all levels)."""
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    spk = {"L1": [], "L2": [], "L3": []}

    def stack(xb, learning):
        x0 = xb[..., None]
        if learning:
            level_pass(x0, d1, POS1, W1, 1, G1, False, True)
        m1, s1 = resid_out(x0, d1, POS1, W1, G1)
        spk["L1"].append(s1)
        if arm == "allresid":
            if learning:
                level_pass(m1, d2, POS2, W2, K1, G2, True, True)
            m2, s2 = resid_out(m1, d2, POS2, W2, G2)
            spk["L2"].append(s2)
            if learning:
                level_pass(m2, d3, POS3, W3, K2, G3, True, True)
            m3, s3 = resid_out(m2, d3, POS3, W3, G3)
            spk["L3"].append(s3)
        else:
            m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
            m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, learning)
        return m2, m3

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            _, m3 = stack(xb, True)
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
            m2, m3 = stack(xb, False)
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

    np.savez(OUTPUT_DIR / f"weights_{arm}.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)
    np.savez(OUTPUT_DIR / f"corrs_{arm}.npz", C=C, owner=owner, yte=yte)

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b1, b2, b3 = Bank(to_np(d1.W)), Bank(to_np(d2.W)), Bank(to_np(d3.W))

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
        ax.imshow(render8(m1r, b1.W), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — {arm}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_{arm}.png", dpi=110)
    plt.close(fig)

    spk_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in spk.items()}
    return dict(arm=arm, train_s=train_s, probeL2=probes["L2"],
                probeL3=probes["L3"], hard=hard, consistent=consistent,
                spk=spk_mean)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for arm in ["L1resid", "allresid"]:
        r = run_arm(arm, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {arm}: probeL2={r['probeL2']:.4f} probeL3={r['probeL3']:.4f} "
              f"hard={r['hard']:.4f} consistent={r['consistent']}/10 "
              f"speakers={r['spk']} train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# Residual competition speech (8x8, K1=512, K2=64, K3=100, cap 3)",
        "",
        "| arm | probe L2 | probe L3 | hard | consistent | speakers L1/L2/L3 | train_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["spk"]
        lines.append(f"| {r['arm']} | {r['probeL2']:.4f} | {r['probeL3']:.4f} | "
                     f"{r['hard']:.4f} | {r['consistent']}/10 | "
                     f"{s['L1']:.2f}/{s['L2']:.2f}/{s['L3']:.2f} | {r['train_s']:.0f} |")
    lines += ["", "References (same geometry): L1-only top3 .8642 / top1 .8590 / "
              "dense .9094; all-layer top3 .7596 / top1 .7092."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
