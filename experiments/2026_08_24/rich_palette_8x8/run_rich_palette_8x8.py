"""Repaired palette hypothesis: 8x8 L1 windows (63 dims) x big K x sparse output.

Geometry: L1 8x8 s1 (grid 21) -> L2 3x3 s2 (grid 10) -> L3 3x3 s1 (grid 8)
-> top over [8x8x100 ; 10]. Grid: K1 in {64, 512} x L1 output in
{dense, top3, top1}. L2/L3 skeleton learning, L4 dense — standard.
Predictions in README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/rich_palette_8x8/run_rich_palette_8x8.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))

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
    PROBE_N,
    RENDER_SQUELCH,
    SIDE,
    TRAIN_N,
    _squelch_pos,
    expand,
    load_data,
)
from gain_feedback import center_norm  # noqa: E402

# ---- 8x8 geometry ---------------------------------------------------------
W1, S1 = 8, 1
W2, S2 = 3, 2
W3, S3 = 3, 1
G1 = (SIDE - W1) // S1 + 1          # 21
G2 = (G1 - W2) // S2 + 1            # 10
G3 = (G2 - W3) // S3 + 1            # 8
CODE3_DIM = G3 * G3 * K3            # 6400
POS1 = [(r, c) for r in range(0, SIDE - W1 + 1, S1) for c in range(0, SIDE - W1 + 1, S1)]
POS2 = [(r, c) for r in range(0, G1 - W2 + 1, S2) for c in range(0, G1 - W2 + 1, S2)]
POS3 = [(r, c) for r in range(0, G2 - W3 + 1, S3) for c in range(0, G2 - W3 + 1, S3)]
_f8 = np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5])
FEATHER8 = np.outer(_f8, _f8)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8" / "results"
EVAL_B = 250
CONFIGS = [(64, "dense"), (64, "top3"), (64, "top1"),
           (512, "dense"), (512, "top3"), (512, "top1")]


def topk_mask(maps, k):
    N, g, _, K = maps.shape
    if k >= K:
        return maps
    flat = maps.reshape(-1, K)
    out = xp.zeros_like(flat)
    work = flat.copy()
    for _ in range(k):
        am = xp.argmax(work, axis=1)[:, None]
        v = xp.take_along_axis(work, am, axis=1)
        xp.put_along_axis(out, am, v, axis=1)
        xp.put_along_axis(work, am, xp.full_like(v, -1.0), axis=1)
    return out.reshape(N, g, g, K)


def l1_out(xb, d1, mode, learning):
    m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
    if mode == "top3":
        return topk_mask(m1, 3)
    if mode == "top1":
        return topk_mask(m1, 1)
    return m1


def top_view(H_flat, lb):
    Hh, okh = _center_norm_rows(H_flat)
    Z = xp.concatenate([Hh, lb], axis=1)
    Zh, okz = _center_norm_rows(Z)
    return Zh, okz & okh


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def render8(code1, W1n):
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = code1.max()
    for gi in range(G1):
        for gj in range(G1):
            seg = _squelch_pos(code1[gi, gj])
            if seg is None or seg.max() < RENDER_SQUELCH * peak:
                continue
            patch = (seg @ W1n).reshape(W1, W1)
            conf = float(seg.max())
            num[gi:gi + W1, gj:gj + W1] += patch * FEATHER8 * conf
            den[gi:gi + W1, gj:gj + W1] += FEATHER8 * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def run_config(k1, mode, Xtr, ytr, Xte, yte, labels_x):
    d1 = Dict(k1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * k1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m1 = l1_out(xb, d1, mode, True)
            m2 = level_pass(m1, d2, POS2, W2, k1, G2, True, True)
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
            m1 = l1_out(xb, d1, mode, False)
            m2 = level_pass(m1, d2, POS2, W2, k1, G2, True, False)
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
    hard = float((owner[(Z @ Wtn.T).argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    W1n = to_np(d1.W)
    Cc = W1n @ W1n.T
    crowd = float(np.abs(Cc[~np.eye(len(Cc), dtype=bool)]).mean())

    xb = xp.asarray(Xte[:EVAL_B], dtype=DTYPE)
    Vf = _windows(xb[..., None], POS1, W1)
    Vh, ok = _center_norm_rows(Vf.reshape(-1, W1 * W1))
    C = to_np(Vh @ d1.W.T)[to_np(ok)]
    C.sort(axis=1)
    margin = float(np.median(C[:, -1] - C[:, -2]))

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
        m2 = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
        m1 = _harden_map(expand(m2, b2, W2, S2, (G1, G1, k1)))
        ax.imshow(render8(m1, W1n), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — 8x8 L1, K1={k1}, output {mode}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_K{k1}_{mode}.png", dpi=110)
    plt.close(fig)

    return dict(k1=k1, mode=mode, train_s=train_s, probeL2=probes["L2"],
                probeL3=probes["L3"], hard=hard, consistent=consistent,
                crowd=crowd, margin=margin)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    only = os.environ.get("GF_ONLY")
    configs = [c for c in CONFIGS
               if not only or f"{c[0]}:{c[1]}" in only.split(",")]
    rows = []
    for k1, mode in configs:
        r = run_config(k1, mode, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE K1={k1} {mode}: probeL2={r['probeL2']:.4f} "
              f"probeL3={r['probeL3']:.4f} hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 crowd={r['crowd']:.3f} "
              f"margin={r['margin']:.3f} train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# Rich palette at 8x8 L1 windows (63 dims after centering)",
        "",
        "| K1 | L1 out | probe L2 | probe L3 | hard | consistent | L1 crowd | top1-2 margin | train_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['k1']} | {r['mode']} | {r['probeL2']:.4f} | "
                     f"{r['probeL3']:.4f} | {r['hard']:.4f} | {r['consistent']}/10 | "
                     f"{r['crowd']:.3f} | {r['margin']:.3f} | {r['train_s']:.0f} |")
    lines += ["", "4x4 reference: dense@512 hard .9066 margin 0.017; "
              "top1@512 hard .7350; crowd 0.348 everywhere.",
              "Figures: generation_K{64,512}_{dense,top3,top1}.png"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
