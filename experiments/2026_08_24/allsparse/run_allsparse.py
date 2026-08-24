"""All-layer sparse speech: sparse output at L1, L2 AND L3, rich palettes.

8x8 geometry, K1=512. Grid: palettes {std K2=64/K3=100, rich K2=256/K3=256}
x output mode {dense, top3, top1} applied to all three levels. L4 dense
learning as standard. Predictions in README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/allsparse/run_allsparse.py
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
    KTOP,
    LAM,
    PROBE_N,
    TRAIN_N,
    load_data,
    expand,
)
from run_rich_palette_8x8 import (  # noqa: E402
    EVAL_B,
    G1,
    G2,
    G3,
    POS1,
    POS2,
    POS3,
    W1,
    W2,
    W3,
    S2,
    S3,
    render8,
    top_view,
    topk_mask,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "allsparse" / "results"
K1 = 512
CONFIGS = [(64, 100, "dense"), (64, 100, "top3"), (64, 100, "top1"),
           (256, 256, "dense"), (256, 256, "top3"), (256, 256, "top1")]
KMAP = {"dense": None, "top3": 3, "top1": 1}


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def mask(maps, mode):
    k = KMAP[mode]
    return maps if k is None else topk_mask(maps, k)


def run_config(k2, k3, mode, Xtr, ytr, Xte, yte, labels_x):
    code3_dim = G3 * G3 * k3
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(k2, W2 * W2 * K1, ETA2)
    d3 = Dict(k3, W3 * W3 * k2, ETA3)
    top = Dict(KTOP, code3_dim + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    def stack(xb, learning):
        m1 = mask(level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning), mode)
        m2 = mask(level_pass(m1, d2, POS2, W2, K1, G2, True, learning), mode)
        m3 = mask(level_pass(m2, d3, POS3, W3, k2, G3, True, learning), mode)
        return m1, m2, m3

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            _, _, m3 = stack(xb, True)
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
            _, m2, m3 = stack(xb, False)
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
    owner = np.argmax(Wtn[:, code3_dim:], axis=1)
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
        z_hat, _ = center_norm(np.concatenate([np.zeros(code3_dim), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    # per-layer median top1-2 margin on dense correlations of one test batch
    margins = {}
    xb = xp.asarray(Xte[:EVAL_B], dtype=DTYPE)
    m1 = mask(level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, False), mode)
    m2 = mask(level_pass(m1, d2, POS2, W2, K1, G2, True, False), mode)
    for name, maps, pos, win, kprev, dic in [
            ("L1", xb[..., None], POS1, W1, 1, d1),
            ("L2", m1, POS2, W2, K1, d2),
            ("L3", m2, POS3, W3, k2, d3)]:
        Vf = _windows(maps, pos, win)
        Vh, ok = _center_norm_rows(Vf.reshape(-1, win * win * kprev))
        Cd = to_np(Vh @ dic.W.T)[to_np(ok)]
        Cd.sort(axis=1)
        margins[name] = float(np.median(Cd[:, -1] - Cd[:, -2]))

    tag = f"K{k2}_{k3}_{mode}"
    np.savez(OUTPUT_DIR / f"weights_{tag}.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)
    np.savez(OUTPUT_DIR / f"corrs_{tag}.npz", C=C, owner=owner, yte=yte)

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
        z_hat, _ = center_norm(np.concatenate([np.zeros(code3_dim), lab]))
        row = Wtn[int(np.argmax(Wtn @ z_hat))]
        c3 = np.maximum(row[:code3_dim], 0.0).reshape(G3, G3, k3)
        m2r = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, k2)))
        m1r = _harden_map(expand(m2r, b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1r, b1.W), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — all-layer {mode}, K2={k2}, K3={k3}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_{tag}.png", dpi=110)
    plt.close(fig)

    return dict(k2=k2, k3=k3, mode=mode, train_s=train_s,
                probeL2=probes["L2"], probeL3=probes["L3"], hard=hard,
                consistent=consistent, margins=margins)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    only = os.environ.get("GF_ONLY")
    configs = [c for c in CONFIGS
               if not only or f"{c[0]}:{c[1]}:{c[2]}" in only.split(",")]
    rows = []
    for k2, k3, mode in configs:
        r = run_config(k2, k3, mode, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        m = r["margins"]
        print(f"DONE K2={k2} K3={k3} {mode}: probeL2={r['probeL2']:.4f} "
              f"probeL3={r['probeL3']:.4f} hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 "
              f"margins L1/L2/L3={m['L1']:.3f}/{m['L2']:.3f}/{m['L3']:.3f} "
              f"train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# All-layer sparse speech (8x8, K1=512)",
        "",
        "| K2/K3 | out (all layers) | probe L2 | probe L3 | hard | consistent | margins L1/L2/L3 | train_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["margins"]
        lines.append(f"| {r['k2']}/{r['k3']} | {r['mode']} | {r['probeL2']:.4f} | "
                     f"{r['probeL3']:.4f} | {r['hard']:.4f} | {r['consistent']}/10 | "
                     f"{m['L1']:.3f}/{m['L2']:.3f}/{m['L3']:.3f} | {r['train_s']:.0f} |")
    lines += ["", "L1-only sparse reference (same geometry): dense .9094 / "
              "top3 .8642 / top1 .8590.",
              "Figures: generation_K{...}.png; weights_*.npz, corrs_*.npz saved."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
