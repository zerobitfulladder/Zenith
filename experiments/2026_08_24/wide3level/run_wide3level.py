"""3-level wide rig: L1 8x8/512 -> wide L2 -> memory on L2's code.

Arms: K2=256 dense, K2=512 dense, K2=256 all-top3 (sparse speech at both
interfaces). L1/L2 learning standard (dense-window / skeleton); memory
dense. Predictions in README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/wide3level/run_wide3level.py
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
    topk_mask,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "wide3level" / "results"
K1 = 512
EVAL_B = 125
CONFIGS = [(256, "dense"), (512, "dense"), (256, "top3")]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def run_config(k2, mode, Xtr, ytr, Xte, yte, labels_x):
    code_dim = G2 * G2 * k2
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(k2, W2 * W2 * K1, ETA2)
    top = Dict(KTOP, code_dim + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    def stack(xb, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        if mode == "top3":
            m1 = topk_mask(m1, 3)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
        if mode == "top3":
            m2 = topk_mask(m2, 3)
        return m2

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m2 = stack(xb, True)
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

    def encode(X):
        C2s = []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            C2s.append(to_np(stack(xb, False).reshape(-1, code_dim)))
        return np.concatenate(C2s)

    C2tr = encode(Xtr[:PROBE_N])
    C2te = encode(Xte)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(C2tr, ytr[:PROBE_N])
    probe = float(clf.score(C2te, yte))

    Wtn = to_np(top.W)
    owner = np.argmax(Wtn[:, code_dim:], axis=1)
    H = C2te - C2te.mean(axis=1, keepdims=True)
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
        z_hat, _ = center_norm(np.concatenate([np.zeros(code_dim), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    np.savez(OUTPUT_DIR / f"weights_K{k2}_{mode}.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), Wtop=Wtn)

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2 = Bank(to_np(d2.W))

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
        z_hat, _ = center_norm(np.concatenate([np.zeros(code_dim), lab]))
        row = Wtn[int(np.argmax(Wtn @ z_hat))]
        c2 = np.maximum(row[:code_dim], 0.0).reshape(G2, G2, k2)
        m1 = _harden_map(expand(_harden_map(c2), b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1, to_np(d1.W)), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — 3-level, K2={k2}, {mode}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_K{k2}_{mode}.png", dpi=110)
    plt.close(fig)

    return dict(k2=k2, mode=mode, train_s=train_s, probe=probe, hard=hard,
                consistent=consistent)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for k2, mode in CONFIGS:
        r = run_config(k2, mode, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE K2={k2} {mode}: probeL2={r['probe']:.4f} hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 train={r['train_s']:.0f}s",
              flush=True)

    lines = [
        "# 3-level wide rig (L1 8x8/512 -> wide L2 -> memory)",
        "",
        "| K2 | out | probe L2 | hard | consistent | train_s |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['k2']} | {r['mode']} | {r['probe']:.4f} | "
                     f"{r['hard']:.4f} | {r['consistent']}/10 | {r['train_s']:.0f} |")
    lines += ["", "4-level references: dense .9094 / probe .9668; all-top3 "
              ".7596; L1-only top3 .8642."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
