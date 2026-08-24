"""F7b: label generation from images on a lam=1.0 rig, four reader styles.

Train the 8x8 dense champion at lam=1.0, then generate labels from
images (zero-label queries) via: top-1 winner, graded completion
(relu correlations x stored label halves), top-10 vote, sharp softmax.
Predictions in README (F7b).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/lam_sweep/run_lam1_labelgen.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import numpy as np

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
    ETA3,
    ETATOP,
    K2,
    K3,
    KTOP,
    TRAIN_N,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    G3,
    POS1,
    POS2,
    POS3,
    W1,
    W2,
    W3,
    top_view,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "lam_sweep" / "results" / "lam1_labelgen"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
LAM1 = 1.0


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr[:TRAIN_N]] = LAM1
    labels_x = xp.asarray(labels_tr)

    def stack(xb, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
        m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, learning)
        return m3.reshape(len(xb), -1)

    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            Zd, okd = top_view(stack(xb, True), lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
                continue
            Ct = Zd @ top.W.T
            winners = xp.argmax(Ct, axis=1)
            cvals = xp.max(Ct, axis=1) * okd
            keep = cvals > 0
            top.update(Zd[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()

    C3s = []
    for s in range(0, len(Xte), EVAL_B):
        xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
        C3s.append(to_np(stack(xb, False)))
    C3te = np.concatenate(C3s)

    Wtn = to_np(top.W)
    np.savez(OUTPUT_DIR / "weights_lam1.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)
    owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)
    onehot = np.zeros((KTOP, 10))
    onehot[np.arange(KTOP), owner] = 1.0
    label_halves = np.maximum(Wtn[:, CODE3_DIM:], 0.0)

    H = C3te - C3te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    C = Z @ Wtn.T

    res = {}
    res["top-1 winner"] = float((owner[C.argmax(axis=1)] == yte).mean())
    S = np.maximum(C, 0.0)
    res["graded completion (stored label halves)"] = \
        float(((S @ label_halves).argmax(axis=1) == yte).mean())
    k = 10
    idx = np.argpartition(C, -k, axis=1)[:, -k:]
    w = np.maximum(np.take_along_axis(C, idx, axis=1), 0.0)
    votes = np.zeros((len(C), 10))
    for j in range(k):
        np.add.at(votes, (np.arange(len(C)), owner[idx[:, j]]), w[:, j])
    res["top-10 vote"] = float((votes.argmax(axis=1) == yte).mean())
    E = np.exp((C - C.max(axis=1, keepdims=True)) / 0.02)
    res["sharp softmax"] = float(((E @ onehot).argmax(axis=1) == yte).mean())

    lines = ["# F7b: label generation from images at lam=1.0 (8x8 dense)",
             "",
             "| reader | accuracy | lam=0.5 reference |",
             "|---|---|---|"]
    refs = {"top-1 winner": ".9102 (F7) / .9080 (completion exp)",
            "graded completion (stored label halves)": "n/a (variant)",
            "top-10 vote": ".8468", "sharp softmax": ".9080"}
    for kk, v in res.items():
        lines.append(f"| {kk} | {v:.4f} | {refs.get(kk, '-')} |")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
