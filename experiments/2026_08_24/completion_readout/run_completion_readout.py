"""Completion readout: generate the label by memory voting vs copy the winner's.

Retrains the three 8x8 K1=512 arms (dense/top3/top1 L1 output) and scores
each with: top-1 (baseline), top-10 vote, all-200 vote. Voting: relu
correlation weights, each memory pushes one-hot(owner). Predictions in
README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/completion_readout/run_completion_readout.py
"""

import os
import sys
import time
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
    LAM,
    TRAIN_N,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    CODE3_DIM,
    EVAL_B,
    G1,
    G2,
    G3,
    POS2,
    POS3,
    W1,
    W2,
    W3,
    l1_out,
    top_view,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "completion_readout" / "results"
K1 = 512
MODES = ["dense", "top3", "top1"]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def train(mode, Xtr, labels_x):
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
            m1 = l1_out(xb, d1, mode, True)
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
    return d1, d2, d3, top, time.time() - t0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for mode in MODES:
        d1, d2, d3, top, train_s = train(mode, Xtr, labels_x)

        C3s = []
        for s in range(0, len(Xte), EVAL_B):
            xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
            m1 = l1_out(xb, d1, mode, False)
            m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, False)
            m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, False)
            C3s.append(to_np(m3.reshape(len(xb), -1)))
        C3te = np.concatenate(C3s)

        Wtn = to_np(top.W)
        owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)
        onehot = np.zeros((KTOP, 10))
        onehot[np.arange(KTOP), owner] = 1.0

        H = C3te - C3te.mean(axis=1, keepdims=True)
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
        Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
        Z -= Z.mean(axis=1, keepdims=True)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        C = Z @ Wtn.T                                   # (N, 200)

        np.savez(OUTPUT_DIR / f"corrs_{mode}.npz", C=C, owner=owner, yte=yte)
        acc_top1 = float((owner[C.argmax(axis=1)] == yte).mean())

        S = np.maximum(C, 0.0)
        acc_all = float(((S @ onehot).argmax(axis=1) == yte).mean())

        k = 10
        idx = np.argpartition(C, -k, axis=1)[:, -k:]
        w_raw = np.maximum(np.take_along_axis(C, idx, axis=1), 0.0)
        kth = w_raw.min(axis=1, keepdims=True)
        w_margin = np.maximum(np.take_along_axis(C, idx, axis=1) - kth, 0.0)

        def vote(weights):
            votes = np.zeros((len(C), 10))
            for j in range(k):
                np.add.at(votes, (np.arange(len(C)), owner[idx[:, j]]),
                          weights[:, j])
            return float((votes.argmax(axis=1) == yte).mean())

        acc_top10 = vote(w_raw)
        acc_margin = vote(w_margin)

        E = np.exp((C - C.max(axis=1, keepdims=True)) / 0.02)
        acc_soft = float(((E @ onehot).argmax(axis=1) == yte).mean())

        rows.append(dict(mode=mode, train_s=train_s, top1=acc_top1,
                         top10=acc_top10, margin=acc_margin, soft=acc_soft,
                         all=acc_all))
        print(f"DONE {mode}: top1={acc_top1:.4f} top10vote={acc_top10:.4f} "
              f"marginvote={acc_margin:.4f} softvote={acc_soft:.4f} "
              f"allvote={acc_all:.4f} train={train_s:.0f}s", flush=True)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()

    lines = [
        "# Completion readout — 8x8, K1=512 (retrained arms)",
        "",
        "| L1 out | top-1 | top-10 raw vote | top-10 margin vote | softmax vote (t=.02) | all-200 vote |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['top1']:.4f} | {r['top10']:.4f} | "
                     f"{r['margin']:.4f} | {r['soft']:.4f} | {r['all']:.4f} |")
    lines += ["", "Voting: relu-correlation weights, each memory pushes "
              "one-hot(owner); class argmax."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
