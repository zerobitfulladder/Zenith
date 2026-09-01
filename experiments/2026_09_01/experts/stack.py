"""The full loop: L2's table shapes L2, and L2 shapes L1.

Two feedback paths, both using structures that already exist:

    T2 -> L2      the class table biases which images each L2 template wins,
                  exactly as T1 biased L1. Templates become purer.

    L2 -> L1      the top-down expectation. An L2 template IS a pattern over
                  (cell x L1 template) -- 16 x 180 numbers -- so W2 read
                  backwards already says "when I am active, these L1 templates
                  should be winning in these cells". No new table needed; the
                  recognition weights ARE the feedback weights.

So layer 1's own table is no longer what tells it what to prefer. The belief now
comes from the layer above, which is the structural point: a stack rather than
two systems that each talk to themselves.

    pass 1   L1 winners on ink -> pooled -> L2 winner -> class belief
    pass 2   L1 winners biased by L2's expectation  -> pooled'
             L2 winner biased by T2 toward the belief
    learn    both winners rotate toward what they won; counts updated

Starting from the trained pieces rather than from scratch, so that a failure is
visible as a decline from a known number instead of a mess. Accuracy is measured
every epoch.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, settle as S, single as SG, pressure as PR, layer2 as L2

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = 4, 10, 1e-12
NC, B1, B2, EPOCHS, IMG_BATCH, ETA_MIN, ALPHA = GRID * GRID, 1.5, 1.5, 3, 128, 0.02, 1.0
import sys
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 1      # how many L2 templates speak
TAU = 0.05
CELL = PR.CELL


def table2(C):
    p_tc = (C + ALPHA) / (C.sum(0, keepdims=True) + ALPHA * len(C))
    p_t = (C.sum(1, keepdims=True) + ALPHA * NL) / (C.sum() + ALPHA * len(C) * NL)
    return (np.log(p_tc) - np.log(p_t)).astype(np.float32)


def l1_pooled_from(win, keep, m, h):
    b = S.SIDE // GRID
    M = np.zeros((m, S.SIDE * S.SIDE, h), np.float32)
    r, c = np.nonzero(keep.reshape(m, -1))
    M[r, c, win.reshape(m, -1)[r, c]] = 1.0
    return M.reshape(m, GRID, b, GRID, b, h).max(axis=(2, 4)).reshape(m, -1)


def forward(W1f, wn, W2, T2, Xb, b1, b2, h):
    """pass 1 then pass 2. Returns L1 winners, pooled code, L2 winner."""
    Q, keep = E.patches(Xb)
    m = len(Q)
    P = Q.reshape(-1, 25).astype(np.float32)
    k = keep.reshape(-1)
    Sc = P @ W1f.T
    err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
    w1 = err.argmin(1)
    pool = L2.cn(l1_pooled_from(w1, keep, m, h)).astype(np.float32)
    j = (pool @ W2.T).argmax(1)
    z = T2[j]
    q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)   # class belief
    if b1 > 0:                                   # L2 -> L1, top-down expectation
        if TOPK == 1:
            E2 = W2[j]
        else:                                    # a mixture, not a single commitment
            s1 = pool @ W2.T
            idx = np.argpartition(-s1, TOPK, axis=1)[:, :TOPK]
            sc = np.take_along_axis(s1, idx, 1)
            wq = np.exp((sc - sc.max(1, keepdims=True)) / TAU)
            wq /= wq.sum(1, keepdims=True)
            E2 = np.einsum('mk,mkd->md', wq, W2[idx], optimize=True)
        exp = E2.reshape(m, NC, h)
        bias = exp[np.repeat(np.arange(m), S.SIDE * S.SIDE), np.tile(CELL, m)]
        w1 = (err - b1 * bias).argmin(1)
        pool = L2.cn(l1_pooled_from(w1, keep, m, h)).astype(np.float32)
    s2 = pool @ W2.T
    if b2 > 0:                                   # T2 -> L2
        s2 = s2 + b2 * (q @ T2.T)
    return w1, k, P, pool, s2.argmax(1)


def evaluate(W1, W2, T2, Xtr, ytr, Xte, yte, b1, b2):
    h = len(W1); W1f = W1[:, 0, :].astype(np.float32); wn = (W1f ** 2).sum(1)
    out = []
    for X in (Xtr, Xte):
        js, pools, w1s, keeps = [], [], [], []
        for a in range(0, len(X), 256):
            w1, k, _, pool, j = forward(W1f, wn, W2, T2, X[a:a + 256], b1, b2, h)
            js.append(j); pools.append(pool)
        out.append((np.concatenate(js), np.concatenate(pools)))
    (jtr, Ptr), (jte, Pte) = out
    C = np.zeros((len(W2), NL)); np.add.at(C, (jtr, ytr), 1.0)
    acc = float((table2(C)[jte].argmax(1) == yte).mean())
    live = C.sum(1) > 0
    pur = float((C[live].max(1) / C[live].sum(1)).mean())
    M = np.stack([Ptr[ytr == c].mean(0) for c in range(NL)])
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), EPS)
    nm = float(((Pte / np.maximum(np.linalg.norm(Pte, axis=1, keepdims=True), EPS))
                @ M.T).argmax(1) == yte).mean() if False else float(
        (((Pte / np.maximum(np.linalg.norm(Pte, axis=1, keepdims=True), EPS)) @ M.T
          ).argmax(1) == yte).mean())
    return acc, pur, nm, C


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    W1 = np.load(OUT / "coadapt.npz")["W"].astype(np.float64)
    d2 = np.load(OUT / "layer2_k200.npz")
    W2, C = d2["W2"].astype(np.float32), d2["counts"]
    T2 = table2(C)
    h, K2 = len(W1), len(W2)
    print(f"L1 {h} templates, L2 {K2} templates, beta1 {B1} beta2 {B2}, "
          f"top-{TOPK} top-down\n", flush=True)
    a, p, n, _ = evaluate(W1, W2, T2, Xtr, ytr, Xte, yte, 0.0, 0.0)
    print(f"  before, no feedback    acc {a:.4f}   L2 purity {p:.3f}   "
          f"L1-code match-avg {n:.4f}", flush=True)
    a, p, n, _ = evaluate(W1, W2, T2, Xtr, ytr, Xte, yte, B1, B2)
    print(f"  before, feedback ON    acc {a:.4f}   L2 purity {p:.3f}   "
          f"L1-code match-avg {n:.4f}", flush=True)

    W1f = W1[:, 0, :].astype(np.float32); wn = (W1f ** 2).sum(1)
    n1 = np.zeros(h); n2 = np.zeros(K2)
    res = []
    rng = np.random.default_rng(11)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), IMG_BATCH):
            ids = order[s:s + IMG_BATCH]
            w1, k, P, pool, j = forward(W1f, wn, W2, T2, Xtr[ids], B1, B2, h)
            np.add.at(C, (j, ytr[ids]), 1.0)
            T2 = table2(C)
            for t in np.unique(w1[k]):
                sel = P[k][w1[k] == t]
                if len(sel) >= 4:
                    n1[t] += len(sel)
                    e = min(max(len(sel) / n1[t], ETA_MIN), 1.0)
                    W1f[t] = E.geo_step(W1f[t:t + 1].astype(np.float64),
                                        sel.astype(np.float64), e)[0].astype(np.float32)
            wn = (W1f ** 2).sum(1)
            for t in np.unique(j):
                m2 = pool[j == t]
                n2[t] += len(m2)
                W2[t] += min(max(len(m2) / n2[t], ETA_MIN), 1.0) * (m2.mean(0) - W2[t])
                W2[t] /= np.linalg.norm(W2[t]) + EPS
        W1e = W1f[:, None, :].astype(np.float64)
        a, p, n, _ = evaluate(W1e, W2, T2, Xtr, ytr, Xte, yte, B1, B2)
        res.append({"epoch": ep + 1, "acc": a, "purity": p, "match_avg": n})
        print(f"  epoch {ep+1}  acc {a:.4f}   L2 purity {p:.3f}   "
              f"L1-code match-avg {n:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT / f"stack_k{TOPK}.npz", W1=W1f, W2=W2, counts=C)
    (OUT / f"stack_k{TOPK}.json").write_text(json.dumps(
        {"rounds": res, "reference": {"L1 alone table": 0.9540,
                                      "L2 no feedback": 0.9540,
                                      "L1 match-avg": 0.9487}}, indent=2))


if __name__ == "__main__":
    main()
