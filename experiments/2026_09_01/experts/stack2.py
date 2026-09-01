"""Feedback again, this time with a spatially resolved signal.

The previous attempt sent ONE number down -- a single whole-image L2 winner --
to steer 331 L1 positions. It helped the code at read time (+2.1) and cost about
a point when trained through, and softening it into a mixture didn't help.

Now L2 emits 36 identities, one per 3x3-region window, and the windows overlap.
So every region of the pooled map is covered by up to 9 windows, each with an
opinion about what should be firing there. Average them and you get a per-region
expectation:

    expect[region, L1 template] = mean over the windows covering that region
                                  of what their winning L2 template expects

Then an L1 patch at some position is biased by the expectation of the region it
sits in -- a local signal, not a global one. Sixty-four regions each steering
their own neighbourhood, instead of one name steering everything.

Same two questions as before, separately: does it help at READ time, and does it
help when you TRAIN through it.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, settle as S, single as SG, layer2 as L2, conv2layer as C2

OUT = Path(__file__).resolve().parent / "results"
PG, WIN, WG, NL, EPS = C2.PG, C2.WIN, C2.WG, 10, 1e-12
K2, B1, EPOCHS, IMG_BATCH, ETA_MIN = 200, 1.5, 3, 128, 0.02
B_ = S.SIDE // PG
REG = ((np.arange(S.SIDE * S.SIDE) // S.SIDE) // B_) * PG + \
      ((np.arange(S.SIDE * S.SIDE) % S.SIDE) // B_)
COVER = np.zeros((PG, PG))
for i in range(WG):
    for j in range(WG):
        COVER[i:i + WIN, j:j + WIN] += 1


def expectation(W2, jwin, h):
    """(m, PG*PG, h) -- what the winning L2 windows expect of each region."""
    m = len(jwin)
    g = np.zeros((m, PG, PG, h), np.float32)
    for i in range(WG):
        for j in range(WG):
            g[:, i:i + WIN, j:j + WIN, :] += W2[jwin[:, i * WG + j]].reshape(
                m, WIN, WIN, h)
    return (g / COVER[None, :, :, None]).reshape(m, PG * PG, h)


def forward(W1f, wn, W2, Xb, b1, h):
    Q, keep = E.patches(Xb)
    m = len(Q)
    P = Q.reshape(-1, 25).astype(np.float32)
    k = keep.reshape(-1)
    Sc = P @ W1f.T
    err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
    w1 = err.argmin(1)
    for _ in range(2 if b1 > 0 else 1):
        P8 = C2.pooled8(np.where(keep.reshape(m, -1), w1.reshape(m, -1), -1), h)
        V = C2.windows(P8)
        j = (L2.cn(V.reshape(-1, V.shape[-1])).astype(np.float32) @ W2.T
             ).argmax(1).reshape(m, -1)
        if b1 <= 0:
            break
        ex = expectation(W2, j, h)                       # (m, 64, h)
        bias = ex[np.repeat(np.arange(m), S.SIDE * S.SIDE), np.tile(REG, m)]
        w1 = (err - b1 * bias).argmin(1)
        b1 = 0                                           # one refinement pass
    return w1, k, P, P8, j


def run(W1f, wn, W2, X, b1, h, chunk=256):
    js, pools = [], []
    for a in range(0, len(X), chunk):
        w1, k, _, P8, j = forward(W1f, wn, W2, X[a:a + chunk], b1, h)
        js.append(j)
        pools.append(P8.reshape(len(P8), -1))
    return np.concatenate(js), np.concatenate(pools)


def evaluate(W1f, wn, W2, Xtr, ytr, Xte, yte, b1, h):
    jtr, Ptr = run(W1f, wn, W2, Xtr, b1, h)
    jte, Pte = run(W1f, wn, W2, Xte, b1, h)
    T, N = C2.table(jtr, ytr, len(W2))
    P = np.arange(WG * WG)
    sc = np.stack([T[jte[a], P].sum(0) for a in range(len(jte))])
    acc = float((sc.argmax(1) == yte).mean())
    M = np.stack([Ptr[ytr == c].mean(0) for c in range(NL)])
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), EPS)
    nm = float(((Pte / np.maximum(np.linalg.norm(Pte, axis=1, keepdims=True), EPS))
                @ M.T).argmax(1) == yte).mean() if False else float(
        (((Pte / np.maximum(np.linalg.norm(Pte, axis=1, keepdims=True), EPS)) @ M.T)
         .argmax(1) == yte).mean())
    return acc, nm


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    W1 = np.load(OUT / "coadapt.npz")["W"].astype(np.float64)
    W2 = np.load(OUT / f"conv2_k{K2}.npz")["W2"].astype(np.float32)
    h = len(W1)
    W1f = W1[:, 0, :].astype(np.float32); wn = (W1f ** 2).sum(1)
    print(f"L1 {h}, L2 {K2} slid over {WG}x{WG} windows, beta {B1}\n", flush=True)
    for b in (0.0, B1):
        a, n = evaluate(W1f, wn, W2, Xtr, ytr, Xte, yte, b, h)
        print(f"  before, feedback {'ON ' if b else 'OFF'}   acc {a:.4f}   "
              f"L1-code match-avg {n:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    n1 = np.zeros(h); res = []
    rng = np.random.default_rng(13)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), IMG_BATCH):
            ids = order[s:s + IMG_BATCH]
            w1, k, P, _, _ = forward(W1f, wn, W2, Xtr[ids], B1, h)
            for t in np.unique(w1[k]):
                sel = P[k][w1[k] == t]
                if len(sel) >= 4:
                    n1[t] += len(sel)
                    e = min(max(len(sel) / n1[t], ETA_MIN), 1.0)
                    W1f[t] = E.geo_step(W1f[t:t + 1].astype(np.float64),
                                        sel.astype(np.float64), e)[0].astype(np.float32)
            wn = (W1f ** 2).sum(1)
        a, n = evaluate(W1f, wn, W2, Xtr, ytr, Xte, yte, B1, h)
        res.append({"epoch": ep + 1, "acc": a, "match_avg": n})
        print(f"  epoch {ep+1}  acc {a:.4f}   L1-code match-avg {n:.4f}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT / "stack2.npz", W1=W1f, W2=W2)
    (OUT / "stack2.json").write_text(json.dumps(
        {"rounds": res, "reference": {"sliding L2 no feedback": 0.9593,
                                      "whole-image L2 feedback, trained": 0.9453}},
        indent=2))


if __name__ == "__main__":
    main()
