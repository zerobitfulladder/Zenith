"""Widen layer 1: more templates, and more than one winner per position.

The realisation: our pipeline expands once (784 pixels -> a 103,680-slot sparse
map) and then compresses hard twice. The readout that wins reads the EXPANDED
code; every compressed one loses. V1 and the cerebellum both expand enormously
and then get away with a shallow readout -- width instead of depth.

Two axes, both cheap at layer 1 (a template is 25 numbers):

    K1     180 -> 400 -> 800 templates. More slots, same ~331 active per
           image, so the code gets sparser and more separable.
    top-k  report the k best templates per position instead of just the winner.
           LEARNING stays top-1 -- only the best rotates -- so this changes the
           message, not the fitting. That is the sparsity-gradient idea from the
           Zenith work, which raised probes to 0.9626/0.9482/0.8914 and which we
           have not been using.

Top-k also adds REDUNDANCY, which may matter more than accuracy: with one winner
per position there is exactly one way to represent a patch, so feedback cannot
nudge the code without damaging the content. With several active there is slack
for a top-down signal to move around in -- a candidate explanation for why every
feedback experiment today cost accuracy.

Training is the co-adaptive loop, vectorised (the per-template Python loop does
not scale to 800), using km.py's update: each winner moves toward the mean of
what it won, with a 1/n step so it settles.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, coadapt as CA

OUT = Path(__file__).resolve().parent / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
GRID, NL, EPS, ALPHA = 6, 10, 1e-12, 1.0
SIDE = S.SIDE
CONFIGS = [(180, 1), (400, 1), (800, 1), (180, 3), (400, 3), (800, 3)]
EPOCHS, IMG_BATCH, BETA, ETA_MIN, MIN_S = 3, 64, 1.5, 0.02, 4


def train(Xtr, ytr, K, rng):
    """Co-adaptive, vectorised. Table biases the assignment; winner-mean update."""
    W = rng.standard_normal((K, 25)).astype(np.float32)
    W -= W.mean(1, keepdims=True); W /= np.linalg.norm(W, axis=1, keepdims=True) + EPS
    N = np.zeros((K, CA.NC, NL)); T = CA.table_from(N); n = np.zeros(K)
    order = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            ids = order[s:s + IMG_BATCH]
            Q, keep = E.patches(Xtr[ids]); m = len(Q)
            P = Q.reshape(-1, 25).astype(np.float32); k = keep.reshape(-1)
            Sc = P @ W.T
            err = 1.0 - Sc * Sc                                  # unit templates
            cells = np.tile(CA.CELL, m)
            w0 = err.argmin(1)
            sc = (T[w0, cells] * k[:, None]).reshape(m, -1, NL).sum(1)
            z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)
            q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            bias = np.einsum('ik,ihk->ih', np.repeat(q, SIDE * SIDE, 0),
                             np.ascontiguousarray(T.transpose(1, 0, 2))[cells],
                             optimize=True)
            win = (err - BETA * bias).argmin(1)
            lab = np.repeat(ytr[ids], SIDE * SIDE)
            np.add.at(N, (win[k], cells[k], lab[k]), 1.0)
            T = CA.table_from(N)
            wv, Pv = win[k], P[k]                                # vectorised update
            cnt = np.bincount(wv, minlength=K)
            sums = np.zeros((K, 25), np.float32)
            np.add.at(sums, wv, Pv)
            live = cnt >= MIN_S
            n[live] += cnt[live]
            eta = np.clip(cnt[live] / n[live], ETA_MIN, 1.0)[:, None]
            W[live] += eta * (sums[live] / cnt[live, None] - W[live])
            W /= np.linalg.norm(W, axis=1, keepdims=True) + EPS
    return W, int((cnt == 0).sum())


def winners_topk(W, X, topk, chunk=200):
    out = np.full((len(X), SIDE * SIDE, topk), -1, np.int16)
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        P = Q.reshape(-1, 25).astype(np.float32)
        Sc = np.abs(P @ W.T)
        idx = np.argpartition(-Sc, topk, axis=1)[:, :topk]
        o = np.argsort(-np.take_along_axis(Sc, idx, 1), axis=1)
        idx = np.take_along_axis(idx, o, 1).reshape(len(Q), -1, topk)
        out[a:a + len(Q)] = np.where(keep[..., None], idx, -1)
    return out


def table_and_score(itr, ytr, ite, K, g):
    cell = B.cellmap(g)
    U = itr.shape[2]
    N = np.zeros((K, g * g, NL))
    for u in range(U):
        r, c = np.nonzero(itr[:, :, u] >= 0)
        np.add.at(N, (itr[r, c, u], cell[c], ytr[r]), 1.0)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * K * NL))
    T = (np.log(pc) - np.log(pm)).astype(np.float32)
    Tc = T[:, cell, :]
    P = np.arange(SIDE * SIDE)
    out = np.zeros((len(ite), NL))
    for a in range(0, len(ite), 400):
        b = ite[a:a + 400]
        for u in range(U):
            bu = b[:, :, u]; v = bu >= 0
            out[a:a + len(b)] += (Tc[np.where(v, bu, 0), P[None, :]] * v[..., None]).sum(1)
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    print(f"{DS}\n  K1   top-k   slots      active    accuracy", flush=True)
    res = {}
    for K, tk in CONFIGS:
        t1 = time.time()
        W, dead = train(Xtr, ytr, K, np.random.default_rng(7))
        itr, ite = winners_topk(W, Xtr, tk), winners_topk(W, Xte, tk)
        acc = float((table_and_score(itr, ytr, ite, K, GRID).argmax(1) == yte).mean())
        act = float((ite >= 0).sum(axis=(1, 2)).mean())
        res[f"K{K}_top{tk}"] = {"acc": acc, "slots": SIDE * SIDE * K, "active": act,
                                "dead": dead, "params": K * 25,
                                "seconds": round(time.time() - t1, 1)}
        print(f"  {K:<5}{tk:<7}{SIDE*SIDE*K:>8,}{act:>10.0f}    {acc:.4f}   "
              f"dead {dead}  ({time.time()-t1:.0f}s)", flush=True)
    res["reference"] = {"K180 top1 grid6 (today's best)": 0.9650 if DS == "mnist" else 0.8293}
    (OUT / f"expand_{DS}.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
