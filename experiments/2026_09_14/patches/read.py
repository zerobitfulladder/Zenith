"""Brute-force read: every patch scores every value of the parent, then the scores are summed.

Three per-patch match scores, over the 25 cells of a window:
  1. ink   S = sum over cells of  x * log T          -- a real log-likelihood (a bag of ink units,
           each landing at a cell drawn from the table).  Summing these over nodes IS Bayes.
  2. shape S = sum over cells of  (x / patch ink) * log T   -- every node weighted equally however
           much ink it holds.  Not a likelihood, so summing is a vote, not Bayes.
  3. corr  S = correlation of the patch with the table over the 25 cells.  A plain similarity.
A blank patch scores 0 under all three: these tables say where ink lands, so no ink means no word.

Also checks, by measurement, that read 1 collapses: the per-node sum is rebuilt as one weight
image per label matched against the raw pixels, and the two are compared number by number.

    python read.py [tag]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import run as R

HERE = Path(__file__).parent
OUT = HERE / "results"
P, G, NODES, CELLS, C = R.P, R.G, R.NODES, R.CELLS, R.C
BATCH = 1000
N_SHOW = 3


def per_node(Xw, T, logT):
    """(n,576,25) patches -> three (n,576,10) score maps."""
    lt = np.ascontiguousarray(logT.transpose(1, 2, 0))                     # (576,25,10)
    s1 = np.matmul(np.ascontiguousarray(Xw.transpose(1, 0, 2)), lt).transpose(1, 0, 2)

    ink = Xw.sum(2)
    Xn = Xw / np.maximum(ink, 1e-8)[:, :, None]
    s2 = np.matmul(np.ascontiguousarray(Xn.transpose(1, 0, 2)), lt).transpose(1, 0, 2)
    s2 = np.where((ink > 1e-6)[:, :, None], s2, 0.0)

    Tc = T - T.mean(2, keepdims=True)
    tn = np.linalg.norm(Tc, axis=2)                                        # (10,576)
    Xc = Xw - Xw.mean(2, keepdims=True)
    xn = np.linalg.norm(Xc, axis=2)                                        # (n,576)
    num = np.matmul(np.ascontiguousarray(Xc.transpose(1, 0, 2)),
                    np.ascontiguousarray(Tc.transpose(1, 2, 0))).transpose(1, 0, 2)
    den = xn[:, :, None] * tn.T[None, :, :]
    s3 = np.where(den > 1e-8, num / np.maximum(den, 1e-12), 0.0)
    return s1.astype(np.float32), s2.astype(np.float32), s3.astype(np.float32)


def weight_images(logT):
    """what read 1 collapses to: one 28x28 weight image per label."""
    W = np.zeros((C, 28, 28), np.float64)
    L = logT.reshape(C, G, G, P, P)
    for r in range(G):
        for c in range(G):
            W[:, r:r + P, c:c + P] += L[:, r, c]
    return W


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
    t0 = time.time()
    Z = np.load(OUT / f"{tag}.npz")
    T = Z["tables"].astype(np.float64)                                     # (10,576,25)
    logT = np.log(np.maximum(T, 1e-12))
    Xtr, ytr, Xte, yte = R.load()
    prior = np.bincount(ytr, minlength=C) / len(ytr)
    logprior = np.log(prior)

    tot = [np.zeros((len(Xte), C)) for _ in range(3)]
    for b in range(0, len(Xte), BATCH):
        Xw = R.windows(Xte[b:b + BATCH]).astype(np.float64)
        for k, s in enumerate(per_node(Xw, T, logT)):
            tot[k][b:b + BATCH] = s.sum(1)
    tot[0] += logprior                                                     # only read 1 has a scale where the prior means anything

    # --- does read 1 really collapse to a linear classifier? ------------------------------------
    W = weight_images(logT)
    lin = Xte.reshape(len(Xte), -1) @ W.reshape(C, -1).T + logprior
    d = np.abs(lin - tot[0])
    collapse_max_abs = float(d.max())
    collapse_rel = float(d.max() / np.abs(tot[0]).mean())
    collapse_same_pred = float((lin.argmax(1) == tot[0].argmax(1)).mean())

    names = ["ink  (sum x.logT)", "shape  (sum x/ink . logT)", "corr  (patch vs table)"]
    accs = [float((t.argmax(1) == yte).mean()) for t in tot]

    # posterior of read 1 (the only one that is a likelihood)
    e = tot[0]
    z = e - e.max(1, keepdims=True); post = np.exp(z); post /= post.sum(1, keepdims=True)
    srt = np.sort(e, 1); gap1 = srt[:, -1] - srt[:, -2]
    conf = post.max(1)

    gaps = []
    for t in tot:
        s = np.sort(t, 1); sc = np.abs(t).mean()
        gaps.append((s[:, -1] - s[:, -2]) / sc)                            # gap in units of the score's own scale

    amb = np.argsort(gap1)[:6]
    clean = np.argsort(-gap1)[:2]
    show = np.concatenate([clean[:1], amb[:N_SHOW - 1]])

    Xw = R.windows(Xte[show]).astype(np.float64)
    maps = per_node(Xw, T, logT)

    res = dict(tag=tag, n_test=len(Xte), seconds=round(time.time() - t0, 2),
               reads={n: round(a, 4) for n, a in zip(names, accs)},
               ink_mean_conf=round(float(conf.mean()), 4),
               ink_frac_conf_over_99=round(float((conf > 0.99).mean()), 4),
               ink_median_gap_nats=round(float(np.median(gap1)), 1),
               ink_frac_gap_under_5_nats=round(float((gap1 < 5).mean()), 4),
               collapse_max_abs_diff=collapse_max_abs, collapse_rel_diff=collapse_rel,
               collapse_same_prediction=collapse_same_pred,
               acc_by_gap_ink={})
    for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        m = (gap1 >= a) & (gap1 < b)
        if m.sum() > 20:
            res["acc_by_gap_ink"][f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((tot[0].argmax(1)[m] == yte[m]).mean()), 3), int(m.sum())]

    json.dump(res, open(OUT / f"{tag}_read.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{tag}_read.npz",
                        maps1=maps[0], maps2=maps[1], maps3=maps[2],
                        Xshow=Xte[show], yshow=yte[show], post_show=post[show].astype(np.float32),
                        tot1=tot[0][show].astype(np.float32), tot2=tot[1][show].astype(np.float32),
                        tot3=tot[2][show].astype(np.float32),
                        conf=conf.astype(np.float32), gap1=gap1.astype(np.float32),
                        W=W.astype(np.float32), yte=yte.astype(np.int8),
                        pred1=tot[0].argmax(1).astype(np.int8))
    print(json.dumps(res, indent=1))
    print(f"[{time.time() - t0:.1f}s] -> {OUT}/{tag}_read.npz")


if __name__ == "__main__":
    main()
