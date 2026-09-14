"""How much is each window's testimony worth?  The tables are never touched.

score(c) = log P(c) + sum over windows of  w[window, label] * (that window's centred vote)

A window's vote for label c is its usual score minus its own average over the ten labels, so a
positive vote means "this window favours c" and negative means "it argues against c".  With every
w = 1 this is exactly the tiles model, unchanged.  w can be read as how many independent
observations that window's testimony is worth: below 1 means "discount it, it is repeating what
other windows already said."

Three arms, same tables throughout:
  flat     w = 1                     -- today's model
  counted  w from how far the label moves that window's table -- no errors involved, pure counting
  learned  w nudged by the rule: when the answer comes out wrong, raise w for the window's vote for
           the true label and lower it for its vote for the label that won.  Magnitude only, one
           step, no derivatives through anything.  Clipped at 0 (a window cannot be trusted
           negatively) and at 1 if CAP (a window cannot be worth more than one observation).

    python importance.py [passes] [eta]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import run as R

OUT = Path(__file__).parent / "results"
NODES, C = R.NODES, R.C
PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0    # upper bound on a window's worth; must be > 1 or only the "decrease" half of the rule can fire


def votes(X, logT, batch=2000):
    """(n,49,10) centred per-window votes."""
    lt = np.ascontiguousarray(logT.transpose(1, 2, 0))
    out = np.zeros((len(X), NODES, C), np.float32)
    for b in range(0, len(X), batch):
        Xw = R.windows(X[b:b + batch]).astype(np.float64)
        s = np.matmul(np.ascontiguousarray(Xw.transpose(1, 0, 2)), lt).transpose(1, 0, 2)
        out[b:b + batch] = (s - s.mean(2, keepdims=True)).astype(np.float32)
    return out


def calib(tot, y):
    pred = tot.argmax(1)
    z = tot - tot.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = np.sort(tot, 1); gap = s[:, -1] - s[:, -2]
    out = dict(acc=round(float((pred == y).mean()), 4), mean_conf=round(float(p.max(1).mean()), 4),
               median_gap=round(float(np.median(gap)), 2),
               frac_gap_under_5=round(float((gap < 5).mean()), 4), by_gap={})
    for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
        m = (gap >= a) & (gap < b)
        if m.sum() > 20:
            out["by_gap"][f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == y[m]).mean()), 3), int(m.sum())]
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = R.load()
    n_c = np.bincount(ytr, minlength=C).astype(np.float64)
    logprior = np.log(n_c / n_c.sum())
    M = R.accumulate(Xtr, ytr, n_c)
    T = M / M.sum(2, keepdims=True)
    logT = np.log(T)

    Vtr, Vte = votes(Xtr, logT), votes(Xte, logT)
    res, W = {}, {}

    W["flat"] = np.ones((NODES, C), np.float32)

    # counted: how far the label moves this window's table, per label.  No errors, no updates.
    tv = 0.5 * np.abs(T - T.mean(0, keepdims=True)).sum(2).T                      # (49,10) in 0..1
    W["counted"] = (tv / tv.mean()).clip(0, CAP).astype(np.float32)

    # learned: the increase/decrease rule, online, one pass at a time
    scale = float(np.abs(Vtr).mean())                                             # so the step size means the same thing whatever the scores are
    Vtr_n = Vtr / scale
    w = np.ones((NODES, C), np.float32)
    rng = np.random.default_rng(0)
    hist = []
    for p in range(PASSES):
        order = rng.permutation(len(Vtr))
        wrong = 0
        for n in order:
            v = Vtr_n[n]                                                          # (49,10), scaled
            sc = (w * v).sum(0) * scale + logprior
            g = int(sc.argmax()); t = int(ytr[n])
            if g != t:
                wrong += 1
                w[:, t] += ETA * v[:, t]                                          # trust the windows that spoke for the truth
                w[:, g] -= ETA * v[:, g]                                          # distrust those that spoke for the winner
                np.clip(w, 0.0, CAP, out=w)
        hist.append(round(1 - wrong / len(order), 4))
        print(f"  pass {p+1}: training accuracy during the pass {hist[-1]:.4f}")
    W["learned"] = w

    for k, ww in W.items():
        res[k] = calib((Vte * ww[None]).sum(1) + logprior, yte)
        res[k]["w_mean"] = round(float(ww.mean()), 3)
        res[k]["w_frac_at_zero"] = round(float((ww <= 1e-6).mean()), 3)
        print(f"{k:<9} acc {res[k]['acc']:.4f}  conf {res[k]['mean_conf']:.4f}  median gap {res[k]['median_gap']:>6.2f}  "
              f"within 5 {res[k]['frac_gap_under_5']*100:>3.0f}%  mean w {res[k]['w_mean']:.2f}  w=0 on {res[k]['w_frac_at_zero']*100:.0f}%")

    res["passes"], res["eta"], res["cap"] = PASSES, ETA, CAP
    res["train_acc_by_pass"] = hist
    res["fitted_scorecard_ceiling"] = 0.9195
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "importance.json", "w"), indent=1)
    np.savez_compressed(OUT / "importance.npz", **{f"w_{k}": v for k, v in W.items()}, tables=T.astype(np.float32))
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
