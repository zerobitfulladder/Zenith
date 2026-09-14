"""Counting whole arrangements, and how many levels a pixel should be cut into.

A window's state is the WHOLE arrangement of its pixels, not one number per position, so the table
never assumes the pixels inside a window are independent -- it stores the combinations it actually
saw.  "Everything off" is one arrangement among the rest, so a blank window speaks like any other.

The price: with L levels per pixel and p pixels in a window there are L^p arrangements to count,
which grows faster in L than it does in the window size.  This run measures that trade directly:
hold the window, vary the levels.

Pixels are cut into L levels by floor(x * L); L = 2 is ink-or-not at 0.5.
Imaginary mass A is spread flat over the whole alphabet (A/S per column) so it means the same thing
whatever the alphabet size.

    python run.py [tag]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"

N_TRAIN, N_TEST, C = 50000, 10000, 10
CONFIGS = [(3, 2), (3, 3), (4, 2), (2, 2), (2, 4), (2, 8)]        # (window side, levels per pixel)
ALPHAS = [1.0, 10.0, 100.0]                               # total imaginary mass per (window, label)


def load(seed=0):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def codes(X, P, L):
    """each window -> one integer naming its whole arrangement."""
    pad = (-28) % P                                                  # pad up to a whole number of windows
    if pad:
        lo = pad // 2
        X = np.pad(X, ((0, 0), (lo, pad - lo), (lo, pad - lo)))
    G = X.shape[1] // P
    q = np.minimum((X * L).astype(np.int64), L - 1)
    q = q.reshape(-1, G, P, G, P).transpose(0, 1, 3, 2, 4).reshape(-1, G * G, P * P)
    pw = (L ** np.arange(P * P)).astype(np.int64)
    return (q * pw).sum(2), G * G


def calib(tot, y):
    pred = tot.argmax(1)
    z = tot - tot.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = np.sort(tot, 1); gap = s[:, -1] - s[:, -2]
    out = dict(acc=round(float((pred == y).mean()), 4), mean_conf=round(float(p.max(1).mean()), 4),
               median_gap=round(float(np.median(gap)), 1),
               frac_gap_under_5=round(float((gap < 5).mean()), 4), by_gap={})
    for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
        m = (gap >= a) & (gap < b)
        if m.sum() > 20:
            out["by_gap"][f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == y[m]).mean()), 3), int(m.sum())]
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "levels"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_c = np.bincount(ytr, minlength=C).astype(np.float64)
    logprior = np.log(n_c / n_c.sum())
    rows = []

    for P, L in CONFIGS:
        S = L ** (P * P)
        ctr, NODES = codes(Xtr, P, L)
        cte, _ = codes(Xte, P, L)
        cnt = np.zeros((NODES, C, S), np.int32)
        for i in range(NODES):
            cnt[i] = np.bincount(ytr * S + ctr[:, i], minlength=C * S).reshape(C, S)

        seen = (cnt.sum(1) > 0).sum(1)
        once = float((cnt.sum(1)[np.arange(NODES)[:, None], ctr.T].T == 1).mean())   # share of test-time... (train codes)
        best = None
        for A in ALPHAS:
            a = A / S
            logP = np.log((cnt + a) / (cnt.sum(2, keepdims=True) + A)).astype(np.float32)
            tot = np.tile(logprior, (len(cte), 1))
            for i in range(NODES):
                tot += logP[i][:, cte[:, i]].T
            r = calib(tot, yte); r["alpha_total"] = A
            if best is None or r["acc"] > best["acc"]:
                best = r
        rows.append(dict(window=f"{P}x{P}", levels=L, alphabet=S, nodes=NODES,
                         numbers=int(NODES * C * S), seen_mean=round(float(seen.mean()), 1),
                         train_share_on_singleton=round(once, 4), **best))
        print(f"{P}x{P}, {L} levels: alphabet {S:>6}, {NODES:>3} windows, {seen.mean():>7.1f} seen/window, "
              f"{once*100:>4.1f}% of training pictures on a once-seen arrangement  ->  acc {best['acc']:.4f} "
              f"(A={best['alpha_total']:g})  conf {best['mean_conf']:.3f}  gap {best['median_gap']:.1f}")
        del cnt

    json.dump(dict(rows=rows, seconds=round(time.time() - t0, 1)), open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
