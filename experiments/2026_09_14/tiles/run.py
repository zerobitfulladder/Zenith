"""Non-overlapping tiles: 4x4 windows at stride 4, which cover 28x28 exactly with no padding.

Why this shape.  With stride 1 every pixel sat under 25 windows and was counted 25 times over, so
the machine was always certain (mean top guess 0.998) and being unsure stopped meaning anything.
At stride 4 every pixel has EXACTLY ONE parent window: each piece of evidence is counted once.
49 windows x 16 cells = 784 = the picture, with nothing shared and nothing left over.

Same build as before otherwise: each window keeps 10 grids, one per value of the parent C, shaped
like the window itself.  Training adds the picture's pixel values into the grid picked by the label,
with a floor under every pixel (fix 2: a grid speaks only as loudly as its evidence).  Reading is
the three scores from before; a blank patch is muted under the shape read (it has nothing to divide
by), which is the version that measured best.

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

P, G = 4, 7                       # 4x4 windows, 7x7 of them, stride 4, no padding, no overlap
NODES, CELLS, C = G * G, P * P, 10
N_TRAIN, N_TEST, BATCH = 50000, 10000, 2000
FLOOR = 0.002                     # under every pixel, in TRAINING only (the reading half does nothing / hurts)


def load(seed=0):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def windows(X):
    """(B,28,28) -> (B,49,16).  Every pixel lands in exactly one window."""
    return X.reshape(-1, G, P, G, P).transpose(0, 1, 3, 2, 4).reshape(-1, NODES, CELLS)


def accumulate(X, y, n_c):
    M = np.zeros((C, NODES * CELLS), np.float64)
    for b in range(0, len(X), BATCH):
        xb, yb = X[b:b + BATCH], y[b:b + BATCH]
        oh = np.zeros((C, len(xb)), np.float32)
        oh[yb, np.arange(len(xb))] = 1.0
        M += oh @ windows(xb).reshape(len(xb), -1)
    M = M.reshape(C, NODES, CELLS) + n_c[:, None, None] * FLOOR      # the floor, collected over training
    return M


def scores(Xte, T, logT, logprior):
    """ink, shape, corr -- totals over the 49 windows."""
    n = len(Xte)
    tot = [np.zeros((n, C)) for _ in range(3)]
    lt_flat = logT.reshape(C, -1).T                                   # (784,10)
    lt = np.ascontiguousarray(logT.transpose(1, 2, 0))                # (49,16,10)
    Tc = T - T.mean(2, keepdims=True); tn = np.linalg.norm(Tc, axis=2)
    Tct = np.ascontiguousarray(Tc.transpose(1, 2, 0))
    for b in range(0, n, BATCH):
        Xw = windows(Xte[b:b + BATCH]).astype(np.float64)
        tot[0][b:b + BATCH] = Xw.reshape(len(Xw), -1) @ lt_flat

        ink = Xw.sum(2); blank = ink < 1e-6
        Xn = Xw / np.maximum(ink, 1e-12)[:, :, None]
        s2 = np.matmul(np.ascontiguousarray(Xn.transpose(1, 0, 2)), lt).transpose(1, 0, 2)
        tot[1][b:b + BATCH] = np.where(blank[:, :, None], 0.0, s2).sum(1)

        Xc = Xw - Xw.mean(2, keepdims=True); xn = np.linalg.norm(Xc, axis=2)
        num = np.matmul(np.ascontiguousarray(Xc.transpose(1, 0, 2)), Tct).transpose(1, 0, 2)
        den = xn[:, :, None] * tn.T[None, :, :]
        tot[2][b:b + BATCH] = np.where(den > 1e-8, num / np.maximum(den, 1e-12), 0.0).sum(1)
    tot[0] += logprior
    return tot


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tiles"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_c = np.bincount(ytr, minlength=C).astype(np.float64)
    prior = n_c / n_c.sum(); logprior = np.log(prior)

    M = accumulate(Xtr, ytr, n_c)
    T = (M / M.sum(2, keepdims=True)).astype(np.float64)
    logT = np.log(T)
    t_train = time.time() - t0

    tot = scores(Xte, T, logT, logprior)
    accs = [float((t.argmax(1) == yte).mean()) for t in tot]

    e = tot[0]
    z = e - e.max(1, keepdims=True); post = np.exp(z); post /= post.sum(1, keepdims=True)
    s = np.sort(e, 1); gap = s[:, -1] - s[:, -2]
    conf = post.max(1); pred = e.argmax(1)

    acc_by_gap = {}
    for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        m = (gap >= a) & (gap < b)
        if m.sum() > 20:
            acc_by_gap[f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == yte[m]).mean()), 3), int(m.sum())]

    mass = M.sum(2)
    spread = 0.5 * np.abs(T - T.mean(0, keepdims=True)).sum(2).mean(0)
    amb = np.argsort(gap)[:12]

    res = dict(tag=tag, patch=P, stride=P, nodes=NODES, cells=CELLS, floor=FLOOR,
               numbers_trained=int(C * NODES * CELLS), n_train=N_TRAIN, n_test=N_TEST,
               train_seconds=round(t_train, 2), seconds=round(time.time() - t0, 2),
               acc_ink=round(accs[0], 4), acc_shape=round(accs[1], 4), acc_corr=round(accs[2], 4),
               mean_conf=round(float(conf.mean()), 4),
               frac_conf_over_99=round(float((conf > 0.99).mean()), 4),
               median_gap=round(float(np.median(gap)), 1),
               frac_gap_under_5=round(float((gap < 5).mean()), 4),
               acc_by_gap=acc_by_gap,
               mass_min=round(float(mass.min()), 1), mass_median=round(float(np.median(mass)), 1))
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{tag}.npz", tables=T.astype(np.float32), mass=M.astype(np.float32),
                        spread=spread.astype(np.float32), conf=conf.astype(np.float32),
                        gap=gap.astype(np.float32), pred=pred.astype(np.int8), yte=yte.astype(np.int8),
                        Xamb=Xte[amb], yamb=yte[amb], post_amb=post[amb].astype(np.float32))
    print(json.dumps(res, indent=1))
    print(f"[{time.time()-t0:.1f}s] -> {OUT}/{tag}.npz")


if __name__ == "__main__":
    main()
