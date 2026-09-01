"""How much of the ceiling is estimation noise? Triple the data and see.

The table holds 400 x 36 x 10 = 144,000 bins and gets ~4.0M patch firings from
12,000 images -- about 28 counts per bin, which is thin. Two things we measured
may be consequences rather than facts about the architecture:

    800 templates lost to 400 (0.9687 vs 0.9707)   twice the bins, half the counts
    T_down feedback was neutral                     6.4M bins, under 1 count each

So: same everything, 12,000 vs 40,000 training images, at both vocabulary sizes.
The TEST SET IS HELD FIXED (the last 3,000 images) across all four runs, so the
comparison is exact rather than approximate -- which also means the 12k rows here
are re-run on the new split rather than quoted from earlier.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E
import gpu_stack as G

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
NL, GRID, SIDE = 10, G.GRID1, G.SIDE
SIZES, KS, N_TEST = [12000, 40000], [400, 800], 3000


def load(ds, n_train):
    d = Path("/home/lavender/Projects/Zenith/data")
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_images.npy").astype(np.float32).reshape(-1, 784)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(0).permutation(len(X))
    X, y = X[p], y[p]
    return (cp.asarray(X[:n_train]), cp.asarray(y[:n_train]),
            cp.asarray(X[-N_TEST:]), cp.asarray(y[-N_TEST:]))


def main():
    print(f"{DS}   test set fixed at the last {N_TEST} images\n", flush=True)
    print("   train    K1    counts/bin   accuracy", flush=True)
    res = {}
    for n in SIZES:
        Xtr, ytr, Xte, yte = load(DS, n)
        for K in KS:
            t0 = time.time()
            G.K1 = K
            W1, _ = G.l1_train(Xtr, ytr, np.random.default_rng(7))
            itr, mtr, ktr = G.l1_code(W1, Xtr)
            ite, mte, kte = G.l1_code(W1, Xte)
            T1, N1 = G.build_table(itr, G.C1, ytr, K, GRID, ktr)
            acc = float((G.score(T1, ite, G.C1, SIDE * SIDE, kte).argmax(1) == yte).mean())
            per_bin = float(N1.sum() / N1.size)
            res[f"n{n}_K{K}"] = {"acc": acc, "counts_per_bin": per_bin,
                                 "bins": int(N1.size), "seconds": round(time.time() - t0, 1)}
            print(f"  {n:<9}{K:<6}{per_bin:<13.1f}{acc:.4f}   ({time.time()-t0:.0f}s)",
                  flush=True)
        del Xtr, ytr, Xte, yte
        cp.get_default_memory_pool().free_all_blocks()
    G.K1 = 400
    (OUT / f"scale_{DS}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
