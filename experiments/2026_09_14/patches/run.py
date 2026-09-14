"""One layer of patch nodes under a single parent whose value is the label.
Each node keeps ten tables, one per value of the parent, each shaped like the patch itself.

The picture (09-14):
  * the lower layer is what the higher layer CAUSED.  The parent C is the cause, the patch nodes
    are its effects: C -> every patch node.
  * ambiguity is not a choice between two parents, it is a spread over the VALUES of one parent.
  * a node looks at a 5x5 window of the image, stride 1, no padding: 24x24 = 576 nodes.
  * a node holds C tables of 5x5 cells.  Training: pick the table by the label, add the patch's
    pixel values into it cell by cell.  Then normalise each table to sum to 1, so it is a
    distribution over where ink lands inside that window given that value of the parent.
  * nothing is shared between nodes.  Each node's tables are counted only from its own window.
  * how the tables are READ is deliberately left open.

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

P, G = 5, 24                      # patch side, nodes per side: (28 - 5)/1 + 1 = 24, stride 1, no padding
NODES, CELLS, C = G * G, P * P, 10
N_TRAIN, N_TEST = 50000, 10000
BATCH = 2000
ALPHA = 1e-3                      # pseudo-ink per cell, so a window that never saw ink is uniform, not 0/0


def load(seed=0):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def windows(X):
    """(B,28,28) -> (B,576,25): every 5x5 window, stride 1, raw pixel values."""
    W = np.lib.stride_tricks.sliding_window_view(X, (P, P), axis=(1, 2))
    return np.ascontiguousarray(W).reshape(len(X), NODES, CELLS)


def accumulate(X, y):
    """The whole of training: pick the table by the label, add the pixel values in."""
    M = np.zeros((C, NODES * CELLS), np.float64)
    for b in range(0, len(X), BATCH):
        xb, yb = X[b:b + BATCH], y[b:b + BATCH]
        oh = np.zeros((C, len(xb)), np.float32)
        oh[yb, np.arange(len(xb))] = 1.0
        M += oh @ windows(xb).reshape(len(xb), -1)
    return M.reshape(C, NODES, CELLS)


def normalise(M):
    return ((M + ALPHA) / (M.sum(2, keepdims=True) + ALPHA * CELLS)).astype(np.float32)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    M = accumulate(Xtr, ytr)                                  # (10,576,25) raw ink sums
    T = normalise(M)                                          # (10,576,25) each sums to 1
    t_train = time.time() - t0

    mass = M.sum(2)                                           # (10,576) ink each node collected per label
    Tbar = T.mean(0, keepdims=True)                           # the node's label-average table
    spread = 0.5 * np.abs(T - Tbar).sum(2).mean(0)            # (576,) how far the label moves the table, 0..1
    ent = -(T * np.log2(np.maximum(T, 1e-12))).sum(2)         # (10,576) bits, uniform = 4.64

    # how redundant stride 1 is: a node against its right neighbour, on the 5x4 they share
    A = T.reshape(C, G, G, P, P)[:, :, :-1, :, 1:]
    B = T.reshape(C, G, G, P, P)[:, :, 1:, :, :-1]
    a, b = A.reshape(-1, 20), B.reshape(-1, 20)
    a, b = a / a.sum(1, keepdims=True), b / b.sum(1, keepdims=True)
    share_tv = float((0.5 * np.abs(a - b).sum(1)).mean())

    res = dict(
        tag=tag, patch=P, stride=1, padding=0, nodes=NODES, cells_per_table=CELLS,
        tables_per_node=C, numbers_trained=int(C * NODES * CELLS),
        n_train=N_TRAIN, n_test=N_TEST, alpha=ALPHA, train_seconds=round(t_train, 2),
        ink_total=round(float(M.sum()), 1),
        mass_min=round(float(mass.min()), 3), mass_median=round(float(np.median(mass)), 1),
        mass_max=round(float(mass.max()), 1),
        tables_under_1_ink=int((mass < 1.0).sum()),
        entropy_bits_mean=round(float(ent.mean()), 3), entropy_bits_min=round(float(ent.min()), 3),
        uniform_bits=round(float(np.log2(CELLS)), 3),
        spread_mean=round(float(spread.mean()), 4), spread_max=round(float(spread.max()), 4),
        neighbour_tv_on_shared=round(share_tv, 4),
    )
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{tag}.npz",
                        mass_raw=M.astype(np.float32), tables=T,
                        spread=spread.astype(np.float32), ent=ent.astype(np.float32),
                        Xte=Xte[:16], yte=yte[:16])
    print(json.dumps(res, indent=1))
    print(f"[{time.time() - t0:.1f}s] -> {OUT}/{tag}.npz")


if __name__ == "__main__":
    main()
