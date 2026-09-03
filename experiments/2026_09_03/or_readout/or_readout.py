"""Position-free codes: what does OR-ing the top-1 winners cost, and buy?

The champion describes an image as one winner per position and counts them in a
table indexed by (template, 6x6 cell, class). The question: collapse the 576
one-hot winners into ONE 400-bit vector with no position at all, and count on
that. Three readouts on the SAME trained templates:

    cells     T[t, cell, y]   the champion: where the template won matters
    count     T[t, y]         position summed out; a template present k times votes k times
    OR        T[t, y]         presence only; each template votes once, the user's proposal

And each readout on the test set shifted by (dx, dy) pixels, to see what the
position-free code is for.

Usage:  uv run python or_readout.py
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import rf_sweep as R

OUT = HERE / "results"
SIZES = [5, 9, 13]
SHIFTS = [(0, 0), (1, 1), (2, 2), (3, 3)]
K, NL = R.K, R.NL


def shift(X, dx, dy):
    """Roll images (n, 784) by dx columns and dy rows, zero-filled."""
    im = cp.asnumpy(X).reshape(-1, 28, 28)
    out = np.zeros_like(im)
    out[:, max(dy, 0):28 + min(dy, 0), max(dx, 0):28 + min(dx, 0)] = \
        im[:, max(-dy, 0):28 + min(-dy, 0), max(-dx, 0):28 + min(-dx, 0)]
    return cp.asarray(out.reshape(len(im), -1), cp.float32)


def logratio(N):
    pc = (N + R.ALPHA) / (N.sum(0, keepdims=True) + R.ALPHA * K)
    pm = (N.sum(1, keepdims=True) + R.ALPHA * NL) / (N.sum() + R.ALPHA * K * NL)
    return cp.log(pc) - cp.log(pm)


def presence(idx, keep, count):
    """(n, K) matrix: number of positions each template won (count) or 1/0 (OR)."""
    n = len(idx)
    M = cp.zeros((n, K), cp.float32)
    rows = cp.repeat(cp.arange(n), idx.shape[1])
    flat = (rows * K + idx.reshape(-1))[keep.reshape(-1)]
    M += cp.bincount(flat, minlength=n * K).reshape(n, K)
    return cp.minimum(M, 1.0) if not count else M


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    Y = cp.eye(NL, dtype=cp.float32)[ytr_g]
    res = {}
    print(f"{'size':>5}{'readout':>8}" + "".join(f"{f'shift {s[0]},{s[1]}':>12}" for s in SHIFTS)
          + f"{'active bits':>13}")
    for ps in SIZES:
        rig = R.Rig(ps)
        _, W = R.run(rig, True, Xtr, ytr_g, Xte, yte_g, 7)
        W = cp.asarray(W)
        itr, ktr = rig.code(W, Xtr)
        T_cells = rig.table_from(rig.build_table(itr, ytr_g, ktr))
        tabs = {}
        for name, cnt in (("count", True), ("OR", False)):
            M = presence(itr, ktr, cnt)
            tabs[name] = logratio((M.T @ Y).astype(cp.float64))
        bits = float(presence(itr, ktr, False).sum(1).mean())
        for name in ("cells", "count", "OR"):
            row = []
            for (dx, dy) in SHIFTS:
                ct = rig.code(W, shift(Xte, dx, dy))
                if name == "cells":
                    acc = rig.accuracy(T_cells, ct, yte_g)
                else:
                    M = presence(ct[0], ct[1], name == "count")
                    acc = float(((M @ tabs[name]).argmax(1) == yte_g).mean())
                row.append(acc)
            res[f"{ps}|{name}"] = row
            print(f"{ps:>5}{name:>8}" + "".join(f"{a:>12.4f}" for a in row)
                  + (f"{bits:>13.0f}/{K}" if name == "OR" else ""), flush=True)
    (OUT / "or_readout.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
