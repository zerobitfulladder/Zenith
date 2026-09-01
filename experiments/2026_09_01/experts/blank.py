"""Does absence need its own symbol, and does the cell grid need to be finer?

A 3 is an 8 with the left side left open; a 7 is a 9 with the loop left open.
The code cannot see that, because blank patches fail the flatness gate and emit
nothing, and the score sums only over positions that FIRED. Absence is silent,
never negative -- so one class's code can be a subset of another's with no
penalty. 7 is the most under-predicted class by a wide margin (-26) and 7->9 is
the largest single confusion (19).

Two dials, crossed:

    blank    give empty positions a symbol of their own (a 31st index), so
             "nothing here" votes like anything else
    grid     4x4 cells of 6x6 positions (as before), 6x6, or 8x8 of 3x3 --
             a gap sitting beside a stroke INSIDE one cell is invisible, so
             the blank symbol may need the finer grid to pay off

Prediction, before running: the open-versus-closed confusions (7->9, 3->8,
6->8, 0->8) should fall while shape confusions (2->3, 4->9) barely move. If
everything improves evenly, this is just added capacity and the story is wrong.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, settle as S

OUT = Path(__file__).resolve().parent / "results"
SIDE, NL, ALPHA = S.SIDE, 10, 1.0
GRIDS = [4, 6, 8]
OPEN_CLOSED = [(7, 9), (3, 8), (6, 8), (0, 8), (9, 8), (1, 8)]
SHAPE = [(2, 3), (4, 9), (3, 2), (9, 4), (5, 3), (2, 8)]


def winners(W, X, chunk=500):
    idx = np.full((len(X), SIDE * SIDE), -1, np.int16)
    for a in range(0, len(X), chunk):
        F, KP = S.fits(W, X[a:a + chunk])
        idx[a:a + len(F)] = np.where(KP, F.argmax(1), -1)
    return idx


def cellmap(g):
    b = SIDE // g
    return (((np.arange(SIDE * SIDE) // SIDE) // b) * g
            + ((np.arange(SIDE * SIDE) % SIDE) // b))


def table(idx, y, nsym, g):
    cell = cellmap(g)
    N = np.zeros((nsym, g * g, NL))
    r, c = np.nonzero(idx >= 0)
    np.add.at(N, (idx[r, c], cell[c], y[r]), 1.0)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * nsym)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * nsym * NL))
    return np.log(pc) - np.log(pm)


def score(idx, T, g, chunk=500):
    Tc = T[:, cellmap(g), :]
    P = np.arange(SIDE * SIDE)
    out = np.zeros((len(idx), NL))
    for a in range(0, len(idx), chunk):
        b = idx[a:a + chunk]
        v = b >= 0
        out[a:a + len(b)] = (Tc[np.where(v, b, 0), P[None, :]] * v[..., None]).sum(1)
    return out


def main():
    t0 = time.time()
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()
    itr, ite = winners(W, Xtr), winners(W, Xte)
    print(f"filled positions per image: {float((ite >= 0).sum(1).mean()):.0f} of {SIDE*SIDE}",
          flush=True)

    res, preds = {}, {}
    for blank in (False, True):
        a_tr = np.where(itr >= 0, itr, h) if blank else itr
        a_te = np.where(ite >= 0, ite, h) if blank else ite
        nsym = h + 1 if blank else h
        for g in GRIDS:
            T = table(a_tr, ytr, nsym, g)
            p = score(a_te, T, g).argmax(1)
            acc = float((p == yte).mean())
            key = f"blank={blank} grid={g}"
            preds[key] = p
            cm = np.zeros((NL, NL), int); np.add.at(cm, (yte, p), 1)
            oc = int(sum(cm[a, b] for a, b in OPEN_CLOSED))
            sh = int(sum(cm[a, b] for a, b in SHAPE))
            res[key] = {"acc": acc, "open_closed_errors": oc, "shape_errors": sh,
                        "cells": g * g, "symbols": nsym,
                        "params": int(nsym * g * g * NL)}
            print(f"  blank {str(blank):<5} grid {g}x{g}  acc {acc:.4f}  "
                  f"open/closed errors {oc:3d}  shape errors {sh:3d}  "
                  f"table {nsym*g*g*NL:,}", flush=True)

    base = res["blank=False grid=4"]
    print(f"\n  baseline (blank off, 4x4): acc {base['acc']:.4f}  "
          f"open/closed {base['open_closed_errors']}  shape {base['shape_errors']}")
    best = max(res, key=lambda k: res[k]["acc"])
    print(f"  best: {best}  {res[best]['acc']:.4f}")
    cm = np.zeros((NL, NL), int); np.add.at(cm, (yte, preds[best]), 1)
    print("\n  the pairs, baseline -> best")
    for a, b in OPEN_CLOSED:
        cm0 = np.zeros((NL, NL), int)
        np.add.at(cm0, (yte, preds["blank=False grid=4"]), 1)
        print(f"    {a}->{b}   {cm0[a,b]:3d} -> {cm[a,b]:3d}   (open/closed)")
    for a, b in SHAPE:
        cm0 = np.zeros((NL, NL), int)
        np.add.at(cm0, (yte, preds["blank=False grid=4"]), 1)
        print(f"    {a}->{b}   {cm0[a,b]:3d} -> {cm[a,b]:3d}   (shape)")
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "blank.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
