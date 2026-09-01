"""What does a trained classifier get on top of the tables?

argmax over the table's ten class scores is the cheapest possible readout. Two
richer things to try, and a distinction that matters:

    the SUMMED evidence is only 10 numbers -- a linear classifier on it can do
    little more than rescale the classes

    the PER-CELL evidence is 36 x 10 = 360 numbers -- it keeps WHICH REGION
    voted for what, which the sum throws away

So the ladder is: argmax, linear on 10, linear on 360, then the same with layer
2's evidence concatenated, then a probe on the raw pooled code as the ceiling.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts")); sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E, readouts as R
import gpu_stack as G, gpu_merge as M, gpu_bidir as BD

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
K1, K2, NL, SIDE, GRID = G.K1, 1000, 10, G.SIDE, G.GRID1


def per_cell(T, win, cells, valid, g):
    """(n, g*g, 10) -- evidence kept separately for each region."""
    n, P = win.shape
    C = cp.tile(cells, n).reshape(n, P)
    ev = T[win, C] * valid[..., None]
    out = cp.zeros((n, g * g, NL), cp.float32)
    for c in range(g * g):
        out[:, c] = (ev * (C == c)[..., None]).sum(1)
    return out


def lin(Atr, ytr, Ate, yte, hid=0):
    net = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax",
                    hidden=hid, epochs=40)
    return float((R.predict_net(net, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    W1, _ = G.l1_train(Xtr, ytr_g, np.random.default_rng(7))
    itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)
    T1, _ = G.build_table(itr, G.C1, ytr_g, K1, GRID, ktr)

    c1tr = per_cell(T1, itr, G.C1, ktr, GRID); c1te = per_cell(T1, ite, G.C1, kte, GRID)
    e1tr, e1te = c1tr.sum(1), c1te.sum(1)
    base = float((e1te.argmax(1) == yte_g).mean())
    print(f"{DS}   L1 {K1} templates, {GRID}x{GRID} table\n", flush=True)
    print(f"  L1 + table, argmax                      {base:.4f}", flush=True)

    g = lambda a: cp.asnumpy(a).reshape(len(a), -1)
    res = {"argmax": base}
    res["linear on 10 summed"] = lin(g(e1tr), ytr, g(e1te), yte)
    print(f"  L1 + table + linear on 10 numbers       {res['linear on 10 summed']:.4f}",
          flush=True)
    res["linear on 360 per-cell"] = lin(g(c1tr), ytr, g(c1te), yte)
    print(f"  L1 + table + linear on 360 (per cell)   {res['linear on 360 per-cell']:.4f}",
          flush=True)

    W2, _ = BD.train_l2(itr, mtr, ytr_g, np.random.default_rng(3))
    jtr, jte = BD.l2_map(W2, itr, mtr), BD.l2_map(W2, ite, mte)
    C2 = G.cellmap(BD.NW, GRID)
    v = cp.ones(jtr.shape, bool)
    T2, _ = G.build_table(jtr, C2, ytr_g, K2, GRID)
    c2tr = per_cell(T2, jtr, C2, v, GRID)
    c2te = per_cell(T2, jte, C2, cp.ones(jte.shape, bool), GRID)
    e2tr, e2te = c2tr.sum(1), c2te.sum(1)
    print(f"\n  L2 + table, argmax                      "
          f"{float((e2te.argmax(1) == yte_g).mean()):.4f}", flush=True)
    res["L2 argmax"] = float((e2te.argmax(1) == yte_g).mean())

    res["linear on 20 summed"] = lin(np.hstack([g(e1tr), g(e2tr)]), ytr,
                                     np.hstack([g(e1te), g(e2te)]), yte)
    print(f"  L1+L2 tables + linear on 20 numbers     {res['linear on 20 summed']:.4f}",
          flush=True)
    res["linear on 720 per-cell"] = lin(np.hstack([g(c1tr), g(c2tr)]), ytr,
                                        np.hstack([g(c1te), g(c2te)]), yte)
    print(f"  L1+L2 tables + linear on 720 (per cell) {res['linear on 720 per-cell']:.4f}",
          flush=True)

    b = SIDE // 4
    def pooled(idx, keep, chunk=256):
        """Chunked: the dense form of all 12,000 at once is 11 GB."""
        out = np.zeros((len(idx), 16 * K1), np.float32)
        for a in range(0, len(idx), chunk):
            e = min(a + chunk, len(idx)); m = e - a
            D = cp.zeros((m, SIDE * SIDE, K1), cp.float32)
            r = cp.repeat(cp.arange(m), SIDE * SIDE)
            D[r, cp.tile(cp.arange(SIDE * SIDE), m), idx[a:e].ravel()] = keep[a:e].ravel()
            out[a:e] = cp.asnumpy(D.reshape(m, 4, b, 4, b, K1).max(axis=(2, 4))
                                  ).reshape(m, -1)
            del D
        return out
    res["probe on raw pooled code"] = lin(pooled(itr, ktr), ytr, pooled(ite, kte), yte)
    print(f"\n  probe on the raw pooled L1 code (ref)   "
          f"{res['probe on raw pooled code']:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    (OUT / f"readouts_ladder_{DS}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
