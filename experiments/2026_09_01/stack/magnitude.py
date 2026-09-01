"""Does the table want to know HOW STRONGLY a template fired?

Layer 1 emits a winner and that patch's contrast. Layer 2 uses the contrast
(graded pooling). The counted table throws it away -- every firing is +1,
whether the stroke was faint or bold.

Three ways to count, same templates:

    count      N[t, cell, class] += 1                        what we do now
    weighted   N[t, cell, class] += magnitude                strong strokes vote more
    binned     N[t, cell, MAGBIN, class] += 1                a faint stroke here and a
                                                             bold one here are different
                                                             events entirely

Prediction from the pixel-counting control, where intensity bins went 0.6377 ->
0.7330 on Fashion and 0.8343 -> 0.8277 on MNIST: magnitude should buy something
on garments, where material and shading carry the class, and nothing on digits,
where ink is essentially binary.
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
K1, NL, ALPHA, EPS = G.K1, 10, 1.0, 1e-12
SIDE, GRID = G.SIDE, G.GRID1
NB = [2, 3, 4]


def table_of(N, K):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(-1, keepdims=True) + ALPHA * NL)
          / (N.sum((0, -1), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def run(itr, mtr, ktr, ytr, ite, mte, kte, yte, mode, nb=1, edges=None):
    n, P = itr.shape
    C = cp.tile(G.C1, n).reshape(n, P)
    lab = cp.repeat(ytr, P).reshape(n, P)
    if mode == "binned":
        btr = cp.clip(cp.searchsorted(edges, mtr.ravel()).reshape(n, P), 0, nb - 1)
        flat = (((itr[ktr] * (GRID * GRID) + C[ktr]) * nb + btr[ktr]) * NL + lab[ktr])
        N = cp.bincount(flat, minlength=K1 * GRID * GRID * nb * NL).reshape(
            K1, GRID * GRID, nb, NL).astype(cp.float64)
        T = table_of(N, K1)
        m2 = len(ite)
        C2 = cp.tile(G.C1, m2).reshape(m2, P)
        bte = cp.clip(cp.searchsorted(edges, mte.ravel()).reshape(m2, P), 0, nb - 1)
        ev = T[ite, C2, bte]
    else:
        w = mtr[ktr] if mode == "weighted" else None
        flat = ((itr[ktr] * (GRID * GRID) + C[ktr]) * NL + lab[ktr])
        N = (cp.bincount(flat, weights=w, minlength=K1 * GRID * GRID * NL)
             .reshape(K1, GRID * GRID, NL).astype(cp.float64))
        T = table_of(N, K1)
        m2 = len(ite)
        C2 = cp.tile(G.C1, m2).reshape(m2, P)
        ev = T[ite, C2]
        if mode == "weighted":
            ev = ev * mte[..., None]
    sc = (ev * kte[..., None]).sum(1)
    return float((sc.argmax(1) == yte).mean()), int(T.size)


def main():
    for DS in ("mnist", "fashion_mnist"):
        t0 = time.time()
        Xtr, ytr, Xte, yte = E.load(DS)
        Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
        Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
        ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
        W1, _ = G.l1_train(Xtr, ytr_g, np.random.default_rng(7))
        itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)
        live = mtr[ktr]
        print(f"\n{DS}   contrast: median {float(cp.median(live)):.3f}  "
              f"90th pct {float(cp.percentile(live, 90)):.3f}", flush=True)
        res = {}
        for mode in ("count", "weighted"):
            a, sz = run(itr, mtr, ktr, ytr_g, ite, mte, kte, yte_g, mode)
            res[mode] = a
            print(f"  {mode:<10} {a:.4f}   table {sz:,}", flush=True)
        for nb in NB:
            q = cp.asarray(np.linspace(0, 100, nb + 1)[1:-1])
            edges = cp.percentile(live, q)
            a, sz = run(itr, mtr, ktr, ytr_g, ite, mte, kte, yte_g, "binned", nb, edges)
            res[f"binned{nb}"] = a
            print(f"  binned {nb}   {a:.4f}   table {sz:,}", flush=True)
        (OUT / f"magnitude_{DS}.json").write_text(json.dumps(res, indent=2))
        print(f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
