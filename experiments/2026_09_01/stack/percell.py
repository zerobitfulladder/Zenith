"""The two untried Fashion moves, and one debt on MNIST.

Fashion sits at 0.8447 -- below logistic regression on raw pixels (0.8512) --
and the two moves that paid on MNIST were never run on it:

    per-cell readout   the table already produces evidence at each of 36 cells;
                       summing it throws away WHICH region voted for what, and
                       a linear map on the 360 per-cell numbers recovered +0.70
                       on MNIST. Fashion's classes disagree about where their
                       information lives (silhouette at the outline, texture in
                       the interior), which is exactly what a flat sum cannot
                       weight.
    width x data       800 templates and 40,000 images each paid on MNIST
                       (0.9730) and neither axis had flattened. Fashion has
                       only ever seen 400 x 12,000.

Grid: K in {400, 800} x train in {12k, 40k}, champion 4-bin magnitude table,
both readouts, 3 seeds, fixed 3,000-image test set (the usual carve). Probe
inputs are standardised -- the ladder's 10-number row taught us what happens
otherwise. Bonus config: MNIST at 800 x 40k with the per-cell readout, since
its published 0.9770 was measured at 400 x 12k only.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E
import gpu_stack as G

OUT = HERE / "results"
ROOT = HERE.parents[2]
SMOKE = "--smoke" in sys.argv
if SMOKE:
    G.EPOCHS1 = 1
NL, ALPHA = 10, 1.0
GG = G.GRID1 * G.GRID1
P = G.SIDE * G.SIDE
SEEDS = [7] if SMOKE else [7, 8, 9]
CONFIGS = ([("fashion_mnist", 4, 400, 4000)] if SMOKE else
           [("fashion_mnist", 4, 400, 12000),
            ("fashion_mnist", 4, 800, 12000),
            ("fashion_mnist", 4, 400, 40000),
            ("fashion_mnist", 4, 800, 40000),
            ("mnist", 1, 800, 40000)])
M36 = (G.C1[:, None] == cp.arange(GG)[None, :]).astype(cp.float32)


def load_split(ds, ntrain):
    """Same permutation and test carve as experts.load; extra train images are
    drawn from BEYOND the test block, so the test set never moves."""
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_images.npy").astype(np.float64).reshape(-1, 784)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(E.SEED).permutation(len(X))
    tr = np.concatenate([p[:12000], p[15000:15000 + max(0, ntrain - 12000)]])[:ntrain]
    te = p[12000:15000]
    return (cp.asarray(X[tr], cp.float32), cp.asarray(y[tr]),
            cp.asarray(X[te], cp.float32), cp.asarray(y[te]))


def table_of(N, K):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(-1, keepdims=True) + ALPHA * NL)
          / (N.sum((0, -1), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def mag_edges(live, nb):
    if nb == 1:
        return cp.zeros(0, cp.float32)
    q = cp.asarray(np.linspace(0, 100, nb + 1)[1:-1])
    return cp.percentile(live, q)


def bins_of(vals, edges, nb):
    return cp.clip(cp.searchsorted(edges, vals.ravel()).reshape(vals.shape),
                   0, nb - 1).astype(cp.int32)


def count_l1(idx, keep, mb, y, nb, K):
    n = len(idx)
    C = cp.tile(G.C1, n).reshape(n, P)
    lab = cp.repeat(y, P).reshape(n, P)
    flat = ((idx[keep] * GG + C[keep]) * nb + mb[keep]) * NL + lab[keep]
    N = cp.bincount(flat, minlength=K * GG * nb * NL)
    return N.reshape(K, GG, nb, NL).astype(cp.float64)


def percell_ev(T, idx, keep, mb, chunk=2048):
    """(n, 36, 10) evidence, kept separated by cell instead of summed away."""
    out = cp.zeros((len(idx), GG, NL), cp.float32)
    for a in range(0, len(idx), chunk):
        m = len(idx[a:a + chunk])
        C = cp.tile(G.C1, m).reshape(m, P)
        ev = T[idx[a:a + chunk], C, mb[a:a + chunk]] * keep[a:a + chunk, :, None]
        out[a:a + m] = cp.einsum('npc,pg->ngc', ev, M36)
    return out


def probe(Etr, ytr, Ete, yte):
    """Standardised linear map on the 360 per-cell numbers."""
    Xtr = cp.asnumpy(Etr.reshape(len(Etr), -1))
    Xte = cp.asnumpy(Ete.reshape(len(Ete), -1))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000)
    clf.fit((Xtr - mu) / sd, cp.asnumpy(ytr))
    return float(clf.score((Xte - mu) / sd, cp.asnumpy(yte)))


def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = []
    for ds, nb, K, ntrain in CONFIGS:
        G.K1 = K
        G.IMG_BATCH = 64 if K >= 800 else 128
        Xtr, ytr, Xte, yte = load_split(ds, ntrain)
        log(f"\n=== {ds}  K={K}  train={len(ytr)}  nb={nb} ===")
        rows = []
        for s in SEEDS:
            t0 = time.time()
            W, _ = G.l1_train(Xtr, ytr, np.random.default_rng(s))
            itr, mtr, ktr = G.l1_code(W, Xtr)
            ite, mte, kte = G.l1_code(W, Xte)
            edges = mag_edges(mtr[ktr], nb)
            mbtr, mbte = bins_of(mtr, edges, nb), bins_of(mte, edges, nb)
            T = table_of(count_l1(itr, ktr, mbtr, ytr, nb, K), K)
            Etr, Ete = (percell_ev(T, itr, ktr, mbtr),
                        percell_ev(T, ite, kte, mbte))
            a_arg = float((Ete.sum(1).argmax(1) == yte).mean())
            a_pc = probe(Etr, ytr, Ete, yte)
            rows.append({"seed": s, "argmax": a_arg, "percell": a_pc})
            log(f"  seed {s}:  argmax {a_arg:.4f}   per-cell {a_pc:.4f}   "
                f"({time.time() - t0:.0f}s)")
        agg = {r: {"mean": float(np.mean([x[r] for x in rows])),
                   "std": float(np.std([x[r] for x in rows]))}
               for r in ("argmax", "percell")}
        log(f"  {ds} K={K} n={ntrain}:  argmax "
            f"{agg['argmax']['mean']:.4f}+-{agg['argmax']['std']:.4f}   "
            f"per-cell {agg['percell']['mean']:.4f}+-{agg['percell']['std']:.4f}")
        results.append({"dataset": ds, "K": K, "ntrain": ntrain, "nb": nb,
                        "per_seed": rows, "agg": agg})

    tag = "_smoke" if SMOKE else ""
    (OUT / f"percell{tag}.json").write_text(json.dumps(results, indent=2))
    (OUT / f"percell{tag}.log").write_text("\n".join(lines))
    log(f"\nwrote {OUT / f'percell{tag}.json'}")


if __name__ == "__main__":
    main()
