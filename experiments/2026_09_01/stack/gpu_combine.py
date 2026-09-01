"""Does layer 2 carry anything layer 1 doesn't?

Every previous test asked whether layer 2 could REPLACE layer 1's readout, by
building the table on layer 2's output and comparing. Seven designs, seven
losses. But that is the wrong question. The right one is whether layer 2 adds
information, and that is answered by summing both tables:

    score(class) = L1 evidence over its ~331 observations (5x5 patches)
                 + w x L2 evidence over its ~441 observations (8-12 px windows)

At w = 0 this reproduces layer 1 exactly, so any gain is unambiguous.

The weight is chosen on a held-out slice of TRAINING data -- 2,000 images the
tables never saw -- not on the test set. The full test sweep is printed anyway
so a narrow spike can be told from a broad plateau; every previous time an extra
evidence stream helped, it helped over a wide range and only at a small weight.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E
import gpu_stack as G
import gpu_merge as M

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
K1, K2, NL, EPS = G.K1, 1000, 10, 1e-12
SIDE, GRID2, WINS = G.SIDE, 6, [4, 6, 8]
EPOCHS2, IMG_BATCH, BETA = 3, 96, 1.5
WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
N_VAL = 2000


def z(ev):
    return (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)


def l2_evidence(itr, mtr, yfit, sets, w, rng):
    """Train an L2 at window w, return its class evidence for each set."""
    _, n = M.merged(itr[:1], mtr[:1], w, 0, 1)
    npos = n * n
    C2 = G.cellmap(n, GRID2)
    W2 = cp.asarray(rng.standard_normal((K2, K1)), cp.float32)
    W2 -= W2.mean(1, keepdims=True); W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N2 = cp.zeros((K2, GRID2 * GRID2, NL)); T2 = G.table_from(N2, K2); n2 = cp.zeros(K2)
    order = np.arange(len(itr))
    for ep in range(EPOCHS2):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            sel = cp.asarray(np.sort(order[s:s + IMG_BATCH])); m = len(sel)
            V = M.merged(itr[sel], mtr[sel], w, 0, m)[0].reshape(-1, K1)
            cells = cp.tile(C2, m)
            k = cp.ones(len(V), bool)
            w0 = (1.0 - (V @ W2.T) ** 2).argmin(1)
            q = G.belief(T2, w0, k, cells, m, npos)
            win = G.compete(V, W2, T2, q, cells, BETA, m, npos)
            lab = cp.repeat(yfit[sel], npos)
            N2 += cp.bincount((win * (GRID2 * GRID2) + cells) * NL + lab,
                              minlength=K2 * GRID2 * GRID2 * NL).reshape(N2.shape)
            T2 = G.table_from(N2, K2)
            G.learn(W2, V, win, n2, K2)
    wmap = lambda I, Mg: cp.concatenate(
        [(1.0 - (M.merged(I, Mg, w, a, min(a + 128, len(I)))[0].reshape(-1, K1) @ W2.T) ** 2
          ).argmin(1).reshape(-1, npos) for a in range(0, len(I), 128)])
    wfit = wmap(itr, mtr)
    T2f, _ = G.build_table(wfit, C2, yfit, K2, GRID2)
    return [z(G.score(T2f, wmap(I, Mg), C2, npos)) for I, Mg in sets]


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    rng = np.random.default_rng(7)
    W1, _ = G.l1_train(Xtr, ytr_g, rng)
    itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)

    nfit = len(itr) - N_VAL
    fit, val = slice(0, nfit), slice(nfit, None)
    T1, _ = G.build_table(itr[fit], G.C1, ytr_g[fit], K1, G.GRID1, ktr[fit])
    e1_val = z(G.score(T1, itr[val], G.C1, SIDE * SIDE, ktr[val]))
    e1_te = z(G.score(T1, ite, G.C1, SIDE * SIDE, kte))
    yv = ytr_g[val]
    print(f"{DS}  L1 alone: val {float((e1_val.argmax(1)==yv).mean()):.4f}  "
          f"test {float((e1_te.argmax(1)==yte_g).mean()):.4f}   ({time.time()-t0:.0f}s)\n",
          flush=True)

    ev = {}
    for w in WINS:
        ev[w] = l2_evidence(itr[fit], mtr[fit], ytr_g[fit],
                            [(itr[val], mtr[val]), (ite, mte)], w, np.random.default_rng(3))
        a = float((ev[w][1].argmax(1) == yte_g).mean())
        print(f"  L2 w={w} alone: test {a:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    combos = {"L1+L2(w=4)": [4], "L1+L2(w=6)": [6], "L1+L2(all three)": WINS}
    print()
    for name, ws in combos.items():
        vv = sum(ev[w][0] for w in ws) / len(ws)
        tt = sum(ev[w][1] for w in ws) / len(ws)
        curve = [(a, float(((e1_te + a * tt).argmax(1) == yte_g).mean())) for a in WEIGHTS]
        vals = [float(((e1_val + a * vv).argmax(1) == yv).mean()) for a in WEIGHTS]
        best = WEIGHTS[int(np.argmax(vals))]
        chosen = float(((e1_te + best * tt).argmax(1) == yte_g).mean())
        res[name] = {"weight_chosen_on_val": best, "test_at_chosen": chosen,
                     "test_sweep": curve}
        print(f"  {name:<20} weight {best:<5} -> TEST {chosen:.4f}")
        print("      sweep: " + "  ".join(f"{a}:{b:.4f}" for a, b in curve), flush=True)

    res["L1_alone_test"] = float((e1_te.argmax(1) == yte_g).mean())
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / f"gpu_combine_{DS}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
