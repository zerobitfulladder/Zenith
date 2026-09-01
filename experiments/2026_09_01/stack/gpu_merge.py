"""L2 templates that merge a whole window into ONE 400-dim vector.

The L1 map is one-hot in depth -- exactly one of 400 channels fires at each
position -- so a w x w window holds only ~0.575 * w^2 non-zeros. Superimposing
all of them into a single 400-vector barely collides (16 draws from 400 collide
about 27% of the time, almost always one pair, and max-pooling keeps the
stronger). So the window costs 400 numbers instead of w^2 x 400.

    before   2 x 2 sub-regions kept separate   4 x 400 = 1,600 per template
    now      the whole window merged           400 per template

Four times cheaper, and it drops sub-region structure I was paying for without
having checked it earned anything. What is given up is WHERE inside the window
each stroke sat -- but the window itself is still positioned, so only fine
position within 8-12 pixels is lost.

No pooling of the window grid either: L2 slides at stride 1 over the raw 24x24
map, so it gets 289-441 observations, more than layer 1's 331.
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
K1, K2, NL, EPS = G.K1, 1000, 10, 1e-12
SIDE, GRID2 = G.SIDE, 6
WINS, EPOCHS2, IMG_BATCH, BETA = [4, 6, 8], 3, 96, 1.5


def merged(idx, mag, w, a, b):
    """(m, npos, K1) -- every wxw window of the L1 map, superimposed."""
    m = b - a
    D = cp.zeros((m, SIDE * SIDE, K1), cp.float32)
    r = cp.repeat(cp.arange(m), SIDE * SIDE)
    D[r, cp.tile(cp.arange(SIDE * SIDE), m), idx[a:b].ravel()] = mag[a:b].ravel()
    D = D.reshape(m, SIDE, SIDE, K1)
    n = SIDE - w + 1
    R = D[:, :n]                                     # separable max: rows, then cols
    for o in range(1, w):
        R = cp.maximum(R, D[:, o:o + n])
    V = R[:, :, :n]
    for o in range(1, w):
        V = cp.maximum(V, R[:, :, o:o + n])
    V = V.reshape(m, n * n, K1)
    V = V - V.mean(-1, keepdims=True)
    return V / cp.maximum(cp.linalg.norm(V, axis=-1, keepdims=True), EPS), n


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    rng = np.random.default_rng(7)
    W1, _ = G.l1_train(Xtr, ytr_g, rng)
    itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)
    T1, _ = G.build_table(itr, G.C1, ytr_g, K1, G.GRID1, ktr)
    a1 = float((G.score(T1, ite, G.C1, SIDE * SIDE, kte).argmax(1) == yte_g).mean())
    print(f"{DS}  L1 alone {a1:.4f}   ({time.time()-t0:.0f}s)\n", flush=True)
    print("   w   rf(px)  L2 pos   nonzero/window   accuracy", flush=True)

    res = {"L1_alone": a1}
    for w in WINS:
        t1 = time.time()
        _, n = merged(itr, mtr, 4, 0, 1)
        Vp, n = merged(itr[:1], mtr[:1], w, 0, 1)
        npos = n * n
        nz = float((Vp != Vp.mean()).sum() / npos) if False else float(w * w * 0.575)
        C2 = G.cellmap(n, GRID2)
        W2 = cp.asarray(rng.standard_normal((K2, K1)), cp.float32)
        W2 -= W2.mean(1, keepdims=True)
        W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
        N2 = cp.zeros((K2, GRID2 * GRID2, NL)); T2 = G.table_from(N2, K2); n2 = cp.zeros(K2)
        order = np.arange(len(Xtr))
        for ep in range(EPOCHS2):
            rng.shuffle(order)
            for s in range(0, len(order), IMG_BATCH):
                sel = cp.asarray(np.sort(order[s:s + IMG_BATCH])); m = len(sel)
                V, _ = merged(itr[sel], mtr[sel], w, 0, m)
                V = V.reshape(-1, K1)
                cells = cp.tile(C2, m)
                k = cp.ones(len(V), bool)
                w0 = (1.0 - (V @ W2.T) ** 2).argmin(1)
                q = G.belief(T2, w0, k, cells, m, npos)
                win = G.compete(V, W2, T2, q, cells, BETA, m, npos)
                lab = cp.repeat(ytr_g[sel], npos)
                N2 += cp.bincount((win * (GRID2 * GRID2) + cells) * NL + lab,
                                  minlength=K2 * GRID2 * GRID2 * NL).reshape(N2.shape)
                T2 = G.table_from(N2, K2)
                G.learn(W2, V, win, n2, K2)
        wmap = lambda I, M: cp.concatenate(
            [(1.0 - (merged(I, M, w, a, min(a + 128, len(I)))[0].reshape(-1, K1) @ W2.T) ** 2
              ).argmin(1).reshape(-1, npos) for a in range(0, len(I), 128)])
        wtr, wte = wmap(itr, mtr), wmap(ite, mte)
        T2f, N2f = G.build_table(wtr, C2, ytr_g, K2, GRID2)
        acc = float((G.score(T2f, wte, C2, npos).argmax(1) == yte_g).mean())
        dead = int((N2f.sum((1, 2)) == 0).sum())
        res[f"w={w}"] = {"acc": acc, "rf_px": w + 4, "positions": npos,
                         "template_numbers": K1, "dead": dead}
        print(f"  {w:<4}{w+4:<8}{npos:<9}{nz:<17.1f}{acc:.4f}   dead {dead}/{K2}  "
              f"({time.time()-t1:.0f}s)", flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    (OUT / f"gpu_merge_{DS}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
