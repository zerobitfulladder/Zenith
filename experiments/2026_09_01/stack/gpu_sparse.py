"""Layer 2 with no pooling: four sampled positions, spread as far as you like.

Pooling was doing two jobs and both were problems. It bought shift tolerance,
and it starved the readout -- 121 observations against layer 1's 331, when the
counted readout's power scales with how many observations it gets.

And broadening a window the ordinary way makes the template bigger, which is
where layer 2's 25-numbers-per-pixel came from.

Sparse sampling decouples the two. A layer-2 template reads the L1 identity at
exactly FOUR positions, arranged as a square of side S:

    (0,0)  (0,S)  (S,0)  (S,S)

so its dimension is fixed at 4 x 400 = 1,600 no matter how wide S makes it,
and the receptive field grows as S + 5 pixels. No pooling anywhere: layer 2
slides at stride 1 over the raw 24 x 24 layer-1 map, giving (24-S)^2 positions --
484 at S=2, up to 256 at S=8, all of them MORE than the 121 the pooled version
had.

That is V2's shape: a broader field built from a conjunction of a few V1
features at relative offsets, rather than from a blurred average of all of them.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp, cupyx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E
import gpu_stack as G

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
STRIDES = [2, 4, 8]
K1, K2, NL, EPS = G.K1, 1000, 10, 1e-12
SIDE, GRID2 = G.SIDE, 6
EPOCHS2, IMG_BATCH, BETA = 3, 96, 1.5


def offsets(S):
    """Four sampled positions and where a window may start."""
    o = np.array([(0, 0), (0, S), (S, 0), (S, S)])
    n = SIDE - S
    base = np.array([(i, j) for i in range(n) for j in range(n)])
    pos = (base[:, None, 0] + o[None, :, 0]) * SIDE + (base[:, None, 1] + o[None, :, 1])
    return cp.asarray(pos), n            # (npos, 4)


def windows(idx, mag, POS, a, b):
    """(m, npos, 4*K1) -- the L1 identity at four sampled positions. No pooling."""
    m = b - a
    D = cp.zeros((m, SIDE * SIDE, K1), cp.float32)
    r = cp.repeat(cp.arange(m), SIDE * SIDE)
    D[r, cp.tile(cp.arange(SIDE * SIDE), m), idx[a:b].ravel()] = mag[a:b].ravel()
    V = D[:, POS, :].reshape(m, POS.shape[0], 4 * K1)
    V = V - V.mean(-1, keepdims=True)
    return V / cp.maximum(cp.linalg.norm(V, axis=-1, keepdims=True), EPS)


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
    print("   S   rf(px)  L2 positions   accuracy", flush=True)

    d = 4 * K1
    res = {"L1_alone": a1}
    for S in STRIDES:
        t1 = time.time()
        POS, n = offsets(S)
        npos = POS.shape[0]
        C2 = G.cellmap(n, GRID2)
        W2 = cp.asarray(rng.standard_normal((K2, d)), cp.float32)
        W2 -= W2.mean(1, keepdims=True)
        W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
        N2 = cp.zeros((K2, GRID2 * GRID2, NL)); T2 = G.table_from(N2, K2); n2 = cp.zeros(K2)
        order = np.arange(len(Xtr))
        for ep in range(EPOCHS2):
            rng.shuffle(order)
            for s in range(0, len(order), IMG_BATCH):
                sel = cp.asarray(np.sort(order[s:s + IMG_BATCH])); m = len(sel)
                V = windows(itr[sel], mtr[sel], POS, 0, m).reshape(-1, d)
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
            [(1.0 - (windows(I, M, POS, a, min(a + 128, len(I))).reshape(-1, d) @ W2.T) ** 2
              ).argmin(1).reshape(-1, npos) for a in range(0, len(I), 128)])
        wtr, wte = wmap(itr, mtr), wmap(ite, mte)
        T2f, N2f = G.build_table(wtr, C2, ytr_g, K2, GRID2)
        acc = float((G.score(T2f, wte, C2, npos).argmax(1) == yte_g).mean())
        dead = int((N2f.sum((1, 2)) == 0).sum())
        res[f"S={S}"] = {"acc": acc, "rf_px": S + 5, "positions": npos, "dead": dead}
        print(f"  {S:<4}{S+5:<8}{npos:<15}{acc:.4f}   dead {dead}/{K2}  "
              f"({time.time()-t1:.0f}s)", flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    (OUT / f"gpu_sparse_{DS}.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
