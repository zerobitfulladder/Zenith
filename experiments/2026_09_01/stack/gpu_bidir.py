"""The L1<->L2 connection as a counted table, read backwards for feedback.

Every feedback we have tried sent back a PROJECTION of layer 2's forward weights,
and every one was blurred -- "template 12 somewhere around here". This counts the
connection instead:

    T_down[L1 template, relative offset, L2 template]
        = log P(L1 template t at offset o | L2 template j) / P(t at o)

Independently counted, so it is not the transpose of the forward pass (this
repo's own law). Base-rate normalised, so a stroke that occurs under every L2
template scores log 1 = 0 and stays silent. And indexed by RELATIVE OFFSET, so
it says "you, at offset 2 inside my window, should be template 12" -- exact,
which no feedback we have tested has been.

Each L1 position sits inside up to 16 L2 windows (w=4, stride 1). Each of those
windows has a winner, and each knows what it expects at that specific offset.
Sum those votes, bias layer 1's selection, re-pick, rebuild the class table,
re-measure.

Weight swept, with 0 reproducing the baseline exactly.
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
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
K1, K2, NL, EPS, ALPHA = G.K1, 1000, 10, 1e-12, 1.0
SIDE, GRID2, W, BETA = G.SIDE, 6, 4, 1.5
EPOCHS2, IMG_BATCH = 3, 96
NW = SIDE - W + 1                                  # 21 window positions a side
WEIGHTS = [0.0, 0.1, 0.3]


def train_l2(itr, mtr, y, rng):
    npos = NW * NW
    C2 = G.cellmap(NW, GRID2)
    W2 = cp.asarray(rng.standard_normal((K2, K1)), cp.float32)
    W2 -= W2.mean(1, keepdims=True); W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N2 = cp.zeros((K2, GRID2 * GRID2, NL)); T2 = G.table_from(N2, K2); n2 = cp.zeros(K2)
    order = np.arange(len(itr))
    for ep in range(EPOCHS2):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            sel = cp.asarray(np.sort(order[s:s + IMG_BATCH])); m = len(sel)
            V = M.merged(itr[sel], mtr[sel], W, 0, m)[0].reshape(-1, K1)
            cells = cp.tile(C2, m)
            w0 = (1.0 - (V @ W2.T) ** 2).argmin(1)
            q = G.belief(T2, w0, cp.ones(len(V), bool), cells, m, npos)
            win = G.compete(V, W2, T2, q, cells, BETA, m, npos)
            lab = cp.repeat(y[sel], npos)
            N2 += cp.bincount((win * (GRID2 * GRID2) + cells) * NL + lab,
                              minlength=K2 * GRID2 * GRID2 * NL).reshape(N2.shape)
            T2 = G.table_from(N2, K2)
            G.learn(W2, V, win, n2, K2)
    return W2, C2


def l2_map(W2, I, Mg, chunk=128):
    return cp.concatenate(
        [(1.0 - (M.merged(I, Mg, W, a, min(a + chunk, len(I)))[0].reshape(-1, K1) @ W2.T) ** 2
          ).argmin(1).reshape(-1, NW * NW) for a in range(0, len(I), chunk)])


def count_down(idx, wmap, chunk=256):
    """N[L1 template, offset, L2 template] over every window and every offset."""
    N = cp.zeros(K1 * W * W * K2, cp.float64)
    off = [(a, b) for a in range(W) for b in range(W)]
    for s in range(0, len(idx), chunk):
        I = idx[s:s + chunk].reshape(-1, SIDE, SIDE)
        J = wmap[s:s + chunk].reshape(-1, NW, NW)
        for o, (a, b) in enumerate(off):
            t = I[:, a:a + NW, b:b + NW].ravel()
            N += cp.bincount(((t * (W * W) + o) * K2 + J.ravel()),
                             minlength=K1 * W * W * K2).astype(cp.float64)
    N = N.reshape(K1, W * W, K2)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K1)
    pm = ((N.sum(2, keepdims=True) + ALPHA * K2)
          / (N.sum((0, 2), keepdims=True) + ALPHA * K1 * K2))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def feedback_bias(Td, wmap, m):
    """(m, 576, K1) -- what the covering L2 windows expect of each L1 position."""
    B = cp.zeros((m, SIDE, SIDE, K1), cp.float32)
    J = wmap.reshape(m, NW, NW)
    for o, (a, b) in enumerate(((x, y) for x in range(W) for y in range(W))):
        B[:, a:a + NW, b:b + NW, :] += Td[:, o, :].T[J]      # (m, NW, NW, K1)
    return B.reshape(m, SIDE * SIDE, K1)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    rng = np.random.default_rng(7 + 100 * SEED)
    W1, _ = G.l1_train(Xtr, ytr_g, rng)
    itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)
    T1, _ = G.build_table(itr, G.C1, ytr_g, K1, G.GRID1, ktr)
    base = float((G.score(T1, ite, G.C1, SIDE * SIDE, kte).argmax(1) == yte_g).mean())
    print(f"{DS} seed {SEED}  L1 alone {base:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    W2, _ = train_l2(itr, mtr, ytr_g, np.random.default_rng(3 + 100 * SEED))
    jtr, jte = l2_map(W2, itr, mtr), l2_map(W2, ite, mte)
    Td = count_down(itr, jtr)
    print(f"  T_down {tuple(Td.shape)} = {Td.size:,} counted numbers   "
          f"range [{float(Td.min()):.2f}, {float(Td.max()):.2f}]   "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    Wf = W1.astype(cp.float32)
    res = {"L1_alone": base}
    print("  weight   L1 winners changed   accuracy", flush=True)
    for wt in WEIGHTS:
        def repick(X, idx, jm, keep, chunk=128):
            out = cp.zeros_like(idx); moved = 0
            for a in range(0, len(X), chunk):
                b = min(a + chunk, len(X)); m = b - a
                Q, _, _ = G.patches(X[a:b])
                err = (1.0 - (Q.reshape(-1, 25) @ Wf.T) ** 2).reshape(m, SIDE * SIDE, K1)
                if wt > 0:
                    err = err - wt * feedback_bias(Td, jm[a:b], m)
                w = err.argmin(2)
                moved += int((w != idx[a:b]).sum())
                out[a:b] = w
            return out, moved / (len(X) * SIDE * SIDE)
        ntr, _ = repick(Xtr, itr, jtr, ktr)
        nte, mv = repick(Xte, ite, jte, kte)
        Tn, _ = G.build_table(ntr, G.C1, ytr_g, K1, G.GRID1, ktr)
        acc = float((G.score(Tn, nte, G.C1, SIDE * SIDE, kte).argmax(1) == yte_g).mean())
        res[f"w={wt}"] = {"acc": acc, "moved": mv}
        print(f"  {wt:<9}{mv*100:>8.1f}%          {acc:.4f}", flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    (OUT / f"gpu_bidir_{DS}_s{SEED}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
