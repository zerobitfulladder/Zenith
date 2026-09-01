"""Two wide layers, small receptive fields, both counted -- on the GPU.

The correction that shaped this: "smaller" meant a smaller RECEPTIVE FIELD with
MORE templates, not fewer templates. Every L2 we built compressed hard -- whole
image to one of 200, then half an image to one of 200 -- and every one lost to
layer 1 alone. So this goes the other way at both levels:

    L1   400 templates over 5x5 pixels          576 positions
    pool 12 x 12 regions of 2 x 2 positions, GRADED (carries patch contrast,
         which the pixel-counting control says matters a lot on Fashion and
         not at all on MNIST)
    L2   1000 templates over 2 x 2 regions = 8 x 8 PIXELS -- barely bigger than
         an L1 patch, a genuinely local part           121 positions
    table T[template, coarse cell, class] at both levels, identical form
    read  sum the lookups: 331 of them at L1, 121 at L2

Both layers are co-adaptive: each table biases which inputs its own templates
win, at beta 1.5, which is the one intervention that has helped all day.

Layer 2 is 1000 x 1600 = 1.6M numbers and 1.8M window vectors have to be matched
against it, so this is ~3 TFLOP per full pass -- minutes on the CPU, about a
second on the 3060.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
K1, K2, NL, EPS, ALPHA = 400, 1000, 10, 1e-12, 1.0
PS, SIDE = 5, 24
PR, RB = 12, 2                      # 12x12 pooled regions of 2x2 positions
WR = 2                              # window = 2x2 regions -> 8x8 pixels
WG = PR - WR + 1                    # 11x11 = 121 window positions
GRID1, GRID2 = 6, 6                 # coarse cells for the two tables
EPOCHS1, EPOCHS2, IMG_BATCH, BETA, ETA_MIN, MIN_S = 3, 4, 128, 1.5, 0.02, 4


def cellmap(n_pos_side, g):
    """Even split of n x n positions into g x g cells, for any n (11 into 6 too)."""
    a = np.arange(n_pos_side * n_pos_side)
    r = (a // n_pos_side) * g // n_pos_side
    c = (a % n_pos_side) * g // n_pos_side
    return cp.asarray(r * g + c)


PIDX = cp.asarray((((np.arange(SIDE)[:, None, None, None] + np.arange(PS)[None, None, :, None]) * 28
                    + (np.arange(SIDE)[None, :, None, None] + np.arange(PS)[None, None, None, :]))
                   ).reshape(SIDE * SIDE, PS * PS))
WIDX = cp.asarray(np.array([[(i + a) * PR + (j + b) for a in range(WR) for b in range(WR)]
                            for i in range(WG) for j in range(WG)]))
C1, C2 = cellmap(SIDE, GRID1), cellmap(WG, GRID2)


def patches(Xb):
    P = Xb[:, PIDX]                                   # (m, 576, 25)
    C = P - P.mean(-1, keepdims=True)
    n = cp.linalg.norm(C, axis=-1)
    return C / cp.maximum(n, EPS)[..., None], n > 0.05, n


def table_from(N, K):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def belief(T, win, keep, cells, m, npos):
    ev = (T[win, cells] * keep[:, None]).reshape(m, npos, NL).sum(1)
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def compete(V, W, T, q, cells, beta, m, npos):
    """One competitive selection, biased by the table toward the belief."""
    err = 1.0 - (V @ W.T) ** 2
    if beta > 0:
        bias = cp.einsum('ik,ihk->ih', cp.repeat(q, npos, 0),
                         cp.ascontiguousarray(T.transpose(1, 0, 2))[cells])
        err = err - beta * bias
    return err.argmin(1)


def learn(W, V, win, n, K):
    cnt = cp.bincount(win, minlength=K)
    sums = cp.zeros((K, V.shape[1]), cp.float32)
    cupyx.scatter_add(sums, win, V)
    live = cnt >= MIN_S
    n[live] += cnt[live]
    eta = cp.clip(cnt[live] / n[live], ETA_MIN, 1.0)[:, None]
    W[live] += eta * (sums[live] / cnt[live, None] - W[live])
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    return int((cnt == 0).sum())


def l1_train(X, y, rng):
    W = cp.asarray(rng.standard_normal((K1, 25)), cp.float32)
    W -= W.mean(1, keepdims=True); W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    N = cp.zeros((K1, GRID1 * GRID1, NL)); T = table_from(N, K1); n = cp.zeros(K1)
    npos = SIDE * SIDE
    order = np.arange(len(X))
    for ep in range(EPOCHS1):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            ids = cp.asarray(order[s:s + IMG_BATCH]); m = len(ids)
            Q, keep, _ = patches(X[ids])
            V = Q.reshape(-1, 25); k = keep.reshape(-1)
            cells = cp.tile(C1, m)
            w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
            q = belief(T, w0, k, cells, m, npos)
            win = compete(V, W, T, q, cells, BETA, m, npos)
            lab = cp.repeat(y[ids], npos)
            flat = (win[k] * (GRID1 * GRID1) + cells[k]) * NL + lab[k]
            N += cp.bincount(flat, minlength=K1 * GRID1 * GRID1 * NL).reshape(N.shape)
            T = table_from(N, K1)
            learn(W, V[k], win[k], n, K1)
        print(f"    L1 epoch {ep+1}/{EPOCHS1}", flush=True)
    return W, T


def l1_code(W, X, chunk=256):
    """Winner and contrast at every position."""
    idx = cp.zeros((len(X), SIDE * SIDE), cp.int32)
    keepm = cp.zeros((len(X), SIDE * SIDE), bool)
    mag = cp.zeros((len(X), SIDE * SIDE), cp.float32)
    for a in range(0, len(X), chunk):
        Q, keep, nrm = patches(X[a:a + chunk])
        m = len(Q)
        w = (1.0 - (Q.reshape(-1, 25) @ W.T) ** 2).argmin(1).reshape(m, -1)
        idx[a:a + m] = w
        keepm[a:a + m] = keep
        mag[a:a + m] = cp.where(keep, nrm, 0.0)
    return idx, mag, keepm


def windows(idx, mag, a, b):
    """(m, 121, 4*K1) graded pooled windows."""
    m = b - a
    D = cp.zeros((m, SIDE * SIDE, K1), cp.float32)
    r = cp.repeat(cp.arange(m), SIDE * SIDE)
    D[r, cp.tile(cp.arange(SIDE * SIDE), m), idx[a:b].ravel()] = mag[a:b].ravel()
    P = D.reshape(m, PR, RB, PR, RB, K1).max(axis=(2, 4)).reshape(m, PR * PR, K1)
    V = P[:, WIDX, :].reshape(m, WG * WG, WR * WR * K1)
    V = V - V.mean(-1, keepdims=True)
    return V / cp.maximum(cp.linalg.norm(V, axis=-1, keepdims=True), EPS)


def build_table(win, cells, y, K, g, valid=None, npos=None):
    """Rebuild the table from the FINAL templates.

    The table accumulated during training is a mixture of statistics taken while
    the templates were still moving and while the bias policy was still growing.
    Measured earlier today: the online table reads 0.66-0.80 where the rebuilt
    one reads 0.9587. So it gets rebuilt once, cleanly, before anything is judged.
    """
    n, P = win.shape
    C = cp.tile(cells, n).reshape(n, P)
    lab = cp.repeat(y, P).reshape(n, P)
    v = cp.ones((n, P), bool) if valid is None else valid
    flat = ((win[v] * (g * g) + C[v]) * NL + lab[v])
    N = cp.bincount(flat, minlength=K * g * g * NL).reshape(K, g * g, NL).astype(cp.float64)
    return table_from(N, K), N


def score(T, win, cells, npos, valid=None):
    """Flat positions contribute nothing, exactly as they contribute no counts."""
    C = cp.tile(cells, len(win)).reshape(len(win), -1)
    ev = T[win, C]
    return ev.sum(1) if valid is None else (ev * valid[..., None]).sum(1)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    print(f"{DS}  L1 {K1} templates (5x5px)  L2 {K2} templates "
          f"({WR*RB+PS-1}x{WR*RB+PS-1}px, {WG*WG} positions)\n", flush=True)

    rng = np.random.default_rng(7)
    W1, T1 = l1_train(Xtr, ytr_g, rng)
    itr, mtr, ktr = l1_code(W1, Xtr); ite, mte, kte = l1_code(W1, Xte)
    T1, _ = build_table(itr, C1, ytr_g, K1, GRID1, ktr)
    a1 = float((score(T1, ite, C1, SIDE * SIDE, kte).argmax(1) == yte_g).mean())
    print(f"  L1 alone: {a1:.4f}   ({time.time()-t0:.0f}s)", flush=True)

    d = WR * WR * K1
    W2 = cp.asarray(rng.standard_normal((K2, d)), cp.float32)
    W2 -= W2.mean(1, keepdims=True); W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N2 = cp.zeros((K2, GRID2 * GRID2, NL)); T2 = table_from(N2, K2); n2 = cp.zeros(K2)
    npos2 = WG * WG
    order = np.arange(len(Xtr))
    for ep in range(EPOCHS2):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            ids = order[s:s + IMG_BATCH]
            sel = cp.asarray(np.sort(ids)); m = len(sel)
            V = cp.concatenate([windows(itr, mtr, int(i), int(i) + 1) for i in sel]
                               ) if False else None
            V = windows(itr[sel], mtr[sel], 0, m)
            Vf = V.reshape(-1, d); cells = cp.tile(C2, m)
            k = cp.ones(len(Vf), bool)
            w0 = (1.0 - (Vf @ W2.T) ** 2).argmin(1)
            q = belief(T2, w0, k, cells, m, npos2)
            win = compete(Vf, W2, T2, q, cells, BETA, m, npos2)
            lab = cp.repeat(ytr_g[sel], npos2)
            flat = (win * (GRID2 * GRID2) + cells) * NL + lab
            N2 += cp.bincount(flat, minlength=K2 * GRID2 * GRID2 * NL).reshape(N2.shape)
            T2 = table_from(N2, K2)
            learn(W2, Vf, win, n2, K2)
        wmap = lambda I, M: cp.concatenate(
            [(1.0 - (windows(I, M, a, min(a + 256, len(I))).reshape(-1, d) @ W2.T) ** 2
              ).argmin(1).reshape(-1, npos2) for a in range(0, len(I), 256)])
        wtr, wte = wmap(itr, mtr), wmap(ite, mte)
        T2f, _ = build_table(wtr, C2, ytr_g, K2, GRID2)
        a2 = float((score(T2f, wte, C2, npos2).argmax(1) == yte_g).mean())
        dead = int((N2.sum((1, 2)) == 0).sum())
        print(f"    L2 epoch {ep+1}/{EPOCHS2}  acc {a2:.4f}  dead {dead}/{K2}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {"dataset": DS, "L1_alone": a1, "L2": a2, "K1": K1, "K2": K2,
           "L2_rf_px": WR * RB + PS - 1, "L2_positions": npos2,
           "params": int(W1.size + W2.size + T1.size + T2.size),
           "seconds": round(time.time() - t0, 1)}
    print(f"\n  {DS}: L1 {a1:.4f}   L1+L2 {a2:.4f}   "
          f"{res['params']:,} numbers   ({res['seconds']:.0f}s)")
    (OUT / f"gpu_stack_{DS}.json").write_text(json.dumps(res, indent=2))
    cp.savez(OUT / f"gpu_stack_{DS}.npz", W1=W1, W2=W2, T1=T1, T2=T2)


if __name__ == "__main__":
    main()
