"""Layer 2 slid over the pooled map, so its output is a MAP rather than a name.

Two problems with the whole-image L2, and the second is the one that matters:

    it ties identity to absolute position -- fine on centred MNIST, fatal on
    the moving-digit scenes from this morning (measured: -16 points)

    it emits ONE number, which is a hopeless signal for steering 331 L1
    positions. A single global winner can only say "the whole image is like
    this". It cannot say "this corner is like this".

So L2 gets a window and slides, exactly as L1 does:

    L1 map          24 x 24 winners over 180 templates
    pool            8 x 8 regions of 3 x 3 positions        -> 8 x 8 x 180
    L2 window       3 x 3 regions = 1,620 numbers, covering 13 x 13 pixels
                    -- about half a digit, a PART rather than a whole
    slide           stride one region -> 6 x 6 = 36 window positions
    L2 templates    K2 of them, SHARED across all 36 positions
    L2 output       a 6 x 6 map of identities

and its table takes the same form as layer 1's, because now there is a position
to condition on again:

    T2[L2 template, window cell, class]

Classification sums the 36 lookups, the same way layer 1 sums its 331.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, settle as S, single as SG, layer2 as L2

OUT = Path(__file__).resolve().parent / "results"
PG, WIN, NL, EPS = 8, 3, 10, 1e-12          # pooled grid, window in regions
WG = PG - WIN + 1                            # 6 x 6 window positions
K2S, EPOCHS, ETA_MIN, ALPHA, PER_IMG = [100, 200, 400], 10, 0.02, 1.0, 4


def magnitudes(X, chunk=256):
    """Each patch's contrast -- the norm before it was normalised away."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.zeros((len(X), S.SIDE * S.SIDE), np.float32)
    for a in range(0, len(X), chunk):
        B = X[a:a + chunk]
        P = sliding_window_view(B, (5, 5), axis=(1, 2)).reshape(len(B), -1, 25)
        out[a:a + len(B)] = np.linalg.norm(P - P.mean(-1, keepdims=True), axis=-1)
    return out


def pooled8(idx, h, mag=None):
    b = S.SIDE // PG
    M = np.zeros((len(idx), S.SIDE * S.SIDE, h), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = 1.0 if mag is None else mag[r, c]
    return M.reshape(len(idx), PG, b, PG, b, h).max(axis=(2, 4))      # (n, 8, 8, h)


def windows(P8):
    """(n, 36, WIN*WIN*h) -- every 3x3-region window of the pooled map."""
    n, _, _, h = P8.shape
    out = np.empty((n, WG * WG, WIN * WIN * h), np.float32)
    for i in range(WG):
        for j in range(WG):
            out[:, i * WG + j] = P8[:, i:i + WIN, j:j + WIN].reshape(n, -1)
    return out


def code(W1, X, graded=False, chunk=256):
    """The pooled map. GRADED carries each patch's contrast instead of a bare 1."""
    idx = SG.winners(W1, X)
    return pooled8(idx, len(W1), magnitudes(X) if graded else None)


def table(idx, y, k):
    N = np.zeros((k, WG * WG, NL))
    r, c = np.nonzero(idx >= 0)
    np.add.at(N, (idx[r, c], c, y[r]), 1.0)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * k)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * k * NL))
    return np.log(pc) - np.log(pm), N


def win_map(W2, P8, chunk=256):
    out = np.empty((len(P8), WG * WG), np.int16)
    for a in range(0, len(P8), chunk):
        V = windows(P8[a:a + chunk])
        n, w, d = V.shape
        Q = L2.cn(V.reshape(-1, d)).astype(np.float32)
        out[a:a + n] = (Q @ W2.T).argmax(1).reshape(n, w)
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    W1 = np.load(OUT / "coadapt.npz")["W"].astype(np.float64)
    P8tr, P8te = code(W1, Xtr), code(W1, Xte)
    print(f"L1 {len(W1)} templates -> pooled {PG}x{PG}, window {WIN}x{WIN} regions "
          f"= {WIN*WIN*len(W1)} numbers, {WG*WG} positions per image\n", flush=True)

    rng = np.random.default_rng(5)
    samp = []
    for a in range(0, len(P8tr), 256):
        V = windows(P8tr[a:a + 256])
        for i in range(len(V)):
            samp.append(V[i, rng.choice(WG * WG, PER_IMG, False)])
    Q = L2.cn(np.concatenate(samp)).astype(np.float32)
    print(f"  {len(Q):,} training windows", flush=True)

    res = {}
    for k in K2S:
        t1 = time.time()
        W2, n = L2.train(Q, k, np.random.default_rng(3))
        mtr, mte = win_map(W2, P8tr), win_map(W2, P8te)
        T, N = table(mtr, ytr, k)
        Tc = T
        sc = np.zeros((len(mte), NL))
        P = np.arange(WG * WG)
        for a in range(0, len(mte), 500):
            b = mte[a:a + 500]
            sc[a:a + len(b)] = Tc[b, P[None, :]].sum(1)
        acc = float((sc.argmax(1) == yte).mean())
        cl = N.sum(1)
        live = cl.sum(1) > 0
        pur = float((cl[live].max(1) / cl[live].sum(1)).mean())
        res[str(k)] = {"acc": acc, "purity": pur, "live": int(live.sum()),
                       "template_params": int(W2.size), "table_params": int(T.size),
                       "seconds": round(time.time() - t1, 1)}
        print(f"  K2={k:<4} acc {acc:.4f}   purity {pur:.3f}   live {int(live.sum())}/{k}   "
              f"templates {W2.size:,}  table {T.size:,}  ({time.time()-t1:.0f}s)", flush=True)
        np.savez_compressed(OUT / f"conv2_k{k}.npz", W2=W2.astype(np.float32), N=N)

    res["reference"] = {"L1 alone counted table": 0.9540,
                        "L2 whole-image K2=200": 0.9540,
                        "L2 whole-image K2=400": 0.9567,
                        "whole-image L2 params": 576000}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "conv2layer.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
