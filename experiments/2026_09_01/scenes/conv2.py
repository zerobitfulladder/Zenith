"""Layer 2, slid across the canvas instead of read off it whole.

The baseline's confound, measured: one digit centred scores 0.9790, the same
digit anywhere on the same canvas scores 0.8205. An identity map ties WHAT to
WHERE, so a digit at a new place is a new pattern and a global readout has to
learn it again at every location.

The fix is the thing this whole design has been claiming and layer 2 wasn't
doing: every layer emits a MAP. So the digit-level templates get a 28x28 window
and that window slides:

    L1 map over the whole canvas          44 x 96 winners, as before
    window                                24 x 24 of those positions
    pooled inside the window              4 x 4 blocks of 6x6  ->  1024 dims
    matched against K2 window templates   -> which template, how strongly
    emitted                               a 3 x 10 grid of identities

Identity is now computed in a window-local frame, so the same digit anywhere
produces the same code in a different cell. Position is carried by which cell,
not by which features fired.

The templates are trained teacher-free by the same spherical k-means, on windows
sampled from the training scenes themselves -- not on isolated centred digits,
which would smuggle in a definition of what an object is.
"""

import json, time
from pathlib import Path
import numpy as np
import scenes as S
import readouts as R

OUT = Path(__file__).resolve().parent / "results"
WIN = S.STAMP                                  # 28
PWIN = WIN - S.PS + 1                          # 24 L1 positions in a window
GW, BW = 4, (WIN - S.PS + 1) // 4              # 4x4 blocks of 6x6, as yesterday
YS = [0, 10, 20]                               # window origins: digit y-centre 14..34
XS = list(range(0, S.W - WIN + 1, 8))          # 10 across
NWIN = len(YS) * len(XS)
K2, EPOCHS = 100, 8
EPS = 1e-12


def _l1_dense(W1, B):
    Q, keep, norm = S.prep(S.patches(B))
    win = np.einsum("ipd,kd->ipk", Q, W1).argmax(-1)
    M = np.zeros((len(B), S.SY * S.SX, len(W1)), np.float32)
    r, c = np.nonzero(keep)
    M[r, c, win[r, c]] = norm[r, c]
    return M.reshape(len(B), S.SY, S.SX, len(W1))


def _windows(M):
    """(b, NWIN, 1024) -- the pooled L1 map inside each sliding window."""
    b, K = len(M), M.shape[-1]
    out = np.empty((b, NWIN, GW * GW * K), np.float32)
    w = 0
    for y0 in YS:
        for x0 in XS:
            T = M[:, y0:y0 + PWIN, x0:x0 + PWIN, :]
            T = T.reshape(b, GW, BW, GW, BW, K).max(axis=(2, 4))
            out[:, w] = T.reshape(b, -1)
            w += 1
    return out


def sample_windows(W1, X, n_per, seed, chunk=32):
    rng = np.random.default_rng(seed)
    out = []
    for a in range(0, len(X), chunk):
        V = _windows(_l1_dense(W1, X[a:a + chunk]))
        for i in range(len(V)):
            out.append(V[i, rng.choice(NWIN, n_per, replace=False)])
    return np.concatenate(out)


def train_l2(V, seed=0):
    """Spherical k-means over window codes. Empty windows are dropped, not learned."""
    e = np.linalg.norm(V, axis=1)
    floor = 0.35 * np.median(e[e > 0])
    Q = R.cn(V[e > floor])
    rng = np.random.default_rng(seed)
    W = R.kmeanspp(Q[rng.choice(len(Q), min(20000, len(Q)), False)], K2, rng)
    n = np.zeros(K2)
    for ep in range(EPOCHS):
        idx = rng.permutation(len(Q))
        for s in range(0, len(idx), 4096):
            B = Q[idx[s:s + 4096]]
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], 0.01) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
    return W, float(floor), int((n == 0).sum())


def digit_map(W1, W2, floor, X, chunk=32):
    """(n, NWIN * K2): at each window, which template won, carrying its score."""
    out = np.zeros((len(X), NWIN, len(W2)), np.float32)
    for a in range(0, len(X), chunk):
        V = _windows(_l1_dense(W1, X[a:a + chunk]))
        e = np.linalg.norm(V, axis=2)
        s = R.cn(V.reshape(-1, V.shape[-1])) @ W2.T
        s = s.reshape(len(V), NWIN, -1)
        win = s.argmax(-1)
        b, w = np.nonzero(e > floor)
        out[a + b, w, win[b, w]] = s[b, w, win[b, w]]
    return out.reshape(len(X), -1)


def main():
    t0 = time.time()
    sp = S.splits(); X, ptr, pte = S.load_digits(); W1 = S.load_w1()
    imgs, _, _ = S.make(4000, sp["train"], X, ptr, 31)
    V = sample_windows(W1, imgs, 4, 32)
    W2, floor, dead = train_l2(V)
    print(f"  L2 {K2} window templates, {len(V)} samples, floor {floor:.3f}, "
          f"dead {dead}   ({time.time()-t0:.0f}s)", flush=True)

    res = {"K2": K2, "windows": [len(YS), len(XS)], "dims": NWIN * K2,
           "floor": floor, "dead": dead}
    for name, gen in (("free", S.make_single),):
        a, ya = gen(6000, X, ptr, 21); b, yb = gen(2000, X, pte, 22)
        A = digit_map(W1, W2, floor, a); B = digit_map(W1, W2, floor, b)
        for lab, hid in (("logistic", 0), ("mlp", 256)):
            n = R.fit_net(A, np.arange(len(A)), None, ya, 10, "softmax",
                          hidden=hid, epochs=25)
            acc = float((R.predict_net(n, B, np.arange(len(B)), None).argmax(1) == yb).mean())
            res[f"single_{name}_{lab}"] = acc
            print(f"  single digit, {name} position, {lab:<9} {acc:.4f}", flush=True)
    res["reference"] = {"flat L2, free position": 0.8605,
                        "flat L2, fixed position": 0.9790}
    res["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(OUT / "l2_windows.npz", W2=W2.astype(np.float32),
                        floor=np.float32(floor))
    (OUT / "conv2.json").write_text(json.dumps(res, indent=2))
    print(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
