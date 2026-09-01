"""Why did the sliding window lose 24 points? Separate the two things it changed.

It changed the FRAME (identity computed in a window, not against the canvas) and
it changed the CODE (a hard winner over K2 templates, instead of a 2048-dim
graded map). Either could be the cost. An oracle picks the correct window, so
window SELECTION is removed from the question too.

    A  oracle window, graded pooled L1 code    frame alone, no quantisation
    B  oracle window, one-hot L2 identity      + quantisation, K2 swept
    C  all windows, one-hot per window         + selection      (what conv2 did)
    D  all windows, top-3 graded per window    + selection, softer code

A near 0.979 says a window-local frame is sound and the loss is all in the code.
A near 0.86 says the frame itself is the problem and the design needs rethinking.
"""

import json, time
from pathlib import Path
import numpy as np
import scenes as S, readouts as R, conv2 as C

OUT = Path(__file__).resolve().parent / "results"
N_TR, N_TE, EP = 6000, 2000, 25
KS = [100, 300]


def single(n, X, bucket, seed):
    """One digit, free position, and we keep where we put it."""
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 10, n)
    imgs = np.zeros((n, S.H, S.W), np.float32)
    xs = rng.integers(S.STAMP // 2, S.W - S.STAMP // 2 + 1, n)
    ys = rng.integers(S.STAMP // 2, S.H - S.STAMP // 2 + 1, n)
    for i in range(n):
        y0, x0 = ys[i] - 14, xs[i] - 14
        t = imgs[i, y0:y0 + 28, x0:x0 + 28]
        np.maximum(t, X[rng.choice(bucket[cls[i]])], out=t)
    return imgs, cls.astype(np.int64), xs, ys


def oracle(xs, ys):
    """Index of the window whose centre is nearest the digit's true centre."""
    cy = np.array([y0 + 14 for y0 in C.YS for _ in C.XS])
    cx = np.array([x0 + 14 for _ in C.YS for x0 in C.XS])
    d = (ys[:, None] - cy) ** 2 + (xs[:, None] - cx) ** 2
    return d.argmin(1)


def wcodes(W1, X, chunk=32):
    out = np.empty((len(X), C.NWIN, C.GW * C.GW * len(W1)), np.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = C._windows(C._l1_dense(W1, X[a:a + chunk]))
    return out


def train_l2(V, k, seed=0):
    e = np.linalg.norm(V, axis=1)
    floor = 0.35 * np.median(e[e > 0])
    Q = R.cn(V[e > floor])
    rng = np.random.default_rng(seed)
    W = R.kmeanspp(Q[rng.choice(len(Q), min(20000, len(Q)), False)], k, rng)
    n = np.zeros(k)
    for ep in range(8):
        idx = rng.permutation(len(Q))
        for s in range(0, len(idx), 4096):
            B = Q[idx[s:s + 4096]]
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], 0.01) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + 1e-12
    return W, float(floor)


def score(A, ya, B, yb, tag, res):
    for lab, hid in (("logistic", 0), ("mlp", 256)):
        n = R.fit_net(A, np.arange(len(A)), None, ya, 10, "softmax", hidden=hid, epochs=EP)
        acc = float((R.predict_net(n, B, np.arange(len(B)), None).argmax(1) == yb).mean())
        res[f"{tag}_{lab}"] = acc
        print(f"  {tag:<28} {lab:<9} {acc:.4f}", flush=True)


def emit(V, W2, floor, topk):
    """(n, NWIN*K2) -- top-k template scores per window, gated on window energy."""
    n, nw, _ = V.shape
    out = np.zeros((n, nw, len(W2)), np.float32)
    e = np.linalg.norm(V, axis=2)
    s = (R.cn(V.reshape(-1, V.shape[-1])) @ W2.T).reshape(n, nw, -1)
    top = np.argsort(-s, axis=2)[:, :, :topk]
    b, w = np.nonzero(e > floor)
    for j in range(topk):
        out[b, w, top[b, w, j]] = s[b, w, top[b, w, j]]
    return out.reshape(n, -1)


def main():
    t0 = time.time(); sp = S.splits(); X, ptr, pte = S.load_digits(); W1 = S.load_w1()
    a, ya, xa, yya = single(N_TR, X, ptr, 21)
    b, yb, xb, yyb = single(N_TE, X, pte, 22)
    Va, Vb = wcodes(W1, a), wcodes(W1, b)
    oa, ob = oracle(xa, yya), oracle(xb, yyb)
    res = {"reference": {"flat L2 free": 0.8605, "flat L2 fixed": 0.9790,
                         "conv2 all-windows K2=100 top1": 0.6145}}
    print(f"  encoded ({time.time()-t0:.0f}s)", flush=True)

    # A -- the frame alone
    score(R.cn(Va[np.arange(len(Va)), oa]), ya,
          R.cn(Vb[np.arange(len(Vb)), ob]), yb, "A oracle window, graded", res)

    tr_imgs, _, _ = S.make(4000, sp["train"], X, ptr, 31)
    Vtr = wcodes(W1, tr_imgs)
    rng = np.random.default_rng(32)
    samp = np.concatenate([Vtr[i, rng.choice(C.NWIN, 4, False)] for i in range(len(Vtr))])
    for k in KS:
        W2, floor = train_l2(samp, k)
        A1 = emit(Va[np.arange(len(Va)), oa][:, None, :], W2, floor, 1)
        B1 = emit(Vb[np.arange(len(Vb)), ob][:, None, :], W2, floor, 1)
        score(A1, ya, B1, yb, f"B oracle window, K2={k} top1", res)
        for topk in (1, 3):
            score(emit(Va, W2, floor, topk), ya, emit(Vb, W2, floor, topk), yb,
                  f"{'C' if topk==1 else 'D'} all windows, K2={k} top{topk}", res)
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "window_control.json").write_text(json.dumps(res, indent=2))
    print(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
