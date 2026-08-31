"""Rung one of the convolutional stack: one unit, slid over the image.

A UNIT is H hypercolumns, each holding K minicolumns (one template each).
At every position all H hypercolumns compete for the same 5x5 patch, exactly
as they competed for whole images before. The same unit is applied at every
position -- one set of weights, shared.

The rule is the settled base and nothing else:

    score(h) = how badly h rebuilds this patch
               - GAMMA * (1/H - win_fraction[h])        <- conscience
    the best-scoring hypercolumn takes one geodesic step toward the patch

There is no label here. A 5x5 patch has no class, so the grading knob reduces
to the drawing score alone. No repulsion, no reluctance, no sleep.

Patches are mean-centred and L2-normalised (the ablation says both earn their
place), and near-flat patches are dropped -- normalising empty background just
amplifies noise into a fake edge.

    python conv1.py [mnist|fashion_mnist] [--full]
"""

import json, resource, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
sys.path.insert(0, str(HERE.parent / "fashion"))
from common import EPS, geo_step                                   # noqa: E402

MEM_CAP_GB = 6
resource.setrlimit(resource.RLIMIT_AS,
                   (MEM_CAP_GB << 30, MEM_CAP_GB << 30))

PS, GAMMA, ETA = 5, 0.3, 0.5
N_TRAIN, N_TEST = 8000, 2000
EPOCHS, PER_IMG, BATCH = 3, 60, 4096
FLOOR = 0.05                      # a patch must have at least this much contrast
SEED = 0
DS = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "mnist"
SWEEP = ([(8, k) for k in (3, 4, 6, 8, 16, 32, 64)] +
         [(h, 8) for h in (4, 16, 32, 64)])       # (H, K); (8,8) is the crossing


def load(name):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


CHUNK_IMG, CHUNK_PATCH = 256, 100_000


def grid(X, stride=1):
    """(n, positions, 25) raw patches, float32 -- this is the array that blew
    the machine up when it was built for every image at once."""
    V = sliding_window_view(X, (PS, PS), axis=(1, 2))[:, ::stride, ::stride]
    return np.ascontiguousarray(V.reshape(len(X), -1, PS * PS), dtype=np.float32)


def prep(P):
    """Centre, drop the flat ones, L2-normalise. Returns patches and a keep mask."""
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    keep = n > FLOOR
    return C / np.maximum(n, EPS)[..., None], keep


def sample_patches(X, rng, per_img):
    out = []
    for a in range(0, len(X), CHUNK_IMG):
        Q, keep = prep(grid(X[a:a + CHUNK_IMG]))
        for i in range(len(Q)):
            idx = np.nonzero(keep[i])[0]
            if len(idx):
                out.append(Q[i, rng.choice(idx, size=min(per_img, len(idx)),
                                           replace=False)])
    return np.concatenate(out)


def errors(W, Q, want_S=True):
    """(n, H) relative rebuild error, and optionally the (n, H, K) coefficients.

    Chunked over patches and one hypercolumn at a time: the intermediate
    (n, H, 25) reconstruction is what filled 25 GB, so it is never built.
    """
    h, k, d = W.shape
    n = len(Q)
    E = np.empty((n, h), np.float32)
    S = np.empty((n, h, k), np.float32) if want_S else None
    for a in range(0, n, CHUNK_PATCH):
        B = Q[a:a + CHUNK_PATCH]
        qn = np.maximum(np.linalg.norm(B, axis=1), EPS)
        for j in range(h):
            s = B @ W[j].T                       # (m, K)
            r = s @ W[j]                         # (m, 25)
            r -= B
            E[a:a + len(B), j] = np.linalg.norm(r, axis=1) / qn
            if want_S:
                S[a:a + len(B), j] = s
    return E, S


def train(Xtr, H, K, rng):
    W = rng.standard_normal((H, K, PS * PS))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    f = np.full(H, 1.0 / H)
    wins = np.zeros(H, np.int64)
    for ep in range(EPOCHS):
        Q = sample_patches(Xtr, rng, PER_IMG)
        Q = Q[rng.permutation(len(Q))]
        for s in range(0, len(Q), BATCH):
            B = Q[s:s + BATCH]
            e, _ = errors(W, B, want_S=False)
            win = (e - GAMMA * (1.0 / H - f)[None]).argmin(1)
            cnt = np.bincount(win, minlength=H)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            wins += cnt
            for h in range(H):
                m = win == h
                if m.sum() >= 2:
                    W[h] = geo_step(W[h], B[m], ETA)
        print(f"    H={H} K={K} epoch {ep+1}/{EPOCHS}  ({len(Q)} patches)", flush=True)
    return W, wins


def evaluate(W, Xte, yte, wins):
    H, K, _ = W.shape
    n = len(Xte)
    F = np.zeros((n, H * K), np.float32)
    tot_err, tot_n, share = 0.0, 0, np.zeros(H, np.int64)
    kept = 0.0
    for a in range(0, n, CHUNK_IMG):
        Xc = Xte[a:a + CHUNK_IMG]
        Q, keep = prep(grid(Xc))
        npos = Q.shape[1]
        kf = keep.reshape(-1)
        kept += kf.mean() * len(Xc)
        e, S = errors(W, Q.reshape(-1, PS * PS)[kf])
        win = e.argmin(1)
        tot_err += float(e[np.arange(len(win)), win].sum()); tot_n += len(win)
        share += np.bincount(win, minlength=H)
        Sw = np.abs(S[np.arange(len(win)), win])
        idx = np.nonzero(kf)[0]
        img = a + idx // npos
        cols = win[:, None] * K + np.arange(K)[None, :]
        np.maximum.at(F, (img[:, None].repeat(K, 1), cols), Sw)
    err_win = tot_err / max(tot_n, 1)
    share = share / max(share.sum(), 1)
    kept = kept / n
    ov = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
                np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                        (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
          for i in range(H) for j in range(i + 1, H)] or [0.0]
    # F above is the message a second rung would receive: the winner's
    # coefficients at each position (graded, never a bare index), max-pooled
    # over the image. A probe only -- not part of the architecture.
    return {"rebuild_err": float(err_win), "dead": int((share == 0).sum()),
            "busiest": float(share.max()), "overlap": float(np.mean(ov)),
            "kept_patches": float(kf.mean())}, F


def probe(F, ytr, Fte, yte):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=400).fit(F, ytr)
    return float((m.predict(Fte) == yte).mean())


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(DS)
    print(f"{DS}: {len(Xtr)} train images, {PS}x{PS} patches, stride 1", flush=True)
    res, Ws = {}, {}
    for H, K in SWEEP:
        rng = np.random.default_rng(SEED + 1)
        W, wins = train(Xtr, H, K, rng)
        r, Fte = evaluate(W, Xte, yte, wins)
        _, Ftr = evaluate(W, Xtr[:4000], ytr[:4000], wins)
        r["probe"] = probe(Ftr, ytr[:4000], Fte, yte)
        r["templates"] = H * K
        res[f"H{H}_K{K}"] = r
        Ws[f"H{H}_K{K}"] = W
        print(f"  -> H={H} K={K}  rebuild {r['rebuild_err']:.4f}  "
              f"dead {r['dead']}/{H}  busiest {r['busiest']*100:.1f}%  "
              f"overlap {r['overlap']:.4f}  probe {r['probe']:.4f}", flush=True)
    np.savez_compressed(OUT / f"conv1_{DS}.npz",
                        **{k: v.astype(np.float32) for k, v in Ws.items()})
    (OUT / f"conv1_{DS}.json").write_text(json.dumps(
        {"dataset": DS, "patch": PS, "gamma": GAMMA, "eta": ETA,
         "epochs": EPOCHS, "per_img": PER_IMG, "results": res,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
