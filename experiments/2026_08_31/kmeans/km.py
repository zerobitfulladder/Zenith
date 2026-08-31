"""Online spherical k-means on MNIST patches. Nothing else.

Patches are mean-centred and L2-normalised, so the inner product between a
patch and a template IS their Pearson correlation, and nearest-centroid and
highest-correlation choose the same winner.

    i   <- argmax_j  w_j . x            the winner, by correlation
    n_i <- n_i + 1
    w_i <- w_i + eta_i (x - w_i),  renormalised,  eta_i = max(1/n_i, ETA_MIN)

The 1/n schedule makes each template the running mean of everything it has won;
the floor stops it freezing so it can still track a changing stream. Templates
are seeded k-means++ style from real patches, not from noise -- seeding from
noise is why units died in every earlier run.

The output of this layer is one index per position. No dense code.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
PS, FLOOR, ETA_MIN = 5, 0.05, 0.01
N_TRAIN, N_TEST, EPOCHS, PER_IMG, SEED = 8000, 2000, 3, 60, 0
EPS = 1e-12
KS = [16, 64, 256]
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"


def load(name):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    return X[p][:N_TRAIN], y[p][:N_TRAIN], X[p][N_TRAIN:N_TRAIN + N_TEST], y[p][N_TRAIN:N_TRAIN + N_TEST]


def patches(X):
    V = sliding_window_view(X, (PS, PS), axis=(1, 2))
    return np.ascontiguousarray(V.reshape(len(X), -1, PS * PS), dtype=np.float64)


def prep(P):
    """Centre, keep only patches with real contrast, normalise."""
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    return C / np.maximum(n, EPS)[..., None], n > FLOOR, n


def sample(X, rng, per_img):
    out = []
    for a in range(0, len(X), 256):
        Q, keep, _ = prep(patches(X[a:a + 256]))
        for i in range(len(Q)):
            idx = np.nonzero(keep[i])[0]
            if len(idx):
                out.append(Q[i, rng.choice(idx, min(per_img, len(idx)), False)])
    return np.concatenate(out)


def kmeanspp(Q, K, rng):
    """Seed from real patches, spread by how badly they are already explained."""
    W = np.empty((K, Q.shape[1]))
    W[0] = Q[rng.integers(len(Q))]
    d2 = 2.0 - 2.0 * (Q @ W[0])
    for k in range(1, K):
        p = np.maximum(d2, 0); s = p.sum()
        i = rng.integers(len(Q)) if s <= 0 else rng.choice(len(Q), p=p / s)
        W[k] = Q[i]
        d2 = np.minimum(d2, 2.0 - 2.0 * (Q @ W[k]))
    return W


def train(Q, K, rng, batch=4096):
    W = kmeanspp(Q[rng.choice(len(Q), min(20000, len(Q)), False)], K, rng)
    n = np.zeros(K)
    for ep in range(EPOCHS):
        for s in range(0, len(Q), batch):
            B = Q[s:s + batch]
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                eta = max(1.0 / n[j], ETA_MIN) * len(m)
                W[j] += min(eta, 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
        print(f"    K={K} epoch {ep+1}/{EPOCHS}", flush=True)
    return W, n


def evaluate(W, X):
    K = len(W)
    tot_r, tot_n, use = 0.0, 0, np.zeros(K)
    for a in range(0, len(X), 256):
        Q, keep, _ = prep(patches(X[a:a + 256]))
        F = Q.reshape(-1, PS * PS)[keep.reshape(-1)]
        S = F @ W.T
        win = S.argmax(1)
        tot_r += float(S[np.arange(len(win)), win].sum()); tot_n += len(win)
        use += np.bincount(win, minlength=K)
    r = tot_r / max(tot_n, 1)
    use /= max(use.sum(), 1)
    return {"correlation": r, "rel_error": float(np.sqrt(max(2 - 2 * r, 0))),
            "dead": int((use == 0).sum()), "busiest": float(use.max()),
            "entropy_bits": float(-(use[use > 0] * np.log2(use[use > 0])).sum()),
            "max_bits": float(np.log2(K)), "use": use.tolist()}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(DS)
    rng = np.random.default_rng(SEED + 1)
    Q = sample(Xtr, rng, PER_IMG)
    print(f"{DS}: {len(Q)} training patches of {PS}x{PS}", flush=True)
    res, Ws = {}, {}
    for K in KS:
        W, n = train(Q, K, np.random.default_rng(SEED + 1))
        r = evaluate(W, Xte)
        res[str(K)] = r; Ws[str(K)] = W
        print(f"  -> K={K:<4} correlation {r['correlation']:.4f}  "
              f"error {r['rel_error']:.4f}  dead {r['dead']}/{K}  "
              f"busiest {r['busiest']*100:.1f}%  "
              f"code {r['entropy_bits']:.2f} of {r['max_bits']:.0f} bits",
              flush=True)
    np.savez_compressed(OUT / f"km_{DS}.npz",
                        **{k: v.astype(np.float32) for k, v in Ws.items()})
    (OUT / f"km_{DS}.json").write_text(json.dumps(
        {"dataset": DS, "patch": PS, "results": res,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
