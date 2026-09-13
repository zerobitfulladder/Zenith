"""Raw input, two budgets, share the input then scale the weights.

input      pixels as they are, 0..1. No centring, no unit length.
weights    nonnegative. Each template's total weight (sum) may not exceed ROW_B.
           Each pixel's total weight across all templates may not exceed its share of
           the total mass, in proportion to how often it carries ink.
settle     as before: drive = template direction . x, inhibition by cosine overlap,
           per-cell thresholds held by homeostasis. Cells on at the end are the code.
learn      (1) reshare the input: pixel p of image i is divided among the cells that
               are on, in proportion to their claim a_t * w_tp; a cell receives its share
           (2) each active cell grows toward the average share it received
           (3) scale: rows to ROW_B if over, columns to their pixel budget if over, twice
No rotation, no residual, no winner, no adoption.

    python share.py <target rate> [passes] [H]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"

N_TRAIN, N_TEST, BATCH = 8000, 2000, 128
H, D = 144, 784
ITERS, DT = 60, 0.2
ETA, N_MAX = 0.05, 200
ROW_B = 1.0
THETA0, THETA_MIN, GAMMA, KAPPA = 0.1, 0.02, 0.1, 0.1


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    perm = np.random.default_rng(seed).permutation(len(X))
    X = X[perm]
    return X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST]


class Layer:
    def __init__(self, rho, rng, ink_freq):
        self.rho = rho
        self.W = rng.random((H, D)).astype(np.float32) * ink_freq[None, :]
        self.W *= ROW_B / self.W.sum(1, keepdims=True)
        self.col_b = (H * ROW_B * ink_freq / ink_freq.sum()).astype(np.float32)     # each pixel's share of the total mass
        self.n = np.zeros(H, np.float32)
        self.theta = np.full(H, THETA0, np.float32)
        self.rate = np.full(H, rho, np.float32)
        self.refresh()

    def refresh(self):
        self.Wd = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-8)    # directions, for reading
        self.G = self.Wd @ self.Wd.T
        np.fill_diagonal(self.G, 0.0)

    def settle(self, X):
        b = X @ self.Wd.T
        u = np.zeros_like(b); a = np.zeros_like(b)
        for _ in range(ITERS):
            u = (1 - DT) * u + DT * (b - a @ self.G)
            a = np.where(u > self.theta[None, :], u, 0.0).astype(np.float32)
        return a

    def learn(self, a, X):
        on = a > 0
        claim = a[:, :, None] * self.W[None, :, :]                                   # (B,H,D) each cell's claim on each pixel
        tot = claim.sum(1, keepdims=True)
        share = np.where(tot > 0, claim / np.maximum(tot, 1e-12), 0.0) * X[:, None, :]   # the input, divided among the cells on
        m = on.sum(0).astype(np.float32)
        p = m > 0
        recv = share.sum(0)[p] / m[p, None]                                          # average share each active cell received
        self.n[p] += m[p]
        eta = np.minimum(1.0, ETA * m[p] / np.minimum(self.n[p], N_MAX) * N_MAX)    # count-based, ETA at maturity
        self.W[p] += eta[:, None] * (recv - self.W[p])
        np.maximum(self.W, 0, out=self.W)
        for _ in range(2):                                                           # scale to the two budgets
            rs = self.W.sum(1)
            over = rs > ROW_B
            self.W[over] *= (ROW_B / rs[over])[:, None]
            cs = self.W.sum(0)
            over = cs > self.col_b
            self.W[:, over] *= (self.col_b[over] / cs[over])[None, :]
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - self.rho) / self.rho), THETA_MIN, 10.0)
        self.refresh()


def support(W, frac=0.9):
    w2 = W ** 2
    o = np.argsort(-w2, 1)
    cs = np.cumsum(np.take_along_axis(w2, o, 1), 1) / np.maximum(w2.sum(1, keepdims=True), 1e-12)
    return (cs < frac).sum(1) + 1


def smoothness(W):
    T = W.reshape(-1, 28, 28); T = T - T.mean((1, 2), keepdims=True)
    v = (T ** 2).sum((1, 2)) + 1e-12
    return ((T[:, :, 1:] * T[:, :, :-1]).sum((1, 2)) + (T[:, 1:, :] * T[:, :-1, :]).sum((1, 2))) / (2 * v)


def main():
    global H
    rho = float(sys.argv[1]); passes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if len(sys.argv) > 3:
        H = int(sys.argv[3])
    tag = f"share_r{rho:g}_h{H}"
    t0 = time.time()
    Xtr, Xte = load(0)
    ink = (Xtr > 0.1).mean(0) + 1e-3
    L = Layer(rho, np.random.default_rng(0), ink)
    rng = np.random.default_rng(1); step = -1
    for _ in range(passes):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            step += 1
            xb = Xtr[order[b:b + BATCH]]
            a = L.settle(xb)
            L.learn(a, xb)
            if step % 20 == 0:
                print(f"  {tag} step {step:3d}  on/img {(a > 0).sum(1).mean():.2f}  theta {L.theta.mean():.3f}  firing {(a > 0).any(0).sum()}/{H}  "
                      f"row sums {L.W.sum(1).mean():.2f}  {time.time()-t0:.0f}s", flush=True)
    A = np.concatenate([L.settle(Xte[b:b + BATCH]) for b in range(0, len(Xte), BATCH)])
    xhat = A @ L.Wd
    cos = (Xte * xhat).sum(1) / np.maximum(np.linalg.norm(Xte, axis=1) * np.linalg.norm(xhat, axis=1), 1e-8)
    unex = 1 - cos ** 2                                                             # share of the image's energy the rebuild cannot account for, best scale
    on = A > 0; uses = on.sum(0); live = uses > 0
    s = support(L.W); sm = smoothness(L.W)
    G = L.G
    res = dict(rho=rho, passes=passes, H=H, unexplained=float(unex.mean()), unexplained_median=float(np.median(unex)),
               silent_images=float((on.sum(1) == 0).mean()), on_per_image=float(on.sum(1).mean()), dead=float(1 - live.mean()),
               support_median=float(np.median(s[live])), wholes=int((s[live] > 60).sum()),
               strokes=int(((s[live] > 20) & (s[live] <= 60)).sum()), dots=int((s[live] <= 20).sum()),
               overlap_mean=float(G[np.triu_indices(H, 1)].mean()), overlap_nearest5=float(np.sort(G, 1)[:, -5:].mean()),
               smoothness_median=float(np.median(sm[live])), distinct_configs=len({frozenset(np.flatnonzero(r)) for r in on}),
               theta_mean=float(L.theta.mean()), t=time.time() - t0)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"{tag}.npz", W=L.W, theta=L.theta, uses=uses, Xte=Xte[:8], xhat=xhat[:8], A8=A[:8])
    print(f"{tag}: unexplained {res['unexplained']:.3f} (median {res['unexplained_median']:.3f}, silent {res['silent_images']*100:.0f}%)  "
          f"on/img {res['on_per_image']:.1f}  dead {res['dead']*100:.0f}%  size {res['support_median']:.0f}px  "
          f"wholes/strokes/dots {res['wholes']}/{res['strokes']}/{res['dots']}  overlap {res['overlap_mean']:.3f} (nearest 5: {res['overlap_nearest5']:.3f})  "
          f"smoothness {res['smoothness_median']:.2f}  configs {res['distinct_configs']}  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
