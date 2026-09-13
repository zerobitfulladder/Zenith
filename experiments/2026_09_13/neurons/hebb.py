"""One layer of cells over the whole image, settling with lateral inhibition, learning
from what the cell saw when it fired. No residual anywhere. No price: each cell holds
its own firing threshold to a target rate (homeostasis).

    settle   u <- (1-dt) u + dt ( W x - (G - I) a )      a = u where u > theta (per cell)
    hebb     an active cell moves toward the activity-weighted mean of the inputs it fired on
    resid    (comparison) an active cell moves toward the residual x - W^T a, weighted by its activity
    theta    rises if a cell fires more often than the target rate, falls if less

    python hebb.py <hebb|resid> <target rate>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

N_TRAIN, N_TEST, BATCH = 8000, 2000, 128
H, D = 256, 784
ITERS, DT = 60, 0.2
N_MAX = 200
THETA0, THETA_MIN, THETA_MAX, GAMMA, KAPPA = 0.05, 0.02, 1.0, 0.1, 0.1


def unit(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-8)


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    perm = np.random.default_rng(seed).permutation(len(X))
    X = X[perm]
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)[perm]
    return unit(X[:N_TRAIN]), unit(X[N_TRAIN:N_TRAIN + N_TEST]), y[N_TRAIN:N_TRAIN + N_TEST]


class Layer:
    def __init__(self, rule, rho, rng):
        self.rule, self.rho = rule, rho
        self.M = (rng.normal(size=(H, D)) / np.sqrt(D)).astype(np.float32)
        self.n = np.zeros(H, np.float32)
        self.theta = np.full(H, THETA0, np.float32)
        self.rate = np.full(H, rho, np.float32)
        self.refresh()

    def refresh(self):
        self.W = unit(self.M)
        self.G = self.W @ self.W.T
        np.fill_diagonal(self.G, 0.0)

    def settle(self, X):
        b = X @ self.W.T
        u = np.zeros_like(b); a = np.zeros_like(b)
        for _ in range(ITERS):
            u = (1 - DT) * u + DT * (b - a @ self.G)
            a = np.where(u > self.theta[None, :], u, 0.0).astype(np.float32)
        return a

    def learn(self, a, X):
        on = a > 0
        m = on.sum(0).astype(np.float32)
        p = m > 0
        asum = np.maximum(a.sum(0), 1e-8)
        if self.rule == "hebb":
            target = (a.T @ X) / asum[:, None]                       # what the cell saw, weighted by how hard it fired
            self.n[p] += m[p]
            eta = np.minimum(1.0, m[p] / np.minimum(self.n[p], N_MAX))
            self.M[p] += eta[:, None] * (target[p] - self.M[p])
        else:
            g = (a.T @ (X - a @ self.W)) / asum[:, None]              # the residual, weighted by activity
            self.n[p] += m[p]
            eta = np.minimum(1.0, m[p] / np.minimum(self.n[p], N_MAX))
            self.M[p] += eta[:, None] * g[p]
        # homeostasis: each cell holds its own rate near the target
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - self.rho) / self.rho), THETA_MIN, THETA_MAX)
        self.refresh()


def support(W, frac=0.9):
    w2 = np.maximum(W, 0) ** 2
    o = np.argsort(-w2, 1)
    cs = np.cumsum(np.take_along_axis(w2, o, 1), 1) / np.maximum(w2.sum(1, keepdims=True), 1e-12)
    return (cs < frac).sum(1) + 1


def main():
    rule, rho = sys.argv[1], float(sys.argv[2])
    tag = f"{rule}_r{rho:g}"
    t0 = time.time()
    Xtr, Xte, yte = load(0)
    L = Layer(rule, rho, np.random.default_rng(0))
    order = np.random.default_rng(1).permutation(len(Xtr))
    for step, b in enumerate(range(0, len(Xtr), BATCH)):
        xb = Xtr[order[b:b + BATCH]]
        a = L.settle(xb)
        L.learn(a, xb)
        if step % 10 == 0:
            print(f"  {tag} step {step:3d}  on/img {(a > 0).sum(1).mean():.2f}  unexpl {((xb - a @ L.W) ** 2).sum(1).mean():.3f}  "
                  f"theta mean {L.theta.mean():.3f}  cells firing this batch {(a > 0).any(0).sum()}  {time.time()-t0:.0f}s", flush=True)
    A = np.concatenate([L.settle(Xte[b:b + BATCH]) for b in range(0, len(Xte), BATCH)])
    xhat = A @ L.W
    unex = ((Xte - xhat) ** 2).sum(1)
    on = A > 0
    uses = on.sum(0)
    s = support(L.W)
    live = uses > 0
    res = dict(rule=rule, rho=rho, unexplained=float(unex.mean()), on_per_image=float(on.sum(1).mean()),
               dead=float(1 - live.mean()), support_median=float(np.median(s[live])),
               wholes=int((s[live] > 60).sum()), strokes=int(((s[live] > 20) & (s[live] <= 60)).sum()), dots=int((s[live] <= 20).sum()),
               distinct_configs=len({frozenset(np.flatnonzero(r)) for r in on}),
               theta_mean=float(L.theta.mean()), uses_entropy=float(-((uses[live] / uses[live].sum()) * np.log2(uses[live] / uses[live].sum())).sum()),
               t=time.time() - t0)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"{tag}.npz", W=L.W, theta=L.theta, uses=uses, Xte=Xte[:8], xhat=xhat[:8], A8=A[:8])
    print(f"{tag}: unexplained {res['unexplained']:.3f}  on/img {res['on_per_image']:.1f}  dead {res['dead']*100:.0f}%  "
          f"support {res['support_median']:.0f}px  wholes/strokes/dots {res['wholes']}/{res['strokes']}/{res['dots']}  "
          f"configs {res['distinct_configs']}  theta {res['theta_mean']:.3f}  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
