"""Whole-image cells, settling with lateral inhibition, Pearson drive, geodesic rotation.

    input      each image mean-centred and unit length; each template the same, so a cell's
               drive is the Pearson correlation between its template and the image
    settle     u <- (1-dt) u + dt ( c - (G - I) a ),  a = u where u > theta (per cell)
    learn      every cell that is on after settling rotates toward what it saw (the
               activity-weighted mean of the images it fired on, centred and unit length)
               by the angle eta * (its correlation), on the unit sphere. No residual.
    threshold  each cell holds its own firing rate near the target (homeostasis)

    python pearson.py <target rate> [passes]
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
H, D = 256, 784
ITERS, DT = 60, 0.2
ETA = 0.05
THETA0, THETA_MIN, GAMMA, KAPPA = 0.1, 0.02, 0.1, 0.1
EPS = 0.0             # division tilt: when a column is scaled to its budget, each cell's share goes to weight^(1+EPS),
                      # so the heavier user of a pixel slowly takes more of it. 0 = proportional (no division).
BUDGET = 0.0          # per-input budget scale: 0 = off; 1 = each axon gets its share of the total weight mass (256)
                      # in proportion to how often its pixel carries ink; 0.5 = half that


def centre_unit(V):
    V = V - V.mean(-1, keepdims=True)
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    perm = np.random.default_rng(seed).permutation(len(X))
    X = X[perm]
    return centre_unit(X[:N_TRAIN]), centre_unit(X[N_TRAIN:N_TRAIN + N_TEST]), X[N_TRAIN:N_TRAIN + N_TEST]


class Layer:
    def __init__(self, rho, rng, ink_freq=None):
        self.rho = rho
        self.W = centre_unit(rng.normal(size=(H, D)).astype(np.float32))
        self.budget = None
        if BUDGET > 0 and ink_freq is not None:
            share = ink_freq / ink_freq.sum()                              # each axon's share of the mass
            self.budget = (BUDGET * np.sqrt(H * share)).astype(np.float32)   # column L2 norm allowed per input line
        self.theta = np.full(H, THETA0, np.float32)
        self.rate = np.full(H, rho, np.float32)
        self.refresh()

    def refresh(self):
        self.G = self.W @ self.W.T
        np.fill_diagonal(self.G, 0.0)

    def settle(self, X):
        c = X @ self.W.T
        u = np.zeros_like(c); a = np.zeros_like(c)
        for _ in range(ITERS):
            u = (1 - DT) * u + DT * (c - a @ self.G)
            a = np.where(u > self.theta[None, :], u, 0.0).astype(np.float32)
        return a, c

    def learn(self, a, c, X):
        on = a > 0
        for t in np.flatnonzero(on.any(0)):
            sel = np.flatnonzero(on[:, t])
            cc = c[sel, t]
            keep = cc > 0
            if not keep.any():
                continue
            w = a[sel[keep], t]
            q = centre_unit((w[:, None] * X[sel[keep]]).sum(0) / w.sum())           # what it saw, weighted by how hard it fired
            wv = self.W[t]
            cq = float(np.clip(wv @ q, -1, 1))
            if cq <= 0 or cq >= 1:
                continue
            th = min(ETA * float(cc[keep].mean()) * keep.sum(), np.arccos(cq))    # angle: eta x correlation per image; never past the target
            tq = np.sqrt(1 - cq * cq)
            self.W[t] = centre_unit((np.cos(th) - cq * np.sin(th) / tq) * wv + (np.sin(th) / tq) * q)
        if self.budget is not None:
            if EPS > 0:                                                    # tilt each column toward its heavier users, norm kept
                col = np.linalg.norm(self.W, axis=0)
                V = np.sign(self.W) * np.abs(self.W) ** (1.0 + EPS)
                self.W = V * (col / np.maximum(np.linalg.norm(V, axis=0), 1e-12))[None, :]
            for _ in range(2):                                             # share and divide each axon's weight, then rows to unit length
                col = np.linalg.norm(self.W, axis=0)
                over = col > self.budget
                self.W[:, over] *= (self.budget[over] / col[over])[None, :]
                self.W = centre_unit(self.W)
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - self.rho) / self.rho), THETA_MIN, 1.0)
        self.refresh()


def smoothness(W):
    """Lag-1 spatial correlation of each template: ~1 for smooth strokes and wholes, low for dither."""
    T = W.reshape(-1, 28, 28)
    T = T - T.mean((1, 2), keepdims=True)
    v = (T ** 2).sum((1, 2)) + 1e-12
    h = (T[:, :, 1:] * T[:, :, :-1]).sum((1, 2))
    vv = (T[:, 1:, :] * T[:, :-1, :]).sum((1, 2))
    return (h + vv) / (2 * v)


def support(W, frac=0.9):
    w2 = np.maximum(W, 0) ** 2
    o = np.argsort(-w2, 1)
    cs = np.cumsum(np.take_along_axis(w2, o, 1), 1) / np.maximum(w2.sum(1, keepdims=True), 1e-12)
    return (cs < frac).sum(1) + 1


def main():
    global BUDGET, H, EPS
    rho = float(sys.argv[1]); passes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    if len(sys.argv) > 4:
        H = int(sys.argv[4])
    EPS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    tag = f"pearson_r{rho:g}" + (f"_b{BUDGET:g}" if BUDGET > 0 else "") + (f"_h{H}" if H != 256 else "") + (f"_e{EPS:g}" if EPS > 0 else "")
    t0 = time.time()
    Xtr, Xte, Xraw = load(0)
    Xraw_tr = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    ink_freq = (Xraw_tr[np.random.default_rng(0).permutation(len(Xraw_tr))[:N_TRAIN]] > 0.1).mean(0) + 1e-3
    L = Layer(rho, np.random.default_rng(0), ink_freq)
    rng = np.random.default_rng(1); step = -1
    for _ in range(passes):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            step += 1
            xb = Xtr[order[b:b + BATCH]]
            a, c = L.settle(xb)
            L.learn(a, c, xb)
            if step % 20 == 0:
                print(f"  {tag} step {step:3d}  on/img {(a > 0).sum(1).mean():.2f}  unexpl {((xb - a @ L.W) ** 2).sum(1).mean():.3f}  "
                      f"theta {L.theta.mean():.3f}  firing {(a > 0).any(0).sum()}/{H}  {time.time()-t0:.0f}s", flush=True)
    A = np.concatenate([L.settle(Xte[b:b + BATCH])[0] for b in range(0, len(Xte), BATCH)])
    xhat = A @ L.W
    unex = ((Xte - xhat) ** 2).sum(1)                       # inputs are unit length, so this is the share unexplained
    on = A > 0; uses = on.sum(0); live = uses > 0
    s = support(L.W)
    sm = smoothness(L.W)
    G = L.G
    res = dict(rho=rho, passes=passes, budget=BUDGET, eps=EPS, smoothness_median=float(np.median(sm[live])), unexplained=float(unex.mean()), unexplained_median=float(np.median(unex)),
               blowups=float((unex > 1).mean()), silent_images=float((on.sum(1) == 0).mean()),
               on_per_image=float(on.sum(1).mean()), dead=float(1 - live.mean()),
               support_median=float(np.median(s[live])), wholes=int((s[live] > 60).sum()),
               strokes=int(((s[live] > 20) & (s[live] <= 60)).sum()), dots=int((s[live] <= 20).sum()),
               overlap_mean=float(G[np.triu_indices(H, 1)].mean()), overlap_nearest5=float(np.sort(G, 1)[:, -5:].mean()),
               distinct_configs=len({frozenset(np.flatnonzero(r)) for r in on}), theta_mean=float(L.theta.mean()), t=time.time() - t0)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"{tag}.npz", W=L.W, theta=L.theta, uses=uses, Xte=Xte[:8], Xraw=Xraw[:8], xhat=xhat[:8], A8=A[:8])
    print(f"{tag}: unexplained {res['unexplained']:.3f} (median {res['unexplained_median']:.3f}, blow-ups {res['blowups']*100:.0f}%, "
          f"silent {res['silent_images']*100:.0f}%)  on/img {res['on_per_image']:.1f}  dead {res['dead']*100:.0f}%  "
          f"size {res['support_median']:.0f}px  wholes/strokes/dots {res['wholes']}/{res['strokes']}/{res['dots']}  "
          f"overlap {res['overlap_mean']:.3f} (nearest 5: {res['overlap_nearest5']:.3f})  smoothness {res['smoothness_median']:.2f}  configs {res['distinct_configs']}  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
