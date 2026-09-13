"""Two layers, same settling, same share-then-scale learning, the label OR-ed into the top's input
as a random sparse pattern over layer 1's lines (no label lines).

layer 1   784 raw pixels -> H1 cells
layer 2   [H1 activities ; 10 label lines x LABEL_W] -> H2 cells. Label present in
          training, absent at read time (top cells then read with the direction of
          their identity part only).
settling  both layers together: layer-1 drive = pixels . directions + top-down drive from
          the top's expectation; layer-2 drive = z . directions; cosine inhibition within
          each layer; per-cell thresholds by homeostasis.
learning  each layer: divide its input among the cells on by claim (a * w), each cell
          moves toward its share, rows scaled to 1, columns scaled to their budget.
label     read from the top: argmax over the label weights of the cells on.
generate  drive the top with the label line alone, settle, expand the top's expectation
          down through layer 1.

    python tower_or.py [passes]
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
H1, H2, D = 144, 100, 784
RHO1, RHO2 = 0.035, 0.03
ITERS, DT = 60, 0.2
ETA, N_MAX = 0.05, 200
THETA0, THETA_MIN, GAMMA, KAPPA = 0.1, 0.02, 0.1, 0.1
BETA = 0.5             # top-down drive, in units of the layer-1 mean threshold
K_ON = 10              # lines on in each class's random label pattern (of H1)


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


class Layer:
    def __init__(self, d, h, rho, rng, freq, d_code=None):
        self.d, self.h, self.rho = d, h, rho
        self.dc = d if d_code is None else d_code
        self.W = rng.random((h, d)).astype(np.float32) * np.maximum(freq, 1e-3)[None, :]
        self.W *= 1.0 / self.W.sum(1, keepdims=True)
        self.freq = np.maximum(freq, 1e-3).astype(np.float32)
        self.n = np.zeros(h, np.float32)
        self.theta = np.full(h, THETA0, np.float32)
        self.rate = np.full(h, rho, np.float32)
        self.refresh()

    @property
    def col_b(self):
        return (self.h * self.freq / self.freq.sum()).astype(np.float32)

    def refresh(self):
        self.Wd = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-8)
        Wc = np.zeros_like(self.W)
        Wc[:, :self.dc] = self.W[:, :self.dc]
        self.Wdc = Wc / np.maximum(np.linalg.norm(Wc, axis=1, keepdims=True), 1e-8)      # identity part only
        self.G = self.Wd @ self.Wd.T
        np.fill_diagonal(self.G, 0.0)
        self.Gc = self.Wdc @ self.Wdc.T
        np.fill_diagonal(self.Gc, 0.0)

    def learn_free(self, a, X):
        """Phase two: an active cell moves toward the activity-weighted mean of what it saw.
        No sharing, no budgets; any synapse may grow as much as it wants."""
        on = a > 0
        m = on.sum(0).astype(np.float32)
        p = m > 0
        if p.any():
            w = a[:, p]
            zbar = (w.T @ X) / np.maximum(w.sum(0)[:, None], 1e-8)
            self.n[p] += m[p]
            eta = np.minimum(1.0, ETA * m[p] / np.minimum(self.n[p], N_MAX) * N_MAX)
            self.W[p] += eta[:, None] * (zbar - self.W[p])
            np.maximum(self.W, 0, out=self.W)
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - self.rho) / self.rho), THETA_MIN, 50.0)
        self.refresh()

    def learn(self, a, X):
        on = a > 0
        claim = a[:, :, None] * self.W[None, :, :]
        tot = claim.sum(1, keepdims=True)
        share = np.where(tot > 0, claim / np.maximum(tot, 1e-12), 0.0) * X[:, None, :]
        m = on.sum(0).astype(np.float32)
        p = m > 0
        if p.any():
            recv = share.sum(0)[p] / m[p, None]
            self.n[p] += m[p]
            eta = np.minimum(1.0, ETA * m[p] / np.minimum(self.n[p], N_MAX) * N_MAX)
            self.W[p] += eta[:, None] * (recv - self.W[p])
            np.maximum(self.W, 0, out=self.W)
        # column budget follows the running frequency of each input line
        self.freq = 0.98 * self.freq + 0.02 * (X > 0).mean(0)
        for _ in range(2):
            rs = self.W.sum(1)
            over = rs > 1.0
            self.W[over] *= (1.0 / rs[over])[:, None]
            cs = self.W.sum(0)
            cb = self.col_b
            over = cs > cb
            self.W[:, over] *= (cb[over] / cs[over])[None, :]
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - self.rho) / self.rho), THETA_MIN, 50.0)
        self.refresh()


class Tower:
    def __init__(self, rng, ink_freq):
        self.L1 = Layer(D, H1, RHO1, rng, ink_freq)
        self.label_w = 1.0
        self.pat = np.zeros((10, H1), np.float32)                      # one random sparse pattern per class, over layer 1's lines
        for c in range(10):
            self.pat[c, rng.choice(H1, K_ON, replace=False)] = 1.0
        self.L2 = Layer(H1, H2, RHO2, rng, np.full(H1, RHO1, np.float32))

    def settle(self, X, y=None, beta=BETA):
        B = len(X)
        lab = self.label_w * self.pat[y] if y is not None else None
        W2, G2 = self.L2.Wd, self.L2.G
        b1 = X @ self.L1.Wd.T
        u1 = np.zeros((B, H1), np.float32); a1 = np.zeros((B, H1), np.float32)
        u2 = np.zeros((B, H2), np.float32); a2 = np.zeros((B, H2), np.float32)
        th1 = float(self.L1.theta.mean())
        for _ in range(ITERS):
            e = a2 @ self.L2.W
            td = beta * th1 * e / np.maximum(e.max(1, keepdims=True), 1e-8)
            u1 = (1 - DT) * u1 + DT * (b1 + td - a1 @ self.L1.G)
            a1 = np.where(u1 > self.L1.theta[None, :], u1, 0.0).astype(np.float32)
            z = np.maximum(a1, lab) if lab is not None else a1               # OR: the label pattern rides on the same lines
            b2 = z @ W2.T
            u2 = (1 - DT) * u2 + DT * (b2 - a2 @ G2)
            a2 = np.where(u2 > self.L2.theta[None, :], u2, 0.0).astype(np.float32)
        return a1, a2, z

    def label_top(self, a2):
        """The top's expectation over layer 1's lines, matched against each class pattern; the ten
        matches inhibit each other until one stands."""
        e = a2 @ self.L2.W
        v = e @ self.pat.T / K_ON
        u = np.zeros_like(v); l = np.zeros_like(v)
        for _ in range(ITERS):
            u = (1 - DT) * u + DT * (v - (l.sum(1, keepdims=True) - l))
            l = np.maximum(u, 0)
        p = l.argmax(1)
        p[(l.max(1) <= 0) | ((a2 > 0).sum(1) == 0)] = -1
        return p

    def settle_down(self, e, gain=1.5):
        """Layer 1 settled under an expectation alone (no pixels): the most expected cell is driven to
        `gain` times its threshold, inhibition decides the rest. Returns settled activities (B,H1)."""
        B = len(e)
        td = gain * self.L1.theta[None, :] * e / np.maximum(e.max(1, keepdims=True), 1e-8)
        u1 = np.zeros((B, H1), np.float32); a1 = np.zeros((B, H1), np.float32)
        for _ in range(ITERS):
            u1 = (1 - DT) * u1 + DT * (td - a1 @ self.L1.G)
            a1 = np.where(u1 > self.L1.theta[None, :], u1, 0.0).astype(np.float32)
        return a1

    def draw_top(self):
        """What each top cell proposes: layer 1 settled under that cell's expectation, painted."""
        a1 = self.settle_down(self.L2.W.copy())
        return a1 @ self.L1.W, a1

    def generate(self):
        """Drive the top with each label line alone, settle the top, settle layer 1 under the top's
        expectation, paint the settled layer-1 cells."""
        z = self.label_w * self.pat
        b2 = z @ self.L2.Wd.T
        u2 = np.zeros_like(b2); a2 = np.zeros_like(b2)
        for _ in range(ITERS):
            u2 = (1 - DT) * u2 + DT * (b2 - a2 @ self.L2.G)
            a2 = np.where(u2 > self.L2.theta[None, :], u2, 0.0).astype(np.float32)
        for c in range(10):                                   # pattern alone clears nothing: take the cells that carry it most
            if not (a2[c] > 0).any():
                sc = self.L2.W @ self.pat[c]
                top = np.argsort(-sc)[:3]
                a2[c, top] = sc[top]
        e = a2 @ self.L2.W
        a1 = self.settle_down(e)
        return a1 @ self.L1.W, a2, a1


def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    passes2 = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # phase two: label on, top learns free
    global K_ON
    if len(sys.argv) > 3:
        K_ON = int(sys.argv[3])
    share2 = len(sys.argv) > 4 and sys.argv[4] == "share"           # phase two: the top keeps sharing its input
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(0)
    ink = (Xtr > 0.1).mean(0)
    tw = Tower(np.random.default_rng(0), ink)
    # the label line counts as one active layer-1 cell: its value tracks the running mean active layer-1 activity
    rng = np.random.default_rng(1); step = -1
    for _ in range(passes):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            step += 1
            i = order[b:b + BATCH]
            a1, a2, z = tw.settle(Xtr[i], None if passes2 > 0 else ytr[i])     # phase one: no label when a phase two follows
            if (a1 > 0).any():
                tw.label_w = 0.9 * tw.label_w + 0.1 * float(a1[a1 > 0].mean())
            tw.L1.learn(a1, Xtr[i])
            tw.L2.learn(a2, z)
            if step % 20 == 0:
                print(f"  step {step:3d}  L1 on {(a1 > 0).sum(1).mean():.1f}  L2 on {(a2 > 0).sum(1).mean():.1f}  "
                      f"label_w {tw.label_w:.2f}  theta1 {tw.L1.theta.mean():.2f} theta2 {tw.L2.theta.mean():.2f}  {time.time()-t0:.0f}s", flush=True)
    for _ in range(passes2):                                        # phase two
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            step += 1
            i = order[b:b + BATCH]
            a1, a2, z = tw.settle(Xtr[i], ytr[i])
            tw.L1.learn(a1, Xtr[i])                                  # layer 1 keeps sharing
            if share2:
                tw.L2.learn(a2, z)                                   # the top keeps sharing: the label's bits get divided like any other line
            else:
                tw.L2.learn_free(a2, z)                              # the top learns what it saw, label included
            if step % 20 == 0:
                print(f"  phase two step {step:3d}  L1 on {(a1 > 0).sum(1).mean():.1f}  L2 on {(a2 > 0).sum(1).mean():.1f}  "
                      f"top label share {(tw.L2.W[:, H1:].sum(1) / np.maximum(tw.L2.W.sum(1), 1e-8)).mean():.2f}  {time.time()-t0:.0f}s", flush=True)
    # held out, label absent, feedback on and off
    res = dict(passes=passes, passes2=passes2, H1=H1, H2=H2, label_w=tw.label_w, modes={})
    for beta, name in ((BETA, "feedback"), (0.0, "no_feedback")):
        A1, A2, P = [], [], []
        for b in range(0, len(Xte), BATCH):
            a1, a2, _ = tw.settle(Xte[b:b + BATCH], None, beta)
            A1.append(a1); A2.append(a2); P.append(tw.label_top(a2))
        A1, A2, P = map(np.concatenate, (A1, A2, P))
        xhat = A1 @ tw.L1.Wd
        cos = (Xte * xhat).sum(1) / np.maximum(np.linalg.norm(Xte, axis=1) * np.linalg.norm(xhat, axis=1), 1e-8)
        cov = P >= 0
        res["modes"][name] = dict(label_top=float((P == yte).mean()), covered=float(cov.mean()),
                                  label_top_when_covered=float((P[cov] == yte[cov]).mean()) if cov.any() else 0.0,
                                  unexplained=float((1 - cos ** 2).mean()), L1_on=float((A1 > 0).sum(1).mean()),
                                  L2_on=float((A2 > 0).sum(1).mean()), L1_dead=float(((A1 > 0).sum(0) == 0).mean()),
                                  L2_dead=float(((A2 > 0).sum(0) == 0).mean()),
                                  L2_configs=len({frozenset(np.flatnonzero(r)) for r in A2 > 0}))
        if name == "feedback":
            keep = dict(A1=A1, A2=A2, P=P, xhat=xhat)
        m = res["modes"][name]
        print(f"{name:12s}: label at top {m['label_top']:.3f} (covered {m['covered']*100:.0f}%, {m['label_top_when_covered']:.3f} when covered)  "
              f"unexplained {m['unexplained']:.3f}  L1 on {m['L1_on']:.1f} dead {m['L1_dead']*100:.0f}%  L2 on {m['L2_on']:.1f} dead {m['L2_dead']*100:.0f}%  "
              f"L2 configs {m['L2_configs']}", flush=True)
    W2 = tw.L2.W
    ident = W2
    members = (ident > 0.5 * ident.max(1, keepdims=True)).sum(1)
    share1 = ident.max(1) / np.maximum(np.linalg.norm(ident, axis=1), 1e-8)
    match = W2 @ tw.pat.T                                                    # weight each cell puts on each class pattern
    lab_share = match.max(1) / np.maximum(W2.sum(1), 1e-8)
    live2 = (keep["A2"] > 0).sum(0) > 0
    res["L2"] = dict(members_median=float(np.median(members[live2])), members_mean=float(members[live2].mean()),
                     wrappers=float((share1[live2] > 0.9).mean()), label_share_median=float(np.median(lab_share[live2])),
                     label_share_mean=float(lab_share[live2].mean()))
    gen, a2g, a1g = tw.generate()
    top_pic, top_a1 = tw.draw_top()
    top1 = ident.max(1) / np.maximum(ident.sum(1), 1e-8)
    res["L2"]["top1_share_median"] = float(np.median(top1[live2]))
    res["L2"]["proposal_cells_median"] = float(np.median((top_a1 > 0).sum(1)[live2]))
    lab = match / np.maximum(match.sum(1, keepdims=True), 1e-12)
    ent = -(lab * np.log2(lab + 1e-12)).sum(1)
    res["L2"]["label_entropy_median_bits"] = float(np.median(ent[live2]))
    tag = f"_or_p{passes}_{passes2}_k{K_ON}" + ("_share" if share2 else "")
    with open(OUT / f"tower{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"tower{tag}.npz", W1=tw.L1.W, W2=W2, theta1=tw.L1.theta, theta2=tw.L2.theta,
             uses1=(keep["A1"] > 0).sum(0), uses2=(keep["A2"] > 0).sum(0), Xte=Xte[:8], yte=yte[:8],
             xhat=keep["xhat"][:8], P8=keep["P"][:8], A1_8=keep["A1"][:8], A2_8=keep["A2"][:8],
             gen=gen, a2g=a2g, a1g=a1g, top_pic=top_pic, top_a1=top_a1, pat=tw.pat)
    print(f"L2: members median {res['L2']['members_median']:.0f} (mean {res['L2']['members_mean']:.1f}), wrappers {res['L2']['wrappers']*100:.0f}%, "
          f"top-1 line share median {res['L2']['top1_share_median']:.2f}, label share median {res['L2']['label_share_median']:.2f}, "
          f"L1 cells a top cell proposes (settled) median {res['L2']['proposal_cells_median']:.0f}  |  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
