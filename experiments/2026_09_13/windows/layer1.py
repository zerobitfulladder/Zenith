"""Layer 1 in terms of neurons.

cells      a grid of windows on the image (WIN x WIN, stride STRIDE), K cells per window.
           A cell sees only its window. No weight sharing. Random start.
matching   Pearson: patch and template are mean-centred and unit length inside the
           window, so a cell's drive is a correlation in [-1, 1].
settling   u <- (1-dt) u + dt ( c - (G - I) a ),  a = u where u > theta (per cell).
           G is the overlap between cells' templates in image space: zero for cells
           whose windows do not overlap, their correlation inside a shared window.
learning   every cell that is on after settling rotates toward the patch it saw, by
           the angle eta * (its correlation), on the unit sphere. Nothing is picked.
threshold  each cell's theta drifts to hold its firing rate near RHO; silent cells
           come in by their threshold falling. No adoption.

    python layer1.py          one pass over 8k, held out: unexplained, cells on, dead, pictures
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
WIN, STRIDE, K = 12, 4, 12
POS = [(r, c) for r in range(0, 28 - WIN + 1, STRIDE) for c in range(0, 28 - WIN + 1, STRIDE)]
NW = len(POS)
H = NW * K
ITERS, DT = 40, 0.25
ETA, INK = 0.05, 0.15                        # rotation rate; minimum patch std to count as inked
RHO, THETA0, THETA_MIN, GAMMA, KAPPA = 0.04, 0.1, 0.02, 0.1, 0.1
PASSES = 1


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def patches(X):
    return np.stack([X[:, r:r + WIN, c:c + WIN].reshape(len(X), -1) for r, c in POS], 1)


def pearson(P):
    m = P.mean(-1, keepdims=True)
    Q = P - m
    s = np.linalg.norm(Q, axis=-1, keepdims=True)
    return Q / np.maximum(s, 1e-8), s[..., 0], (s[..., 0] > INK)


def centre_unit(V):
    V = V - V.mean(-1, keepdims=True)
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-8)


class Layer1:
    def __init__(self, rng):
        self.W = centre_unit(rng.normal(size=(NW, K, WIN * WIN)).astype(np.float32))
        self.theta = np.full((NW, K), THETA0, np.float32)
        self.rate = np.full((NW, K), RHO, np.float32)
        self.refresh()

    def full(self):
        F = np.zeros((NW, K, 28, 28), np.float32)
        for w, (r, c) in enumerate(POS):
            F[w, :, r:r + WIN, c:c + WIN] = self.W[w].reshape(K, WIN, WIN)
        return F.reshape(H, 784)

    def refresh(self):
        F = self.full()
        self.G = F @ F.T
        np.fill_diagonal(self.G, 0.0)

    def settle(self, X):
        Q, s, inked = pearson(patches(X))
        c = np.einsum("bwd,wkd->bwk", Q, self.W) * inked[..., None]
        b = c.reshape(len(X), H)
        u = np.zeros_like(b); a = np.zeros_like(b)
        th = self.theta.reshape(H)[None, :]
        for _ in range(ITERS):
            u = (1 - DT) * u + DT * (b - a @ self.G)
            a = np.where(u > th, u, 0.0).astype(np.float32)
        return dict(a=a.reshape(len(X), NW, K), c=c, Q=Q, s=s, inked=inked)

    def learn(self, st):
        a, c, Q, inked = st["a"], st["c"], st["Q"], st["inked"]
        on = (a > 0) & inked[..., None]
        for w in range(NW):
            for t in range(K):
                sel = np.flatnonzero(on[:, w, t])
                if len(sel) == 0:
                    continue
                cc = c[sel, w, t]
                keep = cc > 0
                if not keep.any():
                    continue
                q = centre_unit(Q[sel[keep], w].mean(0))               # what the cell saw when on, averaged over the batch
                wv = self.W[w, t]
                cq = float(np.clip(wv @ q, -1, 1))
                if cq <= 0 or cq >= 1:
                    continue
                th = min(ETA * float(cc[keep].mean()) * keep.sum(), np.arccos(cq))    # angle: eta x correlation, per image; never past the target
                tq = np.sqrt(1 - cq * cq)
                self.W[w, t] = centre_unit((np.cos(th) - cq * np.sin(th) / tq) * wv + (np.sin(th) / tq) * q)
        self.rate = (1 - GAMMA) * self.rate + GAMMA * on.mean(0)
        self.theta = np.clip(self.theta * np.exp(KAPPA * (self.rate - RHO) / RHO), THETA_MIN, 1.0)
        self.refresh()

    def rebuild(self, st):
        a, s = st["a"], st["s"]
        B = len(a)
        img = np.zeros((B, 28, 28), np.float32); cnt = np.zeros((28, 28), np.float32)
        for w, (r, c) in enumerate(POS):
            rec = (a[:, w] @ self.W[w]) * s[:, w, None]
            img[:, r:r + WIN, c:c + WIN] += rec.reshape(B, WIN, WIN)
            cnt[r:r + WIN, c:c + WIN] += 1
        return img / cnt


def main():
    global RHO, PASSES
    if len(sys.argv) > 1:
        RHO = float(sys.argv[1])
    if len(sys.argv) > 2:
        PASSES = int(sys.argv[2])
    tag = f"r{RHO:g}"
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(0)
    L = Layer1(np.random.default_rng(0))
    L.rate[:] = RHO
    rng = np.random.default_rng(1)
    step = -1
    for ps in range(PASSES):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            step += 1
            xb = Xtr[order[b:b + BATCH]]
            st = L.settle(xb)
            L.learn(st)
            if step % 20 != 0:
                continue
            on = st["a"] > 0
            print(f"  step {step:3d}  cells on/img {on.sum((1, 2)).mean():.1f}  inked windows/img {st['inked'].sum(1).mean():.1f}  "
                  f"best corr in inked windows {st['c'].max(2)[st['inked']].mean():.3f}  theta {L.theta.mean():.3f}  "
                  f"cells firing this batch {on.any(0).sum()}/{H}  {time.time()-t0:.0f}s", flush=True)
    A, C, INKD, REC = [], [], [], []
    for b in range(0, len(Xte), BATCH):
        st = L.settle(Xte[b:b + BATCH])
        A.append(st["a"]); C.append(st["c"]); INKD.append(st["inked"]); REC.append(L.rebuild(st))
    A, C, INKD, REC = map(np.concatenate, (A, C, INKD, REC))
    on = A > 0
    best = np.where(on, C, -np.inf).max(2)
    unexpl_win = 1 - np.where(np.isfinite(best), best, 0.0) ** 2
    per_win_on = on.sum(2)
    live = on.reshape(len(Xte), H).any(0)
    pix_unexpl = ((Xte - REC) ** 2).sum((1, 2)) / (Xte ** 2).sum((1, 2))
    res = dict(cells=H, windows=NW, cells_per_window=K, win=WIN, stride=STRIDE,
               cells_on_per_image=float(on.sum((1, 2)).mean()), inked_windows_per_image=float(INKD.sum(1).mean()),
               cells_on_per_inked_window=float(per_win_on[INKD].mean()),
               windows_with_two_or_more_on=float((per_win_on[INKD] >= 2).mean()),
               unexplained_patch_variance=float(unexpl_win[INKD].mean()),
               inked_windows_left_silent=float((per_win_on[INKD] == 0).mean()),
               pixel_unexplained=float(pix_unexpl.mean()),
               dead_cells=float(1 - live.mean()), theta_mean=float(L.theta.mean()),
               distinct_configs=len({frozenset(np.flatnonzero(r)) for r in on.reshape(len(Xte), H)}), t=time.time() - t0)
    res["rho"], res["passes"] = RHO, PASSES
    with open(OUT / f"layer1_{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"layer1_{tag}.npz", W=L.W, full=L.full(), theta=L.theta, uses=on.reshape(len(Xte), H).sum(0),
             Xte=Xte[:8], rec=REC[:8], A8=A[:8])
    print(f"held out: cells on/img {res['cells_on_per_image']:.1f} over {res['inked_windows_per_image']:.1f} inked windows "
          f"({res['cells_on_per_inked_window']:.2f} per window, {res['windows_with_two_or_more_on']*100:.0f}% with 2+, "
          f"{res['inked_windows_left_silent']*100:.0f}% silent)  unexplained patch variance {res['unexplained_patch_variance']:.3f}  "
          f"pixel unexplained {res['pixel_unexplained']:.3f}  dead cells {res['dead_cells']*100:.0f}%  "
          f"configs {res['distinct_configs']}  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
