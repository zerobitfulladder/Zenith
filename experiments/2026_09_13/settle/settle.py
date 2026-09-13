"""Two layers that settle together. No beam, no enumeration.

Every unit starts with its match to the input. Units that explain the same thing
inhibit each other in proportion to their overlap (explaining away). A hard
threshold is the price: a unit stays on only if what is left of its drive exceeds
it. The layer above feeds its expectation down as extra drive to the units it
predicts. Both layers relax together for ITERS steps; the settled state is the
explanation; the units that are on learn.

    u1 <- (1-dt) u1 + dt ( W1 x  + beta * expected-from-above  - (G1 - I) a1 )     a1 = u1 if u1 > lam1 else 0
    u2 <- (1-dt) u2 + dt ( W2 z                                - (G2 - I) a2 )     a2 = u2 if u2 > lam2 else 0
    z  = [ unit(a1) ; label * LABEL_W ]         (label absent at read time)

This is the locally competitive network, and its fixed points are local minima of
  unexplained energy + price * (units on).

Layer 1 starts from this morning's stroke vocabulary and learns by the sparse-coding
rule (each unit moves toward the part of the residual its activity is responsible for).
Layer 2 groups, three arms:
    avg     a group learns the average of the codes it was on for (the rule that eroded)
    gated   the same, but each image counts in proportion to how well the group fitted
            it (its own fit squared), so a partial match barely writes
    carve   groups are cliques of a pair-count table over layer-1 identities and the
            label; never learned by averaging; a group has at least two members

    python settle.py <avg|gated|carve>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
OWN = HERE.parent / "ownership" / "results"

N_TRAIN, N_TEST, TALLY_N, BATCH = 8000, 2000, 4000, 128
H1, H2, D = 256, 128, 784
LABEL_W = 0.7
PRICE = 0.02
LAM1 = LAM2 = float(np.sqrt(2 * PRICE))      # threshold on the drive that corresponds to the price
ITERS, DT = 60, 0.2
BETA = 0.5 * LAM1                             # a fully expected unit gets half a threshold of extra drive
N_MAX, STALE = 200, 60
HIRE, HIRE_MAX, DUP, MIN_MEMBERS = 0.3, 8, 0.7, 2
WARMUP, MIN_SEED, TAU, MAX_CARVE = 15, 0.5, 0.7, 8
FREEZE_L1 = True      # layer 1 is the stroke vocabulary from ../ownership and stays put; the question is layer 2


def unit(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-8)


class Net:
    def __init__(self, arm, rng):
        self.arm = arm
        Z = np.load(OWN / "weights_soft_l0.02.npz")
        self.M1 = Z["W"].astype(np.float32).copy()
        self.n1 = np.ones(H1, np.float32)
        self.M2 = (rng.normal(size=(H2, H1 + 10)) * 0.01).astype(np.float32)
        self.n2 = np.zeros(H2, np.float32)
        self.last2 = np.full(H2, -10**9)
        self.hv = rng.random(H2)
        self.hired = 0
        self.recycled = 0
        self.dc = H1
        # pair counts over identities, for the carve arm
        self.co = np.zeros((H1 + 10, H1 + 10), np.float32)
        self.seen = np.zeros(H1 + 10, np.float32)
        self.T = 0
        self.mask = np.zeros((H1 + 10, H1 + 10), bool)
        self.refresh()

    def refresh(self):
        self.W1 = unit(self.M1)
        self.G1 = self.W1 @ self.W1.T
        np.fill_diagonal(self.G1, 0.0)
        self.W2 = unit(self.M2)
        self.W2p = np.maximum(self.W2, 0)
        self.W2r = np.zeros_like(self.W2)                    # scored on the identity part only (label absent)
        self.W2r[:, :self.dc] = unit(self.W2[:, :self.dc])
        self.G2 = self.W2 @ self.W2.T
        np.fill_diagonal(self.G2, 0.0)
        self.G2r = self.W2r @ self.W2r.T
        np.fill_diagonal(self.G2r, 0.0)

    # ---- settling ----------------------------------------------------------------
    def settle(self, X, y=None, beta=BETA):
        B = len(X)
        lab = np.zeros((B, 10), np.float32)
        if y is not None:
            lab[np.arange(B), y] = LABEL_W
        W2, G2 = (self.W2, self.G2) if y is not None else (self.W2r, self.G2r)
        live2 = self.n2 > 0
        b1 = X @ self.W1.T
        u1 = np.zeros((B, H1), np.float32); a1 = np.zeros((B, H1), np.float32)
        u2 = np.zeros((B, H2), np.float32); a2 = np.zeros((B, H2), np.float32)
        for it in range(ITERS):
            e = a2 @ self.W2p[:, :H1]
            td = beta * e / np.maximum(e.max(1, keepdims=True), 1e-8)
            u1 = (1 - DT) * u1 + DT * (b1 + td - a1 @ self.G1)
            a1_new = np.where(u1 > LAM1, u1, 0.0).astype(np.float32)
            z = np.concatenate([unit(a1_new), lab], 1)
            b2 = z @ W2.T
            b2[:, ~live2] = 0.0
            u2 = (1 - DT) * u2 + DT * (b2 - a2 @ G2)
            a2_new = np.where(u2 > LAM2, u2, 0.0).astype(np.float32)
            a2_new[:, ~live2] = 0.0
            delta = float(np.abs(a1_new - a1).mean() + np.abs(a2_new - a2).mean())
            a1, a2 = a1_new, a2_new
        return dict(a1=a1, a2=a2, z=z, xhat=a1 @ self.W1, zhat=a2 @ self.W2p, delta=delta)

    def label_top(self, st):
        v = st["a2"] @ self.W2[:, H1:]
        p = v.argmax(1)
        p[(st["a2"] > 0).sum(1) == 0] = -1
        return p

    # ---- learning ----------------------------------------------------------------
    def learn(self, st, X, step):
        a1, a2, z = st["a1"], st["a2"], st["z"]
        B = len(X)
        # layer 1: sparse-coding step, count-based rate, unit norm (off when frozen)
        if not FREEZE_L1:
            res = X - st["xhat"]
            m1 = (a1 > 0).sum(0).astype(np.float32)
            p = m1 > 0
            g = (a1.T @ res) / np.maximum(a1.sum(0)[:, None], 1e-8)
            self.n1[p] += m1[p]
            eta = np.minimum(1.0, m1[p] / np.minimum(self.n1[p], N_MAX))
            self.M1[p] += eta[:, None] * g[p]
        # layer 2
        on2 = a2 > 0
        if self.arm in ("avg", "gated"):
            w = a2.copy()
            if self.arm == "gated":
                fit = np.clip(z @ self.W2.T, 0, 1) ** 2                      # cosine squared: how well the group fitted
                w = w * fit
            cnt = w.sum(0)
            p = cnt > 0
            zbar = (w.T @ z) / np.maximum(cnt[:, None], 1e-8)
            obs = (w / np.maximum(a2.max(0, keepdims=True), 1e-8)).sum(0) if self.arm == "gated" else on2.sum(0).astype(np.float32)
            self.n2[p] += obs[p]
            self.last2[p] = step
            eta = np.minimum(1.0, obs[p] / np.minimum(self.n2[p], N_MAX))
            self.M2[p] += eta[:, None] * (zbar[p] - self.M2[p])
            # hire from what layer 2 left unexplained, never a single-member group
            free = np.flatnonzero((self.n2 == 0) | (step - self.last2 > STALE))
            if len(free):
                R = np.maximum(z - st["zhat"], 0)
                members = (R[:, :H1] > 0.15).sum(1)
                e = (R ** 2).sum(1)
                cand = np.argsort(-e)
                cand = [i for i in cand if e[i] > HIRE and members[i] >= MIN_MEMBERS]
                taken = []
                for i in cand:
                    if len(taken) >= min(len(free), HIRE_MAX):
                        break
                    if taken and (unit(R[i]) @ unit(np.stack(taken)).T).max() > DUP:
                        continue
                    taken.append(R[i])
                for k, r in enumerate(taken):
                    t = free[k]
                    self.M2[t], self.n2[t], self.last2[t] = r, 1, step
                    self.hired += 1
        else:                                                                # carve
            v = np.concatenate([(a1 > 0).astype(np.float32), z[:, H1:] > 0], 1).astype(np.float32)
            self.co += v.T @ v
            self.seen += v.sum(0)
            self.T += B
            m2 = on2.sum(0).astype(np.float32)
            self.n2[m2 > 0] += m2[m2 > 0]
            self.last2[m2 > 0] = step
            if step >= WARMUP:
                free = np.flatnonzero((self.n2 == 0) | (step - self.last2 > STALE))
                covered = (st["zhat"][:, :H1] > 0.1)
                unc = (a1 > 0) & ~covered                                     # identities no group accounts for
                order = np.argsort(-unc.sum(1))
                carved = 0
                for i in order:
                    if carved >= min(len(free), MAX_CARVE) or unc[i].sum() < MIN_MEMBERS:
                        break
                    lump = self.carve(np.flatnonzero(unc[i]))
                    if lump is None:
                        continue
                    lab_ids = H1 + np.flatnonzero(v[i, H1:] > 0)
                    if len(lab_ids):
                        Fl = self.friend(np.concatenate([lump, lab_ids]))[len(lump):, :len(lump)]     # label x members
                        if np.isfinite(Fl).any() and Fl.mean(1).max() > MIN_SEED:
                            lump = np.concatenate([lump, lab_ids[[int(Fl.mean(1).argmax())]]])
                    t = free[carved]
                    mem = np.zeros(H1 + 10, np.float32)
                    mem[lump] = 1.0
                    mem[H1:] *= LABEL_W
                    self.M2[t], self.n2[t], self.last2[t] = mem, 1, step
                    self.hired += 1
                    carved += 1
        self.refresh()

    def friend(self, ids):
        a, T = 0.5, max(self.T, 1.0)
        c = self.co[np.ix_(ids, ids)]
        n = self.seen[ids]
        F = np.log(((c + a) / T) / (((n + a) / T)[:, None] * ((n + a) / T)[None, :]))
        F[self.mask[np.ix_(ids, ids)]] = -np.inf
        np.fill_diagonal(F, -np.inf)
        return F

    def carve(self, ids):
        """A clique of identities that go together more than chance: seed on the friendliest
        unmasked pair, grow while candidates stay within TAU of the seed. Label dims may join."""
        if len(ids) < 2:
            return None
        F = self.friend(ids)
        if not np.isfinite(F).any():
            return None
        i, j = np.unravel_index(int(np.argmax(F)), F.shape)
        seed = F[i, j]
        if seed <= MIN_SEED:
            return None
        cur = [int(i), int(j)]
        while True:
            rest = [k for k in range(len(ids)) if k not in cur]
            if not rest:
                break
            m = np.array([F[k, cur].mean() for k in rest])
            b = int(np.argmax(m))
            if m[b] < TAU * seed:
                break
            cur.append(rest[b])
        lump = ids[np.array(cur)]
        if (lump < H1).sum() < MIN_MEMBERS:
            return None
        self.mask[np.ix_(lump, lump)] = True
        return lump

    

    # ---- use -------------------------------------------------------------------------
    def train(self, X, y, rng, log):
        order = rng.permutation(len(X))
        for step, b in enumerate(range(0, len(X), BATCH)):
            i = order[b:b + BATCH]
            st = self.settle(X[i], y[i])
            self.learn(st, X[i], step)
            if step % 10 == 0:
                log(step, st)

    def codes(self, X, beta):
        A1, A2, P, U, Dl = [], [], [], [], []
        for b in range(0, len(X), BATCH):
            st = self.settle(X[b:b + BATCH], None, beta)
            A1.append(st["a1"]); A2.append(st["a2"]); P.append(self.label_top(st))
            U.append(((X[b:b + BATCH] - st["xhat"]) ** 2).sum(1)); Dl.append(st["delta"])
        return np.concatenate(A1), np.concatenate(A2), np.concatenate(P), np.concatenate(U), float(np.mean(Dl))


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return unit(X[:N_TRAIN]), y[:N_TRAIN], unit(X[N_TRAIN:N_TRAIN + N_TEST]), y[N_TRAIN:N_TRAIN + N_TEST]


def tally(Ftr, ytr, Fte, yte, alpha=1.0):
    N = np.stack([Ftr[ytr == c].sum(0) for c in range(10)], 1).astype(np.float64)
    py = np.bincount(ytr, minlength=10) / len(ytr)
    L = np.log(((N + alpha) / (N.sum(1, keepdims=True) + 10 * alpha)) / py)
    return float(((Fte.astype(np.float64) @ L).argmax(1) == yte).mean())


def probe(Atr, ytr, Ate, yte):
    if Atr.std() == 0:
        return 0.1
    return float(LogisticRegression(max_iter=500).fit(Atr, ytr).score(Ate, yte))


def reuse(A):
    on = A > 0
    uses = on.sum(0); live = uses > 0
    p = uses[live] / max(uses[live].sum(), 1)
    return dict(live=int(live.sum()), on_per_image=float(on.sum(1).mean()), uses_mean=float(uses[live].mean()) if live.any() else 0.0,
                uses_entropy=float(-(p * np.log2(p)).sum()) if live.any() else 0.0,
                distinct_configs=len({frozenset(np.flatnonzero(r)) for r in on}))


def main():
    arm = sys.argv[1]
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(0)
    net = Net(arm, np.random.default_rng(0))

    def log(step, st):
        print(f"  {arm} step {step:3d}  L1 on {(st['a1'] > 0).sum(1).mean():.2f}  L2 on {(st['a2'] > 0).sum(1).mean():.2f}  "
              f"live2 {(net.n2 > 0).sum()}  hired {net.hired}  settle delta {st['delta']:.4f}  {time.time()-t0:.0f}s", flush=True)

    net.train(Xtr, ytr, np.random.default_rng(1), log)
    t_train = time.time() - t0
    A1tr, A2tr, _, _, _ = net.codes(Xtr[:TALLY_N], BETA)
    res = dict(arm=arm, t_train=t_train, modes={})
    for beta, name in ((BETA, "feedback"), (0.0, "no_feedback")):
        A1, A2, P, U, dl = net.codes(Xte, beta)
        cov = P >= 0
        m = dict(label_top=float((P == yte).mean()), label_top_when_covered=float((P[cov] == yte[cov]).mean()) if cov.any() else 0.0,
                 covered=float(cov.mean()), tally_L1=tally(A1tr > 0, ytr[:TALLY_N], A1 > 0, yte), tally_L2=tally(A2tr > 0, ytr[:TALLY_N], A2 > 0, yte),
                 probe_L1=probe(A1tr, ytr[:TALLY_N], A1, yte), probe_L2=probe(A2tr, ytr[:TALLY_N], A2, yte),
                 unexplained=float(U.mean()), settle_delta=dl, reuse_L1=reuse(A1), reuse_L2=reuse(A2))
        res["modes"][name] = m
        if name == "feedback":
            np.savez(OUT / f"codes_{arm}.npz", A1=A1, A2=A2, P=P, y=yte)
        print(f"{arm:6s} {name:12s}: label at top {m['label_top']:.3f} (covered {m['covered']*100:.0f}%)  tally L1 {m['tally_L1']:.3f} L2 {m['tally_L2']:.3f}  "
              f"probe L1 {m['probe_L1']:.3f} L2 {m['probe_L2']:.3f}  unexpl {m['unexplained']:.3f}  L1 on {m['reuse_L1']['on_per_image']:.1f} "
              f"L2 on {m['reuse_L2']['on_per_image']:.1f}  live2 {m['reuse_L2']['live']}  configs L1 {m['reuse_L1']['distinct_configs']} L2 {m['reuse_L2']['distinct_configs']}", flush=True)
    W2p = net.W2p; live2 = net.n2 > 0
    ident = W2p[:, :H1]
    members = (ident > 0.5 * ident.max(1, keepdims=True)).sum(1)
    share1 = ident.max(1) / np.maximum(np.linalg.norm(ident, axis=1), 1e-8)
    lab_share = np.linalg.norm(W2p[:, H1:], axis=1) / np.maximum(np.linalg.norm(W2p, axis=1), 1e-8)
    res.update(L2=dict(live=int(live2.sum()), members_median=float(np.median(members[live2])) if live2.any() else 0.0,
                       members_mean=float(members[live2].mean()) if live2.any() else 0.0,
                       wrappers=float((share1[live2] > 0.9).mean()) if live2.any() else 0.0,
                       label_share_median=float(np.median(lab_share[live2])) if live2.any() else 0.0, hired=net.hired))
    with open(OUT / f"{arm}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"weights_{arm}.npz", W1=net.W1, W2=net.W2, W2p=W2p, n2=net.n2, Xte=Xte[:8], yte=yte[:8])
    print(f"{arm}: L2 {res['L2']['live']} live, members median {res['L2']['members_median']:.0f} (mean {res['L2']['members_mean']:.1f}), "
          f"wrappers {res['L2']['wrappers']*100:.0f}%, label share {res['L2']['label_share_median']:.2f}, hired {net.hired} | train {t_train:.0f}s total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
