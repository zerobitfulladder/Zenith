"""Sleep instead of punishment.

Repulsion put the correction where the incoming data pointed, which on a stream
means always at the old specialists. The negative phase of a Forward-Forward
network does not depend on incoming data at all: the model generates its own
negative examples. Here that is literal -- each hypercolumn dreams from its own
coefficient statistics, and the population is corrected on the dreams.

The dreamer always wins the gate on its own dream (the dream lives in its
subspace), so "who wins" cannot be the criterion. The criterion is a margin:

    for every dream, the ONE hypercolumn of another class that explains it
    best is pushed away from it.

Two criteria were tried and thrown out. A margin against the dreamer catches
nobody: the dream lives in the dreamer's subspace, so its own error is near
zero. A margin against each hypercolumn's habitual error on real images
catches everybody: dreams are low-rank and smooth, so every hypercolumn beats
its habitual error on them. Ranking among the non-owners is invariant to both
-- it asks only who is the nearest competitor, which is exactly the pair that
will be confused on real data.

Everything here is data-free and teacher-free. The statistics are accumulated
online during wake, so nothing is stored.
"""

import numpy as np
from common import center_norm, EPS, geo_step
from dopamine import geo_step_neg

N_IMG = 784


class Coef:
    """Running mean and covariance of each hypercolumn's coefficients."""

    def __init__(self, H, K, decay=0.995):
        self.n = np.zeros(H)
        self.s1 = np.zeros((H, K))
        self.s2 = np.zeros((H, K, K))
        self.en = np.zeros(H)          # how well it usually explains its own
        self.es = np.zeros(H)
        self.decay = decay

    def seen(self, h, e):
        self.en[h] = self.decay * self.en[h] + len(e)
        self.es[h] = self.decay * self.es[h] + float(e.sum())

    def typical(self, h):
        return self.es[h] / self.en[h] if self.en[h] > 0 else np.inf

    def update(self, h, S):
        self.n[h] = self.decay * self.n[h] + len(S)
        self.s1[h] = self.decay * self.s1[h] + S.sum(0)
        self.s2[h] = self.decay * self.s2[h] + S.T @ S

    def ready(self, h, floor=50):
        return self.n[h] > floor

    def sample(self, h, m, temp, rng):
        mu = self.s1[h] / self.n[h]
        cov = self.s2[h] / self.n[h] - np.outer(mu, mu)
        K = len(mu)
        w, V = np.linalg.eigh(cov + 1e-8 * np.eye(K))
        A = V * np.sqrt(np.maximum(w, 0.0))
        return mu[None] + temp * (rng.standard_normal((m, K)) @ A.T)


def errs(W, Q, alive):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return np.where(alive[None], e, np.inf)


def sleep(W, wins, coef, rng, n_dream=48, temp=0.8, slack=0.0,
          eta_neg=0.1, cap=np.pi / 32, eta_pos=0.0, min_s=2):
    """One night. Returns how many corrections were applied."""
    H, K, D = W.shape
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    hs = [h for h in range(H) if alive[h] and coef.ready(h)]
    if len(hs) < 2:
        return 0, 0
    Q, owner = [], []
    for h in hs:
        V = coef.sample(h, n_dream, temp, rng) @ W[h]
        B = np.zeros((n_dream, D)); B[:, :N_IMG] = V[:, :N_IMG]
        Q.append(center_norm(B)); owner += [h] * n_dream
    Q = np.concatenate(Q); owner = np.asarray(owner)
    E = errs(W, Q, alive)
    other = (claim[None, :] != claim[owner][:, None]) & alive[None]
    Eo = np.where(other, E, np.inf)
    rival = Eo.argmin(1)
    ok = np.isfinite(Eo[np.arange(len(Q)), rival])
    close = np.zeros_like(other)
    close[np.arange(len(Q))[ok], rival[ok]] = True
    n_fix = 0
    for t in np.where(close.any(0))[0]:
        m = close[:, t]
        if m.sum() >= min_s:
            W[t] = geo_step_neg(W[t], Q[m], eta_neg, cap)
            n_fix += int(m.sum())
    if eta_pos > 0:                      # optional rehearsal of one's own dreams
        for h in hs:
            m = owner == h
            if m.sum() >= min_s:
                W[h] = geo_step(W[h], Q[m], eta_pos)
    return n_fix, len(Q)
