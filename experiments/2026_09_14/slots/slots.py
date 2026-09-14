"""Two layers, no labels. L1 = whole-image cells, share-then-scale (09-13 neurons/share.py, unchanged),
fed 4-level pixels. L2 = a discrete belief network over the L1 code, learned by counting.

The picture (09-14 discussion):
  * every node has 4 levels (0 = silent); pixels are discretized to 4 levels; no pixel table.
  * L2 nodes are the PARENTS of L1 nodes. Each L1 node has K=4 parent SLOTS.
  * a slot has a learned NAME: a histogram over the L2 nodes of who filled it. The slot's name is
    the histogram's argmax (the four names of a child are distinct).
  * each L1 node keeps one FULL table of counts over the levels of its 4 slots (256 rows) x its own
    4 levels. Rows mean something only once the names have converged; when a name changes, that
    slot's axis of the table is collapsed to its mean (the other three slots keep their counts).
  * L2 root priors: one 4-vector of counts per node. The price per active unit is this prior.
  * inference = settling: mean-field beliefs over L2 levels, each L2 node scored by its prior plus
    the expected log-probability of every child that names it, given the current beliefs of the
    child's other slots. Explaining away comes through the child's table and the prior.
  * learning = counting the settled state: prior counts, table rows, and, for children that were
    on, which on-L2 nodes filled which slot (existing names first, then free slots by affinity).
  * L2 pictures and generations: expected L1 levels through the tables, L1 settled under that
    expectation alone (no pixels), painted. Never a weighted sum.
No labels, no tally, no probe.

    python slots.py [passes_L2] [J]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
sys.path.insert(0, str(ROOT / "experiments/2026_09_13/neurons"))
import share                                                   # noqa: E402  (L1, unchanged)

I, J, K, S = share.H, 100, 4, 4                               # L1 nodes, L2 nodes, slots per child, levels
BATCH = 128
MF_ITERS, MF_DAMP = 12, 0.5
RHO1, PASSES1 = 0.035, 2
PRIOR0 = np.array([2.0, 0.4, 0.3, 0.3], np.float32)           # root prior starts loose (P(on) 0.33) so the loop can close; it is learned from counts
PRIOR_ON_FLOOR = 0.03                                         # a node's prior P(on) may not fall below this (no death spiral)
CPT_ALPHA, LEAK = 2.0, 0.03                                    # pseudo-counts behind the noisy-OR start of each table
LOG_FLOOR = 1e-4


def quantize(X):
    """Pixels to 4 levels: 0, 1/3, 2/3, 1."""
    return (np.minimum((X * 4).astype(np.int32), 3) / 3.0).astype(np.float32)


def train_l1(Xtr, rng):
    ink = (Xtr > 0.1).mean(0) + 1e-3
    L = share.Layer(RHO1, np.random.default_rng(0), ink)
    for _ in range(PASSES1):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            xb = Xtr[order[b:b + BATCH]]
            L.learn(L.settle(xb), xb)
    return L


def settle_all(L, X):
    return np.concatenate([L.settle(X[b:b + BATCH]) for b in range(0, len(X), BATCH)])


class Slots:
    def __init__(self, rng):
        self.rng = rng
        self.prior = np.tile(PRIOR0, (J, 1)).astype(np.float32)                  # (J,S) counts
        self.N = (rng.random((I, K, J)) * 0.05).astype(np.float32)               # slot-name histograms
        self.names = self.pick_names()
        # each child's table starts as a noisy-OR of its (random) parents, weak random strengths
        e = rng.uniform(0.15, 0.45, (I, K)).astype(np.float32)
        lv = np.arange(S, dtype=np.float32) / (S - 1)
        keep = np.ones((I, S, S, S, S), np.float32) * (1 - LEAK)
        for k in range(K):
            shp = [1] * 5; shp[k + 1] = S
            keep = keep * (1 - e[:, k][:, None, None, None, None] * lv.reshape(shp))
        p_on = 1 - keep
        table = np.empty((I, S, S, S, S, S), np.float32)
        table[..., 0] = 1 - p_on
        for s in range(1, S):
            table[..., s] = p_on / (S - 1)
        self.C = CPT_ALPHA * table                                               # (I,S,S,S,S,S) counts
        self.refresh()

    def pick_names(self):
        """Argmax of each slot's histogram, the four names of a child distinct."""
        names = np.zeros((I, K), np.int64)
        for i in range(I):
            taken = []
            for k in range(K):
                h = self.N[i, k].copy(); h[taken] = -1
                names[i, k] = int(h.argmax()); taken.append(names[i, k])
        return names

    def refresh(self):
        self.logP = np.log(np.maximum(self.C / self.C.sum(-1, keepdims=True), LOG_FLOOR)).astype(np.float32)
        pr = self.prior / self.prior.sum(1, keepdims=True)
        on = pr[:, 1:].sum(1)
        low = on < PRIOR_ON_FLOOR
        pr[low, 1:] *= (PRIOR_ON_FLOOR / np.maximum(on[low], 1e-8))[:, None]
        pr[low, 0] = 1 - PRIOR_ON_FLOOR
        self.pr = pr
        self.logprior = np.log(np.maximum(pr, 1e-6)).astype(np.float32)
        self.M = np.stack([np.eye(J, dtype=np.float32)[self.names[:, k]] for k in range(K)])   # (K,I,J) one-hot names

    def settle(self, s):
        """Mean-field beliefs over the L2 levels given the observed L1 levels s (B,I). Returns beliefs (B,J,S)."""
        B = len(s)
        LP = self.logP[np.arange(I)[None, :], :, :, :, :, s]                     # (B,I,S,S,S,S): log P(s_i | slot levels)
        b = np.tile(np.exp(self.logprior), (B, 1, 1))
        b /= b.sum(-1, keepdims=True)
        for _ in range(MF_ITERS):
            q = b[:, self.names, :]                                              # (B,I,K,S) beliefs of each child's slots
            q0, q1, q2, q3 = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]
            msg = (np.einsum('niacde,nic,nid,nie->nia', LP, q1, q2, q3, optimize=True),
                   np.einsum('niacde,nia,nid,nie->nic', LP, q0, q2, q3, optimize=True),
                   np.einsum('niacde,nia,nic,nie->nid', LP, q0, q1, q3, optimize=True),
                   np.einsum('niacde,nia,nic,nid->nie', LP, q0, q1, q2, optimize=True))
            score = np.tile(self.logprior, (B, 1, 1))
            for k in range(K):
                score += np.einsum('nit,ij->njt', msg[k], self.M[k], optimize=True)
            score -= score.max(-1, keepdims=True)
            bn = np.exp(score); bn /= bn.sum(-1, keepdims=True)
            b = (1 - MF_DAMP) * b + MF_DAMP * bn
        return b

    def rows(self, T):
        """Each child's table row index from the L2 MAP levels T (B,J) -> (B,I) in 0..255."""
        t = T[:, self.names]                                                     # (B,I,K)
        return ((t[..., 0] * S + t[..., 1]) * S + t[..., 2]) * S + t[..., 3]

    def learn(self, s, b):
        B = len(s)
        T = b.argmax(-1)                                                         # (B,J) MAP levels
        np.add.at(self.prior, (np.arange(J)[None, :].repeat(B, 0), T), 1.0)
        Cf = self.C.reshape(I, S ** K, S)
        np.add.at(Cf, (np.arange(I)[None, :].repeat(B, 0), self.rows(T), s), 1.0)
        # names: for each child on, the on-L2 nodes fill its slots; existing names first, then free slots by affinity
        for n in range(B):
            on2 = np.flatnonzero(T[n] > 0)
            if len(on2) == 0:
                continue
            for i in np.flatnonzero(s[n] > 0):
                nm = self.names[i]
                filled = np.isin(nm, on2)
                self.N[i, np.flatnonzero(filled), nm[filled]] += 1.0
                free = np.flatnonzero(~filled)
                rest = on2[~np.isin(on2, nm)]
                if len(free) == 0 or len(rest) == 0:
                    continue
                rest = rest[np.argsort(-b[n, rest, 1:].sum(-1))][:len(free)]
                for j in rest:
                    k = free[np.argmax(self.N[i, free, j])]
                    self.N[i, k, j] += 1.0
                    free = free[free != k]
        new = self.pick_names()
        changed = np.argwhere(new != self.names)
        for i, k in changed:                                                     # a renamed slot: collapse its axis of the table
            ax = k + 1
            self.C[i] = np.broadcast_to(self.C[i].mean(axis=k, keepdims=True), self.C[i].shape)
        self.names = new
        self.refresh()
        return len(changed)

    # ---- reading the top ------------------------------------------------------------------------------
    def expected_levels(self, T):
        """Expected L1 level (0..1) of every child under L2 levels T (B,J), through the tables."""
        Cf = self.C.reshape(I, S ** K, S)
        P = Cf / Cf.sum(-1, keepdims=True)
        r = self.rows(T)                                                         # (B,I)
        p = P[np.arange(I)[None, :], r]                                          # (B,I,S)
        return (p * np.arange(S)[None, None, :]).sum(-1) / (S - 1)

    def loglik(self, s, b):
        """Log-probability of the L1 code under the MAP L2 state, per image: sum of table terms + prior of the state."""
        T = b.argmax(-1)
        Cf = self.C.reshape(I, S ** K, S)
        P = np.maximum(Cf / Cf.sum(-1, keepdims=True), LOG_FLOOR)
        r = self.rows(T)
        tab = np.log(P[np.arange(I)[None, :], r, s]).sum(1)
        pri = self.logprior[np.arange(J)[None, :], T].sum(1)
        return tab, pri, T

    def members(self, thresh=0.3):
        """For each L2 node: the children it raises by more than `thresh` (expected level) when it alone is at level 3."""
        T0 = np.zeros((1, J), np.int64)
        base = self.expected_levels(T0)[0]
        mem = []
        for j in range(J):
            T = T0.copy(); T[0, j] = S - 1
            gain = self.expected_levels(T)[0] - base
            mem.append(np.flatnonzero(gain > thresh))
        return mem, base


def settle_down(L, e, gain=1.5):
    """L1 settled under an expectation alone (no pixels): the most expected cell is driven to `gain` times its
    threshold, inhibition decides the rest. (09-13 share_tower convention.)"""
    B = len(e)
    td = gain * L.theta[None, :] * e / np.maximum(e.max(1, keepdims=True), 1e-8)
    u = np.zeros((B, I), np.float32); a = np.zeros((B, I), np.float32)
    for _ in range(share.ITERS):
        u = (1 - share.DT) * u + share.DT * (td - a @ L.G)
        a = np.where(u > L.theta[None, :], u, 0.0).astype(np.float32)
    return a


def main():
    passes2 = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    global J
    if len(sys.argv) > 2:
        J = int(sys.argv[2])
    tag = f"slots_j{J}_p{passes2}"
    t0 = time.time()
    Xtr, Xte = share.load(0)
    Xtr, Xte = quantize(Xtr), quantize(Xte)
    rng = np.random.default_rng(1)
    L = train_l1(Xtr, rng)
    Atr, Ate = settle_all(L, Xtr), settle_all(L, Xte)
    on = Atr[Atr > 0]
    cuts = np.quantile(on, [1 / 3, 2 / 3]).astype(np.float32)                    # L1 activity -> levels 1..3 by terciles
    lev = lambda A: np.where(A > 0, 1 + (A > cuts[0]) + (A > cuts[1]), 0).astype(np.int64)
    Str, Ste = lev(Atr), lev(Ate)
    print(f"L1 trained: on/img {(Atr > 0).sum(1).mean():.2f}, level cuts {cuts[0]:.3f} {cuts[1]:.3f}, {time.time()-t0:.0f}s", flush=True)

    top = Slots(rng)
    names_p1 = None
    step = -1
    log = []
    for p in range(passes2):
        order = rng.permutation(len(Str))
        for bi in range(0, len(Str), BATCH):
            step += 1
            s = Str[order[bi:bi + BATCH]]
            b = top.settle(s)
            nchg = top.learn(s, b)
            if step % 10 == 0:
                T = b.argmax(-1)
                pur = (top.N.max(-1) / np.maximum(top.N.sum(-1), 1e-8)).mean()
                rec = dict(step=step, on2=float((T > 0).mean(0).sum()), live2=int((T > 0).any(0).sum()), renamed=int(nchg), purity=float(pur))
                log.append(rec)
                print(f"  step {step:3d}  L2 on/img {rec['on2']:.2f}  L2 nodes used in batch {rec['live2']}/{J}  "
                      f"renamed slots {nchg}  name purity {pur:.2f}  {time.time()-t0:.0f}s", flush=True)
        if p == 0:
            names_p1 = top.names.copy()

    # ---- held out --------------------------------------------------------------------------------------
    Bte = np.concatenate([top.settle(Ste[b:b + BATCH]) for b in range(0, len(Ste), BATCH)])
    tab, pri, T = top.loglik(Ste, Bte)
    T0 = np.zeros_like(T)
    tab0, pri0, _ = top.loglik(Ste, np.eye(S, dtype=np.float32)[T0])
    marg = np.eye(S)[Str].mean(0) + 1e-3; marg /= marg.sum(-1, keepdims=True)  # the no-L2 baseline: each child independent
    indep = np.log(marg[np.arange(I)[None, :], Ste]).sum(1)
    on2 = T > 0
    uses2 = on2.sum(0)
    mem, base = top.members()
    nmem = np.array([len(m) for m in mem])
    named_by = np.bincount(top.names.ravel(), minlength=J)
    purity = top.N.max(-1) / np.maximum(top.N.sum(-1), 1e-8)
    stable = float((names_p1 == top.names).mean()) if names_p1 is not None else float('nan')
    # top-down rebuild of 8 held-out images, L2 pictures, generations from the prior
    e_rb = top.expected_levels(T[:8]); a_rb = settle_down(L, e_rb)
    Tj = np.zeros((J, J), np.int64); Tj[np.arange(J), np.arange(J)] = S - 1
    e_j = top.expected_levels(Tj); a_j = settle_down(L, e_j)
    pr = top.pr
    Tg = np.stack([[rng.choice(S, p=pr[j]) for j in range(J)] for _ in range(8)])
    e_g = top.expected_levels(Tg); a_g = settle_down(L, e_g)
    # how much evidence ONE child at level 3 gives its named parent alone, against the prior's cost of turning on
    rows0 = np.zeros((I, K), np.int64)
    ev = np.zeros((I, K), np.float32)
    for k in range(K):
        rk = rows0.copy(); rk[:, k] = S - 1
        r1 = ((rk[:, 0] * S + rk[:, 1]) * S + rk[:, 2]) * S + rk[:, 3]
        Pf = np.maximum(top.C.reshape(I, S ** K, S) / top.C.reshape(I, S ** K, S).sum(-1, keepdims=True), LOG_FLOOR)
        ev[:, k] = np.log(Pf[np.arange(I), r1, S - 1]) - np.log(Pf[np.arange(I), 0, S - 1])
    prior_cost = float(np.log(pr[:, 0] / np.maximum(1 - pr[:, 0], 1e-8)).mean())
    res = dict(tag=tag, J=J, passes2=passes2, single_child_evidence_median=float(np.median(ev)), single_child_evidence_mean=float(ev.mean()),
               single_child_clears_prior=float((ev > prior_cost).mean()), prior_cost_nats=prior_cost, l1_on_per_image=float((Ate > 0).sum(1).mean()),
               l2_on_per_image=float(on2.sum(1).mean()), l2_dead=float((uses2 == 0).mean()), l2_silent_images=float((on2.sum(1) == 0).mean()),
               loglik_table=float(tab.mean()), loglik_prior=float(pri.mean()), loglik_alloff_table=float(tab0.mean()),
               loglik_indep=float(indep.mean()),
               members_mean=float(nmem.mean()), members_median=float(np.median(nmem)), members_le1=float((nmem <= 1).mean()),
               members_ge3=float((nmem >= 3).mean()), named_by_mean=float(named_by.mean()),
               name_purity_mean=float(purity.mean()), slots_pure=float((purity > 0.5).mean()), names_stable_p1_to_end=stable,
               prior_on_mean=float(1 - pr[:, 0].mean()), t=time.time() - t0, log=log)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"{tag}.npz", W=L.W, theta=L.theta, uses1=(Ate > 0).sum(0), uses2=uses2, names=top.names, nmem=nmem, named_by=named_by,
             Xte=Xte[:8], A8=Ate[:8], xhat_l1=(Ate[:8] @ L.Wd), a_rb=a_rb, T8=T[:8], a_j=a_j, e_j=e_j, a_g=a_g, Tg=Tg, prior=pr,
             members=np.array([np.pad(m, (0, I - len(m)), constant_values=-1) for m in mem]), C=top.C, N=top.N, ev=ev)
    print(f"{tag}: held out  L2 on/img {res['l2_on_per_image']:.2f}  dead {res['l2_dead']*100:.0f}%  silent imgs {res['l2_silent_images']*100:.0f}%\n"
          f"  log-prob of the L1 code per image: tables under MAP L2 {res['loglik_table']:.1f} (+ prior {res['loglik_prior']:.1f})  "
          f"| all L2 off {res['loglik_alloff_table']:.1f}  | children independent {res['loglik_indep']:.1f}\n"
          f"  members per L2 node: mean {res['members_mean']:.1f} median {res['members_median']:.0f}  <=1: {res['members_le1']*100:.0f}%  >=3: {res['members_ge3']*100:.0f}%  "
          f"named by {res['named_by_mean']:.1f} children\n"
          f"  slot names: purity {res['name_purity_mean']:.2f}  pure(>0.5) {res['slots_pure']*100:.0f}%  unchanged since pass 1 {stable*100:.0f}%  "
          f"prior P(on) {res['prior_on_mean']:.3f}\n"
          f"  one child at level 3 alone gives its named parent {res['single_child_evidence_median']:.1f} nats (median), the prior costs {prior_cost:.1f} to turn on: "
          f"a single child clears it for {res['single_child_clears_prior']*100:.0f}% of slots  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
