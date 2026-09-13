"""Two layers, one search over the tower, the label as an exact stream at the top.

layer 1   784 pixels -> 256 templates. Starts from the stroke vocabulary learned this
          morning (../ownership, soft rule) and keeps learning with the same rule.
          Each pixel is owned by the sharpest active template. Message up = WHICH
          templates are on (identity only).
layer 2   [256 identities ; label x LABEL_W] -> 128 templates, each a group of
          layer-1 identities plus the label they go with. Same search, same
          ownership, residual learning. At read time the label is absent: templates
          are scored on the identity part only, and the label is read from the
          label part of the templates that are on.
tower     read up (1 then 2), expand the top's configuration down into an expected
          layer-1 pattern, make the expected units cheap, read up again. The top's
          opinion biases selection; the pixels decide content.
prices    names cost -log2 of their usage, Huffman shape, mean held at LAM.
          A template that stops paying for itself (decayed energy explained minus
          price goes negative) or is unused for STALE batches is recycled from the
          largest leftover.

python tower.py            trains one pass, scores held out with feedback on and off,
                           saves results/tower.json and results/weights.npz
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

S, BEAM, ROUNDS, PRE, CHILD, NSWAP_DROP, NSWAP_ADD, CHUNK = 20, 4, 30, 16, 4, 2, 6, 8
HIRE, HIRE_MAX, DUP, N_MAX, STALE = 0.3, 8, 0.7, 200, 60
LAM = 0.02
N_TRAIN, N_TEST, TALLY_N, BATCH = 8000, 2000, 4000, 128
H1, H2 = 256, 128
LABEL_W = 0.7          # label stream weight in the top's input: energy 0.49 against identity energy 1
EXPECT = 0.75          # a layer-1 unit the top expects costs (1 - EXPECT) of its price
DECAY = 0.98
MIN_USES = 20          # a template is judged on its value only after this many uses


def unit(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-8)


class Layer:
    def __init__(self, d, h, rule, rng, d_code=None):
        self.d, self.h, self.rule = d, h, rule
        self.dc = d if d_code is None else d_code
        self.M = (rng.normal(size=(h, d)) / np.sqrt(d)).astype(np.float32)
        self.W = unit(self.M)
        self.Wp = np.maximum(self.W, 0)
        self.n = np.zeros(h, np.float32)
        self.last = np.full(h, -10**9)
        self.hv = np.random.default_rng(12345).random(h)
        self.hired = 0
        self.cap_hits = 0
        self.recycled = 0
        self.uses = np.ones(h, np.float32)
        self.images = float(h)
        self.value = np.zeros(h, np.float32)
        self.price = np.full(h, LAM, np.float32)
        self.Pb = None                       # per-image prices for the current batch (B,h), or None
        self.ar_S = np.arange(S)

    # ---- prices -------------------------------------------------------------
    def _pr(self, ids, rows):
        if self.Pb is None:
            return self.price[np.maximum(ids, 0)]
        return self.Pb[rows, np.maximum(ids, 0)]

    def reprice(self):
        p = np.clip(self.uses / self.images, 1e-4, 1.0)
        bits = np.log(1.0 / p)
        live = self.n > 0
        ref = bits[live].mean() if live.any() else bits.mean()
        self.price = (LAM * bits / ref).astype(np.float32)

    def partial(self):
        """Templates scored on the identity part only (the label is absent at read time)."""
        Wr = np.zeros_like(self.Wp)
        Wr[:, :self.dc] = unit(self.Wp[:, :self.dc])
        return Wr

    # ---- exact scoring of configurations ---------------------------------------
    def exact(self, idx, X, rows, Wp):
        B, n, _ = idx.shape
        d = self.d
        owner = np.full((B, n, d), -1, np.int64)
        coef = np.zeros((B, n, S), np.float32)
        xhat = np.zeros((B, n, d), np.float32)
        for j in range(0, n, CHUNK):
            ii = idx[:, j:j + CHUNK]
            c = ii.shape[1]
            act = ii >= 0
            claims = Wp[np.maximum(ii, 0)] * act[..., None]
            ow = claims.argmax(2)
            top = np.take_along_axis(claims, ow[:, :, None, :], 2)[:, :, 0, :]
            ow = np.where(top > 0, ow, S)
            flat = (ow + (np.arange(B * c) * (S + 1)).reshape(B, c, 1)).ravel()
            Xb = np.broadcast_to(X[:, None, :], (B, c, d))
            num = np.bincount(flat, weights=(Xb * top).ravel(), minlength=B * c * (S + 1)).reshape(B, c, S + 1)[:, :, :S]
            den = np.bincount(flat, weights=(top * top).ravel(), minlength=B * c * (S + 1)).reshape(B, c, S + 1)[:, :, :S]
            cf = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
            cf = np.maximum(cf, 0).astype(np.float32)
            cfp = np.concatenate([cf, np.zeros((B, c, 1), np.float32)], 2)
            owner[:, j:j + c] = np.where(ow == S, -1, ow)
            coef[:, j:j + c] = cf
            xhat[:, j:j + c] = np.take_along_axis(cfp, ow, 2) * top
        U = ((X[:, None, :] - xhat) ** 2).sum(2)
        count = (idx >= 0).sum(2)
        names = np.where(idx >= 0, self._pr(idx, rows[:, None, None]), 0.0).sum(2)
        return owner, coef, xhat, U + names, count

    def sig(self, idx):
        return np.where(idx >= 0, self.hv[np.maximum(idx, 0)], 0.0).sum(-1)

    # ---- the search over configurations ------------------------------------------
    def search(self, X, full=True):
        Wp = self.Wp if full else self.partial()
        B, K, h, d = len(X), BEAM, self.h, self.d
        rows0 = np.arange(B)
        idx = np.full((B, K, S), -1, np.int64)
        owner, coef, xhat, cost, count = self.exact(idx, X, rows0, Wp)
        cost[:, 1:] = np.inf
        active = np.ones(B, bool)
        rounds = np.zeros(B, int)
        for r in range(ROUNDS):
            a = np.flatnonzero(active)
            if len(a) == 0:
                break
            rounds[a] += 1
            Xs, ix, ow, cf, xh, co = X[a], idx[a], owner[a], coef[a], xhat[a], cost[a]
            b = len(a)
            ra = a[:, None, None]
            act = ix >= 0
            claims = Wp[np.maximum(ix, 0)] * act[..., None]
            s1 = claims.argmax(2)
            c1 = np.take_along_axis(claims, s1[:, :, None, :], 2)[:, :, 0, :]
            np.put_along_axis(claims, s1[:, :, None, :], -1.0, 2)
            s2 = claims.argmax(2)
            c2 = np.take_along_axis(claims, s2[:, :, None, :], 2)[:, :, 0, :]
            cf2 = np.take_along_axis(cf, s2, 2)
            xhat2 = np.where(c2 > 0, cf2 * c2, 0.0).astype(np.float32)
            e_old = (Xs[:, None, :] - xh) ** 2
            e2 = (Xs[:, None, :] - xhat2) ** 2
            owS = np.where(ow < 0, S, ow)
            flat = (owS + (np.arange(b * K) * (S + 1)).reshape(b, K, 1)).ravel()
            old_u = np.bincount(flat, weights=e_old.ravel(), minlength=b * K * (S + 1)).reshape(b, K, S + 1)[:, :, :S]
            new_u = np.bincount(flat, weights=e2.ravel(), minlength=b * K * (S + 1)).reshape(b, K, S + 1)[:, :, :S]
            lam_s = self._pr(ix, ra)
            drop_cost = np.where(act, co[:, :, None] - old_u + new_u - lam_s, np.inf)
            res = Xs[:, None, :] - xh
            corr = res @ Wp.T
            on = np.zeros((b, K, h), bool)
            for s in range(S):
                m = act[:, :, s]
                on[np.nonzero(m)[0], np.nonzero(m)[1], ix[m, s]] = True
            corr[on] = -np.inf
            cand = np.argpartition(-corr, PRE - 1, axis=2)[:, :, :PRE]
            Wc = Wp[cand]
            lam_c = self._pr(cand, ra)
            add_cost, ac = self._add_cost(Xs, Wc, c1, xh, co, lam_c)
            add_cost[co == np.inf] = np.inf
            add_cost[count[a] >= S] = np.inf
            ds = np.argsort(drop_cost, axis=2)[:, :, :NSWAP_DROP]
            ts = np.argsort(add_cost, axis=2)[:, :, :NSWAP_ADD]
            Wc2 = np.take_along_axis(Wc, ts[..., None], 2)
            lam_c2 = np.take_along_axis(lam_c, ts, 2)
            swap_cost = np.full((b, K, NSWAP_DROP, NSWAP_ADD), np.inf)
            for j in range(NSWAP_DROP):
                sj = ds[:, :, j]
                gone = ow == sj[:, :, None]
                M_minus = np.where(gone, c2, c1)
                xh_minus = np.where(gone, xhat2, xh)
                base = np.take_along_axis(drop_cost, sj[:, :, None], 2)[:, :, 0]
                sc, _ = self._add_cost(Xs, Wc2, M_minus, xh_minus, base, lam_c2)
                swap_cost[:, :, j] = sc
            est = np.concatenate([add_cost, drop_cost, swap_cost.reshape(b, K, -1)], 2)
            pick = np.argpartition(est, CHILD - 1, axis=2)[:, :, :CHILD]
            child = np.repeat(ix[:, :, None, :], CHILD, 2).copy()
            free = np.argmax(ix < 0, axis=2)
            for c in range(CHILD):
                p = pick[:, :, c]
                is_add = p < PRE
                is_drop = (p >= PRE) & (p < PRE + S)
                is_swap = p >= PRE + S
                bi, ki = np.nonzero(is_add & (count[a] < S))
                child[bi, ki, c, free[bi, ki]] = cand[bi, ki, p[bi, ki]]
                bi, ki = np.nonzero(is_drop)
                child[bi, ki, c, p[bi, ki] - PRE] = -1
                bi, ki = np.nonzero(is_swap)
                q = p[bi, ki] - PRE - S
                sj, tj = q // NSWAP_ADD, q % NSWAP_ADD
                child[bi, ki, c, ds[bi, ki, sj]] = cand[bi, ki, ts[bi, ki, tj]]
            child = child.reshape(b, K * CHILD, S)
            child[np.repeat(co == np.inf, CHILD, 1)] = -1
            c_owner, c_coef, c_xhat, c_cost, c_count = self.exact(child, Xs, a, Wp)
            c_cost[np.repeat(co == np.inf, CHILD, 1)] = np.inf
            p_idx = np.concatenate([ix, child], 1)
            p_cost = np.concatenate([co, c_cost], 1)
            p_sig = self.sig(p_idx)
            p_sig[p_cost == np.inf] = np.inf
            o = np.lexsort((p_cost, p_sig), axis=1)
            sg = np.take_along_axis(p_sig, o, 1)
            cs = np.take_along_axis(p_cost, o, 1)
            dup = np.zeros_like(cs, bool)
            dup[:, 1:] = sg[:, 1:] == sg[:, :-1]
            cs[dup] = np.inf
            keep = np.take_along_axis(o, np.argsort(cs, axis=1)[:, :K], 1)
            old_sig = np.sort(np.where(co == np.inf, np.inf, self.sig(ix)), 1)
            bi = np.arange(b)[:, None]
            idx[a] = p_idx[bi, keep]
            cost[a] = np.take_along_axis(np.concatenate([co, c_cost], 1), keep, 1)
            owner[a] = np.concatenate([ow, c_owner], 1)[bi, keep]
            coef[a] = np.concatenate([cf, c_coef], 1)[bi, keep]
            xhat[a] = np.concatenate([xh, c_xhat], 1)[bi, keep]
            count[a] = np.concatenate([count[a], c_count], 1)[bi, keep]
            new_sig = np.sort(np.where(cost[a] == np.inf, np.inf, self.sig(idx[a])), 1)
            active[a[(old_sig == new_sig).all(1)]] = False
        self.cap_hits += int(active.sum())
        best = cost.argmin(1)
        ar = np.arange(B)
        st = dict(idx=idx[ar, best], owner=owner[ar, best], coef=coef[ar, best],
                  xhat=xhat[ar, best], cost=cost[ar, best], count=count[ar, best])
        C = np.zeros((B, h), np.float32)
        act = st["idx"] >= 0
        C[np.nonzero(act)[0], st["idx"][act]] = st["coef"][act]
        st["C"] = C
        return st

    def _add_cost(self, Xs, Wc, Mx, xh, base, lam):
        region = Wc > Mx[:, :, None, :]
        X4 = Xs[:, None, None, :]
        num = (region * X4 * Wc).sum(3)
        den = (region * Wc * Wc).sum(3)
        ac = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        ac = np.maximum(ac, 0).astype(np.float32)
        new_u = (region * (X4 - ac[..., None] * Wc) ** 2).sum(3)
        old_u = (region * ((Xs[:, None, :] - xh) ** 2)[:, :, None, :]).sum(3)
        c = base[:, :, None] - old_u + new_u + lam
        c[ac <= 0] = np.inf
        return c, ac

    # ---- learning from the winning configuration ---------------------------------
    def learn(self, st, X, step):
        idx, owner, coef, xhat = st["idx"], st["owner"], st["coef"], st["xhat"]
        B, h, d = len(X), self.h, self.d
        act = idx >= 0
        oh = owner[:, None, :] == self.ar_S[None, :, None]
        if self.rule == "soft":
            claims = self.Wp[np.maximum(idx, 0)] * act[..., None]
            share = claims / np.maximum(claims.sum(1, keepdims=True), 1e-12)
            target = share * X[:, None, :]
        else:                                                       # order: residual, largest coefficient first
            target = np.zeros((B, S, d), np.float32)
            rank = np.argsort(-np.where(act, coef, -1), axis=1)
            res = X.copy()
            for r in range(S):
                s = rank[:, r]
                m = act[np.arange(B), s]
                if not m.any():
                    break
                bi = np.flatnonzero(m)
                target[bi, s[bi]] = res[bi]
                res[bi] -= coef[bi, s[bi]][:, None] * self.Wp[idx[bi, s[bi]]]
        tt = idx[act]
        tg = target[act]
        Ssum = np.zeros((h, d), np.float32)
        np.add.at(Ssum, tt, tg)
        m = np.bincount(tt, minlength=h).astype(np.float32)
        p = m > 0
        n_before = self.n.copy()
        self.n[p] += m[p]
        self.last[p] = step
        eta = np.minimum(1.0, m[p] / np.minimum(self.n[p], N_MAX))
        self.M[p] += eta[:, None] * (Ssum[p] / m[p, None] - self.M[p])
        # value: energy each template explained on its owned pixels, minus what it charged
        expl = (oh * (X[:, None, :] ** 2 - (X[:, None, :] - xhat[:, None, :]) ** 2)).sum(2)     # (B,S)
        gain = expl[act] - self._pr(idx, np.arange(B)[:, None])[act]
        G = np.zeros(h, np.float32)
        np.add.at(G, tt, gain)
        self.value = DECAY * self.value + G
        # usage, for the prices
        self.uses = DECAY * self.uses + m
        self.images = DECAY * self.images + B
        new = (n_before == 0) & (self.n > 0)
        if new.any():
            self.uses[new] = self.uses[self.n > 0].mean()
        # recycle: never won, stale, or judged and not paying for itself
        judged = (self.n >= MIN_USES) & (self.value < 0)
        free = np.flatnonzero((self.n == 0) | (step - self.last > STALE) | judged)
        self.recycled += int(judged.sum())
        if len(free):
            R = np.maximum(X - xhat, 0)
            e = (R ** 2).sum(1)
            cand = np.argsort(-e)
            cand = cand[e[cand] > HIRE]
            taken = []
            for i in cand:
                if len(taken) >= min(len(free), HIRE_MAX):
                    break
                r = R[i]
                if taken and (unit(r) @ unit(np.stack(taken)).T).max() > DUP:
                    continue
                taken.append(r)
            for k, r in enumerate(taken):
                t = free[k]
                self.M[t], self.n[t], self.last[t], self.value[t] = r, 1, step, 0.0
                self.uses[t] = self.uses[self.n > 0].mean()
                self.hired += 1
        self.W = unit(self.M)
        self.Wp = np.maximum(self.W, 0)
        self.reprice()


class Tower:
    def __init__(self, rng):
        self.L1 = Layer(784, H1, "soft", rng)
        Z = np.load(OWN / "weights_soft_l0.02.npz")
        self.L1.M = Z["W"].astype(np.float32).copy()
        self.L1.W = unit(self.L1.M)
        self.L1.Wp = np.maximum(self.L1.W, 0)
        self.L1.n[:] = 1
        self.L1.last[:] = 0
        self.L2 = Layer(H1 + 10, H2, "order", rng, d_code=H1)

    def z(self, st1, y):
        b = (st1["C"] > 0).astype(np.float32)
        lab = np.zeros((len(b), 10), np.float32)
        if y is not None:
            lab[np.arange(len(y)), y] = LABEL_W
        return np.concatenate([unit(b), lab], 1)

    def expect(self, st2):
        e = st2["C"] @ self.L2.Wp[:, :H1]                     # (B,256) what the top expects below
        return e / np.maximum(e.max(1, keepdims=True), 1e-8)

    def read(self, X, y=None, feedback=True):
        self.L1.Pb = None
        st1 = self.L1.search(X)
        z = self.z(st1, y)
        st2 = self.L2.search(z, full=y is not None)
        if feedback:
            self.L1.Pb = self.L1.price[None, :] * (1.0 - EXPECT * self.expect(st2))
            st1 = self.L1.search(X)
            self.L1.Pb = None
            z = self.z(st1, y)
            st2 = self.L2.search(z, full=y is not None)
        return st1, st2, z

    def label_top(self, st2):
        v = st2["C"] @ self.L2.W[:, H1:]
        pred = v.argmax(1)
        pred[(st2["C"] > 0).sum(1) == 0] = -1
        return pred

    def train(self, X, y, rng, log):
        step = 0
        order = rng.permutation(len(X))
        for b in range(0, len(X), BATCH):
            i = order[b:b + BATCH]
            st1, st2, z = self.read(X[i], y[i], feedback=True)
            self.L1.learn(st1, X[i], step)
            self.L2.learn(st2, z, step)
            step += 1
            if step % 10 == 0:
                log(step, st1, st2)

    def codes(self, X, feedback):
        C1, C2, P, cost = [], [], [], []
        for b in range(0, len(X), BATCH):
            st1, st2, _ = self.read(X[b:b + BATCH], None, feedback)
            C1.append(st1["C"]); C2.append(st2["C"]); P.append(self.label_top(st2))
            cost.append(st1["cost"] + st2["cost"])
        return np.concatenate(C1), np.concatenate(C2), np.concatenate(P), np.concatenate(cost)


# ------------------------------------------------------------------ scoring ----
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


def support(Wp, frac=0.9):
    w2 = Wp ** 2
    o = np.argsort(-w2, 1)
    cs = np.cumsum(np.take_along_axis(w2, o, 1), 1) / np.maximum(w2.sum(1, keepdims=True), 1e-12)
    return (cs < frac).sum(1) + 1


def reuse(C):
    on = C > 0
    uses = on.sum(0)
    live = uses > 0
    p = uses[live] / uses[live].sum()
    sets = {frozenset(np.flatnonzero(r)) for r in on}
    return dict(live=int(live.sum()), uses_mean=float(uses[live].mean()), uses_entropy=float(-(p * np.log2(p)).sum()),
                distinct_configs=len(sets), configs_per_live=float(len(sets) / max(live.sum(), 1)),
                on_per_image=float(on.sum(1).mean()))


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(0)
    tw = Tower(np.random.default_rng(0))

    def log(step, st1, st2):
        print(f"  step {step:3d}  L1 on {st1['count'].mean():.2f}  L2 on {st2['count'].mean():.2f}  "
              f"tower cost {(st1['cost'] + st2['cost']).mean():.3f}  hired {tw.L1.hired}/{tw.L2.hired}  "
              f"recycled {tw.L1.recycled}/{tw.L2.recycled}  {time.time()-t0:.0f}s", flush=True)

    tw.train(Xtr, ytr, np.random.default_rng(1), log)
    t_train = time.time() - t0
    res = dict(t_train=t_train, modes={})
    C1tr, C2tr, _, _ = tw.codes(Xtr[:TALLY_N], feedback=True)
    for fb in (True, False):
        C1, C2, P, cost = tw.codes(Xte, feedback=fb)
        covered = P >= 0
        m = dict(label_top=float((P == yte).mean()), label_top_when_covered=float((P[covered] == yte[covered]).mean()) if covered.any() else 0.0,
                 covered=float(covered.mean()),
                 tally_L1=tally(C1tr > 0, ytr[:TALLY_N], C1 > 0, yte), tally_L2=tally(C2tr > 0, ytr[:TALLY_N], C2 > 0, yte),
                 probe_L1=probe(C1tr, ytr[:TALLY_N], C1, yte), probe_L2=probe(C2tr, ytr[:TALLY_N], C2, yte),
                 tower_cost=float(cost.mean()), reuse_L1=reuse(C1), reuse_L2=reuse(C2))
        res["modes"]["feedback" if fb else "no_feedback"] = m
        if fb:
            np.savez(OUT / "codes_test.npz", C1=C1, C2=C2, P=P, y=yte)
        print(f"{'feedback' if fb else 'no feedback':12s}: label at top {m['label_top']:.3f} (covered {m['covered']*100:.0f}%, "
              f"{m['label_top_when_covered']:.3f} when covered)  tally L1 {m['tally_L1']:.3f} L2 {m['tally_L2']:.3f}  "
              f"probe L1 {m['probe_L1']:.3f} L2 {m['probe_L2']:.3f}  L1 on {m['reuse_L1']['on_per_image']:.1f} L2 on {m['reuse_L2']['on_per_image']:.1f}  "
              f"configs L1 {m['reuse_L1']['distinct_configs']} L2 {m['reuse_L2']['distinct_configs']}", flush=True)
    # what the top templates are
    W2 = tw.L2.Wp
    mem = (W2[:, :H1] > 0.5 * W2[:, :H1].max(1, keepdims=True)).sum(1)
    share1 = W2[:, :H1].max(1) / np.maximum(np.linalg.norm(W2[:, :H1], axis=1), 1e-8)
    lab = W2[:, H1:]
    lab_share = np.linalg.norm(lab, axis=1) / np.maximum(np.linalg.norm(W2, axis=1), 1e-8)
    live2 = tw.L2.n > 0
    s1 = support(tw.L1.Wp)
    live1 = tw.L1.n > 0
    res.update(L2=dict(live=int(live2.sum()), members_median=float(np.median(mem[live2])), members_mean=float(mem[live2].mean()),
                       wrappers=float((share1[live2] > 0.9).mean()), label_share_median=float(np.median(lab_share[live2])),
                       hired=tw.L2.hired, recycled=tw.L2.recycled),
               L1=dict(live=int(live1.sum()), support_median=float(np.median(s1[live1])),
                       wholes=int((s1[live1] > 60).sum()), strokes=int(((s1[live1] > 20) & (s1[live1] <= 60)).sum()),
                       dots=int((s1[live1] <= 20).sum()), hired=tw.L1.hired, recycled=tw.L1.recycled))
    with open(OUT / "tower.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / "weights.npz", W1=tw.L1.W, Wp1=tw.L1.Wp, n1=tw.L1.n, price1=tw.L1.price,
             W2=tw.L2.W, Wp2=tw.L2.Wp, n2=tw.L2.n, price2=tw.L2.price, Xte=Xte[:8], yte=yte[:8])
    print(f"L2: {res['L2']['live']} live, members median {res['L2']['members_median']:.0f}, wrappers {res['L2']['wrappers']*100:.0f}%, "
          f"label share median {res['L2']['label_share_median']:.2f}, hired {tw.L2.hired}, recycled {tw.L2.recycled} | "
          f"L1: support {res['L1']['support_median']:.0f}px, wholes/strokes/dots {res['L1']['wholes']}/{res['L1']['strokes']}/{res['L1']['dots']}, "
          f"hired {tw.L1.hired}, recycled {tw.L1.recycled} | train {t_train:.0f}s total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
