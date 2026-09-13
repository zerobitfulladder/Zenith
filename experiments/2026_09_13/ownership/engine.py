"""One fully connected layer, 784 -> 256, learned by a search over configurations.

A configuration is a set of templates. Each pixel is explained by exactly one
member of the set: the one whose weight on that pixel is largest (ownership by
strength). A template's coefficient is fitted on the pixels it owns, so a small
sharp template is scored on its own region.

    cost(C) = energy of the image left unexplained  +  price * |C|

The search: beam of BEAM configurations; moves are add / drop / swap; children
are estimated cheaply, the best few are scored exactly, the BEAM cheapest
survive, and the search stops when no move on any of them makes the
description shorter. Nothing fixes how many templates are on.

Three learning rules from the winning configuration, same search in all:
    order   each winner, largest coefficient first, learns the residual left by
            the ones before it (ownership by order -- today's rule)
    hard    each winner learns the input on the pixels it owns and zero elsewhere
    soft    contested pixels are split in proportion to weight
Templates are unit norm. Never-winning or stale templates are hired from the
largest unexplained leftover.
"""

import numpy as np

H, D = 256, 784
S = 20            # most templates a configuration may hold (safety cap, reported)
BEAM = 4
ROUNDS = 30       # safety cap on search rounds (reported)
PRE = 16          # add candidates pre-filtered by match with the unexplained part
CHILD = 4         # children scored exactly per beam state
NSWAP_DROP, NSWAP_ADD = 2, 6
HIRE, HIRE_MAX, DUP = 0.3, 8, 0.7
N_MAX, STALE = 200, 60
CHUNK = 8         # configurations scored exactly at a time (memory)
ASSIGN = "weight" # who owns a pixel: "weight" = the sharpest template on it; "fit" = the one predicting it best
FIT_ITERS = 2     # for "fit": reassignment / refit rounds inside the exact scoring
LEARN_ASSIGN = "weight"   # who LEARNS a pixel under the "owned" rule: the reader's owner ("weight"), or the
                          # best-predicting template among those on ("fit": one k-means step over the pixels)


def unit(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-8)


class Ownership:
    def __init__(self, rule, lam, rng):
        assert rule in ("order", "hard", "soft", "owned", "owned-px")
        self.npix = np.zeros((H, D), np.float32)          # per-pixel counts, for the owned-px rule
        self.rule, self.lam = rule, lam
        self.price = np.full(H, lam, np.float32)           # per-template price; uniform unless self-priced
        self.M = (rng.normal(size=(H, D)) / np.sqrt(D)).astype(np.float32)
        self.W = unit(self.M)
        self.Wp = np.maximum(self.W, 0)
        self.n = np.zeros(H, np.float32)
        self.last = np.full(H, -10**9)
        self.hv = np.random.default_rng(12345).random(H)
        self.hired = 0
        self.cap_hits = 0
        self.rounds = []
        self.ar_S = np.arange(S)

    # ------------------------------------------------------------ exact ----
    def exact(self, idx, X):
        """idx (B,n,S) template ids (-1 = empty slot), X (B,d).
        Returns owner slot per pixel (B,n,d) (-1 none), coef (B,n,S), xhat (B,n,d), cost (B,n), count (B,n)."""
        B, n, _ = idx.shape
        owner = np.full((B, n, D), -1, np.int64)
        coef = np.zeros((B, n, S), np.float32)
        xhat = np.zeros((B, n, D), np.float32)
        for j in range(0, n, CHUNK):
            ii = idx[:, j:j + CHUNK]
            c = ii.shape[1]
            act = ii >= 0
            claims = self.Wp[np.maximum(ii, 0)] * act[..., None]                  # (B,c,S,d)
            ow = claims.argmax(2)                                                  # (B,c,d)
            top = np.take_along_axis(claims, ow[:, :, None, :], 2)[:, :, 0, :]
            ow = np.where(top > 0, ow, S)                                          # S = the "nobody" slot
            flat = (ow + (np.arange(B * c) * (S + 1)).reshape(B, c, 1)).ravel()
            Xb = np.broadcast_to(X[:, None, :], (B, c, D))
            num = np.bincount(flat, weights=(Xb * top).ravel(), minlength=B * c * (S + 1)).reshape(B, c, S + 1)[:, :, :S]
            den = np.bincount(flat, weights=(top * top).ravel(), minlength=B * c * (S + 1)).reshape(B, c, S + 1)[:, :, :S]
            cf = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
            cf = np.maximum(cf, 0).astype(np.float32)
            if ASSIGN == "fit":
                # k-means on pixels among the templates that are on: each pixel goes to the template
                # predicting it best (ties to the larger coefficient), then coefficients are refitted
                for _ in range(FIT_ITERS):
                    pred = claims * cf[..., None]                                    # (B,c,S,d)
                    err = (X[:, None, None, :] - pred) ** 2 - 1e-9 * cf[..., None]   # tie -> larger coefficient
                    err[~act] = np.inf
                    err[(cf <= 0)] = np.inf
                    ow = err.argmin(2)
                    top = np.take_along_axis(claims, ow[:, :, None, :], 2)[:, :, 0, :]
                    ow = np.where(top > 0, ow, S)
                    flat = (ow + (np.arange(B * c) * (S + 1)).reshape(B, c, 1)).ravel()
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
        names = np.where(idx >= 0, self.price[np.maximum(idx, 0)], 0.0).sum(2)
        return owner, coef, xhat, U + names, count

    def sig(self, idx):
        return np.where(idx >= 0, self.hv[np.maximum(idx, 0)], 0.0).sum(-1)

    # ----------------------------------------------------------- search ----
    def search(self, X):
        B, K = len(X), BEAM
        idx = np.full((B, K, S), -1, np.int64)
        owner, coef, xhat, cost, count = self.exact(idx, X)
        cost[:, 1:] = np.inf                                   # only slot 0 is a real state at the start
        active = np.ones(B, bool)
        rounds = np.zeros(B, int)
        for r in range(ROUNDS):
            a = np.flatnonzero(active)
            if len(a) == 0:
                break
            rounds[a] += 1
            Xs, ix, ow, cf, xh, co = X[a], idx[a], owner[a], coef[a], xhat[a], cost[a]
            b = len(a)
            act = ix >= 0
            claims = self.Wp[np.maximum(ix, 0)] * act[..., None]                  # (b,K,S,d)
            s1 = claims.argmax(2)                                                  # best slot per pixel
            c1 = np.take_along_axis(claims, s1[:, :, None, :], 2)[:, :, 0, :]
            np.put_along_axis(claims, s1[:, :, None, :], -1.0, 2)                  # hide it, take the next
            s2 = claims.argmax(2)
            c2 = np.take_along_axis(claims, s2[:, :, None, :], 2)[:, :, 0, :]
            cf2 = np.take_along_axis(cf, s2, 2)                                    # coef of the second-best owner, per pixel
            xhat2 = np.where(c2 > 0, cf2 * c2, 0.0).astype(np.float32)
            e_old = (Xs[:, None, :] - xh) ** 2
            e2 = (Xs[:, None, :] - xhat2) ** 2
            owS = np.where(ow < 0, S, ow)
            flat = (owS + (np.arange(b * K) * (S + 1)).reshape(b, K, 1)).ravel()
            old_u_slot = np.bincount(flat, weights=e_old.ravel(), minlength=b * K * (S + 1)).reshape(b, K, S + 1)[:, :, :S]
            new_u_slot = np.bincount(flat, weights=e2.ravel(), minlength=b * K * (S + 1)).reshape(b, K, S + 1)[:, :, :S]
            lam_s = self.price[np.maximum(ix, 0)]                                  # price of each slot's template
            drop_cost = np.where(act, co[:, :, None] - old_u_slot + new_u_slot - lam_s, np.inf)   # (b,K,S)
            # adds: pre-filter by match with the unexplained part, then score by region
            res = Xs[:, None, :] - xh
            corr = res @ self.Wp.T                                                 # (b,K,h)
            on = np.zeros((b, K, H), bool)
            for s in range(S):
                m = act[:, :, s]
                on[np.nonzero(m)[0], np.nonzero(m)[1], ix[m, s]] = True
            corr[on] = -np.inf
            cand = np.argpartition(-corr, PRE - 1, axis=2)[:, :, :PRE]           # (b,K,PRE)
            Wc = self.Wp[cand]                                                     # (b,K,PRE,d)
            add_cost, ac = self._add_cost(Xs, Wc, c1, xh, co, self.price[cand])
            add_cost[co == np.inf] = np.inf
            add_cost[count[a] >= S] = np.inf
            # swaps: the cheapest drops x the best adds, scored on the post-drop map
            ds = np.argsort(drop_cost, axis=2)[:, :, :NSWAP_DROP]                # (b,K,2) slots
            ts = np.argsort(add_cost, axis=2)[:, :, :NSWAP_ADD]                  # (b,K,8) into cand
            Wc2 = np.take_along_axis(Wc, ts[..., None], 2)                         # (b,K,8,d)
            swap_cost = np.full((b, K, NSWAP_DROP, NSWAP_ADD), np.inf)
            for j in range(NSWAP_DROP):
                sj = ds[:, :, j]
                gone = ow == sj[:, :, None]
                M_minus = np.where(gone, c2, c1)
                xh_minus = np.where(gone, xhat2, xh)
                base = np.take_along_axis(drop_cost, sj[:, :, None], 2)[:, :, 0]
                sc, _ = self._add_cost(Xs, Wc2, M_minus, xh_minus, base, np.take_along_axis(self.price[cand], ts, 2))
                swap_cost[:, :, j] = sc
            # gather the children: estimated costs, then the best CHILD per state scored exactly
            est = np.concatenate([add_cost, drop_cost, swap_cost.reshape(b, K, -1)], 2)   # (b,K,PRE+S+16)
            pick = np.argpartition(est, CHILD - 1, axis=2)[:, :, :CHILD]
            child = np.repeat(ix[:, :, None, :], CHILD, 2).copy()                # (b,K,CHILD,S)
            free = np.argmax(ix < 0, axis=2)                                       # first free slot
            for c in range(CHILD):
                p = pick[:, :, c]
                is_add = p < PRE
                is_drop = (p >= PRE) & (p < PRE + S)
                is_swap = p >= PRE + S
                # add
                bi, ki = np.nonzero(is_add & (count[a] < S))
                t = cand[bi, ki, p[bi, ki]]
                child[bi, ki, c, free[bi, ki]] = t
                # drop
                bi, ki = np.nonzero(is_drop)
                child[bi, ki, c, p[bi, ki] - PRE] = -1
                # swap
                bi, ki = np.nonzero(is_swap)
                q = p[bi, ki] - PRE - S
                sj, tj = q // NSWAP_ADD, q % NSWAP_ADD
                slot = ds[bi, ki, sj]
                t = cand[bi, ki, ts[bi, ki, tj]]
                child[bi, ki, c, slot] = t
            child = child.reshape(b, K * CHILD, S)
            child[np.repeat(co == np.inf, CHILD, 1)] = -1                          # children of dead slots are junk
            c_owner, c_coef, c_xhat, c_cost, c_count = self.exact(child, Xs)
            c_cost[np.repeat(co == np.inf, CHILD, 1)] = np.inf
            # pool with the current states, merge duplicates, keep the BEAM cheapest
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
            keep = np.take_along_axis(o, np.argsort(cs, axis=1)[:, :K], 1)        # (b,K) into the pool
            old_sig = np.sort(np.where(co == np.inf, np.inf, self.sig(ix)), 1)
            bi = np.arange(b)[:, None]
            idx[a] = p_idx[bi, keep]
            cost[a] = np.take_along_axis(np.concatenate([co, c_cost], 1), keep, 1)
            owner[a] = np.concatenate([ow, c_owner], 1)[bi, keep]
            coef[a] = np.concatenate([cf, c_coef], 1)[bi, keep]
            xhat[a] = np.concatenate([xh, c_xhat], 1)[bi, keep]
            count[a] = np.concatenate([count[a], c_count], 1)[bi, keep]
            new_sig = np.sort(np.where(cost[a] == np.inf, np.inf, self.sig(idx[a])), 1)
            same = (old_sig == new_sig).all(1)
            active[a[same]] = False
        self.cap_hits += int(active.sum())
        self.rounds.extend(rounds.tolist())
        best = cost.argmin(1)
        ar = np.arange(B)
        return dict(idx=idx[ar, best], owner=owner[ar, best], coef=coef[ar, best],
                    xhat=xhat[ar, best], cost=cost[ar, best], count=count[ar, best])

    def _add_cost(self, Xs, Wc, Mx, xh, base, lam):
        """Cost of adding each candidate in Wc (b,K,m,d) to a state with max-claim map Mx and
        reconstruction xh: the candidate takes the pixels where it is sharper than the owner.
        lam: the price of each candidate, (b,K,m)."""
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

    # --------------------------------------------------------- learning ----
    def learn(self, st, X, step):
        idx, owner, coef = st["idx"], st["owner"], st["coef"]                      # (B,S), (B,d), (B,S)
        B = len(X)
        act = idx >= 0
        oh = owner[:, None, :] == self.ar_S[None, :, None]                          # (B,S,d)
        if self.rule == "hard":
            target = oh * X[:, None, :]
        elif self.rule in ("owned", "owned-px"):
            # the objective's M-step, online: owned pixels move toward the input, unowned pixels do not move.
            # done by making the target equal to the current template on unowned pixels.
            if LEARN_ASSIGN == "fit":
                pred = coef[:, :, None] * self.Wp[np.maximum(idx, 0)]                       # (B,S,d)
                err = (X[:, None, :] - pred) ** 2 - 1e-9 * coef[:, :, None]
                err[~act] = np.inf
                err[coef <= 0] = np.inf
                o = err.argmin(1)                                                          # (B,d)
                o = np.where(np.isfinite(err.min(1)) & (X > 0), o, -1)                     # only ink is worth learning
                oh = o[:, None, :] == self.ar_S[None, :, None]
            cur = self.W[np.maximum(idx, 0)]                                            # (B,S,d)
            target = np.where(oh, X[:, None, :], cur)
        elif self.rule == "soft":
            claims = self.Wp[np.maximum(idx, 0)] * act[..., None]                    # (B,S,d)
            share = claims / np.maximum(claims.sum(1, keepdims=True), 1e-12)
            target = share * X[:, None, :]
        else:                                                                       # order
            target = np.zeros((B, S, D), np.float32)
            rank = np.argsort(-np.where(act, coef, -1), axis=1)                    # largest coefficient first
            res = X.copy()
            for r in range(S):
                s = rank[:, r]
                m = act[np.arange(B), s]
                if not m.any():
                    break
                bi = np.flatnonzero(m)
                target[bi, s[bi]] = res[bi]
                t = idx[bi, s[bi]]
                res[bi] -= coef[bi, s[bi]][:, None] * self.Wp[t]
        tt = idx[act]
        if self.rule == "owned-px":
            # each pixel of a template is a running average over the images where THAT pixel was owned
            ohm = oh[act].astype(np.float32)                                            # (n,d) owned mask
            xs = X[np.nonzero(act)[0]]
            Sx = np.zeros((H, D), np.float32)
            Sc = np.zeros((H, D), np.float32)
            np.add.at(Sx, tt, ohm * xs)
            np.add.at(Sc, tt, ohm)
            pp = Sc > 0
            self.npix[pp] += Sc[pp]
            eta = np.minimum(1.0, Sc[pp] / np.minimum(self.npix[pp], N_MAX))
            self.M[pp] += eta * (Sx[pp] / Sc[pp] - self.M[pp])
            m = np.bincount(tt, minlength=H).astype(np.float32)
            p = m > 0
            self.n[p] += m[p]
            self.last[p] = step
        else:
            tg = target[act]
            Ssum = np.zeros((H, D), np.float32)
            np.add.at(Ssum, tt, tg)
            m = np.bincount(tt, minlength=H).astype(np.float32)
            p = m > 0
            mean_t = Ssum[p] / m[p, None]
            self.n[p] += m[p]
            self.last[p] = step
            eta = np.minimum(1.0, m[p] / np.minimum(self.n[p], N_MAX))
            self.M[p] += eta[:, None] * (mean_t - self.M[p])
        # hire the never-winning and the stale from the largest unexplained leftover
        free = np.flatnonzero((self.n == 0) | (step - self.last > STALE))
        if len(free):
            R = np.maximum(X - st["xhat"], 0)
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
                self.M[t], self.n[t], self.last[t] = r, 1, step
                self.npix[t] = 1.0
                self.hired += 1
        self.W = unit(self.M)
        self.Wp = np.maximum(self.W, 0)

    def fit(self, X, rng, epochs, batch, log=None):
        step = 0
        for ep in range(epochs):
            order = rng.permutation(len(X))
            for b in range(0, len(X), batch):
                xb = X[order[b:b + batch]]
                st = self.search(xb)
                self.learn(st, xb, step)
                step += 1
                if log and step % 20 == 0:
                    log(step, st)
        return self

    def codes(self, X, batch=128):
        C = np.zeros((len(X), H), np.float32)
        out = dict(count=[], cost=[], idx=[], owner=[], xhat=[])
        for b in range(0, len(X), batch):
            st = self.search(X[b:b + batch])
            act = st["idx"] >= 0
            bi = np.nonzero(act)[0]
            C[b + bi, st["idx"][act]] = st["coef"][act]
            for k in out:
                out[k].append(st[k])
        return C, {k: np.concatenate(v) for k, v in out.items()}
