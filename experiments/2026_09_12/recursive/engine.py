"""One vector, one notebook, recursion in time instead of in floors.

The vector has 100 light slots and some blank slots. A blank slot becomes the
name of a pattern. Reading is:

    round 1   recognise patterns among the active slots
              switch OFF the slots they accounted for
              switch ON the slots that name them
    round 2   do it again -- now the active slots are pattern names
    ...       until it stops getting smaller

So a pattern whose members are lights and a pattern whose members are other
patterns are the same kind of object in the same vocabulary, and there is one
set of thresholds because there is one notebook.

The notebook is pair counts over slots, which makes a separate "do these two
patterns go together" table unnecessary: a pattern's name IS a slot, so the
company it keeps is the slot-pair count between names.

Three rules keep it honest:

  compress   a rewrite is only accepted if it leaves FEWER slots active.
             Otherwise the thing churns forever, swapping six slots for six.
  evidence   going up, a pattern may only be used if most of its members are
             actually active. Partial matches are guesses, and guessing is not
             what the up-pass is for.
  propose    guessing happens once, at the top, deliberately: allow a single
             partly-matched pattern, then expand it back down into lights.
             That is the synthesis half.
"""

import numpy as np

EPS = 1e-9


class Recursive:
    def __init__(self, n_lights, n_extra=120, mu=3.0, alpha=0.5, alpha_p=0.05,
                 mint_min=3, mint_max=10, mint_weight=3.0, tau=0.7, min_seed=0.0,
                 dedup=0.7, max_rounds=4, beam=1, max_k=4, match=0.6):
        self.NL, self.M, self.N = n_lights, n_extra, n_lights + n_extra
        self.mu, self.alpha, self.alpha_p = mu, alpha, alpha_p
        self.mint_min, self.mint_max, self.mint_weight = mint_min, mint_max, mint_weight
        self.tau, self.min_seed, self.dedup = tau, min_seed, dedup
        self.max_rounds, self.beam, self.max_k, self.match = max_rounds, beam, max_k, match

        self.pair = np.zeros((self.N, self.N))     # the one notebook
        self.slot = np.zeros(self.N)
        self.T = 0
        self.pcnt = np.zeros((self.M, self.N))     # each pattern's own notebook
        self.puse = np.zeros(self.M)
        self.V = 0
        self._W = self._b = None

    # ---- the notebook ---------------------------------------------------
    def observe(self, v):
        self.T += 1
        idx = np.flatnonzero(v)
        self.slot[idx] += 1
        self.pair[np.ix_(idx, idx)] += 1            # sparse: v is ~24 of 220

    def tables(self, refresh=False):
        if refresh or self._W is None:
            a, s = self.alpha, self.slot
            W = np.log((self.pair + a) * max(self.T, 1)
                       / ((s[:, None] + a) * (s[None, :] + a)))
            np.fill_diagonal(W, 0.0)
            self._W = W
            self._b = (s + a) / (self.T + 2 * a)
        return self._W, self._b

    def probs(self):
        a = self.alpha_p
        return (self.pcnt[:self.V] + a) / (self.puse[:self.V, None] + 1.0 + a)

    def name_of(self, k):
        return self.NL + k

    # ---- recognising ----------------------------------------------------
    def usable(self, v, p, match):
        """A pattern may be used only if `match` of its members are active."""
        if self.V == 0:
            return np.zeros(0, dtype=bool)
        mem = p > 0.5
        n = np.clip(mem.sum(1), 1, None)
        hit = (mem & (v > 0)[None, :]).sum(1)
        return (hit / n) >= match

    def explain(self, v, off_idx, mu=None, p=None, b=None, W=None, match=None):
        """Which patterns account for the active slots. Beam search: carry the
        best `beam` partial answers rather than committing to one."""
        if self.V == 0:
            return []
        p = self.probs() if p is None else p
        if W is None or b is None:
            W, b = self.tables()
        mu = self.mu if mu is None else mu
        match = self.match if match is None else match
        on_idx = np.flatnonzero(v)
        if len(on_idx) == 0:
            return []
        ok = self.usable(v, p, match)
        if not ok.any():
            return []
        names = self.NL + np.arange(self.V)

        beams = [(0.0, [], 1.0 - b[on_idx])]
        done = []
        for _ in range(self.max_k):
            nxt = []
            for sc, S, neg in beams:
                q = 1.0 - neg
                num = 1.0 - neg[None, :] * (1.0 - p[:, on_idx])
                g = np.log(np.clip(num, EPS, None)
                           / np.clip(q, EPS, None)[None, :]).sum(1)
                if len(off_idx):
                    g = g + np.log(np.clip(1.0 - p[:, off_idx], EPS, None)).sum(1)
                if S:
                    g = g + W[np.ix_(names, names[S])].sum(1)
                g = g - mu
                g[~ok] = -np.inf
                g[S] = -np.inf
                order = np.argsort(-g)[:max(self.beam, 1)]
                good = [int(k) for k in order if g[k] > 0]
                if not good:
                    done.append((sc, S))
                for k in good:
                    nxt.append((sc + g[k], S + [k], neg * (1.0 - p[k, on_idx])))
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[0])
            beams = nxt[:max(self.beam, 1)]
        done += [(sc, S) for sc, S, _ in beams]
        return max(done, key=lambda t: t[0])[1] if done else []

    def claimed(self, S, p=None):
        p = self.probs() if p is None else p
        neg = np.ones(self.N)
        for k in S:
            neg = neg * (1.0 - p[k])
        return 1.0 - neg

    # ---- naming something new -------------------------------------------
    def _grow(self, resid, W):
        if len(resid) < 2:
            return resid
        sub = W[np.ix_(resid, resid)].copy()
        np.fill_diagonal(sub, -np.inf)
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        if sub[i, j] <= self.min_seed:
            return resid[[i]]
        seed_w, cur = sub[i, j], [int(i), int(j)]
        while len(cur) < self.mint_max:
            rest = [k for k in range(len(resid)) if k not in cur]
            if not rest:
                break
            m = np.array([sub[k, cur].mean() for k in rest])
            bidx = int(np.argmax(m))
            if m[bidx] < self.tau * seed_w:
                break
            cur.append(rest[bidx])
        return resid[np.array(cur)]

    def _mint(self, v, S, p, W, mint_min):
        resid = np.flatnonzero((v == 1) & (self.claimed(S, p) < 0.5))
        if len(resid) < mint_min or self.V >= self.M:
            return S, False
        lump = self._grow(resid, W)
        if len(lump) < mint_min:
            return S, False
        if self.V:
            ov = (p[:, lump] > 0.5).mean(1)
            if ov.max() >= self.dedup:                # already have a name for it
                k = int(np.argmax(ov))
                return (S + [k] if k not in S else S), False
        k = self.V
        self.pcnt[k] = 0.0
        self.pcnt[k, lump] = self.mint_weight
        self.puse[k] = self.mint_weight
        self.V += 1
        return S + [k], True

    def _credit(self, v, S, p, b):
        if not S:
            return
        on = np.flatnonzero(v)
        Sa = np.array(S)
        w = p[Sa][:, on]
        q = 1.0 - (1.0 - b[on]) * np.prod(1.0 - w, axis=0)
        r = np.clip(w / np.clip(q, EPS, None), 0.0, 1.0)
        self.pcnt[Sa[:, None], on[None, :]] += r
        self.puse[Sa] += 1.0

    # ---- going up -------------------------------------------------------
    def settle(self, v, learn=False, mu=None, match=None):
        trace, minted_here = [], False
        for r in range(self.max_rounds):
            if learn:
                self.observe(v)
            p = self.probs() if self.V else np.zeros((0, self.N))
            W, b = self.tables()
            off = np.flatnonzero(v == 0) if learn else np.array([], dtype=int)
            S = self.explain(v, off, mu=mu, p=p, b=b, W=W, match=match)
            if learn and not minted_here:
                # at most one new name per input, and a group of patterns needs
                # only two members where a group of lights needs three
                S, minted = self._mint(v, S, p, W, 2 if r else self.mint_min)
                if minted:
                    minted_here = True
                    p = self.probs()
            if learn:
                self._credit(v, S, p, b)
            if not S:
                break
            nv = np.zeros(self.N, dtype=np.int8)
            nv[(v == 1) & (self.claimed(S, p) < 0.5)] = 1
            for k in S:
                nv[self.name_of(k)] = 1
            if nv.sum() >= v.sum():          # a rewrite must COMPRESS
                break
            trace.append((int(v.sum()), list(S)))
            v = nv
        return v, trace

    # ---- going back down ------------------------------------------------
    def expand(self, v, thresh=0.5):
        """Replace every pattern name by its members, again and again, until
        only lights are left."""
        out = v.astype(np.int8).copy()
        p = self.probs()
        for _ in range(self.max_rounds + 3):
            names = [s for s in np.flatnonzero(out) if s >= self.NL]
            if not names:
                break
            for s in names:
                out[s] = 0
                if s - self.NL < self.V:
                    out[p[s - self.NL] > thresh] = 1
        return out[:self.NL].astype(bool)

    # ---- the two things you do with it ----------------------------------
    def fit(self, X, warmup=1000):
        for x in X[:warmup]:                      # watch before naming anything
            v = np.zeros(self.N, dtype=np.int8)
            v[:self.NL] = x
            self.observe(v)
        self.tables(refresh=True)
        for i, x in enumerate(X):
            if i % 50 == 0:
                self.tables(refresh=True)
            v = np.zeros(self.N, dtype=np.int8)
            v[:self.NL] = x
            self.settle(v, learn=True)
        self.tables(refresh=True)
        return self

    def complete(self, cue, mu=None, propose=True):
        """Analysis, then synthesis. Settle on what the evidence supports, then
        allow ONE partly-matched pattern at the top -- the guess -- and expand
        it back down into lights."""
        v = np.zeros(self.N, dtype=np.int8)
        v[:self.NL] = cue
        top, trace = self.settle(v, learn=False, mu=mu)
        guess = None
        if propose and self.V:
            p = self.probs()
            W, b = self.tables()
            S = self.explain(top, np.array([], dtype=int), mu=mu, p=p, b=b, W=W,
                             match=0.01)
            new = [k for k in S if not top[self.name_of(k)]]
            if new:
                guess = new[0]
                top = top.copy()
                top[self.name_of(guess)] = 1
        return self.expand(top) | cue, top, trace, guess
