"""Templates are the only memory. No notebook, no pair counts, no base rates.

The vector is a 4,000-slot sparse space. Every light and every template has a
fixed random signature -- 12 slots chosen once and never changed. Light 5 on
means its 12 slots are on. Template 40 fired means its 12 slots are on. Same
space, and a template can learn "light 5 with template 40" as easily as any
pair of lights, because to it they are just slots that co-occur.

Reading is sequential explaining -- the pursuit rule -- with a beam over the
order:

    pick the template that best accounts for the residual
    switch off what it claimed
    pick the next from what is left
    stop when nothing left is worth claiming

Learning follows the sequence the search chose: each chosen template moves
toward the residual it was shown. Nothing else learns. Templates are hired
when the residual is still large after the search, by copying it, and are
refined by what they get picked for after that.

Substitution switches off the claimed slots and switches on the chosen
templates' signatures; the same encoder then reads the result.
"""

import numpy as np


class Encoder:
    def __init__(self, n_lights, M=150, N=4000, k=12, seed=0, rounds=3, beam=3,
                 max_steps=8, min_frac=0.4, propose_frac=0.35, miss_w=0.5,
                 hire_w=3.0, thresh=0.5):
        rng = np.random.default_rng(seed)
        self.NL, self.M, self.N, self.k = n_lights, M, N, k
        self.rounds, self.beam, self.max_steps = rounds, beam, max_steps
        self.min_frac, self.propose_frac = min_frac, propose_frac
        self.miss_w, self.hire_w, self.thresh = miss_w, hire_w, thresh
        self.sigL = np.stack([rng.choice(N, k, replace=False) for _ in range(n_lights)])
        self.sigT = np.stack([rng.choice(N, k, replace=False) for _ in range(M)])
        self.cnt = np.zeros((M, N), np.float32)
        self.use = np.zeros(M, np.float32)
        self.V = 0                                   # templates hired so far

    # ---- codebook ---------------------------------------------------------
    def encode(self, lights):
        v = np.zeros(self.N, bool)
        if lights.any():
            v[self.sigL[lights].ravel()] = True
        return v

    def decode(self, v):
        return v[self.sigL].sum(1) >= 0.6 * self.k

    # ---- what each template expects ---------------------------------------
    def expect(self):
        w = self.cnt[:self.V] / np.maximum(self.use[:self.V], 1)[:, None]
        E = w > self.thresh
        return E, E.sum(1)

    def _score(self, idx, E, nE, exclude):
        present = E[:, idx].sum(1).astype(np.float64)
        frac = present / np.maximum(nE, 1)
        gain = present - self.miss_w * (nE - present)
        ok = (frac >= self.min_frac) & (nE > 0)
        if exclude:
            ok[exclude] = False
        return gain, frac, present, ok

    # ---- sequential explaining, beam over the order -------------------------
    def search(self, v, E, nE):
        beams = [(0.0, [], v, [])]
        done = []
        for _ in range(self.max_steps):
            nxt = []
            for tot, seq, r, hist in beams:
                idx = np.flatnonzero(r)
                if len(idx) == 0 or self.V == 0:
                    done.append((tot, seq, r, hist))
                    continue
                gain, _, _, ok = self._score(idx, E, nE, seq)
                g = np.where(ok & (gain > 0), gain, -np.inf)
                order = np.argsort(-g)[:self.beam]
                cands = [int(t) for t in order if np.isfinite(g[t])]
                if not cands:
                    done.append((tot, seq, r, hist))
                    continue
                for t in cands:
                    nxt.append((tot + g[t], seq + [t], r & ~E[t], hist + [r]))
            if not nxt:
                break
            nxt.sort(key=lambda b: -b[0])
            beams = nxt[:self.beam]
        done += beams
        best = max(done, key=lambda b: b[0])
        return best[1], best[2], best[3]

    def propose(self, v, E, nE):
        """One partly-matched template, allowed only at the top. The guess."""
        idx = np.flatnonzero(v)
        if len(idx) == 0 or self.V == 0:
            return None
        _, frac, present, _ = self._score(idx, E, nE, [])
        ok = (frac >= self.propose_frac) & (frac < 1.0) & (nE > 0)
        if not ok.any():
            return None
        return int(np.argmax(np.where(ok, present, -1.0)))

    def substitute(self, v, seq, E):
        nv = v.copy()
        for t in seq:
            nv &= ~E[t]
            nv[self.sigT[t]] = True
        return nv

    # ---- learning -----------------------------------------------------------
    def hire(self, r):
        # a residual of one signature is not a thing to name -- it is the
        # thing itself. Naming it again is a wrapper, and wrappers breed.
        if r.sum() < 2 * self.k or self.V >= self.M:
            return None
        t = self.V
        self.cnt[t] = r.astype(np.float32) * self.hire_w
        self.use[t] = self.hire_w
        self.V += 1
        return t

    def learn(self, seq, hist):
        for t, r in zip(seq, hist):
            self.cnt[t] += r
            self.use[t] += 1

    def fit(self, X):
        for x in X:
            v = self.encode(x)
            E, nE = self.expect()
            for _ in range(self.rounds):
                seq, resid, hist = self.search(v, E, nE)
                self.learn(seq, hist)
                t = self.hire(resid)
                if t is not None:
                    seq = seq + [t]
                    E, nE = self.expect()
                if not seq:
                    break
                nv = self.substitute(v, seq, E)
                if nv.sum() >= v.sum():          # a rewrite must compress
                    break
                v = nv
        return self

    # ---- reading ------------------------------------------------------------
    def expand(self, v, E, depth=6):
        out = v.copy()
        for _ in range(depth):
            pres = out[self.sigT[:self.V]].sum(1) >= 0.6 * self.k
            if not pres.any():
                break
            for t in np.flatnonzero(pres):
                out[self.sigT[t]] = False
                out |= E[t]
        return out

    def complete(self, lights):
        v = self.encode(lights)
        E, nE = self.expect()
        levels, seqs = [v], []
        for _ in range(self.rounds):
            seq, _, _ = self.search(v, E, nE)
            if not seq:
                break
            nv = self.substitute(v, seq, E)
            if nv.sum() >= v.sum():
                break
            v = nv
            levels.append(v)
            seqs.append(seq)
        p = self.propose(v, E, nE)
        top = v.copy()
        if p is not None:
            top[self.sigT[p]] = True
        return self.decode(self.expand(top, E)) | lights, seqs, p


class EncoderTally(Encoder):
    """The same encoder with one thing added back: a notebook of pair counts
    over THINGS -- lights and templates, 250 in all -- not over the 4,000 slots.

    It is used for exactly one job: when a residual is still large after the
    search, decide which of the things in it belong together, and hire a
    template for that lump rather than for the whole residual.

    That is the information a template cannot get from its own record. A
    template knows what accompanies IT. The notebook knows what accompanies
    EACH OTHER, which is what a group is.

    With the carve doing the grouping, templates no longer need to erode from
    blobs to parts, so the evidence bar for being picked can be raised -- and
    that stops pairs eroding into wrappers.
    """

    def __init__(self, n_lights, min_frac=0.75, alpha=0.5, tau=0.7, min_seed=0.3,
                 lump_min=2, **kw):
        super().__init__(n_lights, min_frac=min_frac, **kw)
        self.sigA = np.vstack([self.sigL, self.sigT])          # every thing
        self.nT = self.NL + self.M
        self.co = np.zeros((self.nT, self.nT), np.float32)
        self.seen = np.zeros(self.nT, np.float32)
        self.T = 0.0
        self.alpha, self.tau, self.min_seed, self.lump_min = alpha, tau, min_seed, lump_min

    def things(self, v):
        """Which lights and which hired templates are present in the vector."""
        pres = v[self.sigA].sum(1) >= 0.6 * self.k
        pres[self.NL + self.V:] = False
        return np.flatnonzero(pres)

    def observe(self, v):
        idx = self.things(v)
        if len(idx) == 0:
            return
        self.T += 1
        self.seen[idx] += 1
        self.co[np.ix_(idx, idx)] += 1

    def friend(self, idx):
        a = self.alpha
        c = self.co[np.ix_(idx, idx)]
        n = self.seen[idx]
        F = np.log((c + a) * max(self.T, 1.0) / ((n[:, None] + a) * (n[None, :] + a)))
        np.fill_diagonal(F, -np.inf)
        return F

    def _grow(self, idx):
        if len(idx) < 2:
            return idx
        F = self.friend(idx)
        i, j = np.unravel_index(int(np.argmax(F)), F.shape)
        if F[i, j] <= self.min_seed:
            return idx[[i]]
        seed, cur = F[i, j], [int(i), int(j)]
        while True:
            rest = [k for k in range(len(idx)) if k not in cur]
            if not rest:
                break
            m = np.array([F[k, cur].mean() for k in rest])
            b = int(np.argmax(m))
            if m[b] < self.tau * seed:
                break
            cur.append(rest[b])
        return idx[np.array(cur)]

    def hire(self, r):
        if self.V >= self.M:
            return None
        idx = self.things(r)
        if len(idx) < self.lump_min:
            return None
        lump = self._grow(idx)
        if len(lump) < self.lump_min:
            return None
        t = self.V
        self.cnt[t] = 0.0
        self.cnt[t, self.sigA[lump].ravel()] = self.hire_w
        self.use[t] = self.hire_w
        self.V += 1
        return t

    def fit(self, X, warmup=1000):
        # watch before naming anything: a carve from an empty notebook is a
        # coin flip, and with the evidence bar raised those early mistakes
        # never erode away
        for x in X[:warmup]:
            self.observe(self.encode(x))
        for x in X:
            v = self.encode(x)
            E, nE = self.expect()
            for _ in range(self.rounds):
                self.observe(v)
                seq, resid, hist = self.search(v, E, nE)
                self.learn(seq, hist)
                t = self.hire(resid)
                if t is not None:
                    seq = seq + [t]
                    E, nE = self.expect()
                if not seq:
                    break
                nv = self.substitute(v, seq, E)
                if nv.sum() >= v.sum():
                    break
                v = nv
        return self
