"""The notebook is the learning target.

A template is a clique carved out of the notebook -- a set of things whose
mutual friendship is high -- and nothing else. It never learns from an input.
Reading uses templates; counting updates the notebook; carving makes templates
from the counts. Learning and reading come apart.

Things are the 100 lights and up to 150 templates. The notebook is pair counts
over things. The vector is one slot per thing.

What this buys, as consequences rather than rules:

  no wrappers      a one-member template explains zero pairs, so the carve
                   (which starts from a PAIR) cannot produce one
  no blobs         growing F1 into {A,B} predicts A-F1 go together; the
                   notebook says chance; the carve refuses
  no duplicates    the pairs inside a carved clique are masked, so the same
                   clique can never be seeded again

Counting is as-understood: the vector is counted AFTER substitution, once per
input, so a part that has a name stops being counted at the light level and
its group forms at the template level instead.
"""

import numpy as np


class Carve:
    def __init__(self, n_lights, M=150, rounds=3, beam=3, max_steps=8,
                 min_frac=0.75, propose_frac=0.35, miss_w=0.5, alpha=0.5,
                 tau=0.7, min_seed=0.3, lump_min_light=3, lump_min_temp=2,
                 warmup=1000, min_age=300):
        self.NL, self.M = n_lights, M
        self.nT = n_lights + M
        self.rounds, self.beam, self.max_steps = rounds, beam, max_steps
        self.min_frac, self.propose_frac, self.miss_w = min_frac, propose_frac, miss_w
        self.alpha, self.tau, self.min_seed = alpha, tau, min_seed
        self.lump_min_light, self.lump_min_temp = lump_min_light, lump_min_temp
        self.warmup = warmup
        # a thing born 20 inputs ago has a 20-input window, and one coincidence
        # in a window that short clears any floor. Lights get the warm-up;
        # templates get this.
        self.min_age = min_age

        self.co = np.zeros((self.nT, self.nT), np.float32)
        self.seen = np.zeros(self.nT, np.float32)
        self.born = np.full(self.nT, -1.0)
        self.T = 0
        self.mask = np.zeros((self.nT, self.nT), bool)      # explained pairs
        self.members = []
        self.V = 0
        self._E = np.zeros((M, self.nT), bool)             # preallocated
        self.E = self._E[:0]

    # ---- the notebook ---------------------------------------------------
    def observe(self, v):
        idx = np.flatnonzero(v)
        self.T += 1
        fresh = idx[self.born[idx] < 0]
        self.born[fresh] = self.T
        self.seen[idx] += 1
        self.co[np.ix_(idx, idx)] += 1

    def friend(self, idx):
        """More-than-chance, each rate taken over the window since the thing
        existed, so template-things are not judged against warm-up time."""
        a = self.alpha
        c = self.co[np.ix_(idx, idx)]
        n = self.seen[idx]
        w = np.maximum(self.T - self.born[idx], 1.0)
        span = np.maximum(self.T - np.maximum(self.born[idx][:, None],
                                              self.born[idx][None, :]), 1.0)
        F = np.log(((c + a) / span) / (((n + a) / w)[:, None] * ((n + a) / w)[None, :]))
        F[self.mask[np.ix_(idx, idx)]] = -np.inf
        young = (self.T - self.born[idx]) < self.min_age
        F[young, :] = -np.inf
        F[:, young] = -np.inf
        np.fill_diagonal(F, -np.inf)
        return F

    # ---- carving: the only way a template comes to exist --------------------
    def carve(self, resid):
        idx = np.flatnonzero(resid)
        if len(idx) < 2 or self.V >= self.M:
            return None
        F = self.friend(idx)
        if not np.isfinite(F).any():
            return None
        i, j = np.unravel_index(int(np.argmax(F)), F.shape)
        seed = F[i, j]
        if seed <= self.min_seed:
            return None
        cur = [int(i), int(j)]
        while True:
            rest = [k for k in range(len(idx)) if k not in cur]
            if not rest:
                break
            m = np.array([F[k, cur].mean() for k in rest])
            b = int(np.argmax(m))
            if m[b] < self.tau * seed:
                break
            cur.append(rest[b])
        lump = idx[np.array(cur)]
        need = self.lump_min_light if lump[0] < self.NL else self.lump_min_temp
        if len(lump) < need:
            return None
        t = self.V
        self.members.append(lump)
        self.mask[np.ix_(lump, lump)] = True
        self.V += 1
        self._E[t, lump] = True
        self.E = self._E[:self.V]                           # a view, no copy
        return t

    # ---- reading ------------------------------------------------------------
    def _score(self, idx, exclude):
        nE = self.E.sum(1)
        present = self.E[:, idx].sum(1).astype(np.float64)
        frac = present / np.maximum(nE, 1)
        gain = present - self.miss_w * (nE - present)
        ok = (frac >= self.min_frac) & (nE > 0)
        if exclude:
            ok[exclude] = False
        return gain, frac, present, ok

    def search(self, v):
        beams = [(0.0, [], v)]
        done = []
        for _ in range(self.max_steps):
            nxt = []
            for tot, seq, r in beams:
                idx = np.flatnonzero(r)
                if len(idx) == 0 or self.V == 0:
                    done.append((tot, seq, r))
                    continue
                gain, _, _, ok = self._score(idx, seq)
                g = np.where(ok & (gain > 0), gain, -np.inf)
                cands = [int(t) for t in np.argsort(-g)[:self.beam] if np.isfinite(g[t])]
                if not cands:
                    done.append((tot, seq, r))
                    continue
                for t in cands:
                    nxt.append((tot + g[t], seq + [t], r & ~self.E[t]))
            if not nxt:
                break
            nxt.sort(key=lambda b: -b[0])
            beams = nxt[:self.beam]
        done += beams
        best = max(done, key=lambda b: b[0])
        return best[1], best[2]

    def propose(self, v):
        idx = np.flatnonzero(v)
        if len(idx) == 0 or self.V == 0:
            return None
        _, frac, present, _ = self._score(idx, [])
        ok = (frac >= self.propose_frac) & (frac < 1.0)
        if not ok.any():
            return None
        return int(np.argmax(np.where(ok, present, -1.0)))

    def substitute(self, v, seq):
        nv = v.copy()
        for t in seq:
            nv &= ~self.E[t]
            nv[self.NL + t] = True
        return nv

    def expand(self, v, depth=6):
        out = v.copy()
        for _ in range(depth):
            names = np.flatnonzero(out[self.NL:self.NL + self.V])
            if len(names) == 0:
                break
            for t in names:
                out[self.NL + t] = False
                out |= self.E[t]
        return out[:self.NL]

    # ---- use ------------------------------------------------------------------
    def fit(self, X):
        for i, x in enumerate(X):
            v = np.zeros(self.nT, bool)
            v[:self.NL] = x
            if i < self.warmup:
                self.observe(v)
                continue
            for _ in range(self.rounds):
                seq, resid = self.search(v)
                t = self.carve(resid)
                if t is not None:
                    seq = seq + [t]
                if not seq:
                    break
                nv = self.substitute(v, seq)
                if nv.sum() >= v.sum():
                    break
                v = nv
            self.observe(v)                     # counted as understood
        return self

    def complete(self, lights):
        v = np.zeros(self.nT, bool)
        v[:self.NL] = lights
        seqs = []
        for _ in range(self.rounds):
            seq, _ = self.search(v)
            if not seq:
                break
            nv = self.substitute(v, seq)
            if nv.sum() >= v.sum():
                break
            v = nv
            seqs.append(seq)
        p = self.propose(v)
        top = v.copy()
        if p is not None:
            top[self.NL + p] = True
        return self.expand(top) | lights, seqs, p
