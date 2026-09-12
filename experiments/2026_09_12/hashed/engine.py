"""No minting decision, no warm-up, no all-or-nothing naming.

Four changes from ../recursive:

  hashed        a friendship between slot a and slot b lives at hash(a,b).
                Nobody decides to create it; its address is a function of what
                it is, so the same pair found twice lands in the same place and
                duplicates cannot happen.

  paired        a group is never hashed as a set. {4,5,6} is hash(4,5) and then
                hash(that, 6). If 6 drops out the first half still fires, so a
                missing member costs the top of the chain rather than all of it.

  soft          a slot holds a strength, not a bit.

  decaying      evidence that stops arriving fades, so a bad friendship picked
                up on input 30 is not permanent. Done with a running scale
                factor rather than by touching every count.

The notebook is sparse -- a neighbour list per slot -- because the hash space
has to be large (20,000 here) and a dense table over that is 3 GB.

What cannot be removed: each round keeps only the strongest few friendships.
24 active slots make 276 pairs, those make 38,000. The cutoff is the design
admitting that the number of possible groups grows faster than any budget.
"""

from collections import defaultdict

import numpy as np

EPS = 1e-9


class Hashed:
    def __init__(self, n_lights, n_hash=20000, K=20, rounds=3, decay=0.9995,
                 alpha=0.5, floor=0.05, min_friend=0.4):
        self.NL, self.NH = n_lights, n_hash
        self.N = n_lights + n_hash
        self.K, self.rounds = K, rounds
        self.decay, self.alpha = decay, alpha
        self.floor, self.min_friend = floor, min_friend

        self.slot = np.zeros(self.N)                     # scaled counts
        self.nbr = defaultdict(lambda: defaultdict(float))
        self.T = 0.0
        self.g = 1.0                                     # forgetting scale
        self.meaning = {}                                # slot -> (a, b)

    # ---- where a friendship lives ---------------------------------------
    def address(self, a, b):
        if a > b:
            a, b = b, a
        h = (a * 2654435761 ^ b * 40503 ^ 0x9E37) % self.NH
        for probe in range(8):
            s = self.NL + (h + probe) % self.NH
            m = self.meaning.get(s)
            if m is None:
                self.meaning[s] = (a, b)
                return s
            if m == (a, b):
                return s
        return self.NL + h                               # give up, share

    # ---- the notebook ---------------------------------------------------
    def observe(self, v):
        idx = np.flatnonzero(v > self.floor)
        if len(idx) < 1:
            return
        w = v[idx] * self.g
        self.T += self.g
        self.slot[idx] += w
        nb = self.nbr
        for p in range(len(idx)):
            i, wi = int(idx[p]), w[p]
            ni = nb[i]
            for q in range(len(idx)):
                if q != p:
                    ni[int(idx[q])] += wi * v[idx[q]]

    def forget(self):
        self.g /= self.decay

    def F(self, i, j):
        """How much more than coincidence slots i and j occur together."""
        a, g = self.alpha, self.g
        c = self.nbr[i].get(j, 0.0) / g
        ni, nj = self.slot[i] / g, self.slot[j] / g
        return np.log((c + a) * max(self.T / g, 1.0) / ((ni + a) * (nj + a)))

    def Fblock(self, act):
        n = len(act)
        M = np.zeros((n, n))
        for p in range(n):
            for q in range(p + 1, n):
                M[p, q] = M[q, p] = self.F(int(act[p]), int(act[q]))
        return M

    # ---- one round up ---------------------------------------------------
    def lift(self, v):
        """Every pair of active slots is a candidate. Score them, keep the
        strongest K that are genuinely friendly, and that is the next vector."""
        act = np.flatnonzero(v > self.floor)
        if len(act) < 2:
            return np.zeros(self.N)
        M = self.Fblock(act)
        strength = np.minimum(v[act][:, None], v[act][None, :]) * M
        iu = np.triu_indices(len(act), k=1)
        s = strength[iu]
        friendly = M[iu] > self.min_friend
        order = np.argsort(-s)
        nv = np.zeros(self.N)
        taken = 0
        for t in order:
            if taken >= self.K or not friendly[t] or s[t] <= self.floor:
                break
            a, b = int(act[iu[0][t]]), int(act[iu[1][t]])
            slot = self.address(a, b)
            nv[slot] = max(nv[slot], min(s[t], 1.0))
            taken += 1
        return nv

    # ---- back down ------------------------------------------------------
    def expand(self, v):
        out = np.zeros(self.NL, dtype=bool)
        stack = [int(s) for s in np.flatnonzero(v > self.floor)]
        seen = set()
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            if s < self.NL:
                out[s] = True
            elif s in self.meaning:
                stack.extend(self.meaning[s])
        return out

    # ---- use ------------------------------------------------------------
    def fit(self, X):
        for x in X:
            v = np.zeros(self.N)
            v[:self.NL] = x
            self.observe(v)
            for _ in range(self.rounds):
                v = self.lift(v)
                if not (v > self.floor).any():
                    break
                self.observe(v)
            self.forget()
        return self

    def climb(self, cue):
        v = np.zeros(self.N)
        v[:self.NL] = cue.astype(float)
        levels = [v]
        for _ in range(self.rounds):
            v = self.lift(v)
            if not (v > self.floor).any():
                break
            levels.append(v)
        return levels

    def complete(self, cue, thresh=1.0, n_guess=1):
        """Climb, then guess. Only slots that have actually been seen beside
        something active are candidates -- the neighbour list gives those
        directly, so nothing is scanned."""
        levels = self.climb(cue)
        top = levels[-1].copy()
        act = [int(s) for s in np.flatnonzero(top > self.floor)]
        guesses = []
        score = defaultdict(float)
        for i in act:
            for j in self.nbr[i]:
                if top[j] <= self.floor:
                    score[j] += self.F(i, j) * top[i]
        for _ in range(n_guess):
            if not score:
                break
            j = max(score.items(), key=lambda kv: kv[1])[0]
            if score[j] <= thresh:
                break
            top[j] = 1.0
            guesses.append(j)
            del score[j]
        return self.expand(top) | cue, levels, guesses
