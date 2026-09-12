"""Two engines over the same sparse vector.

BitTally      counts pairs of BITS.            The Hopfield-with-counts baseline.
PatternEngine counts bits within a PATTERN,    The level-raising proposal.
              and pairs of PATTERNS.

Both complete a partial vector by adding whatever the counts say belongs, and
both stop when nothing left to add is worth its price.
"""

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------
# level 0: pairs of bits
# --------------------------------------------------------------------------
class BitTally:
    def __init__(self, n_bits, alpha=0.5, price=0.0):
        self.N = n_bits
        self.alpha = alpha
        self.price = price

    def fit(self, X):
        X = X.astype(np.float64)
        self.T = len(X)
        self.n = X.sum(0)
        self.C = X.T @ X
        a = self.alpha
        # log [ P(i,j) / P(i)P(j) ] -- the popularity correction
        self.W = np.log((self.C + a) * self.T / ((self.n[:, None] + a) * (self.n[None, :] + a)))
        np.fill_diagonal(self.W, 0.0)
        return self

    def complete(self, cue, max_add=40):
        """cue: bool mask of bits KNOWN to be on. Returns the completed mask."""
        on = cue.copy()
        for _ in range(max_add):
            s = self.W[on].sum(0)
            s[on] = -np.inf
            j = int(np.argmax(s))
            if s[j] <= self.price:
                break
            on[j] = True
        return on


# --------------------------------------------------------------------------
# level 1: patterns, and pairs of patterns
# --------------------------------------------------------------------------
class PatternEngine:
    """A pattern is a tally over bits. The pattern statistics are a tally over
    patterns. Same object, one level up."""

    def __init__(self, n_bits, mu=3.0, use_stats=True, alpha=0.5,
                 max_patterns=80, mint_min=3, prune_every=1000,
                 prune_age=800, prune_use=15, tau=0.7, mint_max=10,
                 alpha_p=0.05, mint_weight=3.0, dedup=0.7, min_seed=0.0, seed=0):
        self.N = n_bits
        self.mu = mu
        self.use_stats = use_stats
        self.alpha = alpha
        self.M = max_patterns
        self.mint_min = mint_min
        self.tau = tau
        self.mint_max = mint_max
        # A pattern's prior over bits it has never seen must sit near zero.
        # At alpha_p = 0.5 a fresh pattern claims p = 0.25 on all ~100 bits,
        # and its penalty for predicting bits that are absent buries it before
        # it is ever picked a second time.
        self.alpha_p = alpha_p
        self.mint_weight = mint_weight
        self.dedup = dedup
        # An ABSOLUTE floor on the seed pair: the two bits must actually keep
        # company, not merely fail to avoid each other. Needed on the upper
        # floor, where two independent things can be the only things left in a
        # residual and get lumped on evidence of nearly zero. Harmful on floor
        # 0, where genuine atoms can have modest internal evidence (the filler
        # atoms here sit at 0.49), so it defaults off and is set per level.
        self.min_seed = min_seed
        self.Wbits = None
        self.prune_every = prune_every
        self.prune_age = prune_age
        self.prune_use = prune_use

        self.cnt = np.zeros((self.M, n_bits))
        self.use = np.zeros(self.M)
        self.co = np.zeros((self.M, self.M))
        self.born = np.zeros(self.M)
        self.V = 0
        self.T = 0
        self.bit_n = np.zeros(n_bits)

    # ---- derived tables -------------------------------------------------
    def probs(self):
        a = self.alpha_p
        return (self.cnt[:self.V] + a) / (self.use[:self.V, None] + 1.0 + a)

    def base(self):
        a = self.alpha
        return (self.bit_n + a) / (self.T + 2 * a)

    def pmi(self):
        a = self.alpha
        u = self.use[:self.V]
        return np.log((self.co[:self.V, :self.V] + a) * max(self.T, 1)
                      / ((u[:, None] + a) * (u[None, :] + a)))

    # ---- inference ------------------------------------------------------
    def explain(self, on_idx, off_idx, p=None, b=None, W=None, max_k=8, mu=None,
                bonus=None):
        """Greedy: add the pattern with the best gain until nothing pays.

        on_idx  bits known to be ON
        off_idx bits known to be OFF (empty during completion -- unknown != off)
        """
        if self.V == 0:
            return []
        p = self.probs() if p is None else p
        b = self.base() if b is None else b
        W = (self.pmi() if self.use_stats else None) if W is None else W
        mu = self.mu if mu is None else mu

        neg = np.ones(self.N)
        neg[on_idx] = 1.0 - b[on_idx]
        S = []
        for _ in range(max_k):
            q_on = 1.0 - neg[on_idx]
            num = 1.0 - neg[on_idx][None, :] * (1.0 - p[:, on_idx])
            g = np.log(np.clip(num, EPS, None) / np.clip(q_on, EPS, None)[None, :]).sum(1)
            if len(off_idx):
                g = g + np.log(np.clip(1.0 - p[:, off_idx], EPS, None)).sum(1)
            if W is not None and S:
                g = g + W[:, S].sum(1)
            if bonus is not None:
                g = g + bonus          # what the floor above expects, in the
            g = g - mu                 # same currency: surprise against chance
            g[S] = -np.inf
            k = int(np.argmax(g))
            if g[k] <= 0:
                break
            S.append(k)
            neg[on_idx] *= (1.0 - p[k, on_idx])
        return S

    def q_of(self, S, p=None, b=None):
        """noisy-OR prediction over all bits, with the base rate as the leak."""
        p = self.probs() if p is None else p
        b = self.base() if b is None else b
        neg = 1.0 - b
        for k in S:
            neg = neg * (1.0 - p[k])
        return 1.0 - neg

    # ---- minting --------------------------------------------------------
    def _grow(self, resid):
        """Carve one coherent lump out of the unexplained bits, using the
        bit-level pair table: start from the strongest pair, keep adding the
        bit with the best average evidence against everything already in, and
        stop when that evidence falls below tau of the seed pair's.

        This is the only place the two levels touch: level 0 tells level 1
        which bits belong in the same thing."""
        if self.Wbits is None or len(resid) < 2:
            return resid[:self.mint_max]
        sub = self.Wbits[np.ix_(resid, resid)].copy()
        np.fill_diagonal(sub, -np.inf)
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        seed_w = sub[i, j]
        if seed_w <= self.min_seed:
            return resid[[i]]
        cur = [int(i), int(j)]
        while len(cur) < self.mint_max:
            rest = [k for k in range(len(resid)) if k not in cur]
            if not rest:
                break
            m = np.array([sub[k, cur].mean() for k in rest])
            b = int(np.argmax(m))
            if m[b] < self.tau * seed_w:
                break
            cur.append(rest[b])
        return resid[np.array(cur)]

    # ---- learning -------------------------------------------------------
    def seed_vocab(self, lumps, weight=20.0):
        """Install a vocabulary by hand, to test what the engine does GIVEN a
        carving rather than what carving it finds."""
        for i, lump in enumerate(lumps):
            self.cnt[i] = 0.0
            self.cnt[i, np.asarray(lump)] = weight
            self.use[i] = weight
            self.born[i] = 0
        self.V = len(lumps)
        return self

    def fit(self, X, Wbits=None, mint=True):
        self.Wbits = Wbits
        self._mint = mint
        for t in range(len(X)):
            x = X[t]
            on_idx = np.flatnonzero(x)
            off_idx = np.flatnonzero(x == 0)
            self.T += 1
            self.bit_n += x

            p = self.probs() if self.V else np.zeros((0, self.N))
            b = self.base()
            W = self.pmi() if (self.use_stats and self.V) else None
            S = self.explain(on_idx, off_idx, p=p, b=b, W=W)

            # mint from whatever no PATTERN claims. Note this deliberately
            # ignores the base rate: a bit that is merely common is not thereby
            # explained, or nothing frequent would ever get a pattern.
            neg = np.ones(self.N)
            for k in S:
                neg = neg * (1.0 - p[k])
            resid = np.flatnonzero((x == 1) & (neg > 0.5))
            if getattr(self, "_mint", True) and len(resid) >= self.mint_min and self.V < self.M:
                lump = self._grow(resid)
                # do not mint a second name for something already named. If an
                # existing pattern covers this lump it simply lost the round --
                # usually to a dropped bit -- so credit it instead.
                dup = -1
                if self.V and len(lump):
                    ov = (p[:, lump] > 0.5).mean(1)
                    if ov.max() >= self.dedup:
                        dup = int(np.argmax(ov))
                if dup >= 0 and dup not in S:
                    S = S + [dup]
                elif dup < 0 and len(lump) >= self.mint_min:
                    k = self.V
                    self.cnt[k] = 0.0
                    self.cnt[k, lump] = self.mint_weight
                    self.use[k] = self.mint_weight
                    self.born[k] = self.T
                    self.V += 1
                    S = S + [k]
                    p = self.probs()

            if not S:
                continue

            # responsibility-weighted update. For a noisy-OR, a cause that
            # fired always turns the bit on, so given the bit IS on,
            #     P(this pattern fired) = p / q,   q = 1 - (1-base) * prod(1-p)
            # Causes are not competing for a fixed budget -- two patterns can
            # both be responsible -- so these do not sum to one.
            Sa = np.array(S)
            w = p[Sa][:, on_idx]
            q_on = 1.0 - (1.0 - b[on_idx]) * np.prod(1.0 - w, axis=0)
            r = np.clip(w / np.clip(q_on, EPS, None), 0.0, 1.0)
            self.cnt[Sa[:, None], on_idx[None, :]] += r
            self.use[Sa] += 1.0
            for i in range(len(Sa)):
                for j in range(i + 1, len(Sa)):
                    self.co[Sa[i], Sa[j]] += 1.0
                    self.co[Sa[j], Sa[i]] += 1.0

            if (getattr(self, "_mint", True) and self.prune_every
                    and self.T % self.prune_every == 0):
                self._prune()
        if getattr(self, "_mint", True):
            self._prune()
        return self

    def _prune(self):
        if self.V == 0:
            return
        age = self.T - self.born[:self.V]
        keep = np.flatnonzero(~((age > self.prune_age) & (self.use[:self.V] < self.prune_use)))
        if len(keep) == self.V:
            return
        v = len(keep)
        self.cnt[:v] = self.cnt[keep]
        self.use[:v] = self.use[keep]
        self.born[:v] = self.born[keep]
        co = self.co[np.ix_(keep, keep)]
        self.co[:, :] = 0.0
        self.co[:v, :v] = co
        self.cnt[v:] = 0.0
        self.use[v:] = 0.0
        self.born[v:] = 0.0
        self.V = v

    # ---- reading --------------------------------------------------------
    def complete(self, cue, mu=None, bonus=None):
        """cue: bool mask of bits known ON. Unknown bits are unknown, not off.

        mu here is the READ price, which need not equal the learning price: it
        sets how much a pattern must be worth before it is added on the
        strength of company alone, with no observed bit of its own."""
        on_idx = np.flatnonzero(cue)
        S = self.explain(on_idx, np.array([], dtype=int), mu=mu, bonus=bonus)
        q = self.q_of(S)
        return (q > 0.5) | cue, S
