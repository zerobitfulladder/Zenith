"""Two floors of the identical engine.

The claim being tested: the union vocabulary that had to be hand-built in
`run.py`'s carving test is simply what a SECOND FLOOR mints on its own.

Floor 0 sees bulbs and learns patterns. Its explanation of a panel -- which
patterns were used -- is itself a sparse on/off vector, over patterns instead
of bulbs. Floor 1 is the same class, run on that. If A and B keep company at
the pattern level, floor 1 should mint a pattern meaning "A-with-B", which is
exactly the lump that made the illegal triple refusable.

Reading goes up then down: floor 0 proposes, floor 1 says what it expects floor
0 to contain, and floor 0 is re-read with that expectation added as a bonus --
in the same currency both floors already use, surprise against chance.
"""

import numpy as np

from engines import EPS, BitTally, PatternEngine


class Stack:
    def __init__(self, n_bits, mu0=3.0, mu1=1.0, use_stats=True,
                 max_patterns=80, max_upper=40):
        self.L0 = PatternEngine(n_bits, mu=mu0, use_stats=use_stats,
                                max_patterns=max_patterns)
        self.mu1 = mu1
        self.use_stats = use_stats
        self.max_upper = max_upper
        self.L1 = None

    # ---- the message between floors ------------------------------------
    def _lift(self, X):
        """Each panel -> which floor-0 patterns explained it, as an on/off
        vector over the floor-0 vocabulary."""
        V0 = self.L0.V
        Z = np.zeros((len(X), V0), dtype=np.int8)
        p, b = self.L0.probs(), self.L0.base()
        W = self.L0.pmi() if self.use_stats else None
        for t, x in enumerate(X):
            S = self.L0.explain(np.flatnonzero(x), np.flatnonzero(x == 0),
                                p=p, b=b, W=W)
            Z[t, S] = 1
        return Z

    def fit(self, X):
        Wbits = BitTally(self.L0.N).fit(X).W
        self.L0.fit(X, Wbits=Wbits)
        self.L0._mint = False                      # floor 0 is frozen from here

        Z = self._lift(X)
        self.Z = Z
        self.L1 = PatternEngine(self.L0.V, mu=self.mu1, use_stats=self.use_stats,
                                max_patterns=self.max_upper, mint_min=2,
                                mint_max=4, tau=0.6, min_seed=0.5)
        self.L1.fit(Z, Wbits=BitTally(self.L0.V).fit(Z).W)
        self.L1._mint = False
        return self

    # ---- reading -------------------------------------------------------
    def complete(self, cue, mu=None, mu_top=None, passes=2):
        """Floor 0 never speculates -- its price stays put across both passes,
        so the only thing that can admit a pattern with no bulbs of its own is
        the floor above vouching for it. Lowering floor 0's price instead would
        let it invent the illegal combination before floor 1 ever saw it."""
        on0 = np.flatnonzero(cue)
        none = np.array([], dtype=int)
        S0 = self.L0.explain(on0, none, mu=mu)
        for _ in range(passes - 1):
            z = np.zeros(self.L0.V, dtype=np.int8)
            z[S0] = 1
            S1 = self.L1.explain(np.flatnonzero(z), none, mu=mu_top)
            q1 = self.L1.q_of(S1)                  # expected floor-0 patterns
            b1 = self.L1.base()
            bonus = np.log(np.clip(q1, EPS, None) / np.clip(b1, EPS, None))
            S0 = self.L0.explain(on0, none, mu=mu, bonus=bonus)
        q = self.L0.q_of(S0)
        return (q > 0.5) | cue, S0

    # ---- what floor 1 learned ------------------------------------------
    def upper_vocab(self, thresh=0.5):
        p = self.L1.probs()
        return [sorted(np.flatnonzero(p[k] > thresh).tolist())
                for k in range(self.L1.V)]
