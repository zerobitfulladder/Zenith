"""Top-down instead of bottom-up.

Everything starts as ONE category covering all the data. It predicts each light
with that light's overall rate -- vague, but never wrong. When a category fails
to account for what lands in it, it splits, and the split is chosen from the
same co-occurrence counts we have been using all along.

Two things follow that the bottom-up engines never managed:

  no leftover        every input lands in some category from the first one
                     onward, so there is nothing unexplained to mint from,
                     nothing to glue noise onto, nothing to cap or prune.

  illegal is empty   `A-with-B-and-C` is not refused by a stored rule. It is
                     simply never created, because no data ever lands there.
                     An absence cannot be overridden by a score.

A category is a combination, and a leaf is one legal combination of everything.
Which is also the cost: the tree pays the PRODUCT of all the independent
choices in the data, where a pairwise notebook pays the sum. That is what
`scaling` in run.py measures.
"""

import numpy as np


class CatTree:
    def __init__(self, alpha=0.5, min_size=40, max_depth=14, bic=0.5):
        self.alpha = alpha
        self.min_size = min_size
        self.max_depth = max_depth
        self.bic = bic                    # price of a split, in parameters
        self.nodes = []

    # ---- how well a category accounts for what is in it ------------------
    def _ll(self, cnt, n):
        """Total log-likelihood of a node's data under its own predictions.
        Works on a single node (1-D) or on every candidate split at once."""
        a = self.alpha
        cnt = np.atleast_2d(cnt)
        n = np.atleast_1d(n).astype(float)
        p = (cnt + a) / (n[:, None] + 2 * a)
        ll = cnt * np.log(p) + (n[:, None] - cnt) * np.log(1.0 - p)
        return ll.sum(1)

    def fit(self, X):
        X = X.astype(np.float64)
        self.I = X.shape[1]
        root = dict(members=np.arange(len(X)), depth=0, left=None, right=None,
                    split=None)
        self.nodes = [root]
        stack = [0]
        while stack:
            ni = stack.pop()
            node = self.nodes[ni]
            m = node["members"]
            Xn = X[m]
            n = len(m)
            cnt = Xn.sum(0)
            node["cnt"], node["n"] = cnt, n
            if n < 2 * self.min_size or node["depth"] >= self.max_depth:
                continue

            # every candidate split at once: row j of C is the light-counts of
            # the members where light j is on, which is exactly co-occurrence
            C = Xn.T @ Xn
            nL = np.diag(C).copy()
            nR = n - nL
            cntR = cnt[None, :] - C
            gain = self._ll(C, nL) + self._ll(cntR, nR) - self._ll(cnt, n)[0]

            # a split buys a whole second set of light probabilities, so charge
            # for them -- otherwise it splits until every leaf is one vector
            price = self.bic * self.I * np.log(max(n, 2))
            ok = (nL >= self.min_size) & (nR >= self.min_size)
            gain = np.where(ok, gain, -np.inf)
            j = int(np.argmax(gain))
            if not np.isfinite(gain[j]) or gain[j] < price:
                continue

            on = Xn[:, j] > 0
            for side, sel in (("left", on), ("right", ~on)):
                self.nodes.append(dict(members=m[sel], depth=node["depth"] + 1,
                                       left=None, right=None, split=None))
                node[side] = len(self.nodes) - 1
                stack.append(len(self.nodes) - 1)
            node["split"] = j
        return self

    # ---- reading ---------------------------------------------------------
    def leaves(self):
        return [i for i, nd in enumerate(self.nodes) if nd["split"] is None]

    def probs(self, i):
        nd = self.nodes[i]
        a = self.alpha
        return (nd["cnt"] + a) / (nd["n"] + 2 * a)

    def base(self):
        nd = self.nodes[0]
        a = self.alpha
        return (nd["cnt"] + a) / (nd["n"] + 2 * a)

    def best_leaf(self, cue):
        """Which category does this scene belong to? Scored on the lights the
        cue says are ON -- the rest are unknown, not off."""
        on = np.flatnonzero(cue)
        b = self.base()
        best, bs = None, -np.inf
        for i in self.leaves():
            p = self.probs(i)
            s = np.log(p[on] / b[on]).sum()
            if s > bs:
                best, bs = i, s
        return best

    def complete(self, cue, thresh=0.5):
        i = self.best_leaf(cue)
        return (self.probs(i) > thresh) | cue, i
