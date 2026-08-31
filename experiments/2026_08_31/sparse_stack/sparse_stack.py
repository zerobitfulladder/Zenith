"""stack3, with layer one swapped for a place code and a sparse column.

Geometry, layer three and both reads are `2026_08_30/stack3/stack3.py`
IMPORTED, not reimplemented, so the 0.57 (wta) and 0.93 (dense) numbers on
the board are produced by the same code they were produced by. Only the
substrate of layers one and two changes:

    L1 in    4x4 patch -> place code (16 channels x nb cells), nonnegative
    L1/L2    k-of-K nonnegative matching pursuit, greedy, MP for learning
    down     code -> place-code vector -> centroid per channel -> pixels

Reading can be the pursuit or the settling; learning is always the pursuit,
because `../sparse_column` measured that settling-as-learner triples the
dictionary's coherence.
"""

import sys
from pathlib import Path

import numpy as np

_H = Path(__file__).resolve().parent
sys.path.insert(0, str(_H.parent / "place_code"))
sys.path.insert(0, str(_H.parent / "sparse_column"))
sys.path.insert(0, str(_H.parents[1] / "2026_08_30" / "stack3"))

from place_code import PlaceCode                                    # noqa: E402
from sparse_column import SparseColumn                              # noqa: E402
from stack3 import (BoundLayer, unit_rows, patches_of, unpatch,     # noqa: E402
                    l2_windows, unwindow, EPS, PATCH, GRID,
                    WIN, L2_GRID)


def calibrate(col, U, target, iters=16):
    """The lambda that makes a settled code as sparse as the pursuit's k."""
    lo, hi = 1e-4, 1.0
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        m = float((col.settle(U, mid, n_iter=48) > 0).sum(1).mean())
        lo, hi = (mid, hi) if m > target else (lo, mid)
    return float(np.sqrt(lo * hi))


class SparseStack:
    """Layers one and two: image in, the 9 layer-two codes out."""

    def __init__(self, k1, k2, kmax, eta, rng, nb=8, read="pursue"):
        self.k1, self.k2, self.kmax, self.read = k1, k2, kmax, read
        self.pc = PlaceCode(PATCH * PATCH, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
        self.l1 = SparseColumn(k1, self.pc.size, kmax=kmax, eta=eta, rng=rng)
        self.l2 = SparseColumn(k2, WIN * WIN * k1, kmax=kmax, eta=eta, rng=rng)
        self.lam1 = self.lam2 = None

    # -- up ---------------------------------------------------------------
    def _l1_in(self, X):
        """(n,28,28) -> (n*49, 16*nb) unit rows. No centring anywhere."""
        return unit_rows(self.pc.encode(patches_of(X)))[0]

    def _code(self, col, U, lam):
        return col.pursue(U)[0] if self.read == "pursue" else col.settle(U, lam)

    def l1_field(self, X):
        C = self._code(self.l1, self._l1_in(X), self.lam1)
        return C.reshape(len(X), GRID, GRID, self.k1)

    def forward(self, X):
        U, _ = unit_rows(l2_windows(self.l1_field(X)))
        C = self._code(self.l2, U, self.lam2)
        return C.reshape(len(X), L2_GRID * L2_GRID * self.k2)

    # -- learning ---------------------------------------------------------
    def train(self, X, epochs, rng, chunk=256):
        for _ in range(epochs):                       # layer one, on patches
            order = rng.permutation(len(X))
            for s in range(0, len(X), chunk):
                U = self._l1_in(X[order[s:s + chunk]])
                self.l1.learn(U[np.linalg.norm(U, axis=1) > EPS])
            self.l1.revive(self._l1_in(X[rng.choice(len(X), 64, replace=False)]))
        if self.read == "settle":
            self.lam1 = calibrate(self.l1, self._l1_in(X[:64]), self.kmax)
        for _ in range(epochs):                       # layer two, on frozen L1
            order = rng.permutation(len(X))
            for s in range(0, len(X), chunk):
                U, _ = unit_rows(l2_windows(self.l1_field(X[order[s:s + chunk]])))
                self.l2.learn(U[np.linalg.norm(U, axis=1) > EPS])
        if self.read == "settle":
            U, _ = unit_rows(l2_windows(self.l1_field(X[:256])))
            self.lam2 = calibrate(self.l2, U, self.kmax)

    # -- down -------------------------------------------------------------
    def backward(self, code, n):
        """9 layer-two codes -> pixels, back down through both rungs."""
        Wv = self.l2.decode(code.reshape(n * L2_GRID * L2_GRID, self.k2))
        F = unwindow(Wv, n, self.k1)
        Pc = self.l1.decode(F.reshape(n * GRID * GRID, self.k1))
        return unpatch(self.pc.decode(Pc), n)

    def templates(self):
        """Layer one's templates as 4x4 patches, for the picture."""
        return self.pc.decode(self.l1.W).reshape(-1, PATCH, PATCH)


class NoCenterBound(BoundLayer):
    """Layer three without the mean-centring.

    `BoundLayer.join` subtracts the row mean, which on a sparse nonnegative
    code turns every silent template slightly negative -- the dense-code
    failure mode, arriving through the back door. This arm prices removing
    it. Everything else is inherited.
    """

    def join(self, img, lab=None):
        V = np.zeros((len(img), self.dim))
        V[:, :self.img_dim] = img
        if lab is not None:
            gain = self.rho * np.linalg.norm(img, axis=1) / \
                np.maximum(np.linalg.norm(lab, axis=1), EPS)
            V[:, self.img_dim:] = lab * gain[:, None]
        return unit_rows(V)[0]

    def classify(self, img, topm=1):
        S = self._bids(unit_rows(img)[0], self.img_ix)
        win = S.argmax(axis=1)
        if topm <= 1:
            prof = self.W[win][:, self.lab_ix]
        else:
            g = np.clip(S, 0.0, None)
            np.put_along_axis(g, np.argpartition(-g, topm, axis=1)[:, topm:],
                              0.0, axis=1)
            prof = g @ self.W[:self.n_boot][:, self.lab_ix]
        return prof.argmax(axis=1), win, prof

    def generate(self, lab, topm=1):
        S = self._bids(unit_rows(lab)[0], self.lab_ix)
        if topm <= 1:
            win = S.argmax(axis=1)
            return self.W[win][:, self.img_ix], win
        g = np.clip(S, 0.0, None)
        np.put_along_axis(g, np.argpartition(-g, topm, axis=1)[:, topm:],
                          0.0, axis=1)
        g /= np.maximum(g.sum(axis=1, keepdims=True), EPS)
        return g @ self.W[:self.n_boot][:, self.img_ix], S.argmax(axis=1)


class GramStack(SparseStack):
    """Layer three compares codes in the dictionary's geometry, not cell by cell.

    `c1 . c2` over code cells asks only "did we pick the same templates" --
    two near-identical templates are different coordinates and contribute
    nothing. `(c1 W2) . (c2 W2) = c1' (W2 W2') c2` asks how similar the
    things they picked actually are, which is the `x1' W'W x2` the dense arm
    got for free (08-30 part two). One matmul, and the code itself stays
    sparse and nameable.
    """

    def forward_gram(self, X):
        C = self.forward(X)
        n, P = len(X), L2_GRID * L2_GRID
        return (C.reshape(n * P, self.k2) @ self.l2.W).reshape(n, -1)

    def backward_gram(self, code, n):
        """Layer three's output is already in the 576-wide window space."""
        F = unwindow(code.reshape(n * L2_GRID * L2_GRID, WIN * WIN * self.k1),
                     n, self.k1)
        Pc = self.l1.decode(F.reshape(n * GRID * GRID, self.k1))
        return unpatch(self.pc.decode(Pc), n)
