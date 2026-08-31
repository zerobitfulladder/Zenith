"""A nonnegative sparse place code: scalars in, bumps out.

Every channel owns a contiguous block of `nb` cells, and each cell stands
for one value on that channel's ladder. A value lights the cells around it
with a raised cosine measured in VALUE units, so the pattern slides
smoothly as the value moves instead of snapping to the nearest cell. That
sliding is the whole source of generalisation: a value halfway between two
cells lights both, and a hair of movement moves the code by a hair.

Blocks are contiguous and disjoint. The older encoders scattered each
channel's cells to random positions in one big array; once no two channels
share a position that scatter is just a permutation of coordinates, and
every operation downstream -- dot products, normalisation, geodesic
rotation -- is unchanged by a permutation. So it is dropped here and
`profiles` becomes a reshape.

Nothing in this file is ever negative.
"""

import numpy as np

EPS = 1e-12


class PlaceCode:
    """(n, n_ch) scalars -> (n, n_ch * nb) nonnegative sparse code."""

    def __init__(self, n_ch, nb, lo=0.0, hi=1.0, halfw=1.5):
        if nb < 2:
            raise ValueError("nb must be at least 2")
        self.n_ch, self.nb, self.halfw = n_ch, nb, halfw
        self.lo = np.broadcast_to(np.asarray(lo, float), (n_ch,)).copy()
        self.hi = np.broadcast_to(np.asarray(hi, float), (n_ch,)).copy()
        t = np.linspace(0.0, 1.0, nb)
        self.centers = self.lo[:, None] + t[None, :] * (self.hi - self.lo)[:, None]
        spacing = (self.hi - self.lo) / (nb - 1)
        self.radius = halfw * spacing              # fall-off, in value units

    @property
    def size(self):
        return self.n_ch * self.nb

    # -- forward ----------------------------------------------------------
    def encode(self, V):
        """(n, n_ch) -> (n, size). A raised-cosine bump per channel."""
        V = np.atleast_2d(np.asarray(V, float))
        d = np.abs(np.clip(V, self.lo, self.hi)[:, :, None] - self.centers[None])
        r = self.radius[None, :, None]
        W = np.where(d < r, 0.5 * (1.0 + np.cos(np.pi * d / r)), 0.0)
        return W.reshape(len(V), self.size)

    def profiles(self, C):
        """(n, size) -> (n, n_ch, nb). Which cells of which channel."""
        return np.asarray(C).reshape(-1, self.n_ch, self.nb)

    # -- backward ---------------------------------------------------------
    def decode(self, C, mode="centroid"):
        """(n, size) -> (n, n_ch). `peak` is top-1; `centroid` is sub-cell."""
        P = self.profiles(C)
        i = P.argmax(axis=2)
        peak = np.take_along_axis(self.centers[None], i[:, :, None], 2)[:, :, 0]
        if mode == "peak":
            return peak
        top = P.max(axis=2, keepdims=True)
        w = np.maximum(P - 0.5 * top, 0.0)          # the run above half peak
        s = w.sum(axis=2)
        cen = (w * self.centers[None]).sum(axis=2) / np.maximum(s, EPS)
        return np.where(s > EPS, cen, peak)


def unit_rows(X):
    """Scale each row to unit length; return the rows and their norms."""
    n = np.linalg.norm(X, axis=1)
    ok = n > EPS
    U = np.zeros_like(X)
    U[ok] = X[ok] / n[ok, None]
    return U, n


def relu_split(X):
    """The ON/OFF split: signed (n, d) -> nonnegative (n, 2d)."""
    X = np.atleast_2d(np.asarray(X, float))
    return np.concatenate([np.maximum(X, 0.0), np.maximum(-X, 0.0)], axis=1)


def patches_of(X, p=4):
    """(n, 28, 28) -> (n * (28/p)^2, p*p) in row-major position order."""
    n, side = len(X), X.shape[1]
    g = side // p
    return (X.reshape(n, g, p, g, p).transpose(0, 1, 3, 2, 4)
             .reshape(n * g * g, p * p))
