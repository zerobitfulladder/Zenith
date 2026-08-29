"""A second layer that carries the wave — and what it is allowed to hear.

The one-layer result was that a single (x, y) pair cannot see the wave.
Nothing about the pair "x = 2.0, y = 1.6" says the curve is on its way
down; that is a property of how y moves as x moves, and it only exists
ACROSS pairs. So layer two is given several pairs at once.

  * A WINDOW is T taps spaced delta apart along x. Each tap is one
    channel of the window's code, and all T are OR'ed into one array
    with disjoint scatters. One vector is a whole PIECE OF CURVE.
  * Same hypercolumn, same competition, same geodesic learning. A
    template is now a stored shape: rising, cresting, falling.
  * Reading is the ordinary partial-cue read, scored against the
    template's norm over the cells actually present (the one-layer run
    showed a bare dot product is not safe here): write the taps you
    know, leave the taps in the hole empty, take the winner, read its
    cells at the empty taps.

WHAT EACH TAP SAYS is the interesting choice, and it is the whole
question of what one layer should send to the next:

  value-relative — the tap holds layer one's answered y MINUS the first
        known tap's y, as a bump. Overlapping, and translation-free in
        y: a descent is the same template wherever it sits.
  value-absolute — the tap holds the answered y itself, as a bump.
        Overlapping, but a descent high up and the same descent low
        down are different templates.
  winner (one-hot) — the tap holds layer one's WINNING MINICOLUMN, one
        cell of K1 lit. This is what a top-1 layer actually emits.
        Neighbouring x values share a cell only when they happen to
        share a winner; there is no graded overlap at all.
  graded — the tap holds layer one's top-m matches with magnitudes
        intact (the project's graded speech). Overlapping again.

Note what the two activation interfaces cannot do: a minicolumn's
identity binds an x to a y absolutely, so no message built out of
minicolumn identities can express "this shape, at whatever height".
The invariance is not lost in transmission — it is unrepresentable.
"""

import sys
from pathlib import Path

import numpy as np

# pop_regress lives in the sibling experiment folder regress/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regress"))
import pop_regress as P                      # noqa: E402


# ----------------------------------------------------- what a tap says ---
class ValueTaps:
    """A tap carries layer one's answer as a bump, absolutely or relatively."""

    def __init__(self, nb=96, halfw=8, dy=5.5, relative=True):
        self.relative = relative
        lo, hi = (-dy, dy) if relative else (P.Y_LO, P.Y_HI)
        self.centers = np.linspace(lo, hi, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        self.dim = nb
        self.name = "value-relative" if relative else "value-absolute"

    def reference(self, ys, known):
        if not self.relative:
            return 0.0
        return float(np.asarray(ys)[int(np.nonzero(known)[0][0])])

    def payload(self, i, y, ref):
        c = self.centers
        d = np.abs(c - np.clip(y - ref, c[0], c[-1]))
        w = np.where(d < self.radius,
                     0.5 * (1.0 + np.cos(np.pi * d / self.radius)), 0.0)
        keep = np.nonzero(w > 1e-3)[0]
        return keep, w[keep].astype(np.float32)

    def decode(self, prof, ref):
        return ref + P.sharpen(prof, self.centers)[1]


class ActivationTaps:
    """A tap carries layer one's own activity: its winner, or its top-m.

    topm=1 with magnitudes dropped is a one-hot — literally what a
    top-1 layer emits. topm>1 with magnitudes kept is graded speech.
    """

    def __init__(self, hc1, enc1, grid, topm=16, onehot=False):
        self.dim = hc1.n_boot
        self.name = "winner (one-hot)" if onehot else f"graded top-{topm}"
        Wy = hc1.W[:hc1.n_boot][:, enc1.pos["y"]]
        self.mini_y = np.array([P.sharpen(Wy[j], enc1.centers["y"])[1]
                                for j in range(hc1.n_boot)])
        self.table = []
        for x in grid:
            idx, vals = enc1.encode_sparse(float(x), None)
            s = hc1.scores(idx, vals, mode="masked")
            if onehot:
                self.table.append((np.array([int(np.argmax(s))]),
                                   np.ones(1, dtype=np.float32)))
            else:
                g = np.clip(s, 0.0, None)
                keep = np.argsort(-g)[:topm]
                keep = keep[g[keep] > 0]
                self.table.append((keep.astype(int),
                                   g[keep].astype(np.float32)))

    def reference(self, ys, known):
        return 0.0

    def payload(self, i, y, ref):
        return self.table[i]

    def decode(self, prof, ref):
        return float(self.mini_y[int(np.argmax(prof))])


# ------------------------------------------------------------ layer two ---
class WindowLayer:
    """A hypercolumn whose input is a window of taps."""

    def __init__(self, taps_code, k=384, taps=21, delta=0.2, size=None,
                 eta=0.05, seed=0):
        self.code, self.T, self.delta = taps_code, taps, delta
        self.dim = taps_code.dim
        need = taps * self.dim
        size = size or int(2 ** np.ceil(np.log2(max(2 * need, 4096))))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)      # disjoint, per the one-layer finding
        self.pos = [perm[i * self.dim:(i + 1) * self.dim] for i in range(taps)]
        self.size = size
        self.hc = P.Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def offsets(self):
        return (np.arange(self.T) - (self.T - 1) // 2) * self.delta

    def encode_sparse(self, idxs, ys, known=None):
        known = (np.ones(self.T, bool) if known is None
                 else np.asarray(known, bool))
        ys = np.asarray(ys, dtype=float)
        ref = self.code.reference(ys, known)
        cells, w = [], []
        for t in np.nonzero(known)[0]:
            c, v = self.code.payload(int(idxs[t]), ys[t], ref)
            cells.append(self.pos[t][c])
            w.append(v)
        cells, w = np.concatenate(cells), np.concatenate(w)
        uniq, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, w)
        return uniq.astype(np.int32), out

    def tap_profile(self, template, t):
        return template[self.pos[t]]

    def shape_image(self, template):
        """(dim, T): a template drawn as the piece of curve it stands for."""
        return np.stack([self.tap_profile(template, t)
                         for t in range(self.T)], axis=1)

    def learn(self, idxs, ys):
        self.hc.learn(*self.encode_sparse(idxs, ys))

    def complete(self, idxs, ys, known):
        """Fill the unknown taps from the winning template."""
        known = np.asarray(known, bool)
        ys = np.asarray(ys, dtype=float)
        ref = self.code.reference(ys, known)
        s = self.hc.scores(*self.encode_sparse(idxs, ys, known), mode="masked")
        w = int(np.argmax(s))
        out = ys.copy()
        prof = np.zeros((self.dim, self.T), dtype=np.float32)
        for t in range(self.T):
            pr = self.tap_profile(self.hc.row(w), t)
            prof[:, t] = pr
            if not known[t]:
                out[t] = self.code.decode(pr, ref)
        return out, w, prof, float(s[w])


# ------------------------------------------------------ the tap grid ---
def tap_grid(delta):
    """Half-delta grid, so every tap of every window lands on it exactly."""
    n = int(round((P.X_HI - P.X_LO) / (delta / 2))) + 1
    return np.linspace(P.X_LO, P.X_HI, n)


def l1_field(hc, enc, xs, mode="masked", read="graded", topm=16):
    """Layer one's answer at every grid point — layer two's input."""
    return P.predict(hc, enc, xs, mode=mode, read=read, topm=topm)["peak"]


def coverage(grid, xtr, delta):
    """A tap is KNOWN if layer one has training data within half a tap of it.

    This is the honest test-time signal — the model knows where it has
    seen data — and it is what marks out the hole, rather than the
    experimenter pointing at it.
    """
    d = np.abs(grid[:, None] - xtr[None, :]).min(axis=1)
    return d <= delta / 2


def window_indices(l2, grid):
    half = (l2.T - 1) // 2
    step = 2                                   # delta = 2 half-delta steps
    return np.arange(half * step, len(grid) - half * step), step, half


def taps_at(a, step, half, T):
    return a + (np.arange(T) - half) * step


def train_l2(l2, grid, field, known, n_windows=12000, seed=0):
    """Learn complete windows — only where every tap has real data behind it."""
    anchors, step, half = window_indices(l2, grid)
    ok = np.array([a for a in anchors
                   if known[taps_at(a, step, half, l2.T)].all()])
    rng = np.random.default_rng(seed)
    for a in rng.choice(ok, size=n_windows):
        ti = taps_at(int(a), step, half, l2.T)
        l2.learn(ti, field[ti])
    return ok


def bridge(l2, grid, field, known, gap):
    """Complete the hole: one window centred on it, rims known, middle blank."""
    _, step, half = window_indices(l2, grid)
    a = int(np.argmin(np.abs(grid - 0.5 * (gap[0] + gap[1]))))
    ti = taps_at(a, step, half, l2.T)
    out, w, prof, sc = l2.complete(ti, field[ti], known[ti])
    return {"x": grid[ti], "y": out, "known": known[ti], "winner": w,
            "profile": prof, "score": sc, "field": field[ti]}


def blank_control(l2, grid, field, known, n_unknown, seed=0, n=16):
    """The same completion where the shape IS in the training data.

    Blank the same number of middle taps in windows that sit inside the
    support. Separates "layer two cannot complete" from "layer two never
    saw this shape".
    """
    anchors, step, half = window_indices(l2, grid)
    rng = np.random.default_rng(seed)
    lo = (l2.T - n_unknown) // 2
    cand = np.array([a for a in anchors
                     if known[taps_at(a, step, half, l2.T)].all()])
    err = []
    for a in rng.choice(cand, size=min(n, len(cand)), replace=False):
        ti = taps_at(int(a), step, half, l2.T)
        m = np.ones(l2.T, bool)
        m[lo:lo + n_unknown] = False
        out, _, _, _ = l2.complete(ti, field[ti], m)
        err.append(out[~m] - P.target(grid[ti][~m]))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
