"""A convolutional layer one — shared templates, small receptive field.

The problem this fixes. In the earlier rig every layer-one template was
tied to one place on the x axis: there was no template meaning "a
descent", only "the descent at x = 1.4" and "the descent at x = -7.6",
and those two are strangers. Measured: two windows over the SAME shape
two periods apart shared 0 of 105 and 188 active templates, and their
codes correlated -0.005 — indistinguishable from unrelated windows. A
code built out of place-specific identities can only ever answer "where
am I", never "what is here", so nothing above it can transfer.

The fix is weight sharing. ONE hypercolumn, a small patch as its input,
applied at every x position. Its templates become a vocabulary of local
shapes with no location attached, and the same shape anywhere produces
the same code.

  * a PATCH is 5 samples of the curve spaced 0.2 apart — 0.8 wide, far
    too small to span the 1.6-wide hole, so layer one cannot cheat,
  * the patch is written relative to ITS OWN MEAN, so its height is
    factored out structurally rather than by a privileged reference tap,
  * each of the 5 samples is a channel of cells with the usual fall-off,
    disjoint positions, OR'ed into one array,
  * one hypercolumn learns patches drawn from every position.

Nothing underneath it is learned: a patch is read straight off the raw
training pairs by local averaging. Where there are no pairs, the sample
is simply missing.
"""

import sys
from pathlib import Path

import numpy as np

# pop_regress lives in the sibling experiment folder regress/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regress"))
import pop_regress as P                      # noqa: E402


# ------------------------------------------------------- the raw signal ---
def sample_raw(grid, xtr, ytr, halfwin=0.1):
    """(values, known) — what the training pairs say at each grid point."""
    val = np.zeros(len(grid))
    known = np.zeros(len(grid), bool)
    for i, x in enumerate(grid):
        m = np.abs(xtr - x) <= halfwin
        if m.any():
            val[i] = float(ytr[m].mean())
            known[i] = True
    return val, known


# ------------------------------------------------------ convolutional L1 ---
class PatchLayer:
    """One shared hypercolumn over a small patch, applied at every x."""

    def __init__(self, k=128, taps=5, step=2, nb=64, halfw=6, dy=2.0,
                 size=1024, eta=0.05, seed=0):
        self.T, self.step, self.nb = taps, step, nb
        self.centers = np.linspace(-dy, dy, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.pos = [perm[i * nb:(i + 1) * nb] for i in range(taps)]
        self.size = size
        self.hc = P.Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def offsets(self, grid):
        d = grid[1] - grid[0]
        return (np.arange(self.T) - (self.T - 1) // 2) * self.step * d

    def taps_at(self, i):
        return i + (np.arange(self.T) - (self.T - 1) // 2) * self.step

    def _bump(self, v):
        c = self.centers
        d = np.abs(c - np.clip(v, c[0], c[-1]))
        w = np.where(d < self.radius,
                     0.5 * (1.0 + np.cos(np.pi * d / self.radius)), 0.0)
        keep = np.nonzero(w > 1e-3)[0]
        return keep, w[keep].astype(np.float32)

    def encode_sparse(self, vals, mask=None, ref_mask=None):
        """A patch, written relative to its own mean.

        `ref_mask` selects which taps define that mean, and it must be
        the SAME set at training and at read time. If the reference is
        the mean of all 5 taps while learning but the mean of the 4
        surviving taps while reading, every query is silently shifted
        against the stored templates — for a falling patch the mean of
        the first four sits above the mean of all five, so the query
        rides high and matches templates that turn upward. Measured: it
        turned a walk across the hole from tracking into drifting away.
        """
        vals = np.asarray(vals, float)
        mask = np.ones(self.T, bool) if mask is None else np.asarray(mask, bool)
        rm = mask if ref_mask is None else np.asarray(ref_mask, bool)
        ref = float(vals[rm].mean())
        cells, w = [], []
        for t in np.nonzero(mask)[0]:
            c, v = self._bump(vals[t] - ref)
            cells.append(self.pos[t][c])
            w.append(v)
        cells, w = np.concatenate(cells), np.concatenate(w)
        uniq, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, w)
        return uniq.astype(np.int32), out, ref

    def valid(self, grid, known):
        half = (self.T - 1) // 2 * self.step
        return np.array([i for i in range(half, len(grid) - half)
                         if known[self.taps_at(i)].all()])

    def train(self, grid, val, known, n=20000, seed=0, ref_mask=None):
        ok = self.valid(grid, known)
        rng = np.random.default_rng(seed)
        for i in rng.choice(ok, size=n):
            idx, w, _ = self.encode_sparse(val[self.taps_at(int(i))],
                                           ref_mask=ref_mask)
            self.hc.learn(idx, w)
        return ok

    # -- what layer one says at a position -----------------------------
    def code(self, grid, val, i, topm=8):
        """(indices, magnitudes, ref) — graded speech over shared shapes."""
        idx, w, ref = self.encode_sparse(val[self.taps_at(int(i))])
        s = self.hc.scores(idx, w, mode="masked")
        g = np.clip(s, 0.0, None)
        keep = np.argsort(-g)[:topm]
        keep = keep[g[keep] > 0]
        return keep.astype(int), g[keep].astype(np.float32), ref

    def winner(self, grid, val, i):
        idx, w, _ = self.encode_sparse(val[self.taps_at(int(i))])
        return int(np.argmax(self.hc.scores(idx, w, mode="masked")))

    def template_patch(self, j):
        """The 5 relative values a template stands for."""
        row = self.hc.row(j)
        return np.array([P.sharpen(row[self.pos[t]], self.centers)[1]
                         for t in range(self.T)])

    def walk(self, grid, val, known, ref_mask):
        """Fill every unknown point, left to right, as a patch's last tap."""
        filled, done = val.copy(), known.copy()
        trace = []
        for q in range(len(grid)):
            if done[q]:
                continue
            ti = self.taps_at(q) - (self.T - 1) // 2 * self.step
            if ti.min() < 0 or ti.max() >= len(grid) or not done[ti[:-1]].all():
                continue
            m = np.array([True] * (self.T - 1) + [False])
            v = filled[ti]
            idx, w, _ = self.encode_sparse(v, m, ref_mask=ref_mask)
            j = int(np.argmax(self.hc.scores(idx, w, mode="masked")))
            shape = self.template_patch(j)
            lvl = float(np.mean(v[m] - shape[m]))
            filled[q] = shape[-1] + lvl
            done[q] = True
            trace.append((float(grid[q]), float(filled[q])))
        return filled, trace


# ---------------------------------------------------------- layer two ---
class CodeWindowLayer:
    """Layer two over layer one's CODES, not over its answers.

    Each tap of the window holds the graded code layer one emits at that
    position — a handful of shared shape-templates with their match
    strengths. Because the vocabulary is shared, the same curve shape
    anywhere produces the same code, so a window here overlaps another
    window by CONTENT. That is the property the place-bound layer one
    could not provide (its margin between same-shape and different-shape
    windows was exactly 0.000).

    Taps sit one grid step apart, so completed patches land on both
    parities of the grid and every missing point ends up constrained.
    """

    def __init__(self, dim, taps=61, step=1, k=384, size=8192, eta=0.05,
                 seed=0):
        self.dim, self.T, self.step = dim, taps, step
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.pos = [perm[i * dim:(i + 1) * dim] for i in range(taps)]
        self.size = size
        self.hc = P.Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def taps_at(self, i):
        return i + (np.arange(self.T) - self.T // 2) * self.step

    def encode_sparse(self, codes, mask=None):
        """codes: list of (indices, magnitudes) per tap; None where absent."""
        mask = (np.array([c is not None for c in codes]) if mask is None
                else np.asarray(mask, bool))
        cells, w = [], []
        for t in np.nonzero(mask)[0]:
            k, m = codes[t]
            cells.append(self.pos[t][k])
            w.append(m)
        cells, w = np.concatenate(cells), np.concatenate(w)
        uniq, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, w)
        return uniq.astype(np.int32), out

    def valid_anchors(self, n_grid, ok):
        half = self.T // 2 * self.step
        return np.array([i for i in range(half, n_grid - half)
                         if ok[self.taps_at(i)].all()])

    def train(self, codes, ok, n_grid, n=12000, seed=0):
        anchors = self.valid_anchors(n_grid, ok)
        rng = np.random.default_rng(seed)
        for i in rng.choice(anchors, size=n):
            ti = self.taps_at(int(i))
            self.hc.learn(*self.encode_sparse([codes[j] for j in ti]))
        return anchors

    def complete(self, codes, ok, centre):
        """Fill the taps with no layer-one code. Returns a template id each."""
        ti = self.taps_at(int(centre))
        mask = ok[ti]
        idx, w = self.encode_sparse([codes[j] if ok[j] else None for j in ti],
                                    mask)
        s = self.hc.scores(idx, w, mode="masked")
        j = int(np.argmax(s))
        row = self.hc.row(j)
        out = {}
        for t in np.nonzero(~mask)[0]:
            out[int(ti[t])] = int(np.argmax(row[self.pos[t]]))
        return out, j, float(s[j])

    def template_shapes(self, j, l1):
        """What one layer-two template stands for, tap by tap."""
        row = self.hc.row(j)
        return [l1.template_patch(int(np.argmax(row[self.pos[t]])))
                for t in range(self.T)]


# ------------------------------------------------------------- decode ---
def stitch(shapes, l1, n_grid, anchor_vals=None, anchor_idx=None):
    """Turn per-position patch SHAPES into absolute values, all at once.

    Every completed patch says how its 5 samples differ from one another
    but not where it sits. Overlapping patches must agree, so collect
    each patch's adjacent-sample differences as equations and solve the
    whole system in one least-squares step, pinned by whatever values are
    already known. No marching, so nothing accumulates drift.
    """
    pts = sorted({int(t) for p in shapes for t in l1.taps_at(p)
                  if 0 <= t < n_grid})
    col = {q: n for n, q in enumerate(pts)}
    rows, rhs = [], []
    for p, s in shapes.items():
        ti = l1.taps_at(p)
        for a in range(len(ti) - 1):
            q0, q1 = int(ti[a]), int(ti[a + 1])
            if q0 not in col or q1 not in col:
                continue
            r = np.zeros(len(pts))
            r[col[q1]], r[col[q0]] = 1.0, -1.0
            rows.append(r)
            rhs.append(float(s[a + 1] - s[a]))
    W = 10.0                                   # pin the known values hard
    for q, v in zip(anchor_idx, anchor_vals):
        if int(q) in col:
            r = np.zeros(len(pts))
            r[col[int(q)]] = W
            rows.append(r)
            rhs.append(W * float(v))
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return {q: float(sol[col[q]]) for q in pts}
