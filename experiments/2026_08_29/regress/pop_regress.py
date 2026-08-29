"""Population-coded regression — the data, the encoder, the hypercolumn.

The question this rig asks: if a number is written down as a BLURRY
patch of cells instead of a single cell, does one hypercolumn become a
regressor? Overlap is the whole bet — two nearby numbers share most of
their active cells, so a value the layer has never seen still lands
mostly on top of values it has seen.

Everything is the single-layer drone's machinery, cut down to two
channels:

  * a value becomes a bump of adjacent cells with a fall-off (brightest
    at the value, tapering to nothing HALFW cells away),
  * each channel owns its own random scatter of NB positions inside ONE
    array, and the channels are OR'ed (element-wise max) into it,
  * ONE hypercolumn learns those vectors; minicolumns compete and only
    the winner learns (geodesic rotation, centred unit templates).

So one sparse vector says: "at this x, the answer was this y."
At test time the y cells are left empty, the x cells alone pick a
winner, and that minicolumn's own y cells are read back out.
"""

import numpy as np

# ----------------------------------------------------------------- data ---
X_LO, X_HI = -6.0, 6.0
Y_LO, Y_HI = -4.6, 4.6
GAP = (1.2, 2.8)        # cut out of training entirely: the interpolation test


def _wave_plus_trend(x):
    """Smooth, non-monotone, with a trend — the wave never exactly repeats."""
    return 1.5 * np.sin(1.3 * x) + 0.35 * x + 0.6 * np.cos(0.7 * x)


def _pure_wave(x):
    """Strictly periodic (period 2*pi/1.3 = 4.83) — the wave DOES repeat, so a
    shape memory can have seen the missing piece somewhere else."""
    return 1.8 * np.sin(1.3 * x) + 0.9 * np.sin(2.6 * x + 0.7)


TARGETS = {"wave_plus_trend": _wave_plus_trend, "pure_wave": _pure_wave}
_F = _wave_plus_trend


def target(x):
    return _F(x)


def configure(x_range=None, y_range=None, gap=None, target_name=None):
    """Point the rig at a different range / hole / function.

    The module constants are read by the encoder and by every picture, so
    setting them here keeps one source of truth for a run.
    """
    global X_LO, X_HI, Y_LO, Y_HI, GAP, _F
    if x_range is not None:
        X_LO, X_HI = x_range
    if y_range is not None:
        Y_LO, Y_HI = y_range
    if gap is not None:
        GAP = tuple(gap)
    if target_name is not None:
        _F = TARGETS[target_name]


def make_dataset(n_train=800, noise=0.08, seed=0, gap=None):
    """Training pairs drawn everywhere EXCEPT the gap, plus three probes."""
    gap = GAP if gap is None else gap
    rng = np.random.default_rng(seed)
    xs = []
    while len(xs) < n_train:
        c = rng.uniform(X_LO, X_HI, 2 * n_train)
        xs += [v for v in c if not (gap[0] <= v <= gap[1])]
    xtr = np.sort(np.array(xs[:n_train]))
    ytr = target(xtr) + rng.normal(0, noise, n_train)

    grid = np.linspace(X_LO, X_HI, 601)                  # for the pictures
    sup = rng.uniform(X_LO, X_HI, 20 * n_train)
    sup = np.sort(sup[(sup < gap[0]) | (sup > gap[1])][:400])
    gapx = np.sort(rng.uniform(gap[0], gap[1], 200))
    return xtr, ytr, grid, sup, gapx


def edge_hold(xs, gap=None):
    """The 'nearest neighbour has nothing better to say' answer: inside the
    hole, hold whatever f was at the nearer rim."""
    gap = GAP if gap is None else gap
    mid = 0.5 * (gap[0] + gap[1])
    return target(np.where(np.asarray(xs) < mid, gap[0], gap[1]))


# -------------------------------------------------------------- encoder ---
CHANNELS = ("x", "y")


class Encoder:
    """One bump per channel, scattered into one array, OR'ed together.

    The bump is a raised cosine measured in VALUE units, not cell units,
    so the pattern slides smoothly as the value moves — a number halfway
    between two cells lights both, and shifting x by a hair shifts the
    code by a hair. That is the whole source of generalization here.
    """

    def __init__(self, size=8192, nb=96, halfw=10, seed=0, disjoint=True):
        self.size, self.nb, self.halfw = size, nb, halfw
        self.centers = {"x": np.linspace(X_LO, X_HI, nb),
                        "y": np.linspace(Y_LO, Y_HI, nb)}
        # fall-off radius: halfw cells, expressed in the channel's units
        self.radius = {c: halfw * (self.centers[c][1] - self.centers[c][0])
                       for c in CHANNELS}
        rng = np.random.default_rng(seed)
        if disjoint:
            # One permutation carved up between the channels, so no two
            # cells ever land on the same array position. Independent
            # per-channel draws collide (3 times at these sizes) and a
            # single shared position puts a phantom peak on the y row of
            # every template whose x bump covers it — which the top-1
            # read then emits as the answer, confidently, for a whole
            # band of x. Measured: those collisions were the ONLY
            # catastrophic errors this rig ever made — RMSE 0.157 ->
            # 0.107 from this line alone at size 8192 (2 collisions),
            # and 0.675 -> 0.160 at size 4096 (3 collisions).
            perm = rng.permutation(size)
            self.pos = {c: perm[i * nb:(i + 1) * nb]
                        for i, c in enumerate(CHANNELS)}
        else:
            self.pos = {c: rng.choice(size, size=nb, replace=False)
                        for c in CHANNELS}

    def bump(self, ch, value):
        """(cells, weights) — brightest at the value, tapering with distance."""
        cent = self.centers[ch]
        r = self.radius[ch]
        d = np.abs(cent - np.clip(value, cent[0], cent[-1]))
        w = np.where(d < r, 0.5 * (1.0 + np.cos(np.pi * d / r)), 0.0)
        keep = np.nonzero(w > 1e-3)[0]
        return keep, w[keep].astype(np.float32)

    def encode_sparse(self, x, y=None):
        """(positions, values) — the OR'ed code, never materialised."""
        items = [("x", x)] if y is None else [("x", x), ("y", y)]
        idx, val = [], []
        for ch, v in items:
            cells, w = self.bump(ch, v)
            idx.append(self.pos[ch][cells])
            val.append(w)
        idx = np.concatenate(idx)
        val = np.concatenate(val)
        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, val)          # OR == max where they collide
        return uniq.astype(np.int32), out

    def encode(self, x, y=None):
        """Dense version (for the pictures)."""
        buf = np.zeros(self.size, dtype=np.float32)
        i, v = self.encode_sparse(x, y)
        buf[i] = v
        return buf

    def profiles(self, vec):
        """(2, nb): the x row and the y row of any vector in this space."""
        return np.stack([vec[self.pos[c]] for c in CHANNELS])

    def decode_x(self, template):
        return float(self.centers["x"][int(np.argmax(template[self.pos["x"]]))])

    def sharpen(self, prof):
        """(peak, centroid) — the blurry answer, and two ways to sharpen it."""
        return sharpen(prof, self.centers["y"])

def sharpen(prof, cent):
    """(peak, centroid) of a blurry profile.

    `peak` is the plain top-1 cell, which is what generation uses
    everywhere else in this project. `centroid` reads the same profile
    finer: centre of mass of the contiguous run of cells above half the
    peak, which recovers sub-cell precision.
    """
    i = int(np.argmax(prof))
    peak = float(cent[i])
    half = 0.5 * float(prof[i])
    lo = i
    while lo > 0 and prof[lo - 1] > half:
        lo -= 1
    hi = i
    while hi < len(cent) - 1 and prof[hi + 1] > half:
        hi += 1
    w = prof[lo:hi + 1] - half
    centroid = (float((w * cent[lo:hi + 1]).sum()) / float(w.sum())
                if w.sum() > 0 else peak)
    return peak, centroid


# --------------------------------------------------------- hypercolumn ---
class Hypercolumn:
    """Minicolumns compete; only the winner learns (geodesic rotation).

    Learning is verbatim the single-layer drone's layer. Templates are
    kept centred and unit-length, which makes scoring exact on the
    sparse code and the geodesic step a scale, a shift and a short add.

    Selection has one added option, because this rig exposed a problem
    the drone hides. The drone scores a partial query (sensory cells
    written, motor cells blank) with a bare dot product, which silently
    assumes every template carries the same norm inside the queried
    cells. It does not: a template that has become VAGUE about the
    missing half keeps its whole norm in the queried half and therefore
    outbids the templates that actually match. Dividing by the
    template's own norm over the queried cells — the same cosine rule,
    applied to the cells that are actually present — removes the bid.
    """

    def __init__(self, k, dim, eta=0.05, seed=0):
        self.k, self.dim, self.eta = k, dim, eta
        self.W = np.zeros((k, dim), dtype=np.float32)
        self.n_boot = 0
        self.wins = np.zeros(k, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def _stats(self, vals):
        m = float(vals.sum()) / self.dim
        n2 = float((vals.astype(np.float64) ** 2).sum()) - self.dim * m * m
        return m, float(np.sqrt(max(n2, 0.0)))

    def row(self, i):
        return self.W[i]

    def scores(self, idx, vals, mode="dot", floor=0.25):
        """Every minicolumn's bid for this (possibly partial) code.

        `floor` is the contrast floor on the masked rule. Dividing by a
        template's norm over the queried cells is right when every
        template has real mass there, but a template that holds almost
        NOTHING at those cells gets its tiny (often noise) overlap
        divided by a tiny norm and bids absurdly high. The floor caps
        the amplification at templates holding less than `floor` of the
        best-covered template's mass. floor=0 is the bare masked rule,
        floor=1 is the plain dot product.
        """
        sub = self.W[:self.n_boot][:, idx]
        raw = sub @ vals
        if mode == "masked":
            n = np.linalg.norm(sub, axis=1)
            raw = raw / np.maximum(n, floor * float(n.max()) + 1e-9)
        return raw

    def compete(self, idx, vals, mode="dot"):
        m, n = self._stats(vals)
        if n < 1e-9 or self.n_boot == 0:
            return -1, 0.0
        s = self.scores(idx, vals, mode)
        i = int(np.argmax(s))
        return i, float(s[i]) / (n if mode == "dot" else 1.0)

    def learn(self, idx, vals):
        m, n = self._stats(vals)
        if n < 1e-9:
            return -1
        if self.n_boot < self.k:                    # adopt
            x = np.zeros(self.dim, dtype=np.float32)
            x[idx] = vals
            noise = ((0.02 / np.sqrt(self.dim))
                     * self.rng.standard_normal(self.dim)).astype(np.float32)
            w = x - float(x.mean()) + noise
            w = w - float(w.mean())
            self.W[self.n_boot] = w / (float(np.linalg.norm(w)) + 1e-9)
            self.wins[self.n_boot] += 1
            self.n_boot += 1
            return self.n_boot - 1
        raw = self.W[:, idx] @ vals
        i = int(np.argmax(raw))
        c = float(raw[i]) / n
        if c <= 0.0 or c >= 1.0:
            return i
        th = self.eta * c
        tn = float(np.sqrt(max(1.0 - c * c, 1e-12)))
        a = float(np.cos(th) - c * np.sin(th) / tn)
        b = float(np.sin(th) / tn)
        row = self.W[i]
        row *= a
        row -= b * m / n
        row[idx] += (b / n) * vals
        self.wins[i] += 1
        if self.wins[i] % 128 == 0:                 # tame float drift
            row -= float(row.mean())
            row /= float(np.linalg.norm(row)) + 1e-9
        return i


# ------------------------------------------------------------ the model ---
def train(enc, xtr, ytr, k=256, eta=0.05, epochs=40, seed=1):
    """Learn the bound pairs: one code per (x, y), one winner per code."""
    hc = Hypercolumn(k, enc.size, eta=eta, seed=seed)
    rng = np.random.default_rng(seed)
    order = np.arange(len(xtr))
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            hc.learn(*enc.encode_sparse(float(xtr[i]), float(ytr[i])))
    return hc


def predict(hc, enc, xs, mode="dot", read="top1", topm=16):
    """x in, y out: compete on the x cells alone, read the winner's y cells.

    read="top1"   — the winner's own y cells (what generation does).
    read="graded" — the project's graded speech applied to the read: every
                    minicolumn with a positive match contributes its y
                    cells, magnitudes intact (topm caps how many).
    """
    xs = np.asarray(xs, dtype=float)
    ypos = enc.pos["y"]
    Wy = hc.W[:hc.n_boot][:, ypos]
    out = {"peak": np.zeros(len(xs)), "centroid": np.zeros(len(xs)),
           "winner": np.zeros(len(xs), dtype=np.int64),
           "match": np.zeros(len(xs)),
           "field": np.zeros((enc.nb, len(xs)), dtype=np.float32)}
    for j, x in enumerate(xs):
        idx, vals = enc.encode_sparse(float(x), None)
        if hc.n_boot == 0:
            continue
        s = hc.scores(idx, vals, mode)
        w = int(np.argmax(s))
        out["winner"][j] = w
        out["match"][j] = float(s[w])
        if read == "top1":
            prof = Wy[w]
        else:
            g = np.clip(s, 0.0, None)
            if topm and topm < len(g):
                keep = np.argpartition(-g, topm)[:topm]
                m = np.zeros_like(g)
                m[keep] = g[keep]
                g = m
            prof = g @ Wy
        out["field"][:, j] = prof
        out["peak"][j], out["centroid"][j] = enc.sharpen(prof)
    return out
