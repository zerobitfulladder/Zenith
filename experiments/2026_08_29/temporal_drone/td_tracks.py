r"""Per-signal tracks: every quantity gets its own hierarchy.

There is no input/output distinction. The two thrust values are tracks
exactly like the sensors — the only difference is that at inference we
are generating them rather than reading them, so their part of the top
layer's vector is blank and gets read back out. Same partial-cue read
used everywhere else in this project.

    signal        dx   dy   tilt   collective   differential
                  |    |     |         |             |
    layer 1     one hypercolumn per track, over 5 taps of THAT signal
                  |    |     |         |             |
    layer 2     one hypercolumn per track, over 7 layer-1 codes
                  |    |     |         |             |
    layer 3      \___ all five tracks' codes, OR'ed together ___/

Layer three's input comes only from the layer below it — nothing is
injected from outside the hierarchy.

TRACKS KEEP ABSOLUTE VALUES. In the wave rig each patch was written
relative to its own mean, so a shape could be reused at any height. Here
absolute position is exactly what says which way to tilt, so subtracting
it would discard the load-bearing signal. A track over 5 taps carries
the level AND the slope, so velocity and angular rate are still implicit
— they are the slope of position and angle — and nothing is lost by not
encoding them separately (measured: adding them does not move the
nearest-neighbour bound at all).

Why per-signal tracks at all — measured, no training involved, on the
nearest-neighbour bound for the steering command:

    raw 6 state channels        error 0.0420, still improving with data
    per-signal tracks           error 0.0336, flat
    collective, all tracks      0.2816
    collective, vertical only   0.2028

So the tracks are ~25% better at the capacity we can afford, though they
plateau: 8x more per-track capacity (48 -> 384) moved the floor from
0.0305 to 0.0303, i.e. not at all. Good here, would lose to raw channels
somewhere past ~10k stored moments.

Messages up are graded (top-m with magnitudes). Generation is top-1.
"""

import numpy as np

import sl_drone as W

TRACKS = ("dx", "dy", "tilt", "coll", "diff")
NT = len(TRACKS)
MOTOR = (3, 4)                      # coll, diff — generated, not sensed
SCALE = (8.0, 8.0, np.pi, None, 2.5)

T1, ST1 = 5, 3                      # layer 1: 5 taps, 3 ticks apart (240 ms)
T2, ST2 = 7, 6                      # layer 2: 7 codes, 6 ticks apart (720 ms)
HIST = (T2 - 1) * ST2 + (T1 - 1) * ST1 + 2
RING = (T2 - 1) * ST2 + 1

NB1 = 24
CEN1 = np.linspace(-1.0, 1.0, NB1)
RAD1 = 3 * (CEN1[1] - CEN1[0])


def sense(s, tgt):
    return np.array([np.clip((s[0] - tgt[0]) / 8.0, -1, 1),
                     np.clip((s[1] - tgt[1]) / 8.0, -1, 1),
                     np.clip(W.wrap(s[4]) / np.pi, -1, 1)], dtype=np.float32)


def motor_of(lv):
    """(left, right) levels -> normalised (collective, differential)."""
    a, b = W.LEVELS[lv[0]], W.LEVELS[lv[1]]
    return np.array([(0.5 * (a + b)) / W.TMAX * 2 - 1,
                     np.clip((0.5 * (b - a)) / 2.5, -1, 1)], dtype=np.float32)


def levels_of(coll_n, diff_n):
    c = (coll_n + 1) / 2 * W.TMAX
    d = diff_n * 2.5
    return (int(np.argmin(np.abs(W.LEVELS - np.clip(c - d, 0, W.TMAX)))),
            int(np.argmin(np.abs(W.LEVELS - np.clip(c + d, 0, W.TMAX)))))


def bump(v, cen=CEN1, rad=RAD1):
    d = np.abs(cen - np.clip(v, cen[0], cen[-1]))
    w = np.where(d < rad, 0.5 * (1.0 + np.cos(np.pi * d / rad)), 0.0)
    k = np.nonzero(w > 1e-3)[0]
    return k, w[k].astype(np.float32)


def sharpen(prof, cen):
    i = int(np.argmax(prof))
    half = 0.5 * float(prof[i])
    lo = hi = i
    while lo > 0 and prof[lo - 1] > half:
        lo -= 1
    while hi < len(cen) - 1 and prof[hi + 1] > half:
        hi += 1
    w = prof[lo:hi + 1] - half
    return float(cen[i]) if w.sum() <= 0 else float(
        (w * cen[lo:hi + 1]).sum() / w.sum())


class Hyper:
    """Competing templates; one winner learns by geodesic rotation."""

    def __init__(self, k, dim, eta=0.05, seed=0):
        self.W = np.zeros((k, dim), np.float32)
        self.k, self.dim, self.eta = k, dim, eta
        self.n = 0
        self.wins = np.zeros(k, np.int64)
        self.rng = np.random.default_rng(seed)

    def scores(self, idx, vals, floor=0.25):
        """Cosine between each template and the input, over the cells the
        input actually writes.

        Both norms are needed. Dividing only by the template's norm gives
        a number that is not a correlation — it ranged to +3.3 here — and
        then "keep the positive matches" is meaningless: centred templates
        with little mass in the queried cells produce a large negative
        baseline, every score comes out negative, and the graded message
        is empty. The floor still caps the amplification for templates
        that barely cover the query at all.
        """
        if self.n == 0:
            return np.zeros(0, np.float32)
        sub = self.W[:self.n][:, idx]
        nrm = np.linalg.norm(sub, axis=1)
        vn = float(np.linalg.norm(vals)) + 1e-9
        return (sub @ vals) / (np.maximum(nrm, floor * nrm.max() + 1e-9) * vn)

    def learn(self, idx, vals):
        m = float(vals.sum()) / self.dim
        n2 = float((vals.astype(np.float64) ** 2).sum()) - self.dim * m * m
        nn = float(np.sqrt(max(n2, 0.0)))
        if nn < 1e-9:
            return
        if self.n < self.k:
            x = np.zeros(self.dim, np.float32)
            x[idx] = vals
            w = x - float(x.mean())
            w += (0.02 / np.sqrt(self.dim)) * self.rng.standard_normal(
                self.dim).astype(np.float32)
            w -= float(w.mean())
            self.W[self.n] = w / (np.linalg.norm(w) + 1e-9)
            self.wins[self.n] += 1
            self.n += 1
            return
        raw = self.W[:, idx] @ vals
        i = int(np.argmax(raw))
        c = float(raw[i]) / nn
        if not (0.0 < c < 1.0):
            return
        th = self.eta * c
        tn = float(np.sqrt(max(1 - c * c, 1e-12)))
        a = float(np.cos(th) - c * np.sin(th) / tn)
        b = float(np.sin(th) / tn)
        row = self.W[i]
        row *= a
        row -= b * m / nn
        row[idx] += (b / nn) * vals
        self.wins[i] += 1
        if self.wins[i] % 128 == 0:
            row -= float(row.mean())
            row /= np.linalg.norm(row) + 1e-9


def graded(hc, idx, vals, m):
    s = hc.scores(idx, vals)
    if s.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    g = np.clip(s, 0.0, None)
    k = np.argsort(-g)[:m]
    k = k[g[k] > 0]
    return k.astype(np.int64), g[k].astype(np.float32)


class Track:
    """One signal's own two-layer hierarchy."""

    def __init__(self, k1=64, k2=128, topm=8, seed=0):
        self.k1, self.k2, self.topm = k1, k2, topm
        self.l1 = Hyper(k1, T1 * NB1, seed=seed)
        self.l2 = Hyper(k2, T2 * k1, seed=seed + 1)

    def enc1(self, win):
        cells, w = [], []
        for t in range(T1):
            k, v = bump(win[t])
            cells.append(t * NB1 + k)
            w.append(v)
        return np.concatenate(cells), np.concatenate(w)

    def enc2(self, codes):
        cells, w = [], []
        for t, (ci, cv) in enumerate(codes):
            if len(ci):
                cells.append(t * self.k1 + ci)
                w.append(cv)
        if not cells:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(cells), np.concatenate(w)

    # -- reading a stored template back down to a value ---------------
    def value_of_l1(self, i):
        row = self.l1.l1 if False else self.l1.W[i]
        return sharpen(row[(T1 - 1) * NB1:T1 * NB1], CEN1)

    def value_of_l2(self, j):
        row = self.l2.W[j]
        last = row[(T2 - 1) * self.k1: T2 * self.k1]
        return self.value_of_l1(int(np.argmax(last)))


class Stack:
    """Five tracks, and one top layer that binds their codes.

    Per tick: the sensor tracks advance and layer three is asked for a
    command with the motor tracks' slots left EMPTY. The winner's own
    motor slots are the answer — decoded down through the motor track
    (layer-2 template -> its last layer-1 code -> that template's last
    tap), the ordinary hardened read, one level at a time. Then the
    command is committed to history and the motor tracks advance too, so
    next tick they carry it as context.
    """

    def __init__(self, k1=64, k2=128, k3=4096, topm=8, eta=0.05, seed=0):
        self.tr = [Track(k1, k2, topm, seed=seed + 7 * c) for c in range(NT)]
        self.k2, self.topm = k2, topm
        self.l3 = Hyper(k3, NT * k2, eta=eta, seed=seed + 99)
        self.hist = np.zeros((HIST, NT), np.float32)
        self.rings = [[(np.zeros(0, np.int64), np.zeros(0, np.float32))]
                      * RING for _ in range(NT)]
        self.n = 0
        self.frozen = False
        self.every = 5

    def reset(self):
        self.hist[:] = 0.0
        self.rings = [[(np.zeros(0, np.int64), np.zeros(0, np.float32))]
                      * RING for _ in range(NT)]
        self.n = 0

    def ready(self):
        return self.n >= HIST

    def _advance(self, c, teach):
        idx = HIST - 1 - (T1 - 1 - np.arange(T1)) * ST1
        i, v = self.tr[c].enc1(self.hist[idx, c])
        if teach:
            self.tr[c].l1.learn(i, v)
        self.rings[c] = self.rings[c][1:] + [graded(self.tr[c].l1, i, v,
                                                    self.topm)]

    def _l2(self, c, teach):
        taps = [self.rings[c][-1 - (T2 - 1 - i) * ST2] for i in range(T2)]
        i, v = self.tr[c].enc2(taps)
        if len(i) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        if teach:
            self.tr[c].l2.learn(i, v)
        return graded(self.tr[c].l2, i, v, self.topm)

    def push_sensors(self, sv):
        self.hist[:-1] = self.hist[1:]
        self.hist[-1, :3] = sv
        self.hist[-1, 3:] = 0.0
        self.n += 1
        teach = (not self.frozen) and (self.n % self.every == 0)
        if self.n >= HIST:
            for c in range(3):
                self._advance(c, teach)

    def commit_motor(self, lv):
        self.hist[-1, 3:] = motor_of(lv)
        if self.n >= HIST:
            teach = (not self.frozen) and (self.n % self.every == 0)
            for c in MOTOR:
                self._advance(c, teach)

    def codes(self, which, teach=False):
        return {c: self._l2(c, teach) for c in which}

    def encode(self, codes):
        cells, w = [], []
        for c, (ci, cv) in codes.items():
            if len(ci) == 0:
                continue
            nrm = float(np.linalg.norm(cv))
            if nrm < 1e-9:
                continue
            cells.append(c * self.k2 + ci)
            w.append(cv * (np.sqrt(1.0 / NT) / nrm))
        if not cells:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        cells = np.concatenate(cells)
        w = np.concatenate(w).astype(np.float32)
        u, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(u), np.float32)
        np.maximum.at(out, inv, w)
        return u.astype(np.int32), out

    def act(self):
        if not self.ready() or self.l3.n == 0:
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        i, v = self.encode(self.codes((0, 1, 2)))
        if len(i) == 0:
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        s = self.l3.scores(i, v)
        j = int(np.argmax(s))                       # generation is top-1
        row = self.l3.W[j]
        vals = [self.tr[c].value_of_l2(
                int(np.argmax(row[c * self.k2:(c + 1) * self.k2])))
                for c in MOTOR]
        return levels_of(vals[0], vals[1]), float(s[j])

    def learn(self, teach=True):
        if not self.ready():
            return None
        i, v = self.encode(self.codes(range(NT), teach=teach))
        if len(i) == 0:
            return None
        self.l3.learn(i, v)
        return i, v


def load_policy(npz, cfg):
    """Hook for the shared viewer at the repo root."""
    st = Stack(k1=int(cfg["k1"]), k2=int(cfg["k2"]), k3=int(cfg["k3"]),
               topm=int(cfg["topm"]), seed=int(cfg.get("seed", 0)))
    for c in range(NT):
        st.tr[c].l1.W = npz[f"t{c}l1"]
        st.tr[c].l1.n = int(npz[f"t{c}n1"])
        st.tr[c].l2.W = npz[f"t{c}l2"]
        st.tr[c].l2.n = int(npz[f"t{c}n2"])
    st.l3.W = npz["l3"]
    st.l3.n = int(npz["l3n"])
    st.frozen = True

    def act(s, tgt, lv_prev):
        st.push_sensors(sense(s, tgt))
        lv, _ = st.act()
        st.commit_motor(lv)
        return lv
    return act
