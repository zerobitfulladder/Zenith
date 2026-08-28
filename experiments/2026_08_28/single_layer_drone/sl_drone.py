"""Single-layer sparse-OR drone — world, oracle, encoder, layer.

The user's architecture, self-contained (no imports from earlier rigs):

  * Sensing is a full IMU plus its integrations, and the target offset:
        dx, dy      target offset          (foveated cells)
        vx, vy      velocity (integrated)  (uniform cells)
        ax, ay      acceleration (raw)     (uniform cells)
        tilt        angle (integrated)     (foveated ring)
        gyro        angular rate (raw)     (uniform cells)
  * Every channel is encoded the same way: a bump of ACTIVE adjacent
    cells with a sharp fall-off, so near values share most of their
    bits and far values share none. Each channel owns its own random
    scatter of positions inside ONE giant array, and every channel is
    OR'ed (element-wise max) into it. Collisions are possible and
    tolerated; the array is sized so they stay rare.
  * The two motor commands (left, right thrust) are encoded in exactly
    the same way and OR'ed into the SAME giant array. So one sparse
    vector says: "in this state, these thrusts were commanded."
  * ONE hypercolumn learns those vectors. No tracks, no cascade, no
    tally, no second layer.
  * At inference the sensory bits are written and the motor bits are
    left empty; the winning minicolumn's own motor bits are read back
    out and drive the thrusters.
"""

import numpy as np

# ---------------------------------------------------------------- world ---
M, GRAV, ARM_L, INERT, DT = 1.0, 9.8, 0.2, 0.02, 0.02
TMAX = 8.0
HOVER = GRAV * M / 2.0
NLEV = 13
_u = np.linspace(-1.0, 1.0, NLEV)
LEVELS = np.where(_u < 0, HOVER - np.abs(_u) ** 1.7 * HOVER,
                  HOVER + np.abs(_u) ** 1.7 * (TMAX - HOVER))
EP_CAP = 800


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def accel(s, lv):
    """World-frame acceleration the IMU would report."""
    tt = LEVELS[lv[0]] + LEVELS[lv[1]]
    return -tt * np.sin(s[4]) / M, tt * np.cos(s[4]) / M - GRAV


def physics(s, lv):
    x, y, xd, yd, ph, phd = s
    t1, t2 = LEVELS[lv[0]], LEVELS[lv[1]]
    xa, ya = accel(s, lv)
    pa = ARM_L * (t2 - t1) / INERT
    return np.array([x + DT * xd, y + DT * yd, xd + DT * xa,
                     yd + DT * ya, ph + DT * phd, phd + DT * pa])


def any_init(rng):
    return (np.array([rng.uniform(-4, 4), rng.uniform(-4, 4),
                      rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)]),
            np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)]))


def at_goal(s, tgt):
    return (np.hypot(s[0] - tgt[0], s[1] - tgt[1]) < 0.25
            and np.hypot(s[2], s[3]) < 0.3 and abs(wrap(s[4])) < 0.17)


# --------------------------------------------------------------- oracle ---
PHIMAX, DMAX = 0.6, 2.5
GAINS = (0.15, 0.3, 2.0, 0.6, 1.2, 1.8)     # kpx, kdx, ka, kda, ky, kvy


def teacher(s, tgt):
    """Cascaded PD autopilot -> the two thrust levels (rate ~1.00)."""
    kpx, kdx, ka, kda, ky, kvy = GAINS
    ex, ey = s[0] - tgt[0], s[1] - tgt[1]
    ph = wrap(s[4])
    phi_des = np.clip(kpx * ex + kdx * s[2], -PHIMAX, PHIMAX)
    coll = np.clip(HOVER - ky * ey - kvy * s[3], 0.3, TMAX)
    diff = np.clip(ka * (phi_des - ph) - kda * s[5], -DMAX, DMAX)
    t1 = np.clip(coll - diff, 0, TMAX)
    t2 = np.clip(coll + diff, 0, TMAX)
    return (int(np.argmin(np.abs(LEVELS - t1))),
            int(np.argmin(np.abs(LEVELS - t2))))


# -------------------------------------------------------------- encoder ---
NB = 128                       # cells per channel
ACTIVE = np.array([0.08, 0.40, 1.0, 0.40, 0.08], dtype=np.float32)
HALF = len(ACTIVE) // 2


def _fov(n, lo, hi):
    u = np.linspace(-1.0, 1.0, n)
    span = max(abs(lo), abs(hi))
    return np.sign(u) * np.abs(u) ** 1.7 * span


CHANNELS = [
    ("dx", -8.0, 8.0, "fov"),
    ("dy", -8.0, 8.0, "fov"),
    ("vx", -5.0, 5.0, "lin"),
    ("vy", -5.0, 5.0, "lin"),
    ("ax", -20.0, 20.0, "lin"),
    ("ay", -20.0, 20.0, "lin"),
    ("tilt", -np.pi, np.pi, "ring"),
    ("gyro", -8.0, 8.0, "lin"),
]
MOTOR_CH = ["mL", "mR"]
ALL_CH = [c[0] for c in CHANNELS] + MOTOR_CH


def _centers(kind, lo, hi):
    if kind == "lin":
        return np.linspace(lo, hi, NB)
    if kind == "fov":
        return _fov(NB, lo, hi)
    u = (np.arange(NB) + 0.5) / NB * 2 - 1        # ring, foveated at 0
    return np.sign(u) * np.pi * np.abs(u) ** 1.7


CENTERS = {n: _centers(k, lo, hi) for n, lo, hi, k in CHANNELS}
KINDS = {n: k for n, lo, hi, k in CHANNELS}
# motor channels: cells follow the (foveated) thrust ladder shape
_mu = np.linspace(-1.0, 1.0, NB)
MCENT = np.where(_mu < 0, HOVER - np.abs(_mu) ** 1.7 * HOVER,
                 HOVER + np.abs(_mu) ** 1.7 * (TMAX - HOVER))
for m in MOTOR_CH:
    CENTERS[m] = MCENT
    KINDS[m] = "lin"


class Encoder:
    """Bump-per-channel, scattered into one giant array, OR'ed together."""

    def __init__(self, size, seed=0):
        self.size = size
        rng = np.random.default_rng(seed)
        self.pos = {c: rng.choice(size, size=NB, replace=False)
                    for c in ALL_CH}
        self.motor_pos = np.concatenate([self.pos[m] for m in MOTOR_CH])

    def _cells(self, ch, value):
        cent = CENTERS[ch]
        if KINDS[ch] == "ring":
            i = int(np.argmin(np.abs(wrap(value - cent))))
            cells = (np.arange(i - HALF, i + HALF + 1)) % NB
        else:
            i = int(np.argmin(np.abs(np.clip(value, cent[0], cent[-1])
                                     - cent)))
            cells = np.clip(np.arange(i - HALF, i + HALF + 1), 0, NB - 1)
        return cells

    def _items(self, sens, motors):
        items = [(c, sens[c]) for c, _, _, _ in CHANNELS]
        if motors is not None:
            items += [(MOTOR_CH[0], LEVELS[motors[0]]),
                      (MOTOR_CH[1], LEVELS[motors[1]])]
        return items

    def encode_sparse(self, sens, motors=None):
        """(positions, values) — the OR'ed code, never materialised."""
        idx = np.concatenate([self.pos[ch][self._cells(ch, v)]
                              for ch, v in self._items(sens, motors)])
        val = np.tile(ACTIVE, len(self._items(sens, motors)))
        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, val)        # OR == max on collisions
        return uniq.astype(np.int32), out

    def encode(self, sens, motors=None):
        """Dense version (for pictures and the viewer)."""
        buf = np.zeros(self.size, dtype=np.float32)
        idx, val = self.encode_sparse(sens, motors)
        buf[idx] = val
        return buf

    def read_motors(self, template):
        """Project a template's motor bits back to two thrust levels."""
        out = []
        for m in MOTOR_CH:
            prof = template[self.pos[m]]
            val = float(CENTERS[m][int(np.argmax(prof))])
            out.append(int(np.argmin(np.abs(LEVELS - val))))
        return tuple(out)


def sense(s, tgt, lv_prev):
    ax, ay = accel(s, lv_prev)
    return {"dx": s[0] - tgt[0], "dy": s[1] - tgt[1],
            "vx": s[2], "vy": s[3], "ax": ax, "ay": ay,
            "tilt": wrap(s[4]), "gyro": s[5]}


# ---------------------------------------------------------- hypercolumn ---
class Hypercolumn:
    """Minicolumns compete; only the winner learns (geodesic rotation).

    Sparse-native, so it runs the same on CPU or GPU. Templates are kept
    centred (sum zero), which buys two exact shortcuts:

      score_i = <template_i, raw code> / ||centred code||
                — only the ~50 active positions are ever touched;
      the geodesic step becomes  w <- a*w - (b*m/n) + (b/n)*code
                — a scale, a scalar shift, and a 50-element add, with
                  a = cos(th) - c*sin(th)/sqrt(1-c^2), b = sin(th)/
                  sqrt(1-c^2). No dense query vector is ever built, and
                  the row stays exactly centred and unit-length.
    """

    def __init__(self, k, dim, eta=0.05, seed=0, consol=0.0, gpu=False):
        self.k, self.dim, self.eta = k, dim, eta
        # consol > 0: a minicolumn's step shrinks as it wins more
        # (plastic when young, consolidated when experienced) — the
        # antidote to a fixed hypercolumn re-tiling itself around
        # whatever states the pupil happens to be visiting lately.
        self.consol = consol
        self.gpu = gpu
        if gpu:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        self.W = self.xp.zeros((k, dim), dtype=np.float32)
        self.n_boot = 0
        self.wins = np.zeros(k, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    # -- helpers ----------------------------------------------------
    def _stats(self, vals):
        m = float(vals.sum()) / self.dim
        n2 = float((vals.astype(np.float64) ** 2).sum()) - self.dim * m * m
        return m, float(np.sqrt(max(n2, 0.0)))

    def _dev(self, a):
        return self.xp.asarray(a) if self.gpu else a

    def row(self, i):
        r = self.W[i]
        return r.get() if self.gpu else r

    def weights_numpy(self):
        return self.W.get() if self.gpu else self.W

    # -- interface --------------------------------------------------
    def compete(self, idx, vals):
        m, n = self._stats(vals)
        if n < 1e-9 or self.n_boot == 0:
            return -1, 0.0
        raw = self.W[:self.n_boot][:, self._dev(idx)] @ self._dev(vals)
        i = int(self.xp.argmax(raw))
        return i, float(raw[i]) / n

    def learn(self, idx, vals):
        xp = self.xp
        m, n = self._stats(vals)
        if n < 1e-9:
            return -1
        gi, gv = self._dev(idx), self._dev(vals)
        if self.n_boot < self.k:                    # adopt
            x = xp.zeros(self.dim, dtype=np.float32)
            x[gi] = gv
            noise = self._dev(((0.02 / np.sqrt(self.dim))
                               * self.rng.standard_normal(self.dim)
                               ).astype(np.float32))
            w = x - float(x.mean()) + noise
            w = w - float(w.mean())
            self.W[self.n_boot] = w / (float(xp.linalg.norm(w)) + 1e-9)
            self.wins[self.n_boot] += 1
            self.n_boot += 1
            return self.n_boot - 1
        raw = self.W[:, gi] @ gv
        i = int(xp.argmax(raw))
        c = float(raw[i]) / n
        if c <= 0.0 or c >= 1.0:
            return i
        th = self.eta * c
        if self.consol > 0:
            th /= 1.0 + self.wins[i] / self.consol
        tn = float(np.sqrt(max(1.0 - c * c, 1e-12)))
        a = float(np.cos(th) - c * np.sin(th) / tn)
        b = float(np.sin(th) / tn)
        row = self.W[i]
        row *= a
        row -= b * m / n
        row[gi] += (b / n) * gv
        self.wins[i] += 1
        if self.wins[i] % 128 == 0:                 # tame float drift
            row -= float(row.mean())
            row /= float(xp.linalg.norm(row)) + 1e-9
        return i

    def nudge(self, i, idx, vals, scale):
        """Rotate minicolumn i TOWARD the code (scale > 0) or AWAY from
        it (scale < 0). The third factor of a three-factor rule: which
        minicolumn, what happened, and whether it went well."""
        xp = self.xp
        if i < 0 or scale == 0.0:
            return
        m, n = self._stats(vals)
        if n < 1e-9:
            return
        gi, gv = self._dev(idx), self._dev(vals)
        row = self.W[i]
        c = float(row[gi] @ gv) / n
        if c <= 0.0 or c >= 1.0:
            return
        th = self.eta * float(scale) * c             # signed step
        if self.consol > 0:
            th /= 1.0 + self.wins[i] / self.consol
        tn = float(np.sqrt(max(1.0 - c * c, 1e-12)))
        a = float(np.cos(th) - c * np.sin(th) / tn)
        b = float(np.sin(th) / tn)
        row *= a
        row -= b * m / n
        row[gi] += (b / n) * gv
        self.nudges = getattr(self, "nudges", 0) + 1
        if self.nudges % 256 == 0:          # renormalise rarely: the two
            row -= float(row.mean())        # reductions are half the cost
            row /= float(xp.linalg.norm(row)) + 1e-9
