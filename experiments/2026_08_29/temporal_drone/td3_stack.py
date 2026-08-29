"""Three layers over time, each seeing roughly 3x the span below it.

    layer 1   N  ticks  (12 = 240 ms)   local motion shape
    layer 2   3N ticks  (36 = 720 ms)   a phrase of motion shapes
    layer 3   9N ticks  (90 = 1.8 s)    the whole manoeuvre — where the
                                        goal is, where we are in the
                                        approach, and what to command

Layers one and two are convolutional: one shared hypercolumn each,
applied at every tick, emitting a graded code. Layer three is the top —
it binds a window of layer-two codes, plus the absolute state now, to
the two thrusts.

WHY EACH GROUP IS NORMALISED. Measured on the two-layer version: the
code history carried 99.2% of the input vector's energy, the absolute
state 0.5%, and the motor command 0.3%. Codes carry raw match scores
(up to ~27); bumps are capped at 1.0. Two things follow, and both were
visible in the flying: template selection was blind to which side of the
target the drone was on, and — worse — the winner during LEARNING was
chosen almost without regard to the command, so one template absorbed
moments wanting opposite tilts and rotated to their average. Averaging a
left tilt and a right tilt gives symmetric thrust, which is exactly the
under-tilting that was measured (|L-R| 0.9 against the oracle's 2.0).

So every group is scaled to a designed share of the vector's energy.
Scaling a whole group by one constant leaves the relative magnitudes
inside it untouched — the graded message is preserved; only its
loudness relative to the other groups changes.

PARTIAL WINDOWS. Early in an episode the deeper windows are not full
yet. Rather than sit blind for 1.8 s, the missing taps are simply left
empty and the ordinary partial-cue read handles it — the same mechanism
used everywhere else in this project.
"""

import numpy as np

import sl_drone as W
from td_stack import (SIGNALS, NSIG, SENSOR_SIG, Hypercolumn, bump,  # noqa: F401
                      sharpen, norm_state, norm_motor, denorm_motor)

# ---- timescales, in ticks (DT = 0.02 s) ----------------------------------
S1, T1 = 3, 5          # layer 1: 5 taps, 3 apart  -> 12 ticks (240 ms)
S2, T2 = 6, 7          # layer 2: 7 taps, 6 apart  -> 36 ticks (720 ms)
S3, T3 = 15, 7         # layer 3: 7 taps, 15 apart -> 90 ticks (1.8 s)

HIST = (T1 - 1) * S1 + 2                 # raw ticks a layer-1 window needs
RING1 = (T2 - 1) * S2 + 1                # layer-1 codes layer 2 needs
RING2 = (T3 - 1) * S3 + 1                # layer-2 codes layer 3 needs

# ---- the command, encoded as collective + differential ------------------
# Measured on the previous encoding (one channel per rotor): the frozen
# read reproduced the differential with error sd 0.48 against a signal sd
# of 0.48 — signal-to-noise 1, i.e. no usable steering at all. Two causes,
# both fixed here.
#   * The differential was never read; it was the DIFFERENCE of two
#     independently read rotors, so their read errors added.
#   * It spanned ~2.9 cells of a 48-cell axis, because the axis covered
#     0-8 N while the quantity that steers has sd 0.48 N.
# So the two rotors are re-expressed as collective and differential, each
# with its own channel, and each axis is FOVEATED — cells packed where the
# quantity actually lives (around hover for collective, around zero for
# differential) and sparse out at the extremes.
NB_M = 48
DIFF_SPAN = 2.5


def _fov(n, lo_span, hi_span, centre=0.0, exp=1.7):
    u = np.linspace(-1.0, 1.0, n)
    return np.where(u < 0, centre - np.abs(u) ** exp * lo_span,
                    centre + np.abs(u) ** exp * hi_span)


COLL_C = _fov(NB_M, W.HOVER, W.TMAX - W.HOVER, centre=W.HOVER)
DIFF_C = _fov(NB_M, DIFF_SPAN, DIFF_SPAN)
MOTOR_C = (COLL_C, DIFF_C)


def bump_idx(centers, v, halfw=4):
    """Bump in index space — the axes are foveated, so cell spacing varies."""
    i = int(np.argmin(np.abs(centers - v)))
    k = np.clip(np.arange(i - halfw, i + halfw + 1), 0, len(centers) - 1)
    d = np.abs(k - i) / (halfw + 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * d))
    k, w = np.unique(k), w[np.unique(k, return_index=True)[1]]
    return k, w.astype(np.float32)


def levels_to_cd(lv):
    """(left, right) thrust levels -> (collective, differential)."""
    a, b = W.LEVELS[lv[0]], W.LEVELS[lv[1]]
    return 0.5 * (a + b), 0.5 * (b - a)


def cd_to_levels(c, d):
    """(collective, differential) -> the two thrust levels."""
    a = np.clip(c - d, 0.0, W.TMAX)
    b = np.clip(c + d, 0.0, W.TMAX)
    return (int(np.argmin(np.abs(W.LEVELS - a))),
            int(np.argmin(np.abs(W.LEVELS - b))))


# ---- how the layer-three vector is split --------------------------------
SHARE_CODES, SHARE_STATE, SHARE_MOTOR = 0.50, 0.25, 0.25


def _assemble(parts, size):
    """OR groups into one sparse vector, each scaled to its energy share."""
    cells, vals = [], []
    for pos, w, share in parts:
        if len(pos) == 0:
            continue
        n = float(np.linalg.norm(w))
        if n < 1e-9:
            continue
        cells.append(pos)
        vals.append(w * (np.sqrt(share) / n))
    if not cells:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    cells = np.concatenate(cells)
    vals = np.concatenate(vals).astype(np.float32)
    u, inv = np.unique(cells, return_inverse=True)
    out = np.zeros(len(u), dtype=np.float32)
    np.maximum.at(out, inv, vals)
    return u.astype(np.int32), out


class ConvLayer:
    """A shared hypercolumn over a window of taps, applied at every tick.

    Layer one's taps are raw signals (written relative to each signal's
    own mean over the window, so a template is a shape and not a place).
    Layer two's taps are the codes layer one emitted.
    """

    def __init__(self, taps, tap_dim, k, size, raw=False, nb=24, halfw=3,
                 span=0.8, eta=0.05, seed=0):
        self.T, self.dim, self.raw = taps, tap_dim, raw
        self.nb, self.centers = nb, np.linspace(-span, span, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        need = taps * tap_dim
        assert need <= size, f"{need} > {size}"
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.pos = [perm[i * tap_dim:(i + 1) * tap_dim] for i in range(taps)]
        self.size = size
        self.hc = Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def encode_raw(self, win):
        """win: (T, NSIG). Each signal minus its own mean over the window."""
        rel = win - win.mean(axis=0, keepdims=True)
        cells, w = [], []
        for c in range(NSIG):
            for t in range(self.T):
                k, v = bump(self.centers, self.radius, rel[t, c])
                cells.append(self.pos[t][c * self.nb + k])
                w.append(v)
        return _assemble([(np.concatenate(cells), np.concatenate(w), 1.0)],
                         self.size)

    def encode_codes(self, codes):
        """codes: list of (indices, magnitudes), one per tap; may be empty."""
        cells, w = [], []
        for t, (ci, cv) in enumerate(codes):
            if len(ci) == 0:
                continue
            cells.append(self.pos[t][ci])
            w.append(cv)
        if not cells:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        return _assemble([(np.concatenate(cells), np.concatenate(w), 1.0)],
                         self.size)

    def encode(self, x):
        return self.encode_raw(x) if self.raw else self.encode_codes(x)

    def learn(self, x):
        i, v = self.encode(x)
        if len(i):
            self.hc.learn(i, v)

    def code(self, x, topm):
        i, v = self.encode(x)
        if len(i) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        s = self.hc.scores(i, v)
        if s.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        g = np.clip(s, 0.0, None)
        keep = np.argsort(-g)[:topm]
        keep = keep[g[keep] > 0]
        return keep.astype(np.int64), g[keep].astype(np.float32)


class CommandLayer:
    """The top: layer-two codes + the state now -> the two thrusts."""

    def __init__(self, k2_dim, k=4096, nb=48, halfw=4, size=8192, eta=0.05,
                 seed=0):
        self.k2_dim, self.nb = k2_dim, nb
        self.centers = np.linspace(-1.0, 1.0, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        need = T3 * k2_dim + SENSOR_SIG * nb + 2 * NB_M
        assert need <= size, f"{need} > {size}"
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.code_pos = [perm[i * k2_dim:(i + 1) * k2_dim] for i in range(T3)]
        b = T3 * k2_dim
        self.state_pos = [perm[b + i * nb: b + (i + 1) * nb]
                          for i in range(SENSOR_SIG)]
        b += SENSOR_SIG * nb
        self.motor_pos = [perm[b + i * NB_M: b + (i + 1) * NB_M]
                          for i in range(2)]
        self.size = size
        self.hc = Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def encode(self, codes, st, motor=None):
        cc, cw = [], []
        for t, (ci, cv) in enumerate(codes):
            if len(ci):
                cc.append(self.code_pos[t][ci])
                cw.append(cv)
        sc, sw = [], []
        for i in range(SENSOR_SIG):
            k, v = bump(self.centers, self.radius, st[i])
            sc.append(self.state_pos[i][k])
            sw.append(v)
        parts = [(np.concatenate(cc) if cc else np.zeros(0, int),
                  np.concatenate(cw) if cw else np.zeros(0, np.float32),
                  SHARE_CODES),
                 (np.concatenate(sc), np.concatenate(sw), SHARE_STATE)]
        if motor is not None:                 # motor = (collective, diff)
            mc, mw = [], []
            for i in range(2):
                k, v = bump_idx(MOTOR_C[i], motor[i])
                mc.append(self.motor_pos[i][k])
                mw.append(v)
            parts.append((np.concatenate(mc), np.concatenate(mw),
                          SHARE_MOTOR))
        return _assemble(parts, self.size)

    def learn(self, codes, st, motor):
        i, v = self.encode(codes, st, motor)
        if len(i):
            self.hc.learn(i, v)

    def act(self, codes, st):
        if self.hc.n_boot == 0:
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        i, v = self.encode(codes, st, None)
        if len(i) == 0:
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        s = self.hc.scores(i, v)
        j = int(np.argmax(s))
        row = self.hc.row(j)
        c = sharpen(row[self.motor_pos[0]], MOTOR_C[0])
        d = sharpen(row[self.motor_pos[1]], MOTOR_C[1])
        return cd_to_levels(c, d), float(s[j])


def _empty():
    return (np.zeros(0, np.int64), np.zeros(0, np.float32))


class Stack3:
    """Three layers, the raw history, and one code ring per level."""

    def __init__(self, k1=128, k2=256, k3=4096, m1=6, m2=6, seed=0,
                 learn_every=5):
        # Ticks are 20 ms apart, so consecutive moments are near-duplicates.
        # Learning every one of them fills the memory with 4096 snapshots of
        # a handful of flights (layer three hit its capacity by tick 4096).
        # Learning one in `learn_every` gives adoption genuinely different
        # experiences to store. History still advances every tick — only
        # what gets LEARNED is subsampled.
        self.learn_every = learn_every
        self.l1 = ConvLayer(T1, 24 * NSIG, k1, 4096, raw=True, seed=seed)
        self.l2 = ConvLayer(T2, k1, k2, 2048, raw=False, seed=seed + 10)
        self.l3 = CommandLayer(k2, k=k3, seed=seed + 20)
        self.m1, self.m2 = m1, m2
        self.hist = np.zeros((HIST, NSIG), dtype=np.float32)
        self.ring1 = [_empty()] * RING1
        self.ring2 = [_empty()] * RING2
        self.n = 0
        self.frozen = False

    def reset_history(self):
        self.hist[:] = 0.0
        self.ring1 = [_empty()] * RING1
        self.ring2 = [_empty()] * RING2
        self.n = 0

    def _win1(self):
        idx = HIST - 1 - (T1 - 1 - np.arange(T1)) * S1
        win = np.empty((T1, NSIG), dtype=np.float32)
        win[:, :SENSOR_SIG] = self.hist[idx, :SENSOR_SIG]
        win[:, SENSOR_SIG:] = self.hist[idx - 1, SENSOR_SIG:]
        return win

    def push(self, st, lv_prev):
        self.hist[:-1] = self.hist[1:]
        self.hist[-1, :SENSOR_SIG] = st
        self.hist[-1, SENSOR_SIG:] = norm_motor(lv_prev)
        self.n += 1
        if self.n < HIST:
            return
        win = self._win1()
        teach = (not self.frozen) and (self.n % self.learn_every == 0)
        if teach:
            self.l1.learn(win)
        self.ring1 = self.ring1[1:] + [self.l1.code(win, self.m1)]
        taps1 = [self.ring1[-1 - (T2 - 1 - i) * S2] for i in range(T2)]
        if teach:
            self.l2.learn(taps1)
        self.ring2 = self.ring2[1:] + [self.l2.code(taps1, self.m2)]

    def codes_now(self):
        return [self.ring2[-1 - (T3 - 1 - i) * S3] for i in range(T3)]

    def state_now(self):
        return self.hist[-1, :SENSOR_SIG]

    def ready(self):
        return self.n >= HIST + 1

    def act(self):
        if not self.ready():
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        return self.l3.act(self.codes_now(), self.state_now())

    def learn_live(self, lv):
        if not self.ready():
            return None
        c, s, m = (self.codes_now(), self.state_now().copy(),
                   np.array(levels_to_cd(lv), dtype=np.float32))
        self.l3.learn(c, s, m)
        return c, s, m

    def learn_replay(self, moment):
        self.l3.learn(*moment)


def load_policy(npz, cfg):
    """Hook for the shared viewer at the repo root."""
    st = Stack3(k1=int(cfg["k1"]), k2=int(cfg["k2"]), k3=int(cfg["k3"]),
                seed=int(cfg.get("seed", 0)))
    for lay, a, b in ((st.l1, "W1", "n1"), (st.l2, "W2", "n2"),
                      (st.l3, "W3", "n3")):
        lay.hc.W = npz[a]
        lay.hc.n_boot = int(npz[b])
    st.frozen = True

    def act(s, tgt, lv_prev):
        st.push(norm_state(s, tgt), lv_prev)
        return st.act()[0]
    return act
