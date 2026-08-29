"""A two-layer temporal stack for the drone.

The single-layer rig bound one sensor snapshot to one thrust command.
A snapshot cannot say which way things are going — "hovering at
dx=+1.2" looks identical whether the drone is converging on the target
or sailing past it — so the memory could only ever answer with whatever
it stored at the nearest-looking state.

Here the convolution is over TIME instead of over x:

  signals   dx, dy (offset to the target), tilt, and the two thrusts.
            Velocity, angular rate and acceleration are NOT encoded:
            they are the slope and curvature of these across the window,
            so the shape already carries them. The two thrust signals
            lag one tick, so the command being predicted never appears
            in its own input.
  layer 1   ONE shared hypercolumn over a short window of recent ticks,
            applied at every tick. Each signal is written relative to
            its own mean over the window, so layer one learns the SHAPE
            of the last quarter second, free of where it happened.
  layer 2   a longer window of layer-one codes, OR'ed together with the
            absolute state now and the thrusts to command. Inference is
            the usual partial read: write the history and the state,
            leave the thrust cells empty, take the winner's own.

Replay. A competitive memory has no dataset — it tracks the recent
stream, and the pupil's stream is mostly emergencies, which is how the
single-layer rig drifted to full power while its retrieval quality
stayed flat. A reservoir keeps a uniform sample of every moment ever
lived and re-presents a few each tick, which is what the batched rebuild
was doing implicitly.
"""

import numpy as np

import sl_drone as W

# ------------------------------------------------------------- signals ---
# name, scale (the value that maps to 1.0)
SIGNALS = (("dx", 8.0), ("dy", 8.0), ("tilt", np.pi),
           ("mL", None), ("mR", None))
NSIG = len(SIGNALS)
SENSOR_SIG = 3                      # dx, dy, tilt — the ones read from state
STRIDE = 3                          # ticks between taps (60 ms)
T1 = 5                              # taps in a layer-one window (240 ms)
T2 = 7                              # layer-one codes in a layer-two window
HIST = (T2 - 1) * STRIDE + (T1 - 1) * STRIDE + 2


def norm_state(s, tgt):
    """(dx, dy, tilt) as numbers in [-1, 1]."""
    return np.array([np.clip((s[0] - tgt[0]) / 8.0, -1, 1),
                     np.clip((s[1] - tgt[1]) / 8.0, -1, 1),
                     np.clip(W.wrap(s[4]) / np.pi, -1, 1)])


def norm_motor(lv):
    """Two thrust levels as numbers in [-1, 1]."""
    return np.array([W.LEVELS[lv[0]] / W.TMAX * 2 - 1,
                     W.LEVELS[lv[1]] / W.TMAX * 2 - 1])


def denorm_motor(u):
    t = (np.clip(u, -1, 1) + 1) / 2 * W.TMAX
    return int(np.argmin(np.abs(W.LEVELS - t)))


def bump(centers, radius, v):
    d = np.abs(centers - np.clip(v, centers[0], centers[-1]))
    w = np.where(d < radius, 0.5 * (1.0 + np.cos(np.pi * d / radius)), 0.0)
    k = np.nonzero(w > 1e-3)[0]
    return k, w[k].astype(np.float32)


def sharpen(prof, centers):
    i = int(np.argmax(prof))
    half = 0.5 * float(prof[i])
    lo = hi = i
    while lo > 0 and prof[lo - 1] > half:
        lo -= 1
    while hi < len(centers) - 1 and prof[hi + 1] > half:
        hi += 1
    w = prof[lo:hi + 1] - half
    if w.sum() <= 0:
        return float(centers[i])
    return float((w * centers[lo:hi + 1]).sum() / w.sum())


# --------------------------------------------------------- hypercolumn ---
class Hypercolumn:
    """Competing templates; one winner learns by geodesic rotation."""

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

    def scores(self, idx, vals, mode="masked", floor=0.25):
        if self.n_boot == 0:
            return np.zeros(0, dtype=np.float32)
        sub = self.W[:self.n_boot][:, idx]
        raw = sub @ vals
        if mode == "masked":
            n = np.linalg.norm(sub, axis=1)
            raw = raw / np.maximum(n, floor * float(n.max()) + 1e-9)
        return raw

    def learn(self, idx, vals):
        m, n = self._stats(vals)
        if n < 1e-9:
            return -1
        if self.n_boot < self.k:
            x = np.zeros(self.dim, dtype=np.float32)
            x[idx] = vals
            noise = ((0.02 / np.sqrt(self.dim))
                     * self.rng.standard_normal(self.dim)).astype(np.float32)
            w = x - float(x.mean()) + noise
            w -= float(w.mean())
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
        if self.wins[i] % 128 == 0:
            row -= float(row.mean())
            row /= float(np.linalg.norm(row)) + 1e-9
        return i


# ------------------------------------------- layer one: shapes in time ---
class MotionLayer:
    """One shared hypercolumn over a short window of recent ticks."""

    def __init__(self, k=128, nb=24, halfw=3, span=0.8, size=2048,
                 eta=0.05, seed=0):
        self.nb = nb
        self.centers = np.linspace(-span, span, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.pos = [perm[i * nb:(i + 1) * nb] for i in range(NSIG * T1)]
        self.size = size
        self.hc = Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def encode(self, win):
        """win: (T1, NSIG) normalised. Each signal minus its own mean."""
        rel = win - win.mean(axis=0, keepdims=True)
        cells, w = [], []
        for c in range(NSIG):
            for t in range(T1):
                k, v = bump(self.centers, self.radius, rel[t, c])
                cells.append(self.pos[c * T1 + t][k])
                w.append(v)
        cells, w = np.concatenate(cells), np.concatenate(w)
        u, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(u), dtype=np.float32)
        np.maximum.at(out, inv, w)
        return u.astype(np.int32), out

    def learn(self, win):
        self.hc.learn(*self.encode(win))

    def code(self, win, topm=6):
        idx, w = self.encode(win)
        s = self.hc.scores(idx, w)
        if s.size == 0:                     # nothing learned yet — say nothing
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        g = np.clip(s, 0.0, None)
        keep = np.argsort(-g)[:topm]
        keep = keep[g[keep] > 0]
        return keep.astype(np.int64), g[keep].astype(np.float32)


# ------------------------- layer two: code history + state -> thrusts ---
class CommandLayer:
    """Codes of the recent past, the state now, and the thrusts to send."""

    def __init__(self, k1, k=2048, nb=48, halfw=5, size=4096, eta=0.05,
                 seed=0):
        self.k1, self.nb = k1, nb
        self.centers = np.linspace(-1.0, 1.0, nb)
        self.radius = halfw * (self.centers[1] - self.centers[0])
        need = T2 * k1 + (SENSOR_SIG + 2) * nb
        assert need <= size, f"need {need} slots, have {size}"
        rng = np.random.default_rng(seed)
        perm = rng.permutation(size)
        self.code_pos = [perm[i * k1:(i + 1) * k1] for i in range(T2)]
        base = T2 * k1
        self.state_pos = [perm[base + i * nb: base + (i + 1) * nb]
                          for i in range(SENSOR_SIG)]
        base += SENSOR_SIG * nb
        self.motor_pos = [perm[base + i * nb: base + (i + 1) * nb]
                          for i in range(2)]
        self.size = size
        self.hc = Hypercolumn(k, size, eta=eta, seed=seed + 1)

    def encode(self, codes, st, motor=None):
        cells, w = [], []
        for t, (ci, cv) in enumerate(codes):
            cells.append(self.code_pos[t][ci])
            w.append(cv)
        for i in range(SENSOR_SIG):
            k, v = bump(self.centers, self.radius, st[i])
            cells.append(self.state_pos[i][k])
            w.append(v)
        if motor is not None:
            for i in range(2):
                k, v = bump(self.centers, self.radius, motor[i])
                cells.append(self.motor_pos[i][k])
                w.append(v)
        cells, w = np.concatenate(cells), np.concatenate(w)
        u, inv = np.unique(cells, return_inverse=True)
        out = np.zeros(len(u), dtype=np.float32)
        np.maximum.at(out, inv, w)
        return u.astype(np.int32), out

    def learn(self, codes, st, motor):  # noqa: D401
        self.hc.learn(*self.encode(codes, st, motor))

    def act(self, codes, st):
        """Partial read: no motor cells written, winner's own are the answer."""
        if self.hc.n_boot == 0:
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        idx, w = self.encode(codes, st, None)
        s = self.hc.scores(idx, w)
        i = int(np.argmax(s))
        row = self.hc.row(i)
        lv = tuple(denorm_motor(sharpen(row[self.motor_pos[j]], self.centers))
                   for j in range(2))
        return lv, float(s[i])


# ------------------------------------------------------------- replay ---
class Reservoir:
    """A uniform sample of every moment ever lived, in fixed memory.

    Reservoir sampling: fill the first `cap` slots, then for moment k
    overwrite a random slot with probability cap/k. What is stored is a
    uniform sample of the WHOLE history, not the recent part — a plain
    ring buffer would be just as dominated by the pupil's latest
    emergencies as the memory already is.

    It stores layer-one CODES rather than raw ticks, so replaying a
    moment costs one learn instead of re-running layer one. That is
    sound only because layer one is frozen once it has a vocabulary
    (see `run_td.py`); otherwise the stored indices would drift out from
    under their meanings.
    """

    def __init__(self, cap=50000, topm=6, seed=0):
        self.cap, self.topm = cap, topm
        self.ci = np.zeros((cap, T2, topm), dtype=np.int64)
        self.cv = np.zeros((cap, T2, topm), dtype=np.float32)
        self.st = np.zeros((cap, SENSOR_SIG), dtype=np.float32)
        self.mo = np.zeros((cap, 2), dtype=np.float32)
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def _write(self, slot, codes, st, mo):
        self.ci[slot] = 0
        self.cv[slot] = 0.0
        for t, (ci, cv) in enumerate(codes):
            m = min(len(ci), self.topm)
            self.ci[slot, t, :m] = ci[:m]
            self.cv[slot, t, :m] = cv[:m]
        self.st[slot] = st
        self.mo[slot] = mo

    def add(self, codes, st, mo):
        self.n += 1
        if self.n <= self.cap:
            self._write(self.n - 1, codes, st, mo)
        elif self.rng.random() < self.cap / self.n:
            self._write(int(self.rng.integers(self.cap)), codes, st, mo)

    def draw(self):
        """One remembered moment, as (codes, state, motor)."""
        top = min(self.n, self.cap)
        if top == 0:
            return None
        i = int(self.rng.integers(top))
        codes = [(self.ci[i, t], self.cv[i, t]) for t in range(T2)]
        return codes, self.st[i], self.mo[i]


# -------------------------------------------------------------- policy ---
CODE_HIST = (T2 - 1) * STRIDE + 1


class Stack:
    """Both layers, the rolling history, and the cached code ring.

    Only ONE layer-one code is computed per tick. The layer-two window
    needs codes at t, t-3, ... t-18, and six of those seven were already
    computed on earlier ticks, so they are kept in a ring and reused.
    """

    def __init__(self, k1=128, k2=2048, topm=6, seed=0):
        self.l1 = MotionLayer(k=k1, seed=seed)
        self.l2 = CommandLayer(k1=k1, k=k2, seed=seed + 10)
        self.topm = topm
        self.hist = np.zeros((HIST, NSIG), dtype=np.float32)
        self.ring = [(np.zeros(0, dtype=np.int64),
                      np.zeros(0, dtype=np.float32))] * CODE_HIST
        self.filled = 0
        self.l1_frozen = False

    # -- history -------------------------------------------------------
    def reset_history(self):
        """New episode — the drone teleports, so no window may span it."""
        self.hist[:] = 0.0
        self.ring = [(np.zeros(0, dtype=np.int64),
                      np.zeros(0, dtype=np.float32))] * CODE_HIST
        self.filled = 0

    def push(self, st, lv_prev):
        """One tick: the state now, and what was commanded last tick."""
        self.hist[:-1] = self.hist[1:]
        self.hist[-1, :SENSOR_SIG] = st
        self.hist[-1, SENSOR_SIG:] = norm_motor(lv_prev)
        self.filled = min(self.filled + 1, HIST + CODE_HIST)
        if self.filled >= HIST:
            win = self.l1_window(self.hist, HIST - 1)
            if not self.l1_frozen:
                self.l1.learn(win)
            self.ring = self.ring[1:] + [self.l1.code(win, self.topm)]

    def ready(self):
        return self.filled >= HIST + CODE_HIST - 1

    # -- turning history into the two layers' inputs -------------------
    @staticmethod
    def l1_window(hist, end):
        """T1 taps ending at `end`; motors lag one tick (no label leak)."""
        idx = end - (T1 - 1 - np.arange(T1)) * STRIDE
        win = np.empty((T1, NSIG), dtype=np.float32)
        win[:, :SENSOR_SIG] = hist[idx, :SENSOR_SIG]
        win[:, SENSOR_SIG:] = hist[idx - 1, SENSOR_SIG:]
        return win

    def codes_now(self):
        return [self.ring[-1 - (T2 - 1 - i) * STRIDE] for i in range(T2)]

    def state_now(self):
        return self.hist[-1, :SENSOR_SIG]

    # -- use -----------------------------------------------------------
    def act(self):
        if not self.ready():
            return (W.NLEV // 2, W.NLEV // 2), 0.0
        return self.l2.act(self.codes_now(), self.state_now())

    def learn_live(self, lv):
        """Bind what is happening now to what the oracle would command."""
        if not self.ready():
            return None
        codes, st, mo = self.codes_now(), self.state_now(), norm_motor(lv)
        self.l2.learn(codes, st, mo)
        return codes, st, mo

    def learn_replay(self, moment):
        codes, st, mo = moment
        self.l2.learn(codes, st, mo)


def load_policy(npz, cfg):
    """Hook for the shared viewer at the repo root."""
    st = Stack(k1=int(cfg["k1"]), k2=int(cfg["k2"]), topm=int(cfg["topm"]),
               seed=int(cfg.get("seed", 0)))
    st.l1.hc.W = npz["W1"]
    st.l1.hc.n_boot = int(npz["n1"])
    st.l2.hc.W = npz["W2"]
    st.l2.hc.n_boot = int(npz["n2"])
    st.l1_frozen = True

    def act(s, tgt, lv_prev):
        st.push(norm_state(s, tgt), lv_prev)
        lv, _ = st.act()
        return lv
    return act
