"""Two tracks into one memory: what is seen, and what was done about it.

    sensory track   dx, dy, angle          -> one hypercolumn
    motor track     the two thrust levels  -> one hypercolumn
    layer two       [sensory code ; motor code] concatenated, one
                    hypercolumn, so a template is a whole
                    "in this situation, these thrusts" pair

At inference the motor half is left empty. The sensory half alone picks
a winner -- scored only over the cells that are actually written -- and
the winner's own motor half is read back out and flown.

Each value becomes a bump of adjacent cells with a sharp fall-off, so
near values share most of their cells and far values share none. The
three sensory channels get disjoint blocks of one array and the two
motor channels get disjoint blocks of another: no two channels ever land
on the same cell, because a shared cell puts a phantom peak on the motor
row of every template whose sensory bump covers it.

Note what the sensory track is NOT given: no velocity and no angular
rate. The oracle it learns from uses both. A memoryless policy on
position and angle alone cannot see how fast it is already moving, so
some of the gap to the oracle below is structural, not a training
failure.
"""

import sys
from pathlib import Path

import numpy as np

_SLD = Path(__file__).resolve().parents[2] / "2026_08_28" / "single_layer_drone"
sys.path.insert(0, str(_SLD))
import sl_drone as W                                        # noqa: E402

NB = 128
ACTIVE = np.array([0.08, 0.40, 1.0, 0.40, 0.08], dtype=np.float32)
HALF = len(ACTIVE) // 2
SENS_CH = ["dx", "dy", "tilt"]
MOTOR_CH = ["mL", "mR"]
EPS = 1e-9


def _fov(n, lo, hi):
    u = np.linspace(-1.0, 1.0, n)
    return np.sign(u) * max(abs(lo), abs(hi)) * np.abs(u) ** 1.7


def _ring(n):
    u = (np.arange(n) + 0.5) / n * 2 - 1
    return np.sign(u) * np.pi * np.abs(u) ** 1.7


CENTERS = {"dx": _fov(NB, -8.0, 8.0), "dy": _fov(NB, -8.0, 8.0),
           "tilt": _ring(NB), "mL": W.LEVELS.copy(), "mR": W.LEVELS.copy()}
RING = {"dx": False, "dy": False, "tilt": True, "mL": False, "mR": False}


class Track:
    """One array, a few channels, each channel a bump. Disjoint by design."""

    def __init__(self, channels, size, seed):
        self.channels, self.size = channels, size
        perm = np.random.default_rng(seed).permutation(size)
        n = {c: len(CENTERS[c]) for c in channels}
        cut = np.cumsum([0] + [n[c] for c in channels])
        assert cut[-1] <= size, "array too small for these channels"
        self.pos = {c: perm[cut[i]:cut[i + 1]] for i, c in enumerate(channels)}

    def _cells(self, ch, value):
        cent = CENTERS[ch]
        if RING[ch]:
            i = int(np.argmin(np.abs(W.wrap(value - cent))))
            return np.arange(i - HALF, i + HALF + 1) % len(cent)
        i = int(np.argmin(np.abs(np.clip(value, cent[0], cent[-1]) - cent)))
        return np.clip(np.arange(i - HALF, i + HALF + 1), 0, len(cent) - 1)

    def encode(self, values):
        """(positions, magnitudes) for a dict of channel -> value."""
        idx = np.concatenate([self.pos[c][self._cells(c, values[c])]
                              for c in self.channels])
        val = np.tile(ACTIVE, len(self.channels))
        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(out, inv, val)
        return uniq.astype(np.int32), out

    def read(self, vec, ch):
        """The value a template's cells for one channel are asserting."""
        return float(CENTERS[ch][int(np.argmax(vec[self.pos[ch]]))])


def sense(s, tgt):
    return {"dx": s[0] - tgt[0], "dy": s[1] - tgt[1], "tilt": W.wrap(s[4])}


def motor(levels):
    return {"mL": W.LEVELS[levels[0]], "mR": W.LEVELS[levels[1]]}


def graded(hc, idx, vals, topm):
    """A layer's speech: its best few matches, magnitudes intact."""
    n = float(np.sqrt(max((vals.astype(np.float64) ** 2).sum()
                          - vals.sum() ** 2 / hc.dim, 0.0)))
    if n < EPS or hc.n_boot == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    g = np.clip((hc.W[:hc.n_boot][:, idx] @ vals) / n, 0.0, None)
    keep = np.argsort(-g)[:topm]
    keep = keep[g[keep] > 0]
    return keep.astype(np.int32), g[keep].astype(np.float32)


class TwoTrack:
    """The whole thing: two tracks below, one bound memory on top."""

    def __init__(self, k_sens=1024, k_mot=169, k_top=4096, topm=8,
                 eta=0.05, rho=1.0, seed=0):
        self.sens_track = Track(SENS_CH, 1024, seed)
        self.mot_track = Track(MOTOR_CH, 512, seed + 1)
        self.l1s = W.Hypercolumn(k_sens, self.sens_track.size, eta, seed + 2)
        self.l1m = W.Hypercolumn(k_mot, self.mot_track.size, eta, seed + 3)
        self.k_sens, self.k_mot, self.topm, self.rho = k_sens, k_mot, topm, rho
        self.dim = k_sens + k_mot
        self.l2 = W.Hypercolumn(k_top, self.dim, eta, seed + 4)
        self.sens_ix = np.arange(k_sens)
        self.mot_ix = np.arange(k_sens, self.dim)

    # -- the two halves of layer two's input ------------------------------
    def sens_code(self, sens):
        i, v = self.sens_track.encode(sens)
        return graded(self.l1s, i, v, self.topm)

    def mot_code(self, lv):
        i, v = self.mot_track.encode(motor(lv))
        return graded(self.l1m, i, v, self.topm)

    def join(self, sc, mc):
        """Concatenate, with the motor half scaled to rho x the sensory half."""
        si, sv = sc
        mi, mv = mc
        ns = float(np.linalg.norm(sv)) + EPS
        nm = float(np.linalg.norm(mv)) + EPS
        return (np.concatenate([si, mi + self.k_sens]).astype(np.int32),
                np.concatenate([sv, mv * (self.rho * ns / nm)]).astype(np.float32))

    # -- reading with the motor half left empty ---------------------------
    def _bids(self, idx, vals, floor=0.25):
        sub = self.l2.W[:self.l2.n_boot][:, idx]
        nrm = np.linalg.norm(sub, axis=1)
        return (sub @ vals) / np.maximum(nrm, floor * float(nrm.max()) + EPS)

    def act(self, s, tgt, read="graded"):
        """Sensory in, thrust levels out. The motor cells are never written."""
        si, sv = self.sens_code(sense(s, tgt))
        if len(si) == 0 or self.l2.n_boot == 0:
            h = int(np.argmin(np.abs(W.LEVELS - W.HOVER)))
            return h, h
        win = int(np.argmax(self._bids(si, sv)))
        half = self.l2.W[win][self.mot_ix]
        if read == "top1":
            vec = self.l1m.W[int(np.argmax(half))]
        else:                       # every motor template it half-believes
            vec = np.clip(half, 0.0, None) @ self.l1m.W[:self.l1m.n_boot]
        return tuple(int(np.argmin(np.abs(W.LEVELS - self.mot_track.read(vec, c))))
                     for c in MOTOR_CH)

    # -- persistence ------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path, Ws=self.l1s.W[:self.l1s.n_boot], Wm=self.l1m.W[:self.l1m.n_boot],
            W2=self.l2.W[:self.l2.n_boot],
            wins2=self.l2.wins[:self.l2.n_boot], wins_s=self.l1s.wins[:self.l1s.n_boot])

    @classmethod
    def from_npz(cls, npz, cfg):
        m = cls(k_sens=int(cfg["k_sens"]), k_mot=int(cfg["k_mot"]),
                k_top=int(cfg["k_top"]), topm=int(cfg["topm"]),
                rho=float(cfg["rho"]), seed=int(cfg["seed"]))
        for hc, key in ((m.l1s, "Ws"), (m.l1m, "Wm"), (m.l2, "W2")):
            A = npz[key]
            hc.W[:len(A)] = A
            hc.n_boot = len(A)
        return m


def load_policy(npz, cfg):
    """What viewer.py calls. Returns fn(state, target, lv_prev) -> (l, r)."""
    m = TwoTrack.from_npz(npz, cfg)
    read = cfg.get("read", "graded")
    return lambda s, tgt, lv_prev: m.act(s, tgt, read=read)
