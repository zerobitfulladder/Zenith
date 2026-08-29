"""Viewer hook for the GPU chase rig (gpu_chase.py checkpoints).

Single-world numpy replica of the forward path: six foveated sensors,
per-signal tracks, sensory-only top layer, cetele duty-cycle read.
No learning here — this only flies a saved checkpoint.
"""

import numpy as np

import sl_drone as W

NS, NB = 6, 24
T1, ST1, T2, ST2 = 5, 3, 7, 6
HIST = (T2 - 1) * ST2 + (T1 - 1) * ST1 + 2
RING = (T2 - 1) * ST2 + 1
K1, K2, TOPM = 64, 128, 8
PRES = 0.5
NLEV = W.NLEV
NA = NLEV * NLEV
CEN = np.linspace(-1, 1, NB).astype(np.float32)
RAD = 3 * (2.0 / (NB - 1))
FOV = [(0.25, 30.0), (0.25, 30.0), (0.15, 20.0), (0.15, 20.0),
       (0.05, np.pi), (0.1, 10.0)]
FDEN = np.array([np.arcsinh(hi / lo) for lo, hi in FOV], np.float32)
FSC = np.array([lo for lo, _ in FOV], np.float32)
_ls = np.arange(NA) // NLEV
_rs = np.arange(NA) % NLEV
A_COLL = 0.5 * (W.LEVELS[_ls] + W.LEVELS[_rs])
A_DIFF = 0.5 * (W.LEVELS[_rs] - W.LEVELS[_ls])
HOVER_LV = (NLEV // 2, NLEV // 2)


def bump(v):
    d = np.abs(CEN - v)
    w = np.where(d < RAD, 0.5 * (1 + np.cos(np.pi * d / RAD)), 0.0)
    return np.where(w > 1e-3, w, 0.0).astype(np.float32)


def masked(Wm, x, floor=0.25):
    num = Wm @ x
    nrm = np.sqrt(np.maximum((Wm * Wm) @ (x > 0).astype(np.float32),
                             1e-18))
    den = np.maximum(nrm, floor * nrm.max() + 1e-9)
    return num / (den * (np.linalg.norm(x) + 1e-9))


def graded(sc, m=TOPM):
    g = np.clip(sc, 0, None)
    if len(g) > m:
        th = np.partition(g, len(g) - m)[-m]
        g = np.where(g >= th, g, 0)
    return g


class View:
    def __init__(self, npz):
        self.W1 = [np.asarray(npz[f"l1_{c}"]) for c in range(NS)]
        self.W2 = [np.asarray(npz[f"l2_{c}"]) for c in range(NS)]
        self.W3 = np.asarray(npz["l3"])
        self.C = np.asarray(npz["C"])
        self.reset()

    def reset(self):
        self.hist = np.zeros((HIST, NS), np.float32)
        self.ring = np.zeros((RING, NS, K1), np.float32)
        self.nw = 0

    def step(self, s, tgt):
        raw = np.array([s[0] - tgt[0], s[1] - tgt[1], s[2], s[3],
                        W.wrap(s[4]), s[5]], np.float32)
        sv = np.clip(np.arcsinh(raw / FSC) / FDEN, -1, 1)
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = sv
        self.nw += 1
        ready = self.nw >= HIST
        idx = HIST - 1 - (T1 - 1 - np.arange(T1)) * ST1
        self.ring[:-1] = self.ring[1:]
        self.ring[-1] = 0
        codes = np.zeros((NS, K2), np.float32)
        for c in range(NS):
            if ready:
                x1 = np.concatenate([bump(v)
                                     for v in self.hist[idx, c]])
                self.ring[-1, c] = graded(masked(self.W1[c], x1))
                taps = self.ring[RING - 1 - (T2 - 1 - np.arange(T2))
                                 * ST2, c]
                codes[c] = graded(masked(self.W2[c], taps.ravel()))
        parts = []
        for c in range(NS):
            pb = bump(sv[c])
            parts.append(pb / (np.linalg.norm(pb) + 1e-9)
                         * np.sqrt(PRES / NS))
        for c in range(NS):
            n = np.linalg.norm(codes[c])
            parts.append(codes[c] / (n + 1e-9)
                         * np.sqrt((1 - PRES) / NS) if n > 1e-9
                         else codes[c])
        cue = np.concatenate(parts)
        j = int(np.argmax(masked(self.W3, cue)))
        return self.read(j)

    def read(self, j):
        """Duty-cycle mean over the whole row (override to vary)."""
        row = self.C[j]
        tot = row.sum()
        if tot <= 0:
            return HOVER_LV
        w = row / tot
        return snap(float(w @ A_COLL), float(w @ A_DIFF))


def snap(c_, d_):
    t1 = np.clip(c_ - d_, 0, W.TMAX)
    t2 = np.clip(c_ + d_, 0, W.TMAX)
    return (int(np.argmin(np.abs(W.LEVELS - t1))),
            int(np.argmin(np.abs(W.LEVELS - t2))))


def load_policy(npz, cfg):
    v = View(npz)
    prev = {"p": None}

    def act(s, tgt, lv_prev):
        p = (float(s[0]), float(s[1]), float(tgt[0]), float(tgt[1]))
        if (prev["p"] is None
                or np.hypot(p[0] - prev["p"][0],
                            p[1] - prev["p"][1]) > 2.0
                or abs(p[2] - prev["p"][2]) > 1e-9
                or abs(p[3] - prev["p"][3]) > 1e-9):
            v.reset()
        prev["p"] = p
        return v.step(s, tgt)
    return act
