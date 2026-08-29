"""Viewer hook for the attention rig (gpu_attend weights).

Single-world numpy replica: three tracks (dx, dy, angle; 240/720 ms),
present bumps, the learned attention shares gating the cue (pruned to
the selected columns), the cetele's duty-mean velocity command at
10 Hz, smoothing, the soft reflex. Honest note: this policy approaches
perfectly and fails the hold — that behaviour is what you're watching.
"""

import numpy as np

import sl_drone as W
from chase_view import masked, snap

NS, NB = 3, 24
T1, ST1, T2, ST2 = 5, 3, 7, 6
HIST = (T2 - 1) * ST2 + (T1 - 1) * ST1 + 2
RING = (T2 - 1) * ST2 + 1
K1, K2 = 64, 128
CEN = np.linspace(-1, 1, NB).astype(np.float32)
RAD = 3 * (2.0 / (NB - 1))
FOVS = [(0.25, 30.0), (0.25, 30.0), (0.05, float(np.pi))]
KVX, KVY, PHIMAX, KA, KDA = 0.30, 1.80, 0.6, 2.0, 0.6
ALPHA = 0.15


def fov(v, scale, rim):
    return float(np.clip(np.arcsinh(v / scale)
                         / np.arcsinh(rim / scale), -1, 1))


def bump(v):
    d = np.abs(CEN - v)
    w = np.where(d < RAD, 0.5 * (1 + np.cos(np.pi * d / RAD)), 0.0)
    return np.where(w > 1e-3, w, 0.0).astype(np.float32)


def graded(sc, m=8):
    g = np.clip(sc, 0, None)
    if len(g) > m:
        th = np.partition(g, len(g) - m)[-m]
        g = np.where(g >= th, g, 0)
    return g


def decode_vc(row):
    r = np.clip(row, 0, None).reshape(2, NB)
    out = []
    for h in r:
        u = float((h * CEN).sum() / h.sum()) if h.sum() > 0 else 0.0
        out.append(0.2 * np.sinh(u * np.arcsinh(8.0 / 0.2)))
    return out


class AttendView:
    def __init__(self, npz):
        self.W1 = [np.asarray(npz[f"l1_{c}"]) for c in range(NS)]
        self.W2 = [np.asarray(npz[f"l2_{c}"]) for c in range(NS)]
        self.Wo = np.asarray(npz["Wo"])
        self.Co = np.asarray(npz["Co"])
        self.cols = np.asarray(npz["cols"])
        self.shares = np.asarray(npz["shares"])
        self.vcu = np.array([decode_vc(np.asarray(npz["Bo"])[u])
                             for u in range(32)], np.float32)
        self.reset()

    def reset(self):
        self.hist = np.zeros((HIST, NS), np.float32)
        self.ring = np.zeros((RING, NS, K1), np.float32)
        self.nw = 0
        self.vc = np.zeros(2, np.float32)
        self.vc_raw = np.zeros(2, np.float32)
        self.t = 0

    def cue(self, s, tgt):
        raw = [s[0] - tgt[0], s[1] - tgt[1], W.wrap(s[4])]
        sv = np.array([fov(raw[i], *FOVS[i]) for i in range(NS)],
                      np.float32)
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
            pb = pb / (np.linalg.norm(pb) + 1e-9)
            parts.append(pb * np.sqrt(max(self.shares[c], 0.0)))
        for c in range(NS):
            n = np.linalg.norm(codes[c])
            cc = codes[c] / n if n > 1e-9 else codes[c]
            parts.append(cc * np.sqrt(max(self.shares[NS + c], 0.0)))
        return np.concatenate(parts)[self.cols]

    def step(self, s, tgt):
        cue = self.cue(s, tgt)
        if self.t % 5 == 0:
            j = int(np.argmax(masked(self.Wo, cue)))
            row = self.Co[j]
            if row.sum() > 0:
                self.vc_raw = (row / row.sum()) @ self.vcu
        self.vc = self.vc + ALPHA * (self.vc_raw - self.vc)
        self.t += 1
        ph = W.wrap(s[4])
        phi_des = np.clip(KVX * (s[2] - self.vc[0]), -PHIMAX, PHIMAX)
        coll = np.clip((W.HOVER + KVY * (self.vc[1] - s[3]))
                       / max(np.cos(ph), 0.5), 0.3, W.TMAX)
        diff = np.clip(KA * (phi_des - ph) - KDA * s[5], -2.5, 2.5)
        return snap(coll, diff)


def load_policy(npz, cfg):
    v = AttendView(npz)
    prev = {"p": None}

    def act(s, tgt, lv_prev):
        p = (float(tgt[0]), float(tgt[1]), float(s[0]), float(s[1]))
        if (prev["p"] is None or p[:2] != prev["p"][:2]
                or np.hypot(p[2] - prev["p"][2],
                            p[3] - prev["p"][3]) > 2.0):
            v.reset()
        prev["p"] = p
        return v.step(s, tgt)
    return act
