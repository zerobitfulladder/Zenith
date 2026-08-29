"""Viewer hook for the from-scratch pure-RL drone (pure_rl.npz).

No teacher ever touched this policy: self-organised situation
templates, an innate 7x7 velocity vocabulary (with "stay"), Q-values
learned from distance progress and success bonus alone. Greedy read.
"""

import numpy as np

import sl_drone as W

NB = 24
CEN = np.linspace(-1, 1, NB).astype(np.float32)
RAD = 3 * (2.0 / (NB - 1))
KVX, KVY, PHIMAX, KA, KDA = 0.30, 1.80, 0.6, 2.0, 0.6


def fovb(v):
    u = float(np.clip(np.arcsinh(v / 0.25) / np.arcsinh(30.0 / 0.25),
                      -1, 1))
    d = np.abs(CEN - u)
    w = np.where(d < RAD, 0.5 * (1 + np.cos(np.pi * d / RAD)), 0.0)
    return np.where(w > 1e-3, w, 0.0).astype(np.float32)


class V:
    def __init__(self, npz):
        self.Wm = np.asarray(npz["Wm"])
        self.Q = np.asarray(npz["Q"])
        self.VCU = np.asarray(npz["VCU"])
        self.vc = np.zeros(2, np.float32)
        self.vcr = np.zeros(2, np.float32)
        self.t = 0

    def step(self, s, tgt):
        if self.t % 5 == 0:
            k = np.concatenate([fovb(s[0] - tgt[0]),
                                fovb(s[1] - tgt[1])])
            j = int(np.argmax(k @ self.Wm.T))
            self.vcr = self.VCU[int(self.Q[j].argmax())]
        self.vc = self.vc + 0.15 * (self.vcr - self.vc)
        self.t += 1
        ph = W.wrap(s[4])
        phd = np.clip(KVX * (s[2] - self.vc[0]), -PHIMAX, PHIMAX)
        coll = np.clip((W.HOVER + KVY * (self.vc[1] - s[3]))
                       / max(np.cos(ph), 0.5), 0.3, W.TMAX)
        diff = np.clip(KA * (phd - ph) - KDA * s[5], -2.5, 2.5)
        t1 = np.clip(coll - diff, 0, W.TMAX)
        t2 = np.clip(coll + diff, 0, W.TMAX)
        return (int(np.argmin(np.abs(W.LEVELS - t1))),
                int(np.argmin(np.abs(W.LEVELS - t2))))


def load_policy(npz, cfg):
    v = V(npz)
    prev = {"tgt": None}

    def act(s, tgt, lv_prev):
        tg = (float(tgt[0]), float(tgt[1]))
        if prev["tgt"] != tg:
            v.t = 0
        prev["tgt"] = tg
        return v.step(s, tgt)
    return act
