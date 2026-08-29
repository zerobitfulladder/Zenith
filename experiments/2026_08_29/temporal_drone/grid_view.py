"""Viewer hook for the grid-partition ablation policy (grid16.npz).

The whole policy: foveated (dx, dy) -> a fixed 16x16 cell -> that
cell's cetele duty-mean -> smoothing -> the soft reflex. No learned
partition anywhere — the champion-beating tabular-BC baseline
(0.99 / 0.033 m / median 229). Law #13's exhibit A.
"""

import numpy as np

import sl_drone as W

KVX, KVY, PHIMAX, KA, KDA = 0.30, 1.80, 0.6, 2.0, 0.6


def warp(v):
    return np.clip(np.arcsinh(v / 0.25) / np.arcsinh(30.0 / 0.25),
                   -1, 1)


class GV:
    def __init__(self, npz):
        C = np.asarray(npz["C"])
        CN = C / np.maximum(C.sum(1, keepdims=True), 1e-9)
        self.VCT = CN @ np.asarray(npz["VCU"])   # (256, 2) precomputed
        self.vc = np.zeros(2, np.float32)
        self.vcr = np.zeros(2, np.float32)
        self.t = 0

    def step(self, s, tgt):
        if self.t % 5 == 0:
            ix = int(np.clip((warp(s[0] - tgt[0]) + 1) / 2 * 16, 0, 15))
            iy = int(np.clip((warp(s[1] - tgt[1]) + 1) / 2 * 16, 0, 15))
            self.vcr = self.VCT[ix * 16 + iy]
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
    v = GV(npz)
    prev = {"tgt": None}

    def act(s, tgt, lv_prev):
        tg = (float(tgt[0]), float(tgt[1]))
        if prev["tgt"] != tg:
            v.t = 0
        prev["tgt"] = tg
        return v.step(s, tgt)
    return act
