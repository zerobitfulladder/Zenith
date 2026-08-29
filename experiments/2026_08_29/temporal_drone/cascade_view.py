"""Viewer hook: the cascade chaser — learned outer, reflex inner.

The memory does the cognitive part: 256 sensory templates over the
foveated position error, each with its cetele over 32 velocity-command
prototypes, duty-mean read, recomputed at 10 Hz. The inner loop is the
fixed attitude reflex (the teacher's own nested-P law) — the user's
call: "low level precise motor control is a dumb processor either way."
Measured as outer_only in gpu_cascade reports: 0.98-1.00 strict EVAL.
"""

import numpy as np

import sl_drone as W

NB = 24
CEN = np.linspace(-1, 1, NB).astype(np.float32)
RAD = 3 * (2.0 / (NB - 1))
K_VC = 32
POS_P, AMAX = 1.0, 3.0
KVX, KVY, KA, KDA = 0.30, 1.80, 2.0, 0.6
PHIMAX, DMAX = 0.6, 2.5
OUTER_EVERY = 5


def fov(v, scale, rim):
    return float(np.clip(np.arcsinh(v / scale) / np.arcsinh(rim / scale),
                         -1, 1))


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


def decode_vc(row):
    r = np.clip(row, 0, None).reshape(2, NB)
    out = []
    for h in r:
        u = float((h * CEN).sum() / h.sum()) if h.sum() > 0 else 0.0
        out.append(0.2 * np.sinh(u * np.arcsinh(8.0 / 0.2)))
    return out


DEFAULT_GAINS = {"PHV": KVX, "CV": KVY, "PHIMAX": PHIMAX,
                 "KAp": KA, "KAd": KDA, "ALPHA": 0.15}


class View:
    def __init__(self, npz, pid_inner=False, gains=None):
        self.pid = pid_inner
        self.g = dict(DEFAULT_GAINS, **(gains or {}))
        self.Wo = np.asarray(npz["Wo"])
        self.Co = np.asarray(npz["Co"])
        self.vc_units = np.array([decode_vc(np.asarray(npz["Bo"])[u])
                                  for u in range(K_VC)], np.float32)
        self.vc = np.zeros(2, np.float32)
        self.vc_raw = np.zeros(2, np.float32)
        self.t = 0

    def outer(self, s, tgt):
        key = np.concatenate([bump(fov(s[0] - tgt[0], 0.25, 30.0)),
                              bump(fov(s[1] - tgt[1], 0.25, 30.0))])
        j = int(np.argmax(masked(self.Wo, key)))
        if getattr(self, "VCT", None) is not None:
            return self.VCT[j]           # reward-tilted read
        row = self.Co[j]
        if row.sum() <= 0:
            return np.zeros(2, np.float32)
        w = row / row.sum()
        return w @ self.vc_units

    def step(self, s, tgt):
        if self.t % OUTER_EVERY == 0:
            self.vc_raw = self.outer(s, tgt)
        # brainstem smoothing: the memory speaks in 10 Hz steps of a
        # 32-word vocabulary; the integrator turns steps into ramps
        # (the self-inflicted "wind" was the servo executing each step)
        self.vc = self.vc + self.g["ALPHA"] * (self.vc_raw - self.vc)
        self.t += 1
        ph = W.wrap(s[4])
        if self.pid:
            axd = 4.084 * (self.vc[0] - s[2])
            ayd = 4.084 * (self.vc[1] - s[3])
            phi_des = np.clip(-0.071 * axd, -0.696, 0.696)
            coll = np.clip(0.5 * (9.8 + ayd) / max(np.cos(ph), 0.5),
                           0.3, W.TMAX)
            diff = np.clip(6.426 * (phi_des - ph) - 0.943 * s[5],
                           -2.5, 2.5)
        else:
            g = self.g
            phi_des = np.clip(g["PHV"] * (s[2] - self.vc[0]),
                              -g["PHIMAX"], g["PHIMAX"])
            coll = (W.HOVER + g["CV"] * (self.vc[1] - s[3])) / max(
                np.cos(ph), 0.5)
            coll = np.clip(coll, 0.3, W.TMAX)
            diff = np.clip(g["KAp"] * (phi_des - ph)
                           - g["KAd"] * s[5], -DMAX, DMAX)
        t1 = np.clip(coll - diff, 0, W.TMAX)
        t2 = np.clip(coll + diff, 0, W.TMAX)
        return (int(np.argmin(np.abs(W.LEVELS - t1))),
                int(np.argmin(np.abs(W.LEVELS - t2))))


def load_policy(npz, cfg):
    v = View(npz, pid_inner=cfg.get("oracle") == "pid",
             gains=cfg.get("inner_gains"))
    vt = cfg.get("value_tilt")
    if vt:
        z = np.load(vt)
        beta = float(cfg.get("beta", 0.1))
        CN = v.Co / np.maximum(v.Co.sum(1, keepdims=True), 1e-9)
        tilt = CN * np.exp(np.clip(beta * z["W"], -20, 20))
        tilt = tilt / np.maximum(tilt.sum(1, keepdims=True), 1e-9)
        VCT = tilt @ v.vc_units
        base = CN @ v.vc_units
        moving = np.linalg.norm(base, axis=1) > 0.5
        v.VCT = np.where(moving[:, None], VCT, base).astype(np.float32)
    prev = {"tgt": None}

    def act(s, tgt, lv_prev):
        tg = (float(tgt[0]), float(tgt[1]))
        if prev["tgt"] != tg:
            v.t = 0                     # new target: fresh outer tick
        prev["tgt"] = tg
        return v.step(s, tgt)
    return act
