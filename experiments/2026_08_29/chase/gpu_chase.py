"""Target chasing on the GPU: the split architecture, 64 worlds at once.

Everything the balance rung proved, carried to the position task:

    sensors    dx, dy (target-relative), vx, vy, tilt, gyro — SIX
               channels, each with its own arcsinh fovea (position
               scale 0.25 m rim 30; velocity 0.15/20; tilt 0.05/pi;
               gyro 0.1/10). The speedometer law and the 8 m clip
               lesson, both honoured.
    tracks     per-signal L1 (5 taps, 240 ms, K=64) and L2 (7 codes,
               720 ms, K=128), graded top-8 up — the td_tracks
               machinery, batched.
    top layer  SENSORY-ONLY templates (the split): present bumps at
               50% of the cue + track codes at 50%. K3 = 8192.
    cetele     C[template, 13*13] counts of the teacher's (l, r)
               pairs, notched by the same masked question the read
               asks. Read = duty-cycle expectation (mean, snapped).
    training   teacher phase, then tight-envelope DAgger (pupil flies
               the cetele read, teacher labels). Teacher episodes run
               to the cap, so hover-at-target dwell data exists.

GPU strategy: the loop is a closed-loop simulation, so parallelism is
across WORLDS — B independent drones; every score is one batched
matmul. Masked-cosine scores replicate td_tracks.Hyper.scores exactly
(support-indicator matmul for per-template masked norms, floor 0.25).
Geodesic learning applies per-winner sequentially within the batch
(B<=64 small row ops; the Aug-28 B=2 lesson says batch-averaging is
the thing to avoid, not batching).

Run:  .venv/bin/python experiments/2026_08_29/chase/gpu_chase.py
Env:  GC_B GC_TICKS GC_PHASE1 GC_STAGE1 GC_STAGE2 GC_K3 GC_REPORT
      GC_EVERY GC_TAG GC_CPU (force numpy)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if os.environ.get("GC_CPU", "0") == "1":
    xp = np
    def asnumpy(a):
        return a
else:
    import cupy as xp
    asnumpy = xp.asnumpy

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get('GC_TAG', '').lstrip("_")
B = int(os.environ.get("GC_B", "64"))
TICKS = int(os.environ.get("GC_TICKS", "4000000"))      # experience
PHASE1 = int(os.environ.get("GC_PHASE1", "600000"))
STAGE1 = int(os.environ.get("GC_STAGE1", "40000"))
STAGE2 = int(os.environ.get("GC_STAGE2", "80000"))
K3 = int(os.environ.get("GC_K3", "8192"))
REPORT = int(os.environ.get("GC_REPORT", "400000"))
EVERY = int(os.environ.get("GC_EVERY", "5"))

# ---- world constants (sl_drone, replicated) --------------------------
M_, GRAV, ARM_L, INERT, DT = 1.0, 9.8, 0.2, 0.02, 0.02
HOVER, TMAX, NLEV = 4.9, 8.0, 13
_u = np.linspace(-1.0, 1.0, NLEV)
LEVELS = xp.asarray(np.where(_u < 0, HOVER - np.abs(_u) ** 1.7 * HOVER,
                             HOVER + np.abs(_u) ** 1.7 * (TMAX - HOVER)),
                    dtype=xp.float32)
EP_CAP = 800
NA = NLEV * NLEV
HOVER_L = NLEV // 2

# teacher gains (fast_oracle)
POS_P, AMAX, KVX, KVY = 1.0, 3.0, 0.30, 1.80
KA, KDA, PHIMAX, DMAX = 2.0, 0.6, 0.6, 2.5

# encoding
NS, NB = 6, 24
T1, ST1, T2, ST2 = 5, 3, 7, 6
HIST = (T2 - 1) * ST2 + (T1 - 1) * ST1 + 2          # 50
RING = (T2 - 1) * ST2 + 1                            # 37
K1, K2, TOPM = 64, 128, 8
D1, D2 = T1 * NB, T2 * K1
DPRES = NS * NB
D3 = DPRES + NS * K2
PRES = 0.5
CEN = xp.asarray(np.linspace(-1, 1, NB), dtype=xp.float32)
RAD = 3 * (2.0 / (NB - 1))
FOV = [(0.25, 30.0), (0.25, 30.0), (0.15, 20.0), (0.15, 20.0),
       (0.05, np.pi), (0.1, 10.0)]
FDEN = xp.asarray([np.arcsinh(hi / lo) for lo, hi in FOV],
                  dtype=xp.float32)
FSC = xp.asarray([lo for lo, _ in FOV], dtype=xp.float32)

_ls = xp.arange(NA) // NLEV
_rs = xp.arange(NA) % NLEV
A_COLL = 0.5 * (LEVELS[_ls] + LEVELS[_rs])
A_DIFF = 0.5 * (LEVELS[_rs] - LEVELS[_ls])


def wrap(a):
    return ((a + xp.pi) % (2 * xp.pi)) - xp.pi


def physics(s, l_idx, r_idx):
    t1, t2 = LEVELS[l_idx], LEVELS[r_idx]
    tt = t1 + t2
    out = xp.empty_like(s)
    out[:, 0] = s[:, 0] + DT * s[:, 2]
    out[:, 1] = s[:, 1] + DT * s[:, 3]
    out[:, 2] = s[:, 2] + DT * (-tt * xp.sin(s[:, 4]) / M_)
    out[:, 3] = s[:, 3] + DT * (tt * xp.cos(s[:, 4]) / M_ - GRAV)
    out[:, 4] = s[:, 4] + DT * s[:, 5]
    out[:, 5] = s[:, 5] + DT * (ARM_L * (t2 - t1) / INERT)
    return out


def teacher(s, tgt):
    ex, ey = s[:, 0] - tgt[:, 0], s[:, 1] - tgt[:, 1]

    def vcmd(e):
        a = xp.abs(e)
        d = AMAX / (POS_P * POS_P)
        v = xp.where(a <= d, POS_P * a,
                     xp.sqrt(xp.maximum(2 * AMAX * (a - 0.5 * d), 0)))
        return -xp.sign(e) * v
    ph = wrap(s[:, 4])
    phi_des = xp.clip(KVX * (s[:, 2] - vcmd(ex)), -PHIMAX, PHIMAX)
    coll = (HOVER + KVY * (vcmd(ey) - s[:, 3])) / xp.maximum(
        xp.cos(ph), 0.5)
    coll = xp.clip(coll, 0.3, TMAX)
    diff = xp.clip(KA * (phi_des - ph) - KDA * s[:, 5], -DMAX, DMAX)
    t1 = xp.clip(coll - diff, 0, TMAX)
    t2 = xp.clip(coll + diff, 0, TMAX)
    li = xp.argmin(xp.abs(LEVELS[None, :] - t1[:, None]), axis=1)
    ri = xp.argmin(xp.abs(LEVELS[None, :] - t2[:, None]), axis=1)
    return li.astype(xp.int32), ri.astype(xp.int32)


def at_goal(s, tgt):
    return ((xp.hypot(s[:, 0] - tgt[:, 0], s[:, 1] - tgt[:, 1]) < 0.25)
            & (xp.hypot(s[:, 2], s[:, 3]) < 0.3)
            & (xp.abs(wrap(s[:, 4])) < 0.17))


def sense(s, tgt):
    raw = xp.stack([s[:, 0] - tgt[:, 0], s[:, 1] - tgt[:, 1],
                    s[:, 2], s[:, 3], wrap(s[:, 4]), s[:, 5]], axis=1)
    return xp.clip(xp.arcsinh(raw / FSC[None, :]) / FDEN[None, :],
                   -1, 1).astype(xp.float32)


def bump(v):
    """(B, n) values in [-1,1] -> (B, n, NB) raised-cosine bumps."""
    d = xp.abs(CEN[None, None, :] - v[:, :, None])
    w = xp.where(d < RAD, 0.5 * (1 + xp.cos(xp.pi * d / RAD)), 0.0)
    return xp.where(w > 1e-3, w, 0.0).astype(xp.float32)


class Layer:
    """td_tracks.Hyper, batched read / per-winner sequential learn."""

    def __init__(self, k, dim, eta=0.05, seed=0):
        self.W = xp.zeros((k, dim), xp.float32)
        self.k, self.dim, self.eta = k, dim, eta
        self.n = 0
        self.wins = xp.zeros(k, xp.int64)
        self.rng = np.random.default_rng(seed)

    def scores(self, X, floor=0.25):
        if self.n == 0:
            return xp.zeros((X.shape[0], 0), xp.float32)
        Wn = self.W[:self.n]
        num = X @ Wn.T
        nrm = xp.sqrt(xp.maximum((X > 0).astype(xp.float32)
                                 @ (Wn * Wn).T, 1e-18))
        den = xp.maximum(nrm, floor * nrm.max(axis=1, keepdims=True)
                         + 1e-9)
        vn = xp.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return num / (den * vn)

    def learn_rows(self, X):
        """Batched geodesic learning: winners deduped within the batch
        (first occurrence wins; no averaging — the B=2 lesson), all
        row updates applied as one vectorised gather/scatter."""
        nb = X.shape[0]
        if nb == 0:
            return
        if self.n < self.k:
            take = min(self.k - self.n, nb)
            xb = X[:take]
            w = xb - xb.mean(axis=1, keepdims=True)
            w = w + xp.asarray(
                (0.02 / np.sqrt(self.dim))
                * self.rng.standard_normal((take, self.dim)),
                dtype=xp.float32)
            w = w - w.mean(axis=1, keepdims=True)
            w = w / (xp.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
            self.W[self.n:self.n + take] = w
            self.wins[self.n:self.n + take] += 1
            self.n += take
            X = X[take:]
            if X.shape[0] == 0:
                return
        m = X.mean(axis=1)
        n2 = (X * X).sum(axis=1) - self.dim * m * m
        nn = xp.sqrt(xp.maximum(n2, 1e-18))
        raw = X @ self.W[:self.n].T
        wi = xp.argmax(raw, axis=1)
        c = raw[xp.arange(X.shape[0]), wi] / nn
        good = (c > 0) & (c < 1) & (nn > 1e-9)
        wi_np, good_np = asnumpy(wi), asnumpy(good)
        seen, keep = set(), []
        for ii in range(len(wi_np)):
            if good_np[ii] and wi_np[ii] not in seen:
                seen.add(int(wi_np[ii]))
                keep.append(ii)
        if not keep:
            return
        k = xp.asarray(np.asarray(keep))
        wi, c, nn, m, Xk = wi[k], c[k], nn[k], m[k], X[k]
        th = self.eta * c
        tn = xp.sqrt(xp.maximum(1 - c * c, 1e-12))
        a = xp.cos(th) - c * xp.sin(th) / tn
        bb = xp.sin(th) / tn
        rows = (a[:, None] * self.W[wi] + (bb / nn)[:, None] * Xk
                - (bb * m / nn)[:, None])
        self.wins[wi] += 1
        rn = self.wins[wi] % 128 == 0
        if bool(rn.any()):
            rr = rows[rn]
            rr = rr - rr.mean(axis=1, keepdims=True)
            rows[rn] = rr / (xp.linalg.norm(rr, axis=1, keepdims=True)
                             + 1e-9)
        self.W[wi] = rows


def graded(sc, m=TOPM):
    g = xp.maximum(sc, 0)
    if g.shape[1] == 0:
        return g
    if g.shape[1] > m:
        th = xp.partition(g, g.shape[1] - m, axis=1)[:, -m][:, None]
        g = xp.where(g >= th, g, 0)
    return g


class Rig:
    def __init__(self, seed=0):
        self.l1 = [Layer(K1, D1, seed=seed + 7 * c) for c in range(NS)]
        self.l2 = [Layer(K2, D2, seed=seed + 7 * c + 1)
                   for c in range(NS)]
        self.l3 = Layer(K3, D3, seed=seed + 99)
        self.C = xp.zeros((K3, NA), xp.float32)
        self.hist = xp.zeros((B, HIST, NS), xp.float32)
        self.ring = xp.zeros((B, RING, NS, K1), xp.float32)
        self.nw = xp.zeros(B, xp.int64)
        self.stage = 2
        self.frozen = False

    def reset(self, mask):
        self.hist[mask] = 0
        self.ring[mask] = 0
        self.nw = xp.where(xp.asarray(mask), 0, self.nw)

    def push(self, sv, learn_mask):
        self.hist = xp.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = sv
        self.nw += 1
        ready = self.nw >= HIST
        idx = HIST - 1 - (T1 - 1 - np.arange(T1)) * ST1
        self.ring = xp.roll(self.ring, -1, axis=1)
        self.ring[:, -1] = 0
        codes2 = xp.zeros((B, NS, K2), xp.float32)
        for c in range(NS):
            X1 = bump(self.hist[:, idx, c]).reshape(B, D1)
            if self.stage == 0 and not self.frozen:
                lm = learn_mask & asnumpy(ready)
                if lm.any():
                    self.l1[c].learn_rows(X1[xp.asarray(lm)])
            s1 = self.l1[c].scores(X1)
            if s1.shape[1]:
                g1 = xp.zeros((B, K1), xp.float32)
                g1[:, :s1.shape[1]] = graded(s1)
                g1 = g1 * ready[:, None]
                self.ring[:, -1, c] = g1
            taps = self.ring[:, RING - 1 - (T2 - 1 - np.arange(T2))
                             * ST2, c]
            X2 = taps.reshape(B, D2)
            if self.stage == 1 and not self.frozen:
                lm = learn_mask & asnumpy(ready) \
                    & (asnumpy(xp.abs(X2).sum(axis=1)) > 0)
                if lm.any():
                    self.l2[c].learn_rows(X2[xp.asarray(lm)])
            s2 = self.l2[c].scores(X2)
            if s2.shape[1]:
                g2 = xp.zeros((B, K2), xp.float32)
                g2[:, :s2.shape[1]] = graded(s2)
                codes2[:, c] = g2 * ready[:, None]
        # ---- assemble the L3 cue ------------------------------------
        pb = bump(sv)                                   # (B, NS, NB)
        pn = xp.linalg.norm(pb.reshape(B, NS, NB), axis=2,
                            keepdims=True) + 1e-9
        pb = pb / pn * np.sqrt(PRES / NS)
        cn = xp.linalg.norm(codes2, axis=2, keepdims=True)
        codes2 = xp.where(cn > 1e-9,
                          codes2 / (cn + 1e-9)
                          * np.sqrt((1 - PRES) / NS), 0)
        cue = xp.concatenate([pb.reshape(B, DPRES),
                              codes2.reshape(B, NS * K2)], axis=1)
        return cue

    def winners(self, cue):
        sc = self.l3.scores(cue)
        if sc.shape[1] == 0:
            return xp.full(B, -1, xp.int64)
        return xp.argmax(sc, axis=1)

    def act(self, w):
        """Duty-cycle expectation from the cetele; hover w/o support."""
        rows = self.C[xp.clip(w, 0, K3 - 1)]
        tot = rows.sum(axis=1)
        ok = (tot > 0) & (w >= 0)
        ws = rows / xp.maximum(tot, 1)[:, None]
        c = ws @ A_COLL
        d = ws @ A_DIFF
        t1 = xp.clip(c - d, 0, TMAX)
        t2 = xp.clip(c + d, 0, TMAX)
        li = xp.argmin(xp.abs(LEVELS[None] - t1[:, None]), axis=1)
        ri = xp.argmin(xp.abs(LEVELS[None] - t2[:, None]), axis=1)
        li = xp.where(ok, li, HOVER_L).astype(xp.int32)
        ri = xp.where(ok, ri, HOVER_L).astype(xp.int32)
        return li, ri, ok


def init_worlds(rng, n=B):
    s = xp.asarray(np.stack([
        rng.uniform(-4, 4, n), rng.uniform(-4, 4, n),
        rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n),
        rng.uniform(-1.0, 1.0, n), rng.uniform(-1.5, 1.5, n)],
        axis=1), dtype=xp.float32)
    tgt = xp.asarray(np.stack([rng.uniform(-3, 3, n),
                               rng.uniform(-3, 3, n)], axis=1),
                     dtype=xp.float32)
    return s, tgt


def evaluate(rig, rng, steps=EP_CAP):
    was = rig.frozen
    rig.frozen = True
    s, tgt = init_worlds(rng)
    rig.reset(np.ones(B, bool))
    hold = xp.zeros(B, xp.int32)
    done = xp.zeros(B, bool)
    best = xp.full(B, 1e9, xp.float32)
    for t in range(steps):
        cue = rig.push(sense(s, tgt), np.zeros(B, bool))
        w = rig.winners(cue)
        li, ri, _ = rig.act(w)
        s = physics(s, li, ri)
        best = xp.minimum(best, xp.hypot(s[:, 0] - tgt[:, 0],
                                         s[:, 1] - tgt[:, 1]))
        g = at_goal(s, tgt)
        hold = xp.where(g, hold + 1, 0)
        done = done | (hold >= 10)
    rig.frozen = was
    return float(done.mean()), float(best.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Rig()
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(99)

    # harness check: the teacher through the batched world must succeed
    s, tgt = init_worlds(rng)
    hold = xp.zeros(B, xp.int32)
    done = xp.zeros(B, bool)
    for t in range(EP_CAP):
        li, ri = teacher(s, tgt)
        s = physics(s, li, ri)
        g = at_goal(s, tgt)
        hold = xp.where(g, hold + 1, 0)
        done = done | (hold >= 10)
    print(f"teacher through batched harness: {float(done.mean()):.2f} "
          f"(must be ~1.00)", flush=True)
    if float(done.mean()) < 0.95:
        return

    s, tgt = init_worlds(rng)
    rig.reset(np.ones(B, bool))
    ep_tick = np.zeros(B, np.int64)
    pupil = np.zeros(B, bool)
    episodes = 0
    agree = []
    stats, best = [], {"score": -1.0, "tick": 0}
    t0 = time.time()
    steps = TICKS // B

    for step in range(steps):
        te = step * B
        rig.stage = 0 if te < STAGE1 else (1 if te < STAGE2 else 2)
        lm = (np.arange(B) + step) % EVERY == 0
        cue = rig.push(sense(s, tgt), lm)
        lab_l, lab_r = teacher(s, tgt)
        w = rig.winners(cue) if rig.l3.n else xp.full(B, -1, xp.int64)
        if rig.stage == 2:
            lml = xp.asarray(lm) & (rig.nw >= 1)
            if bool(lml.any()):
                rig.l3.learn_rows(cue[lml])
            valid = w >= 0
            if bool(valid.any()):
                a = lab_l * NLEV + lab_r
                flat = xp.clip(w, 0, K3 - 1) * NA + a
                upd = xp.zeros(K3 * NA, xp.float32)
                if xp is np:
                    np.add.at(upd, asnumpy(flat[valid]), 1.0)
                else:
                    import cupyx
                    cupyx.scatter_add(upd, flat[valid], 1.0)
                rig.C += upd.reshape(K3, NA)
            li, ri, ok = rig.act(w)
            agree.append(float(((li == lab_l) & (ri == lab_r))
                               [valid].mean()) if bool(valid.any())
                         else 0.0)
            pm = xp.asarray(pupil) & ok
            fl = xp.where(pm, li, lab_l)
            fr = xp.where(pm, ri, lab_r)
        else:
            fl, fr = lab_l, lab_r
        s = physics(s, fl, fr)
        ep_tick += 1
        spd = asnumpy(xp.hypot(s[:, 2], s[:, 3]))
        gy = np.abs(asnumpy(s[:, 5]))
        dist = asnumpy(xp.hypot(s[:, 0] - tgt[:, 0], s[:, 1] - tgt[:, 1]))
        blown = pupil & ((spd > 8) | (gy > 6) | (dist > 15))
        over = (ep_tick >= EP_CAP) | blown
        if over.any():
            episodes += int(over.sum())
            ns, ntgt = init_worlds(rng, int(over.sum()))
            om = xp.asarray(over)
            s[om] = ns
            tgt[om] = ntgt
            rig.reset(over)
            ep_tick[over] = 0
            pupil[over] = (te >= PHASE1) & (np.random.default_rng(
                episodes).random(int(over.sum())) < 0.5)

        if (step + 1) % max(1, REPORT // B) == 0:
            ev, near = evaluate(rig, erng)
            row = {"exp_ticks": te + B, "episodes": episodes,
                   "EVAL": round(ev, 3), "closest": round(near, 2),
                   "agree": round(float(np.mean(agree)), 3)
                   if agree else None,
                   "l3": int(rig.l3.n),
                   "support": round(float((asnumpy(rig.C).sum(1) > 0)
                                          .mean()), 3),
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree = []
            side = {"kind": "module", "module": "chase_view",
                    "dir": "experiments/2026_08_29/temporal_drone",
                    "tick": te + B, "score": row["EVAL"]}
            np.savez(OUT / "checkpoint.npz",
                     **{f"l1_{c}": asnumpy(rig.l1[c].W)
                        for c in range(NS)},
                     **{f"l2_{c}": asnumpy(rig.l2[c].W)
                        for c in range(NS)},
                     l3=asnumpy(rig.l3.W), C=asnumpy(rig.C))
            with open(OUT / "checkpoint.json", "w") as f:
                json.dump(side, f)
            # tie-break EVAL by closest-approach so weights_best keeps
            # tracking progress while the strict gate is still shut
            if (row["EVAL"], -row["closest"]) > (best["score"],
                                                 -best.get("closest", 99)):
                best = {"score": row["EVAL"], "closest": row["closest"],
                        "tick": te + B}
                os.replace(OUT / "checkpoint.npz",
                           OUT / "weights_best.npz")
                with open(OUT / "weights_best.json", "w") as f:
                    json.dump(side, f)
                print(f"  new best {best['score']} @ {te + B}",
                      flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "reports": stats}, f, indent=2)

    print(f"done — best {best['score']} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
