"""The deep outer (user's spec): layers back in the champion's seat.

    channels   dx, dy, angle — target-relative position and tilt.
               NO velocity channels: motion exists only in history,
               so the temporal hierarchy has a real job.
    L1         per channel: 5 taps, 3 ticks apart (240 ms), K=64,
               graded top-8 code. L2: 7 codes, 6 apart (720 ms),
               K=128, top-8. The td_tracks machinery, batched.
    L3         [present bumps (50%) | 3 tracks' L2 codes (50%)],
               K=512 — with the SAME cetele over the SAME 32-word
               velocity vocabulary, duty-mean read at 10 Hz, the
               same smoothing and the same soft reflex underneath.

    The bar: the memoryless champion (100/100, closest 0.063 m,
    median 266). The question: can the layered temporal cue carry
    the same outer job — and does history help or does the delay
    disease return?

Run:  .venv/bin/python experiments/2026_08_29/deep_outer/gpu_deep_outer.py
Env:  GO_TICKS GO_REPORT GO_TAG (stages fixed inside)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chase"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascade"))
from gpu_chase import (xp, asnumpy, physics, at_goal, wrap,   # noqa: E402
                       LEVELS, bump, Layer, init_worlds, graded,
                       NB, EP_CAP, TMAX)
import gpu_cascade as C                                       # noqa: E402
from gpu_cascade import (vcmd, code_vc, snap_levels,          # noqa: E402
                         inner_label, fov, OUTER_EVERY)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get('GO_TAG', '').lstrip("_")
B = C.B
TICKS = int(os.environ.get("GO_TICKS", "1200000"))
REPORT = int(os.environ.get("GO_REPORT", "200000"))
# BANKS: the vocabulary's curriculum. 20000/128 worlds = 156 ticks
# per world — the teacher needs ~170 to ARRIVE, so the bank never
# heard a single "stay" command: the drunk-pilot bug. Default now
# spans full episodes, dwell included.
BANKS = int(os.environ.get("GO_BANKS", "240000"))
STG1, STG2, STG3 = BANKS + 40000, BANKS + 120000, BANKS + 220000
NS = int(os.environ.get("GO_NS", "3"))
T1, ST1, T2, ST2 = 5, 3, 7, 6
HIST = (T2 - 1) * ST2 + (T1 - 1) * ST1 + 2
RING = (T2 - 1) * ST2 + 1
K1, K2, K3 = 64, 128, 512
PRES = 0.5
DPRES = NS * NB
D3 = DPRES + NS * K2
FOVS = [(0.25, 30.0), (0.25, 30.0), (0.05, float(np.pi))]
ALPHA = 0.15


class DeepOuter:
    def __init__(self, seed=0):
        self.l1 = [Layer(K1, T1 * NB, seed=seed + 7 * c)
                   for c in range(NS)]
        self.l2 = [Layer(K2, T2 * K1, seed=seed + 7 * c + 1)
                   for c in range(NS)]
        self.top = C.Stage(K3, D3, 32, seed + 99)
        self.hist = xp.zeros((B, HIST, NS), xp.float32)
        self.ring = xp.zeros((B, RING, NS, K1), xp.float32)
        self.nw = xp.zeros(B, xp.int64)
        self.shares = None      # 2*NS block shares (pres..., ctx...);
        self._last_codes = None  # None = the fixed 50/50 default

    def reset(self, mask):
        m = xp.asarray(mask)
        self.hist[m] = 0
        self.ring[m] = 0
        self.nw = xp.where(m, 0, self.nw)

    def sense(self, S, T):
        raw = [S[:, 0] - T[:, 0], S[:, 1] - T[:, 1],
               wrap(S[:, 4])][:NS]
        return xp.stack([fov(raw[i], *FOVS[i]) for i in range(NS)],
                        axis=1)

    def push(self, sv, lm, stage):
        self.hist = xp.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = sv
        self.nw += 1
        ready = self.nw >= HIST
        idx = HIST - 1 - (T1 - 1 - np.arange(T1)) * ST1
        self.ring = xp.roll(self.ring, -1, axis=1)
        self.ring[:, -1] = 0
        codes = xp.zeros((B, NS, K2), xp.float32)
        for c in range(NS):
            X1 = bump(self.hist[:, idx, c]).reshape(B, T1 * NB)
            if stage == 0:
                sel = xp.asarray(lm) & ready
                if bool(sel.any()):
                    self.l1[c].learn_rows(X1[sel])
            s1 = self.l1[c].scores(X1)
            if s1.shape[1]:
                g1 = xp.zeros((B, K1), xp.float32)
                g1[:, :s1.shape[1]] = graded(s1)
                self.ring[:, -1, c] = g1 * ready[:, None]
            taps = self.ring[:, RING - 1 - (T2 - 1 - np.arange(T2))
                             * ST2, c]
            X2 = taps.reshape(B, T2 * K1)
            if stage == 1:
                sel = (xp.asarray(lm) & ready
                       & (xp.abs(X2).sum(axis=1) > 0))
                if bool(sel.any()):
                    self.l2[c].learn_rows(X2[sel])
            s2 = self.l2[c].scores(X2)
            if s2.shape[1]:
                g2 = xp.zeros((B, K2), xp.float32)
                g2[:, :s2.shape[1]] = graded(s2)
                codes[:, c] = g2 * ready[:, None]
        self._last_codes = codes
        pb = bump(sv)
        pn = xp.linalg.norm(pb, axis=2, keepdims=True) + 1e-9
        pb = pb / pn
        cn = xp.linalg.norm(codes, axis=2, keepdims=True)
        codes = xp.where(cn > 1e-9, codes / (cn + 1e-9), 0)
        if self.shares is None:
            ps = [PRES / NS] * NS
            cs = [(1 - PRES) / NS] * NS
        else:
            ps, cs = self.shares[:NS], self.shares[NS:]
        for c in range(NS):
            pb[:, c] *= np.sqrt(max(ps[c], 0.0))
            codes[:, c] *= np.sqrt(max(cs[c], 0.0))
        return xp.concatenate([pb.reshape(B, DPRES),
                               codes.reshape(B, NS * K2)], axis=1)

    def state(self):
        return (self.hist.copy(), self.ring.copy(), self.nw.copy())

    def restore(self, st):
        self.hist, self.ring, self.nw = st


def evaluate(rig, vc_units, rng):
    saved = rig.state()
    S, T = init_worlds(rng, B)
    rig.reset(np.ones(B, bool))
    vc = xp.zeros((B, 2), xp.float32)
    vc_raw = xp.zeros((B, 2), xp.float32)
    hold = xp.zeros(B, xp.int32)
    tdone = xp.full(B, 9999, xp.int32)
    best = xp.full(B, 1e9, xp.float32)
    for t in range(EP_CAP):
        cue = rig.push(rig.sense(S, T), np.zeros(B, bool), 3)
        if t % OUTER_EVERY == 0:
            w = rig.top.winners(cue)
            v, ok = rig.top.read_mean(w, vc_units)
            vc_raw = xp.where(ok[:, None], v, 0)
        vc = vc + ALPHA * (vc_raw - vc)
        li, ri = snap_levels(*inner_label(S, vc))
        S = physics(S, li, ri)
        best = xp.minimum(best, xp.hypot(S[:, 0] - T[:, 0],
                                         S[:, 1] - T[:, 1]))
        g = at_goal(S, T)
        hold = xp.where(g, hold + 1, 0)
        tdone = xp.where((hold >= 10) & (tdone > 9000), t, tdone)
    rig.restore(saved)
    ok = tdone < 9000
    med = float(xp.median(tdone[ok])) if bool(ok.any()) else -1
    return float(ok.mean()), float(best.mean()), med


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = DeepOuter()
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(99)
    S, T = init_worlds(rng, B)
    ep_tick = np.zeros(B, np.int64)
    stats, best = [], {"score": -1.0, "closest": 99.0}
    vc_units = None
    t0 = time.time()

    for step in range(TICKS // B):
        te = step * B
        stage = 0 if te < STG1 else (1 if te < STG2 else
                                     (2 if te < STG3 else 3))
        lm = (np.arange(B) + step) % 5 == 0
        sv = rig.sense(S, T)
        cue = rig.push(sv, lm, stage)
        lab_vc = xp.stack([vcmd(S[:, 0] - T[:, 0]),
                           vcmd(S[:, 1] - T[:, 1])], axis=1)
        if te < BANKS:
            rig.top.bank.learn_rows(code_vc(lab_vc)[xp.asarray(lm)])
        if stage == 2:
            sel = xp.asarray(lm) & (rig.nw >= 1)
            if bool(sel.any()):
                rig.top.top.learn_rows(cue[sel])
        if stage == 3:
            if vc_units is None:
                vc_units = np.array(
                    [C.decode_bank(rig.top.bank.W[u], 0.2, 8.0,
                                   0.2, 8.0) for u in range(32)],
                    np.float32)
            w = rig.top.winners(cue)
            rig.top.notch(w, code_vc(lab_vc), xp.ones(B, bool))
        li, ri = snap_levels(*inner_label(S, lab_vc))
        S = physics(S, li, ri)
        ep_tick += 1
        over = ep_tick >= EP_CAP
        if over.any():
            n = int(over.sum())
            ns_, nt = init_worlds(rng, n)
            om = xp.asarray(over)
            S[om] = ns_
            T[om] = nt
            rig.reset(over)
            ep_tick[over] = 0

        if (step + 1) % max(1, REPORT // B) == 0 and vc_units is not None:
            ev, near, med = evaluate(rig, vc_units, erng)
            row = {"exp_ticks": te + B, "EVAL": round(ev, 3),
                   "closest": round(near, 2), "median_ticks": med,
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            np.savez(OUT / "checkpoint.npz",
                     **{f"l1_{c}": asnumpy(rig.l1[c].W)
                        for c in range(NS)},
                     **{f"l2_{c}": asnumpy(rig.l2[c].W)
                        for c in range(NS)},
                     Wo=asnumpy(rig.top.top.W),
                     Bo=asnumpy(rig.top.bank.W),
                     Co=asnumpy(rig.top.C))
            if (ev, -near) > (best["score"], -best["closest"]):
                best = {"score": ev, "closest": near,
                        "median": med, "tick": te + B}
                import shutil
                shutil.copy(OUT / "checkpoint.npz",
                            OUT / "weights_best.npz")
                print(f"  new best {ev} (closest {near:.2f}, "
                      f"median {med:.0f})", flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "reports": stats}, f,
                          indent=2)

    print(f"done — best {best} ({(time.time() - t0) / 60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
