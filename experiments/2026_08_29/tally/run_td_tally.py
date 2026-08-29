"""The tally (cetele) on the balance rig: proposer / chooser split.

Beside the frozen 0.72 top layer, each template j gets:

    C[j, 13*13]   the cetele — counts of executed (left, right) level
                  pairs while j was the winner. A record of deeds;
                  reward never touches it.
    W[j, 13*13]   the values — dopamine's ledger, advantage-updated
                  only for the EXECUTED entry.
    V[j]          the scalar critic — the state BASELINE (its right
                  job; as a chooser it was refuted twice).

Phase N (notch): DAgger-style mixed episodes (teacher flies half,
  the frozen pupil flies half, the teacher labels every tick), and
  the teacher's command is notched into the winner's cetele.
  GATE: the pure mode read (argmax count) must reproduce ~0.72 —
  the cetele has to carry the policy before values may tilt it.

Phase V (value): teacher OFF — the pupil flies alone, acting by the
  mode with exploration (10% a random supported action, 5% a +-1
  neighbour jitter, so roads not taken get notches too). TD error
  delta = r + gamma*V[j'] - V[j] is the burst; V learns from it with
  traces; W[j, executed] learns alpha_w * delta with its own traces.
  Selection during learning stays mode+exploration (no W feedback:
  one clean policy-improvement step, not a moving target).

Then the sweep: greedy read = argmax(log1p(C) + beta*W) over
  supported entries, beta = 0 is the pure mode. If any beta > 0
  beats it, that is improvement earned from the drone's own
  experience alone.

Run:  .venv/bin/python experiments/2026_08_29/tally/run_td_tally.py
Env:  TT_CKPT TT_NOTCH TT_RL TT_GAMMA TT_LAM TT_AV TT_AW TT_EPS
      TT_NEIGH TT_EVAL_EPS TT_TAG
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "temporal_drone"))      # shared rig modules
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))

import sl_drone as W                       # noqa: E402
import td_tracks as B                      # noqa: E402
import td3_stack as M                      # noqa: E402
import td_bound as T                       # noqa: E402
import td_balance as L                     # noqa: E402

CKPT = Path(os.environ.get(
    "TT_CKPT", str(HERE.parent / "balance" / "results" / "dagger3" / "weights_best.npz")))
OUT = HERE / "results" / os.environ.get('TT_TAG', '').lstrip("_")
NOTCH = int(os.environ.get("TT_NOTCH", "120000"))
RL = int(os.environ.get("TT_RL", "200000"))
GAMMA = float(os.environ.get("TT_GAMMA", "0.98"))
LAM = float(os.environ.get("TT_LAM", "0.9"))
AV = float(os.environ.get("TT_AV", "0.1"))          # critic step
AW = float(os.environ.get("TT_AW", "0.05"))         # tally-value step
EPS = float(os.environ.get("TT_EPS", "0.10"))       # supported explore
NEIGH = float(os.environ.get("TT_NEIGH", "0.05"))   # neighbour jitter
EVAL_EPS = int(os.environ.get("TT_EVAL_EPS", "50"))
NA = W.NLEV * W.NLEV
HOVER_LV = (W.NLEV // 2, W.NLEV // 2)


def a_of(lv):
    return lv[0] * W.NLEV + lv[1]


def lv_of(a):
    return (a // W.NLEV, a % W.NLEV)


class Rig:
    def __init__(self):
        cfg = json.load(open(CKPT.with_suffix(".json")))
        self.st = T.load_stack(np.load(CKPT), cfg)
        self.sense = L.make_sense(ns=cfg["ns"], vwarp=cfg["vwarp"],
                                  fovea=cfg.get("fovea", False))
        k = self.st.l3.k
        self.C = np.zeros((k, NA), np.float64)
        self.Wv = np.zeros((k, NA), np.float64)
        self.V = np.zeros(k, np.float64)

    def winner(self):
        st = self.st
        if st._codes is None or st.l3.n == 0:
            return -1
        i, v = st._sensor_cells()
        if len(i) == 0:
            return -1
        return int(np.argmax(st.l3.scores(i, v)))

    def joint_winner(self, lab):
        """The winner the LEARNING rule would pick: full vector with
        the executed command written, raw-dot argmax — mirrors
        Hyper.learn. MEASURED NECESSITY: notching by the sensor-only
        winner re-mixes the action-variants that the joint competition
        separated (cetele vs cells steering corr 0.55, ~0.06 N
        state-locked bias, EVAL 0.00 vs 0.72). The cetele row must be
        conditioned the way the cells were."""
        st = self.st
        if st._codes is None or st.l3.n == 0:
            return -1
        i, v = st._sensor_cells()
        if len(i) == 0:
            return -1
        c, d = M.levels_to_cd(lab)
        kc, wc = M.bump_idx(M.COLL_C, c)
        kd, wd = M.bump_idx(M.DIFF_C, d)
        half = np.sqrt(st.cmd_share / 2.0)
        cells = np.concatenate([i, st.dim_sensor + kc,
                                st.dim_sensor + T.NCMD + kd])
        vals = np.concatenate(
            [v, wc * (half / (np.linalg.norm(wc) + 1e-9)),
             wd * (half / (np.linalg.norm(wd) + 1e-9))]).astype(np.float32)
        raw = st.l3.W[:st.l3.n][:, cells] @ vals
        return int(np.argmax(raw))

    def cells_act(self, j):
        st = self.st
        row = st.l3.W[j]
        ds = st.dim_sensor
        pc = np.clip(row[ds:ds + T.NCMD], 0, None)
        pd = np.clip(row[ds + T.NCMD:], 0, None)
        c = B.sharpen(pc, M.COLL_C) if pc.max() > 0 else W.HOVER
        d = B.sharpen(pd, M.DIFF_C) if pd.max() > 0 else 0.0
        return M.cd_to_levels(c, d)

    def tally_act(self, j, beta):
        """Expected action under value-tilted duty weights.

        MEASURED (gate N, first run): the MODE read scores 0.02 vs the
        0.72 baseline with full support — within a template's watch the
        distribution is ~unimodal-but-skewed (mostly hover, a minority
        of corrections) and the DUTY CYCLE is the control signal; the
        mode collapses it to the majority action. Mode belongs at
        decision boundaries (the pole); regulation wants the mean.
        beta tilts the duty weights exponentially toward better-valued
        entries; beta = 0 is the pure count-weighted mean.
        """
        cnt = self.C[j]
        sup = np.nonzero(cnt)[0]
        if len(sup) == 0:
            return self.cells_act(j), False
        w = cnt[sup] * np.exp(np.clip(beta * self.Wv[j][sup], -10, 10))
        w /= w.sum()
        ls, rs = sup // W.NLEV, sup % W.NLEV
        c = float((w * (0.5 * (W.LEVELS[ls] + W.LEVELS[rs]))).sum())
        d = float((w * (0.5 * (W.LEVELS[rs] - W.LEVELS[ls]))).sum())
        return M.cd_to_levels(c, d), True


def evaluate(rig, beta, n, seed=99):
    rng = np.random.default_rng(seed)
    ok, unsup = [], 0
    for _ in range(n):
        s = L.air_init(rng)
        rig.st.reset()
        bal = []
        for t in range(L.EP_LEN):
            rig.st.push(rig.sense(s))
            j = rig.winner()
            if j < 0:
                lv = HOVER_LV
            else:
                lv, hit = rig.tally_act(j, beta)
                unsup += 0 if hit else 1
            s = W.physics(s, lv)
            bal.append(L.balanced(s))
        ok.append(all(bal[-150:]))
    return float(np.mean(ok)), unsup


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Rig()
    rng = np.random.default_rng(0)
    t0 = time.time()
    rep = {"ckpt": str(CKPT)}

    # ---- phase N: notch the cetele -------------------------------------
    s = L.air_init(rng)
    rig.st.reset()
    ep_tick, episodes = 0, 0
    teacher_ep = True
    for t in range(NOTCH):
        rig.st.push(rig.sense(s))
        j = rig.winner()
        lab = L.teacher(s)
        jn = rig.joint_winner(lab)
        if jn >= 0:
            rig.C[jn, a_of(lab)] += 1.0
        fly = lab if (teacher_ep or j < 0) else rig.cells_act(j)
        s = W.physics(s, fly)
        ep_tick += 1
        if ep_tick >= L.EP_LEN or np.hypot(s[2], s[3]) > 12:
            episodes += 1
            teacher_ep = episodes % 2 == 0
            s = L.air_init(rng)
            rig.st.reset()
            ep_tick = 0
    ev, unsup = evaluate(rig, 0.0, EVAL_EPS)
    rep["gate_mode_EVAL"] = round(ev, 3)
    rep["gate_unsupported_ticks"] = unsup
    rep["notch_support"] = round(float((rig.C.sum(1) > 0).mean()), 3)
    print(f"phase N done ({(time.time() - t0) / 60:.1f} min): cetele-read "
          f"EVAL {ev:.2f} (baseline 0.72), unsupported ticks {unsup}, "
          f"templates with support {rep['notch_support']:.0%}", flush=True)

    # ---- phase V: teacher off, pupil explores, values learn ------------
    eV = np.zeros_like(rig.V)
    eW = np.zeros_like(rig.Wv)
    s = L.air_init(rng)
    rig.st.reset()
    ep_tick, j_prev, a_prev, r_prev = 0, -1, -1, 0.0
    for t in range(RL):
        rig.st.push(rig.sense(s))
        j = rig.winner()
        if j < 0:
            lv, a = HOVER_LV, -1
        else:
            sup = np.nonzero(rig.C[j])[0]
            u = rng.random()
            if len(sup) and u < EPS:
                a = int(rng.choice(sup))
                lv = lv_of(a)
            else:
                lv, hit = rig.tally_act(j, 0.0)
                if u < EPS + NEIGH:
                    lv = (int(np.clip(lv[0] + rng.integers(-1, 2),
                                      0, W.NLEV - 1)),
                          int(np.clip(lv[1] + rng.integers(-1, 2),
                                      0, W.NLEV - 1)))
                a = a_of(lv)
            jn = rig.joint_winner(lv)   # deeds get notched, always —
            if jn >= 0:                 # attributed as learning would
                rig.C[jn, a] += 1.0
        s = W.physics(s, lv)
        r = 1.0 if L.balanced(s) else 0.0
        if j_prev >= 0 and j >= 0:
            delta = r_prev + GAMMA * rig.V[j] - rig.V[j_prev]
            eV *= GAMMA * LAM
            eV[j_prev] = 1.0
            rig.V += AV * delta * eV
            eW *= GAMMA * LAM
            if a_prev >= 0:
                eW[j_prev, a_prev] = 1.0
            rig.Wv += AW * delta * eW
        j_prev, a_prev, r_prev = j, a, r
        ep_tick += 1
        if ep_tick >= L.EP_LEN or np.hypot(s[2], s[3]) > 12:
            s = L.air_init(rng)
            rig.st.reset()
            ep_tick, j_prev, a_prev = 0, -1, -1
            eV[:] = 0.0
            eW[:] = 0.0
    print(f"phase V done ({(time.time() - t0) / 60:.1f} min): "
          f"V mean {rig.V.mean():.2f}, |W| mean "
          f"{np.abs(rig.Wv[rig.C > 0]).mean():.3f}", flush=True)

    # ---- the sweep ------------------------------------------------------
    rows = []
    for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
        ev, _ = evaluate(rig, beta, EVAL_EPS)
        rows.append({"beta": beta, "EVAL": round(ev, 3)})
        print(f"  beta {beta:<5} EVAL {ev:.2f}", flush=True)
    rep["sweep"] = rows
    rep["mins"] = round((time.time() - t0) / 60, 1)
    np.savez(OUT / "tally.npz", C=rig.C, W=rig.Wv, V=rig.V)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
