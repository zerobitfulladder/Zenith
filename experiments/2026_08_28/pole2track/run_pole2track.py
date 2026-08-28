"""Exp 17 — deep two-track swing-up pole (user spec), the run_pole_tracks retry.

Sensory track: [bumps(theta_t) ; bumps(theta_t-1)] — 64 periodic
overlapping Gaussian bumps per slice; velocity lives in the lagged
pair (today's direction-selective pattern; no trails, no EMA lag).
Motor track: the bipolar force thermometer. Each track has its own
bank; both terminate at a top unification memory over
cn([cn(sensory code) ; GM * cn(motor code)]) — motor half small gain
(key must dominate). Learning: imitation of the self-calibrating
energy-pump + PD-catch expert, then DAgger rounds. Evaluation: angle
in, sensory climbs, top retrieves, stored motor half reprojects
through the motor bank, force out. Recovery gate: upright-and-still
hold from uniform-random states (wide track, no walls — metrology law).

Arms (the parked bisect plan, one run):
  flat   top over [raw sensory ; GM*thermometer] — no banks; isolates
         the lagged-angle channel against the old trails config
         (47-58/100 with DAgger)
  ident  identity sensory + REAL motor bank (which recoding breaks?)
  full   both banks — the true two-track (history: parked at 0/100)

Motor readback in both modes (hard = winner motor unit's template;
graded = population sum) — sharp decisions want hard reads.

Run:  .venv/bin/python experiments/2026_08_28/pole2track/run_pole2track.py
Env:  P2_DEMOS (300), P2_ROUNDS (4), P2_EVAL (100), P2_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_swingup"))

import run_recon3 as R                                  # noqa: E402
from run_pole_ema import (                              # noqa: E402
    LEVELS, NB_F, THERMO_N, force_code, physics)
from run_pole_swingup import (                          # noqa: E402
    NB_A, angle_code, any_init, cn, energy, run_episode, wrap)


def _lqr_gain():
    """Discrete LQR for THIS simulator: Euler linearization about
    upright, u in [-1,1] units (force = u*FMAX)."""
    from run_pole_ema import DT, FMAX, G as GRAV, LP, MC, MP
    from scipy.linalg import solve_discrete_are
    mt = MC + MP
    den = LP * (4.0 / 3.0 - MP / mt)
    dth = GRAV / den                       # d(thetadd)/d(theta)
    dfu = -FMAX / (mt * den)               # d(thetadd)/du
    dxu = FMAX / mt - (MP * LP / mt) * dfu
    dxth = -(MP * LP / mt) * dth
    A = np.array([[0, 1, 0, 0],
                  [0, 0, dxth, 0],
                  [0, 0, 0, 1],
                  [0, 0, dth, 0]])
    B = np.array([[0.0], [dxu], [0.0], [dfu]])
    Ad = np.eye(4) + DT * A
    Bd = DT * B
    Q = np.diag([0.02, 0.05, 5.0, 1.0])
    Rm = np.array([[1.0]])
    P = solve_discrete_are(Ad, Bd, Q, Rm)
    K = np.linalg.solve(Rm + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K[0]


_KLQR = _lqr_gain()


def make_lqr_expert(ke, cent):
    """Energy-shaping swing-up + Riccati-optimal catch. State s is the
    teacher's privilege (oracles see everything; pupils see angles)."""
    def expert(s):
        th = wrap(s[2])
        if abs(th) < 0.5 and abs(s[3]) < 2.8:
            sv = np.array([s[0], s[1], th, s[3]])
            u = float(np.clip(-_KLQR @ sv, -1, 1))
        else:
            drive = s[3] * np.cos(th)
            sgn = np.sign(drive) if abs(drive) > 0.05 else 1.0
            u = ke * (0.0 - energy(s)) * sgn
            u += cent * np.clip(-0.05 * s[0] - 0.1 * s[1], -0.2, 0.2)
            u = float(np.clip(u, -1, 1))
        return int(np.argmin(np.abs(LEVELS - u)))
    return expert


def make_teacher_from_cfg(cfg):
    if cfg.get("type") == "lqr":
        return make_lqr_expert(cfg["ke"], cfg["cent"])
    return make_expert2(*cfg["params"])


def make_expert2(sign_k, kpump, cang, kp, kd):
    """Parameterized demonstrator: energy pump + PD catch."""
    def expert(s):
        th = wrap(s[2])
        if abs(th) < cang and abs(s[3]) < 2.5:
            u = np.clip(kp * th + kd * s[3], -1, 1)
        else:
            drive = s[3] * np.cos(th)
            if abs(drive) < 0.05:
                drive = 1.0
            u = np.clip(sign_k * kpump * (0.0 - energy(s)) * np.sign(drive),
                        -1, 1)
        return int(np.argmin(np.abs(LEVELS - u)))
    return expert

OUTPUT_DIR = HERE / "results" / os.environ.get('P2_TAG', '').lstrip('_')
N_DEMOS = int(os.environ.get("P2_DEMOS", "300"))
ROUNDS = int(os.environ.get("P2_ROUNDS", "4"))
EVAL_N = int(os.environ.get("P2_EVAL", "100"))
ARMS = os.environ.get("P2_ARMS", "flat,ident,full").split(",")
# Lag baselines for the delayed angle slices. Lag-1 measured ALIASED:
# at catch speeds opposite swing directions correlate 0.996 (one step
# moves ~3 deg under ~7 deg bumps). Two-hands law: short lag resolves
# fast spins, long lag resolves slow ones. Default [4, 16].
LAGS = [int(v) for v in os.environ.get("P2_LAGS", "4,16").split(",")]
MAXLAG = max(LAGS)
ROLLOUTS = 60
EPOCHS = 3
K_S, K_M = 256, 32
K_TOP = int(os.environ.get("P2_KTOP", "512"))
GM = 0.3                            # motor cargo gain (key must dominate)
GE = 0.5                            # efference trail gain in the KEY half
ALPHA_EFF = 0.4
ETA_S, ETA_M, ETA_TOP = 0.05, 0.05, 0.05
SDIM = (1 + len(LAGS)) * NB_A + NB_F   # angle slices + efference trail
EP_CAP = 700


# Foveated angle encoding: bump centers packed densely near upright
# (theta=0) so "balanced" and "slightly drifting" occupy different
# rows — required for zero-force and correction tallies to separate.
_u = (np.arange(NB_A) + 0.5) / NB_A * 2 - 1
FOV_C = np.sign(_u) * np.pi * np.abs(_u) ** 1.7
_sp = np.abs(np.diff(np.concatenate([FOV_C, [FOV_C[0] + 2 * np.pi]])))
FOV_SIG = 1.2 * _sp


def angle_code_fov(th):
    d = wrap(th - FOV_C)
    v = np.exp(-0.5 * (d / FOV_SIG) ** 2).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class AngleHist:
    """Past angles (delay lines, foveated bumps) + efference trail
    (fading echo of the commands issued — the motor track's KEY
    contribution)."""

    def __init__(self):
        self.buf = None
        self.eff = np.zeros(NB_F, np.float32)

    def see(self, th):
        if self.buf is None:
            self.buf = [th] * (MAXLAG + 1)
        self.buf.append(th)
        self.buf = self.buf[-(MAXLAG + 1):]

    def did(self, level):
        self.eff = ALPHA_EFF * force_code(level) + (1 - ALPHA_EFF) * self.eff

    def code(self):
        parts = [angle_code_fov(self.buf[-1])]
        parts += [angle_code_fov(self.buf[-1 - L]) for L in LAGS]
        parts.append(GE * self.eff)
        return cn(np.concatenate(parts))


def motor_in(level):
    return cn(force_code(level))


class Bank:
    def __init__(self, k, dim, eta, seed):
        self.layer = R.Layer(k, dim, eta, np.random.default_rng(seed))

    def learn(self, x_hat):
        self.layer.learn(x_hat, self.layer.forward(x_hat))

    def code(self, x_hat):
        return np.maximum(self.layer.forward(x_hat), 0.0)

    @property
    def W(self):
        return self.layer.W


def build_arm(arm, samples, seed):
    """samples: list of (s_in, level). Staged: banks, then top."""
    rng = np.random.default_rng(seed)
    bs = Bank(K_S, SDIM, ETA_S, seed) if arm == "full" else None
    bm = Bank(K_M, NB_F, ETA_M, seed + 1) if arm != "flat" else None
    if bs is not None:
        for _ in range(2):
            for i in rng.permutation(len(samples)):
                bs.learn(samples[i][0])
    if bm is not None:
        for lvl in list(range(NB_F + 1)) * 40:
            bm.learn(motor_in(lvl))

    def halves(s_hat, level):
        sh = cn(bs.code(s_hat)) if bs is not None else s_hat
        if bm is not None:
            mh = cn(bm.code(motor_in(level))) if level is not None \
                else np.zeros(K_M)
        else:
            mh = motor_in(level) if level is not None else np.zeros(NB_F)
        return sh, mh

    sdim = K_S if arm == "full" else SDIM
    mdim = K_M if arm != "flat" else NB_F
    # Action BESIDE the metric (hypercolumn law: label IN the metric
    # fails): rows match on sensory keys only; each row's action is the
    # tally of motor codes it won, read out after retrieval.
    top = R.Layer(K_TOP, sdim, ETA_TOP, np.random.default_rng(seed + 2))
    for _ in range(EPOCHS):
        for i in rng.permutation(len(samples)):
            s_hat, lvl = samples[i]
            sh, _ = halves(s_hat, lvl)
            top.learn(sh, top.forward(sh))
    tally = np.zeros((K_TOP, mdim))
    counts = np.zeros(K_TOP)
    for s_hat, lvl in samples:
        sh, mh = halves(s_hat, lvl)
        w = int(np.argmax(top.forward(sh)))
        tally[w] += mh
        counts[w] += 1
    tally /= np.maximum(counts, 1)[:, None]
    return {"arm": arm, "bs": bs, "bm": bm, "top": top,
            "sdim": sdim, "mdim": mdim, "tally": tally}


def act(model, s_hat, read="hard"):
    bs, bm, top = model["bs"], model["bm"], model["top"]
    sh = cn(bs.code(s_hat)) if bs is not None else s_hat
    mh = model["tally"][int(np.argmax(top.forward(sh)))]
    if bm is None:
        dirn = mh / (np.linalg.norm(mh) + 1e-9)
    elif read == "hard":
        u = int(np.argmax(mh))
        dirn = bm.W[u] / (np.linalg.norm(bm.W[u]) + 1e-9)
    else:
        d = np.maximum(mh, 0.0) @ bm.W
        dirn = d / (np.linalg.norm(d) + 1e-9)
    return int(np.argmax(THERMO_N @ dirn))


def make_policy(model, read):
    hist = AngleHist()

    def policy(s, t):
        if t == 0:
            hist.__init__()
        hist.see(s[2])
        lvl = act(model, hist.code(), read)
        hist.did(lvl)
        return lvl
    return policy


def tf_agreement(model, data, read, rng, n=3000):
    idx = rng.choice(len(data), size=min(n, len(data)), replace=False)
    return float(np.mean([act(model, data[i][0], read) == data[i][1]
                          for i in idx]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # The LQR teacher: audition pump strength x centering, pick best.
    best_cfg, best_ok = None, -1
    for ke in (-1.0, -5.0, -2.0, -3.0, 1.0, 2.0):
        for cent in (0.0, 1.0):
            ex = make_lqr_expert(ke, cent)
            ok = sum(int(run_episode(lambda s, t: ex(s), any_init(
                np.random.default_rng(50)))[1]) for _ in range(25))
            if ok > best_ok:
                best_cfg, best_ok = {"type": "lqr", "ke": ke,
                                     "cent": cent}, ok
    expert = make_teacher_from_cfg(best_cfg)
    erng = np.random.default_rng(60)
    exp_rate = np.mean([run_episode(lambda s, t: expert(s),
                                    any_init(erng))[1] for _ in range(60)])
    print(f"teacher {best_cfg} audition {best_ok}/25, rate {exp_rate:.2f}",
          flush=True)
    best_cfg["rate"] = float(exp_rate)
    with open(OUTPUT_DIR / "expert.json", "w") as f:
        json.dump(best_cfg, f)

    samples = []
    n = 0
    while n < N_DEMOS:
        _, ok, trace = run_episode(lambda s, t: expert(s), any_init(rng))
        if not ok:
            continue
        n += 1
        hist = AngleHist()
        for s, lvl in trace:
            hist.see(s[2])
            samples.append((hist.code(), lvl))
            hist.did(lvl)
        # balance tail: recovery demos END at the catch, so the dataset
        # is pump-rich and balance-poor — exactly where the pupil was
        # seen releasing all force. Keep the expert balancing (with
        # small shoves) so the catch region gets data mass.
        s_c = physics(trace[-1][0], LEVELS[trace[-1][1]])
        for _ in range(int(os.environ.get("P2_TAIL", "120"))):
            if rng.random() < 0.08:
                s_c[3] += (1 if rng.random() < 0.5 else -1) \
                    * rng.uniform(0.8, 2.0)
            hist.see(s_c[2])
            lvl = expert(s_c)
            samples.append((hist.code(), lvl))
            hist.did(lvl)
            s_c = physics(s_c, LEVELS[lvl])
    print(f"demos: {N_DEMOS} recoveries, {len(samples)} samples", flush=True)

    res = {"n_demos": N_DEMOS, "rounds": ROUNDS, "eval_n": EVAL_N,
           "lags": LAGS, "k": [K_S, K_M, K_TOP]}
    best = {"score": -1}
    for arm in ARMS:
        data = list(samples)
        for rnd in range(ROUNDS):
            model = build_arm(arm, data, seed=5 + rnd)
            reads = ("hard",) if arm == "flat" else ("hard", "graded")
            scores = {}
            for read in reads:
                vrng = np.random.default_rng(100 + rnd)
                oks = sum(int(run_episode(make_policy(model, read),
                                          any_init(vrng))[1])
                          for _ in range(EVAL_N))
                scores[read] = oks
                res[f"{arm}_r{rnd}_{read}"] = oks
                res[f"{arm}_r{rnd}_{read}_tf"] = round(tf_agreement(
                    model, data, read, np.random.default_rng(rnd)), 3)
            print(f"[{arm}] round {rnd}: {scores} (data {len(data)})",
                  flush=True)
            for read, oks in scores.items():
                if oks > best["score"]:
                    best = {"score": oks, "arm": arm, "read": read,
                            "round": rnd, "model": model}
            if rnd == ROUNDS - 1:
                break
            read0 = max(scores, key=lambda r: scores[r])
            crng = np.random.default_rng(200 + rnd)
            for _ in range(ROLLOUTS):
                s = any_init(crng)
                hist = AngleHist()
                for t in range(EP_CAP):
                    hist.see(s[2])
                    x = hist.code()
                    data.append((x, expert(s)))
                    lvl = act(model, x, read0)
                    hist.did(lvl)
                    s = physics(s, LEVELS[lvl])
                    if abs(s[0]) > 30.0 or (abs(wrap(s[2])) < np.deg2rad(1.5)
                                            and abs(s[3]) < 0.25):
                        break

    m = best["model"]
    np.savez(OUTPUT_DIR / "weights_best.npz",
             Wtop=m["top"].W, tally=m["tally"],
             Ws=(m["bs"].W if m["bs"] is not None else np.zeros((1, 1))),
             Wm=(m["bm"].W if m["bm"] is not None else np.zeros((1, 1))))
    res["best"] = {k: v for k, v in best.items() if k != "model"}
    res["best"]["lags"] = LAGS
    with open(OUTPUT_DIR / "config_best.json", "w") as f:
        json.dump(res["best"], f)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
