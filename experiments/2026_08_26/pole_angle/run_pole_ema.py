"""Pole recovery by imitation: angle-only sensing, EMA trails, continuous
thermometer-coded force.

User spec: the network senses ONLY the pole angle (population bumps).
Instantaneous angle is direction-ambiguous (falling vs recovering look
identical) — the EMA trails must carry velocity by themselves ("the
network is temporal inherently"). Force is continuous in [-1,1],
discretized to 9 levels, thermometer-coded over 8 bits (fill-from-left
= strength; all 0 = full left, all 1 = full right). Trails on BOTH the
sensed angle and the executed force (efference history). No dopamine:
pure imitation of a PD demonstrator. Demos = RECOVERY episodes: random
recoverable start, episode ends once upright-and-still is held.

Joint state per step: [fast angle trail ; 0.5*slow angle trail ;
0.5*force trail ; 0.5*NEXT force] — standard rule, top-1 read at
control time, next-force half decoded against the 9 canonical codes.

Arms: trails (alpha_f=0.4) vs NO-TRAIL control (alpha_f=1.0, trail =
instantaneous snapshot) — the figure-8 ablation in motor form.
Predictions (before running):
1. Trail arm recovers from most recoverable starts despite angle-only
   sensing — the trails reconstruct direction/velocity.
2. No-trail arm fails badly (direction-blind: cannot tell falling
   from recovering at the same angle).

Track is wide (|x|<10): angle-only sensing cannot regulate cart
position, so the wall is out of scope by design.

Run:  .venv/bin/python experiments/2026_08_26/pole_angle/run_pole_ema.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "pole_angle" / "results"
G, MC, MP, LP, DT, FMAX = 9.8, 1.0, 0.1, 0.5, 0.02, 10.0
TH_FAIL, X_WIDE = np.deg2rad(15), 10.0
NB_A = 24                            # angle bumps over +-15 deg
AC = np.linspace(-TH_FAIL, TH_FAIL, NB_A)
ASIG = 1.2 * (AC[1] - AC[0])
NB_F = 8                             # thermometer bits -> 9 force levels
LEVELS = np.linspace(-1.0, 1.0, NB_F + 1)
GS, GF, GN = 0.5, 0.5, 0.5
DIM = NB_A + NB_A + NB_F + NB_F
K = 256
ALPHA_S = 0.08
DEMO_TRIES, DEMO_CAP = 300, 300
EPOCHS = 3
EVAL_N = 100
RECOVER_TH, RECOVER_TDH, RECOVER_HOLD = np.deg2rad(1.0), 0.2, 10

# Bipolar thermometer: level k -> first k bits +1, rest -1. Equal norm
# for every level; no level aliases the empty query half (the 0/1
# version's all-zero "full left" was correlation-invisible and its
# units aliased empty-next queries — measured catastrophic).
THERMO = -np.ones((NB_F + 1, NB_F), dtype=np.float32)
for k in range(NB_F + 1):
    THERMO[k, :k] = 1.0
THERMO_N = THERMO / (np.linalg.norm(THERMO, axis=1, keepdims=True) + 1e-9)


def physics(s, u):
    x, xd, th, thd = s
    force = float(u) * FMAX
    ct, st = np.cos(th), np.sin(th)
    tmp = (force + MP * LP * thd ** 2 * st) / (MC + MP)
    tha = (G * st - ct * tmp) / (LP * (4.0 / 3.0 - MP * ct ** 2 / (MC + MP)))
    xa = tmp - MP * LP * tha * ct / (MC + MP)
    return np.array([x + DT * xd, xd + DT * xa, th + DT * thd, thd + DT * tha])


def angle_code(th):
    v = np.exp(-0.5 * ((th - AC) / ASIG) ** 2).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def force_code(level):
    return THERMO_N[level]


def expert_level(s):
    u = np.clip(9.0 * s[2] + 2.0 * s[3], -1, 1)
    return int(np.argmin(np.abs(LEVELS - u)))


def recoverable_init(rng):
    return np.array([0.0, rng.uniform(-1.0, 1.0),
                     rng.uniform(-np.deg2rad(10), np.deg2rad(10)),
                     rng.uniform(-1.2, 1.2)])


def run_episode(policy, s0, cap=DEMO_CAP):
    """policy(obs_th, trails)->level; returns (steps, recovered, trace)."""
    s = s0.copy()
    hold = 0
    trace = []
    for t in range(cap):
        level = policy(s, t)
        trace.append((s.copy(), level))
        s = physics(s, LEVELS[level])
        if abs(s[2]) > TH_FAIL or abs(s[0]) > X_WIDE:
            return t + 1, False, trace
        if abs(s[2]) < RECOVER_TH and abs(s[3]) < RECOVER_TDH:
            hold += 1
            if hold >= RECOVER_HOLD:
                return t + 1, True, trace
        else:
            hold = 0
    return cap, False, trace


class Trails:
    def __init__(self, alpha_f):
        self.af = alpha_f
        self.Fa = np.zeros(NB_A, np.float32)
        self.Sa = np.zeros(NB_A, np.float32)
        self.Ff = np.zeros(NB_F, np.float32)

    def see(self, th):
        a = angle_code(th)
        self.Fa = self.af * a + (1 - self.af) * self.Fa
        self.Sa = ALPHA_S * a + (1 - ALPHA_S) * self.Sa

    def did(self, level):
        f = force_code(level)
        self.Ff = self.af * f + (1 - self.af) * self.Ff

    def joint(self, next_level=None):
        nf = (force_code(next_level) if next_level is not None
              else np.zeros(NB_F, np.float32))
        return cn(np.concatenate(
            [self.Fa, GS * self.Sa, GF * self.Ff, GN * nf])).astype(np.float32)


def train_arm(alpha_f, demos, seed):
    bank = SeqDict(K, DIM, 0.05, seed=seed)
    for _ in range(EPOCHS):
        for trace in demos:
            tr = Trails(alpha_f)
            for s, level in trace:
                tr.see(s[2])
                bank.step(tr.joint(next_level=level))
                tr.did(level)
    return bank


def make_policy(bank, alpha_f):
    tr = Trails(alpha_f)

    def policy(s, t):
        if t == 0:
            tr.__init__(alpha_f)
        tr.see(s[2])
        q = tr.joint()
        row = bank.W[int(np.argmax(bank.W @ q))]
        nf = row[NB_A * 2 + NB_F:]          # bipolar: no relu
        level = int(np.argmax(THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
        tr.did(level)
        return level
    return policy


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    demos, exp_ok = [], 0
    for _ in range(DEMO_TRIES):
        n, ok, trace = run_episode(lambda s, t: expert_level(s),
                                   recoverable_init(rng))
        if ok:
            demos.append(trace)
            exp_ok += 1
    print(f"expert demos: {exp_ok}/{DEMO_TRIES} recovered "
          f"(mean len {np.mean([len(d) for d in demos]):.0f})", flush=True)

    results = {}
    for name, af in [("trails", 0.4), ("no-trail", 1.0)]:
        bank = train_arm(af, demos, seed=5)
        pol = make_policy(bank, af)
        erng = np.random.default_rng(7)
        oks, lens = 0, []
        for _ in range(EVAL_N):
            n, ok, _ = run_episode(pol, recoverable_init(erng))
            oks += int(ok)
            lens.append(n)
        results[name] = (oks, float(np.mean(lens)))
        print(f"{name}: recovered {oks}/{EVAL_N} "
              f"(mean episode {np.mean(lens):.0f} steps)", flush=True)
        if name == "trails":
            np.savez(OUTPUT_DIR / "weights.npz", W=bank.W,
                     alpha_f=af, alpha_s=ALPHA_S)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Angle-only pole recovery: trails vs no-trail",
         "",
         f"Expert: {exp_ok}/{DEMO_TRIES} demo recoveries.",
         "",
         "| arm | recovered | mean episode |",
         "|---|---|---|"]
        + [f"| {n} | {r[0]}/{EVAL_N} | {r[1]:.0f} |"
           for n, r in results.items()]
        + ["", "weights.npz = trails arm (for pole_viewer.py)"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
