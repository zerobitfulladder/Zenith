"""Recovery from ANY state: full-circle sensing + swing-up imitation.

User's observation-turned-spec: "fallen" was an arbitrary label — on a
wide track every cart-pole state is recoverable (rock to pump energy,
then catch). Sensing goes periodic: 64 angle bumps around the full
circle (hanging down is a percept, not a void). Demonstrator =
self-calibrating energy swing-up + PD catch. Demos = recoveries from
uniform-random states (any angle, spin up to +-3 rad/s), ending at
upright-and-still. Trails on angle + efference as before; bipolar
thermometer force; DAgger rounds against drift. K=512.

Run:  .venv/bin/python experiments/2026_08_26/pole_swingup/run_pole_swingup.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402
from run_pole_ema import (  # noqa: E402
    G,
    LP,
    MP,
    LEVELS,
    NB_F,
    THERMO_N,
    force_code,
    physics,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "pole_swingup" / "results"
NB_A = 64
AC = np.linspace(-np.pi, np.pi, NB_A, endpoint=False)
ASIG = 1.2 * (2 * np.pi / NB_A)
GS, GF, GN = 0.5, 0.5, 0.5
# LEAKY-SUM traces only (user's form): T = x + gamma*T — unnormalized,
# so the PRESENT always enters at weight 1.0 and the past is a fading
# tail behind it; no gamma can demote "now" below the past (the EMA's
# phase-lag trap is impossible by construction). Two gammas on the
# angle = the two clock hands (motion + phase context); one on force.
# MEASURED VERDICT (three-way): dedicated present channel + separate
# EMA tail = 58/100; single leaky-sum gamma .5 = 13; gamma .25 = 19.
# Present and past belong in SEPARATE channels (precision channel +
# context channel) — summing them smears the present bump into a ridge
# at any mixing ratio. Biology concurs: transient + sustained
# populations, not one summed signal. Restored winning config below.
GT = 0.6
ALPHA_TRAIL, ALPHA_SLOW = 0.4, 0.08
DIM = NB_A + NB_A + NB_A + NB_F + NB_F
K = 512
ALPHA_F, ALPHA_S = 0.4, 0.08
X_WIDE = 30.0
EP_CAP = 700
RECOVER_TH, RECOVER_TDH, RECOVER_HOLD = np.deg2rad(1.5), 0.25, 10
N_DEMOS = 300
EPOCHS = 3
ROUNDS = 4
ROLLOUTS = 60
EVAL_N = 100


def wrap(th):
    return ((th + np.pi) % (2 * np.pi)) - np.pi


def angle_code(th):
    d = wrap(th - AC)
    v = np.exp(-0.5 * (d / ASIG) ** 2).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def energy(s):
    return 0.5 * MP * (LP * s[3]) ** 2 + MP * G * LP * (np.cos(wrap(s[2])) - 1)


def make_expert(sign_k):
    def expert(s):
        th = wrap(s[2])
        if abs(th) < 0.35 and abs(s[3]) < 2.5:
            u = np.clip(9.0 * th + 2.0 * s[3], -1, 1)
        else:
            drive = s[3] * np.cos(th)
            if abs(drive) < 0.05:
                drive = 1.0
            u = np.clip(sign_k * 3.0 * (0.0 - energy(s)) * np.sign(drive),
                        -1, 1)
        return int(np.argmin(np.abs(LEVELS - u)))
    return expert


def any_init(rng):
    return np.array([0.0, rng.uniform(-1, 1),
                     rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3)])


def run_episode(policy, s0, cap=EP_CAP):
    s = s0.copy()
    hold = 0
    trace = []
    for t in range(cap):
        level = policy(s, t)
        trace.append((s.copy(), level))
        s = physics(s, LEVELS[level])
        if abs(s[0]) > X_WIDE:
            return t + 1, False, trace
        if abs(wrap(s[2])) < RECOVER_TH and abs(s[3]) < RECOVER_TDH:
            hold += 1
            if hold >= RECOVER_HOLD:
                return t + 1, True, trace
        else:
            hold = 0
    return cap, False, trace


class Trails:
    """Present channel + separate EMA tails (the measured winner)."""

    def __init__(self):
        self.A = np.zeros(NB_A, np.float32)
        self.Fa = np.zeros(NB_A, np.float32)
        self.Sa = np.zeros(NB_A, np.float32)
        self.Ff = np.zeros(NB_F, np.float32)

    def see(self, th):
        a = angle_code(th)
        self.A = a
        self.Fa = ALPHA_TRAIL * a + (1 - ALPHA_TRAIL) * self.Fa
        self.Sa = ALPHA_SLOW * a + (1 - ALPHA_SLOW) * self.Sa

    def did(self, level):
        f = force_code(level)
        self.Ff = ALPHA_TRAIL * f + (1 - ALPHA_TRAIL) * self.Ff

    def joint(self, next_level=None):
        nf = (force_code(next_level) if next_level is not None
              else np.zeros(NB_F, np.float32))
        return cn(np.concatenate(
            [self.A, GT * self.Fa, GS * self.Sa, GF * self.Ff, GN * nf]
        )).astype(np.float32)


def make_policy(bank):
    tr = Trails()

    def policy(s, t):
        if t == 0:
            tr.__init__()
        tr.see(s[2])
        q = tr.joint()
        row = bank.W[int(np.argmax(bank.W @ q))]
        nf = row[NB_A * 3 + NB_F:]
        level = int(np.argmax(THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
        tr.did(level)
        return level
    return policy


def train_on(zs, seed):
    bank = SeqDict(K, DIM, 0.05, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(zs)):
            bank.step(zs[i])
    return bank


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Self-calibrate the swing-up sign.
    best_sign, best_ok = 1, -1
    for sign_k in (1, -1):
        ex = make_expert(sign_k)
        ok = sum(int(run_episode(lambda s, t: ex(s), any_init(
            np.random.default_rng(50)))[1]) for _ in range(20))
        print(f"expert sign {sign_k:+d}: {ok}/20", flush=True)
        if ok > best_ok:
            best_sign, best_ok = sign_k, ok
    expert = make_expert(best_sign)
    erng = np.random.default_rng(60)
    exp_rate = np.mean([run_episode(lambda s, t: expert(s),
                                    any_init(erng))[1] for _ in range(50)])
    print(f"expert (sign {best_sign:+d}): {exp_rate:.2f} recovery rate",
          flush=True)

    zs = []
    n = 0
    while n < N_DEMOS:
        _, ok, trace = run_episode(lambda s, t: expert(s), any_init(rng))
        if not ok:
            continue
        n += 1
        tr = Trails()
        for s, lvl in trace:
            tr.see(s[2])
            zs.append(tr.joint(next_level=lvl))
            tr.did(lvl)
    print(f"demos: {N_DEMOS} recoveries, {len(zs)} samples", flush=True)

    bank = None
    for rnd in range(ROUNDS):
        bank = train_on(zs, seed=5 + rnd)
        pol = make_policy(bank)
        vrng = np.random.default_rng(100 + rnd)
        oks = sum(int(run_episode(pol, any_init(vrng))[1])
                  for _ in range(EVAL_N))
        print(f"round {rnd}: recovered {oks}/{EVAL_N} "
              f"(dataset {len(zs)})", flush=True)
        if rnd == ROUNDS - 1:
            break
        crng = np.random.default_rng(200 + rnd)
        for _ in range(ROLLOUTS):
            tr = Trails()
            s = any_init(crng)
            for t in range(EP_CAP):
                tr.see(s[2])
                q = tr.joint()
                row = bank.W[int(np.argmax(bank.W @ q))]
                nf = row[NB_A * 3 + NB_F:]
                lvl = int(np.argmax(
                    THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
                zs.append(tr.joint(next_level=expert(s)))
                tr.did(lvl)
                s = physics(s, LEVELS[lvl])
                if abs(s[0]) > X_WIDE:
                    break
                if (abs(wrap(s[2])) < RECOVER_TH
                        and abs(s[3]) < RECOVER_TDH):
                    break

    np.savez(OUTPUT_DIR / "weights.npz", W=bank.W)
    print("weights.npz saved (swingup_viewer-ready)", flush=True)

    (OUTPUT_DIR / "report.md").write_text(
        f"# Swing-up from any state\n\nExpert rate {exp_rate:.2f}; "
        f"see stdout for round-by-round pupil recovery.\n")
    print("Report written.")


if __name__ == "__main__":
    main()
