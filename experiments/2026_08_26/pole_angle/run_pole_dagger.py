"""Fixing the drift: the teacher labels the pupil's own visited states.

Diagnosis (measured): teacher-forced agreement 0.81 exact / 0.97
within-1 near upright, yet closed-loop recovery 5/100 — pure covariate
shift: the pupil's own +-1-level slips create trail-contexts no demo
contains. Cure: iterate — roll the pupil closed-loop, snapshot the
exact joint contexts it experiences (its OWN efference trail included),
label each with the expert's correction, aggregate, retrain (DAgger).

Run:  .venv/bin/python experiments/2026_08_26/pole_angle/run_pole_dagger.py
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

from run_temporal_digits import SeqDict  # noqa: E402
from run_pole_ema import (  # noqa: E402
    DIM,
    K,
    NB_A,
    NB_F,
    OUTPUT_DIR,
    THERMO_N,
    Trails,
    expert_level,
    recoverable_init,
    run_episode,
)

ROUNDS = 4
ROLLOUTS = 80
EVAL_N = 100
EPOCHS = 3
ALPHA_F = 0.4


def train_on(zs, seed):
    bank = SeqDict(K, DIM, 0.05, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(zs)):
            bank.step(zs[i])
    return bank


def make_policy(bank):
    tr = Trails(ALPHA_F)

    def policy(s, t):
        if t == 0:
            tr.__init__(ALPHA_F)
        tr.see(s[2])
        q = tr.joint()
        row = bank.W[int(np.argmax(bank.W @ q))]
        nf = row[NB_A * 2 + NB_F:]
        level = int(np.argmax(THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
        tr.did(level)
        return level
    return policy


def main():
    rng = np.random.default_rng(0)

    # Round 0 data: expert demos, contexts built with expert's own trail.
    zs = []
    n_demo = 0
    while n_demo < 300:
        n, ok, trace = run_episode(lambda s, t: expert_level(s),
                                   recoverable_init(rng))
        if not ok:
            continue
        n_demo += 1
        tr = Trails(ALPHA_F)
        for s, lvl in trace:
            tr.see(s[2])
            zs.append(tr.joint(next_level=lvl))
            tr.did(lvl)
    print(f"round 0: {len(zs)} expert-demo samples", flush=True)

    bank = None
    for rnd in range(ROUNDS):
        bank = train_on(zs, seed=5 + rnd)
        pol = make_policy(bank)
        erng = np.random.default_rng(100 + rnd)
        oks = sum(int(run_episode(pol, recoverable_init(erng))[1])
                  for _ in range(EVAL_N))
        print(f"round {rnd}: recovered {oks}/{EVAL_N} "
              f"(dataset {len(zs)})", flush=True)
        if rnd == ROUNDS - 1:
            break
        # Collect the pupil's OWN contexts, labeled by the expert.
        crng = np.random.default_rng(200 + rnd)
        new = 0
        for _ in range(ROLLOUTS):
            tr = Trails(ALPHA_F)
            s = recoverable_init(crng)
            for t in range(150):
                tr.see(s[2])
                q = tr.joint()
                row = bank.W[int(np.argmax(bank.W @ q))]
                nf = row[NB_A * 2 + NB_F:]
                lvl_pup = int(np.argmax(
                    THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
                zs.append(tr.joint(next_level=expert_level(s)))
                new += 1
                tr.did(lvl_pup)                 # pupil's action drives trail
                from run_pole_ema import LEVELS, physics, TH_FAIL, X_WIDE
                s = physics(s, LEVELS[lvl_pup])
                if abs(s[2]) > TH_FAIL or abs(s[0]) > X_WIDE:
                    break
        print(f"  collected {new} corrected samples", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz", W=bank.W,
             alpha_f=ALPHA_F, alpha_s=0.08)
    print("weights.npz updated (viewer-ready)", flush=True)


if __name__ == "__main__":
    main()
