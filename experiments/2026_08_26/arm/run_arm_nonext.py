"""Arm reaching, USER'S NO-NEXT-SLOT design: time from the lag itself.

Association = plain co-occurrence [target ; present pose ; trails] —
no dedicated next-half. Inference queries with the PRESENT half empty
and the trails filled: the winner is a demonstrated moment whose
trails match ours, and its stored PRESENT pose — which sits AHEAD of
those trails by the integration lag — is the emission. The temporal
advance comes from the leaky integration's internal offset, not from
an explicit future slot.

Predictions: works if the chain self-sustains (emitted pose shifts our
trails onto the next stored moment); risk = freezing at memory-cell
centroids (the absolute-pose disease). Reaching is a regulator
(lag-tolerant), so this is the mechanism's fairest arena.

Run:  .venv/bin/python experiments/2026_08_26/arm/run_arm_nonext.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "arm"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402
from run_arm import (  # noqa: E402
    EP_CAP,
    EVAL_N,
    JC,
    Motor,
    N_DEMOS,
    NB_J,
    TDIM,
    ZONE_R,
    HOLD,
    angles_code,
    expert_step,
    rand_angles,
    rand_target,
    target_code,
    tip,
    wrap,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "arm" / "results" / "nonext"
GP, GTR, GSL = 1.0, 0.6, 0.5
PDIM = 2 * NB_J                       # present pose bumps
TRDIM = 2 * 2 * NB_J                  # fast + slow trails
DIM = TDIM + PDIM + TRDIM
K = 512
EPOCHS = 3
ROUNDS = 3
ROLLOUTS = 60


def decode_pose(half):
    out = []
    for j in range(2):
        w = np.maximum(half[j * NB_J:(j + 1) * NB_J], 0.0)
        out.append(np.arctan2((w * np.sin(JC)).sum(), (w * np.cos(JC)).sum()))
    return np.array(out)


def build_z(tc, m, with_present=True):
    p = m.P if with_present else np.zeros(PDIM, np.float32)
    return cn(np.concatenate(
        [tc, GP * p, GTR * m.Fa, GSL * m.Sa])).astype(np.float32)


def make_policy(uni):
    def policy(th, tgt, m):
        q = build_z(target_code(tgt), m, with_present=False)
        row = uni.W[int(np.argmax(uni.W @ q))]
        return decode_pose(row[TDIM:TDIM + PDIM])
    return policy


def run_episode(policy, th0, tgt, cap=EP_CAP):
    th = th0.copy()
    m = Motor(th)
    hold = 0
    for t in range(cap):
        th = policy(th, tgt, m)
        m.see(th)
        if np.linalg.norm(tip(th) - tgt) < ZONE_R:
            hold += 1
            if hold >= HOLD:
                return True, t + 1
        else:
            hold = 0
    return False, cap


def train_on(zs, seed):
    uni = SeqDict(K, DIM, 0.05, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(zs)):
            uni.step(zs[i])
    return uni


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    zs = []
    n_ok = 0
    for _ in range(N_DEMOS):
        tgt, th = rand_target(rng), rand_angles(rng)
        m = Motor(th)
        tc = target_code(tgt)
        hold, ok, poses = 0, False, []
        for t in range(EP_CAP):
            nxt = expert_step(th, tgt)
            th = nxt
            m.see(th)                     # present now = the moved pose
            poses.append(build_z(tc, m))  # store [tgt; present; trails]
            if np.linalg.norm(tip(th) - tgt) < ZONE_R:
                hold += 1
                if hold >= HOLD:
                    ok = True
                    break
            else:
                hold = 0
        if ok:
            n_ok += 1
            zs.extend(poses)
    print(f"expert demos: {n_ok} reached, {len(zs)} samples", flush=True)

    uni = None
    for rnd in range(ROUNDS):
        uni = train_on(zs, seed=5 + rnd)
        pol = make_policy(uni)
        vrng = np.random.default_rng(100 + rnd)
        oks = sum(int(run_episode(pol, rand_angles(vrng),
                                  rand_target(vrng))[0])
                  for _ in range(EVAL_N))
        print(f"round {rnd}: reached {oks}/{EVAL_N} (dataset {len(zs)})",
              flush=True)
        if rnd == ROUNDS - 1:
            break
        crng = np.random.default_rng(200 + rnd)
        for _ in range(ROLLOUTS):
            tgt, th = rand_target(crng), rand_angles(crng)
            m = Motor(th)
            tc = target_code(tgt)
            for t in range(EP_CAP):
                q = build_z(tc, m, with_present=False)
                row = uni.W[int(np.argmax(uni.W @ q))]
                th_pup = decode_pose(row[TDIM:TDIM + PDIM])
                # expert-corrected sample from the pupil's context:
                nxt_exp = expert_step(th, tgt)
                m2 = Motor(th)
                m2.Fa, m2.Sa = m.Fa.copy(), m.Sa.copy()
                m2.see(nxt_exp)
                zs.append(build_z(tc, m2))
                th = th_pup
                m.see(th)
                if np.linalg.norm(tip(th) - tgt) < ZONE_R:
                    break

    np.savez(OUTPUT_DIR / "weights.npz", W=uni.W)
    print("weights saved", flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Arm reaching, no-next-slot (advance-by-lag)\n\n"
        "See stdout; compare arm/results/delta (delta-command form).\n")
    print("Report written.")


if __name__ == "__main__":
    main()
