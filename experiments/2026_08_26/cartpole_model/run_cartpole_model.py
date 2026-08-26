"""Cart-pole, two-phase: babble the world model, then give it a wish.

Phase 1 (warm-up, no goal): random actions; units learn
[state ; action ; NEXT state] — consequence associations. Dynamics are
STATIONARY, so unlike the RL run's value chase, the surprise gate can
close: prediction error should FALL (the calibration curve).

Phase 2 (goal, subtle): the goal is a STATE — upright, centered,
still — encoded like any percept. Control = ask the memory, per
action, "what does the world look like next?" and take the action
whose predicted future most resembles the wish. The goal lives at
INFERENCE (a standing query), never in the weights — the F-law's
placement of top-down signals. The only plasticity anywhere is
surprise-gated dynamics refinement.

Predictions (before running):
1. Phase-1 prediction error falls (stationary target -> calibration;
   direct fix of the RL run's binding constraint).
2. Goal-similarity greedy control clearly beats the RL plateau (~100);
   the goal code's velocity channels give free damping.
3. Babbling visits wild states -> stress test at least beats
   imitation-pure (165).
4. Watch-item: one-step myopia (overshoot oscillation) -> depth-2 arm
   if needed.

Run:  .venv/bin/python experiments/2026_08_26/cartpole_model/run_cartpole_model.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "cartpole"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_temporal_digits import cn  # noqa: E402
from run_cartpole import (  # noqa: E402
    EVAL_EPS,
    EVAL_T,
    F,
    SDIM,
    encode,
    failed,
    init_state,
    physics,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "cartpole_model" / "results"
GA, GN = 0.5, 1.0
DIM = SDIM + 2 + SDIM
K = 256
BABBLE_EPS, BABBLE_CAP = 400, 200
THETA_CAP = 0.3
THETA_GAIN = 1.5
EYE2 = np.eye(2, dtype=np.float32)
GOAL = encode(np.zeros(4))


class DynBank:
    def __init__(self, seed):
        self.W = np.zeros((K, DIM), dtype=np.float32)
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def predict(self, code, a):
        q = cn(np.concatenate([code, GA * EYE2[a], np.zeros(SDIM, np.float32)])
               ).astype(np.float32)
        if self.n == 0:
            return -1, np.zeros(SDIM, np.float32)
        u = int(np.argmax(self.W[:self.n] @ q))
        nxt = np.maximum(self.W[u, SDIM + 2:], 0.0)
        return u, nxt / (np.linalg.norm(nxt) + 1e-9)

    def learn(self, code, a, next_code):
        u, pred = self.predict(code, a)
        err = 1.0 - float(pred @ next_code)
        target = cn(np.concatenate([code, GA * EYE2[a], GN * next_code])
                    ).astype(np.float32)
        if self.n < K:
            w = target + (0.05 / np.sqrt(DIM)) * self.rng.standard_normal(
                DIM).astype(np.float32)
            self.W[self.n] = cn(w)
            self.n += 1
            return err
        w = self.W[u]
        cw = float(w @ target)
        tau = target - cw * w
        tn = np.linalg.norm(tau)
        if tn > 1e-9:
            th = min(THETA_GAIN * err + 0.005, THETA_CAP)
            self.W[u] = cn(w * np.cos(th) + (tau / tn) * np.sin(th))
        return err


def goal_policy(bank):
    def pol(s):
        code = encode(s)
        scores = [float(bank.predict(code, a)[1] @ GOAL) for a in (0, 1)]
        return int(np.argmax(scores))
    return pol


def rollout(pol, rng, tmax):
    s = init_state(rng)
    for t in range(tmax):
        s = physics(s, F if pol(s) == 1 else -F)
        if failed(s):
            return t + 1
    return tmax


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = DynBank(seed=0)
    rng = np.random.default_rng(1)

    errs = []
    for _ in range(BABBLE_EPS):
        s = init_state(rng)
        ep_err = []
        for _ in range(BABBLE_CAP):
            a = int(rng.integers(2))
            code = encode(s)
            s = physics(s, F if a == 1 else -F)
            ep_err.append(bank.learn(code, a, encode(s)))
            if failed(s):
                break
        errs.append(float(np.mean(ep_err)))
    print(f"babble: prediction error {errs[0]:.3f} (first ep) -> "
          f"{np.mean(errs[-20:]):.3f} (last 20)", flush=True)

    pol = goal_policy(bank)
    erng = np.random.default_rng(999)
    lens = [rollout(pol, erng, EVAL_T) for _ in range(EVAL_EPS)]
    print(f"goal-query control: mean {np.mean(lens):.0f}, "
          f"median {np.median(lens):.0f} / {EVAL_T} "
          f"(random 23, RL 97, imitation 491)", flush=True)

    def stress(pol):
        s = np.array([0, 0, np.deg2rad(7), 0], dtype=float)
        for t in range(400):
            if t == 140:
                s[3] += 1.2
            if t == 260:
                s[3] += -1.4
            s = physics(s, F if pol(s) == 1 else -F)
            if failed(s):
                return t + 1
        return 400
    st = stress(pol)
    print(f"stress (7deg + kicks): {st}/400 "
          f"(imitation pure=165, shoved=273, RL=108)", flush=True)

    w = 10
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(np.convolve(errs, np.ones(w) / w, "valid"))
    ax.set_xlabel("babble episode")
    ax.set_ylabel("mean prediction error")
    fig.suptitle("World-model calibration during babbling "
                 "(stationary target: surprise CAN close)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Two-phase: babbled world model + goal-as-query control",
         "",
         f"K={K}, {BABBLE_EPS} babble episodes. Prediction error "
         f"{errs[0]:.3f} -> {np.mean(errs[-20:]):.3f}.",
         f"Goal-query control: mean {np.mean(lens):.0f}, median "
         f"{np.median(lens):.0f} / {EVAL_T} (random 23, RL 97, "
         "imitation 491).",
         f"Stress: {st}/400 (imitation pure 165, shoved 273, RL 108).",
         "",
         "Figure: calibration.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
