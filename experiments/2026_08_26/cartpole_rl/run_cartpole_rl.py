"""Cart-pole WITHOUT imitation: expectation-gated rotation (the morning's
dopamine mechanism, implemented in pure layers).

Each unit stores [sensor code ; g*action ; g*EXPECTED-OUTCOME] — the
expectation is a population-coded pathway in the unit's own weights.
Acting: query per action with the outcome half empty; decode each
winner's stored expectation; take the action whose owner expects more
(eps-greedy early). Learning: at episode end, every visited decision's
owning unit rotates toward the experienced truth [state ; action ;
actual outcome], with THETA SCALED BY THE PREDICTION ERROR — surprise
moves weights, fulfilled expectation moves nothing. No demonstrator,
no tally, no side structures: knowledge, policy, and expectation all
live in the same rows, moved by the same geodesic step.

Run:  .venv/bin/python experiments/2026_08_26/cartpole_rl/run_cartpole_rl.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "cartpole_rl" / "results"
NV = 11                              # value bumps over normalized return
VC = np.linspace(0.0, 1.0, NV)
VSIG = 1.2 * (VC[1] - VC[0])
GA, GV = 0.5, 0.5
DIM = SDIM + 2 + NV
K = 256
HORIZON = 200                        # return = min(steps survived, H)/H
EPISODES = 600
EP_CAP = 500
EPS0, EPS1 = 0.30, 0.02
THETA_CAP = 0.3
THETA_GAIN = 0.8                     # theta = min(GAIN*|delta| + 0.01, cap)
EYE2 = np.eye(2, dtype=np.float32)


def vcode(g):
    v = np.exp(-0.5 * ((g - VC) / VSIG) ** 2).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def vdecode(half):
    p = np.maximum(half, 0.0)
    s = p.sum()
    return float((p * VC).sum() / s) if s > 1e-9 else 0.5


class Bank:
    def __init__(self, seed):
        self.W = np.zeros((K, DIM), dtype=np.float32)
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def query(self, code, a):
        z = cn(np.concatenate([code, GA * EYE2[a], np.zeros(NV, np.float32)])
               ).astype(np.float32)
        if self.n == 0:
            return -1, 0.5
        u = int(np.argmax(self.W[:self.n] @ z))
        return u, vdecode(self.W[u, SDIM + 2:])

    def learn(self, code, a, ret):
        target = cn(np.concatenate([code, GA * EYE2[a], GV * vcode(ret)])
                    ).astype(np.float32)
        u, expected = self.query(code, a)
        delta = abs(ret - expected)
        if self.n < K:
            w = target + (0.05 / np.sqrt(DIM)) * self.rng.standard_normal(
                DIM).astype(np.float32)
            self.W[self.n] = cn(w)
            self.n += 1
            return delta
        w = self.W[u]
        cw = float(w @ target)
        tau = target - cw * w
        tn = np.linalg.norm(tau)
        if tn > 1e-9:
            th = min(THETA_GAIN * delta + 0.01, THETA_CAP)
            self.W[u] = cn(w * np.cos(th) + (tau / tn) * np.sin(th))
        return delta


def policy_greedy(bank):
    def pol(s):
        code = encode(s)
        return int(np.argmax([bank.query(code, a)[1] for a in (0, 1)]))
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
    bank = Bank(seed=0)
    rng = np.random.default_rng(1)
    curve, surprise = [], []

    for ep in range(EPISODES):
        eps = EPS0 + (EPS1 - EPS0) * ep / EPISODES
        s = init_state(rng)
        visited = []
        for t in range(EP_CAP):
            code = encode(s)
            if rng.random() < eps or bank.n < K:
                a = int(rng.integers(2))
            else:
                a = int(np.argmax([bank.query(code, x)[1] for x in (0, 1)]))
            visited.append((code, a))
            s = physics(s, F if a == 1 else -F)
            if failed(s):
                break
        T = len(visited)
        deltas = []
        for i, (code, a) in enumerate(visited):
            ret = min(T - i, HORIZON) / HORIZON
            deltas.append(bank.learn(code, a, ret))
        curve.append(T)
        surprise.append(float(np.mean(deltas)))
        if (ep + 1) % 50 == 0:
            erng = np.random.default_rng(100 + ep)
            pol = policy_greedy(bank)
            ev = np.mean([rollout(pol, erng, EVAL_T) for _ in range(10)])
            print(f"ep {ep + 1}: train len {np.mean(curve[-50:]):.0f}, "
                  f"surprise {np.mean(surprise[-50:]):.3f}, "
                  f"greedy eval {ev:.0f}/{EVAL_T}", flush=True)

    pol = policy_greedy(bank)
    erng = np.random.default_rng(999)
    lens = [rollout(pol, erng, EVAL_T) for _ in range(EVAL_EPS)]
    print(f"FINAL greedy: mean {np.mean(lens):.0f}, "
          f"median {np.median(lens):.0f} / {EVAL_T}", flush=True)

    # Stress test: 7-deg start + kicks (same protocol as imitation coda)
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
    print(f"stress (7deg + kicks): {stress(pol)}/400  "
          f"(imitation pure=165, shoved=273)", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    w = 20
    sm = np.convolve(curve, np.ones(w) / w, "valid")
    axes[0].plot(sm)
    axes[0].set_ylabel("episode length (smoothed)")
    axes[1].plot(np.convolve(surprise, np.ones(w) / w, "valid"))
    axes[1].set_ylabel("mean |prediction error|")
    axes[1].set_xlabel("episode")
    fig.suptitle("Self-taught cart-pole: performance up, surprise down")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Cart-pole without imitation: expectation-gated rotation",
         "",
         f"K={K}, {EPISODES} self-played episodes, theta = "
         f"min({THETA_GAIN}*|delta|+0.01, {THETA_CAP}).",
         f"Final greedy: mean {np.mean(lens):.0f}, median "
         f"{np.median(lens):.0f} / {EVAL_T} (random floor 23; imitation 491).",
         f"Stress: {stress(pol)}/400 (imitation pure 165, shoved 273).",
         "",
         "Figure: learning_curve.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
