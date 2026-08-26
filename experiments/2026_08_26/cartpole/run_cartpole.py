"""Pole balancing by imitation: sensors in, motor commands out.

Physics: classic cart-pole (Euler, dt=0.02; fail at |theta|>12deg or
|x|>2.4). Demonstrator: hand-tuned bang-bang PD reflex. Sensory
encoding: semantic overlap — 16 overlapping Gaussian tuning bumps per
state variable (x, x_dot, theta, theta_dot) -> 64-dim population code.
Motor: left/right one-hot. The bank learns cn([sensor code ; 0.5*action])
with the standard rule; control queries with the action half empty and
applies the retrieved action — the bridge pattern, motor edition.

Arms: demos {pure expert | expert shoved randomly 15% of steps
(recoveries demonstrated)} x reads {top-1 | graded population vote}.
Predictions (before running):
1. Imitation works: nearest-prototype over population codes
   approximates the reflex; hundreds of steps.
2. SPAN law, control edition: pure-expert demos cover only the narrow
   balanced tube -> the pupil dies soon after drifting off it;
   perturbed demos include recoveries -> near-expert survival.

Run:  .venv/bin/python experiments/2026_08_26/cartpole/run_cartpole.py
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
from PIL import Image

from run_temporal_digits import SeqDict, cn  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "cartpole" / "results"
G, MC, MP, LP, DT, F = 9.8, 1.0, 0.1, 0.5, 0.02, 10.0
TH_FAIL, X_FAIL = np.deg2rad(12), 2.4
NB = 16                              # tuning bumps per variable
RANGES = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]
SDIM = 4 * NB
KU, ETA = 200, 0.05
DEMO_EPS, DEMO_T = 80, 400
EPOCHS = 2
EVAL_EPS, EVAL_T = 50, 500
P_SHOVE = 0.15


def physics(s, force):
    x, xd, th, thd = s
    ct, st = np.cos(th), np.sin(th)
    tmp = (force + MP * LP * thd ** 2 * st) / (MC + MP)
    tha = (G * st - ct * tmp) / (LP * (4.0 / 3.0 - MP * ct ** 2 / (MC + MP)))
    xa = tmp - MP * LP * tha * ct / (MC + MP)
    return np.array([x + DT * xd, xd + DT * xa, th + DT * thd, thd + DT * tha])


def failed(s):
    return abs(s[2]) > TH_FAIL or abs(s[0]) > X_FAIL


def expert(s):
    x, xd, th, thd = s
    return 1 if (0.5 * x + 1.0 * xd + 12.0 * th + 2.5 * thd) > 0 else 0


def encode(s):
    parts = []
    for v, (lo, hi) in zip(s, RANGES):
        centers = np.linspace(lo, hi, NB)
        sig = 1.2 * (centers[1] - centers[0])
        parts.append(np.exp(-0.5 * ((v - centers) / sig) ** 2))
    code = np.concatenate(parts).astype(np.float32)
    return code / (np.linalg.norm(code) + 1e-9)


def init_state(rng):
    return rng.uniform(-0.05, 0.05, 4)


def rollout_policy(policy, rng, tmax):
    s = init_state(rng)
    for t in range(tmax):
        s = physics(s, F if policy(s) == 1 else -F)
        if failed(s):
            return t + 1
    return tmax


def collect_demos(rng, shove):
    data = []
    for _ in range(DEMO_EPS):
        s = init_state(rng)
        for _ in range(DEMO_T):
            a = expert(s)
            if shove and rng.random() < P_SHOVE:
                a = int(rng.integers(2))
            data.append((encode(s), a))
            s = physics(s, F if a == 1 else -F)
            if failed(s):
                break
    return data


def train_bank(data, rng, seed):
    eye = np.eye(2, dtype=np.float32)
    bank = SeqDict(KU, SDIM + 2, ETA, seed=seed)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(data)):
            code, a = data[i]
            bank.step(cn(np.concatenate([code, 0.5 * eye[a]])
                         ).astype(np.float32))
    return bank


def make_policy(bank, mode):
    def policy(s):
        q = cn(np.concatenate([encode(s), np.zeros(2, np.float32)])
               ).astype(np.float32)
        sc = bank.W @ q
        if mode == "top1":
            row = bank.W[int(np.argmax(sc))]
            return int(np.argmax(row[SDIM:]))
        w = np.maximum(sc, 0.0) ** 4
        vote = (w[:, None] * bank.W[:, SDIM:]).sum(axis=0)
        return int(np.argmax(vote))
    return policy


def render_episode(policy, rng, tmax=300):
    frames = []
    s = init_state(rng)
    for _ in range(tmax):
        img = np.zeros((40, 80), dtype=np.float32)
        img[34, :] = 0.25
        cx = int(np.clip(40 + s[0] / X_FAIL * 36, 4, 76))
        img[30:34, max(0, cx - 4):cx + 4] = 0.7
        for i in range(22):
            px = int(round(cx + np.sin(s[2]) * i))
            py = int(round(30 - np.cos(s[2]) * i))
            if 0 <= px < 80 and 0 <= py < 40:
                img[py, px] = 1.0
        frames.append(img)
        s = physics(s, F if policy(s) == 1 else -F)
        if failed(s):
            break
    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    base = np.mean([rollout_policy(expert, rng, EVAL_T)
                    for _ in range(EVAL_EPS)])
    rand = np.mean([rollout_policy(
        lambda s: int(rng.integers(2)), rng, EVAL_T) for _ in range(EVAL_EPS)])
    print(f"expert baseline: {base:.0f} steps; random: {rand:.0f}", flush=True)

    rows = []
    best = None
    for shove in (False, True):
        data = collect_demos(np.random.default_rng(1), shove)
        bank = train_bank(data, np.random.default_rng(2), seed=7)
        for mode in ("top1", "graded"):
            pol = make_policy(bank, mode)
            erng = np.random.default_rng(3)
            lens = [rollout_policy(pol, erng, EVAL_T) for _ in range(EVAL_EPS)]
            name = f"{'shoved' if shove else 'pure'} {mode}"
            rows.append((name, float(np.mean(lens)), float(np.median(lens))))
            print(f"{name}: mean {np.mean(lens):.0f}, "
                  f"median {np.median(lens):.0f} / {EVAL_T}", flush=True)
            if best is None or np.mean(lens) > best[1]:
                best = ((shove, mode), np.mean(lens), pol)

    frames = render_episode(best[2], np.random.default_rng(9))
    imgs = [Image.fromarray((np.kron(f, np.ones((6, 6))) * 255
                             ).astype(np.uint8)) for f in frames]
    imgs[0].save(OUTPUT_DIR / "balance.gif", save_all=True,
                 append_images=imgs[1:], duration=30, loop=0)
    print(f"gif: {len(frames)} frames (best arm)", flush=True)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Cart-pole by imitation (population-coded sensors -> actions)",
         "",
         f"Expert baseline {base:.0f}, random {rand:.0f} (cap {EVAL_T}).",
         "",
         "| arm | mean | median |",
         "|---|---|---|"]
        + [f"| {n} | {m:.0f} | {md:.0f} |" for n, m, md in rows]
        + ["", "balance.gif = best arm episode."]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
