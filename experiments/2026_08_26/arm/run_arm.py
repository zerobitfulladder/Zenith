"""Two-axis arm reaching: target-zone code + motor-angle code -> next angles.

Kinematic world (no momentum/damping/friction): the arm IS its angles.
Sensor track: zone center (x,y) as a 12x12 grid of overlapping 2D
Gaussian bumps (semantic: near targets share bumps). Motor track: two
joints, 16 circular bumps each, present + fast/slow EMA trails (the
channel-separation winner). Unification learns cn([target ; motor
state ; NEXT angles]); inference queries with the next-half empty and
the arm becomes the decoded angles (population-vector per joint).
Demonstrator: greedy +-3deg step descent on tip distance. DAgger.

Run:  .venv/bin/python experiments/2026_08_26/arm/run_arm.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "arm" / "results" / "delta"
L1, L2 = 1.0, 1.0
WS = 2.2                              # workspace half-width
GRID = 12                             # 12x12 target bumps
GC = np.linspace(-WS, WS, GRID)
GSIG = 1.2 * (GC[1] - GC[0])
NB_J = 16                             # circular bumps per joint
JC = np.linspace(-np.pi, np.pi, NB_J, endpoint=False)
JSIG = 1.2 * (2 * np.pi / NB_J)
A_F, A_S = 0.4, 0.08
GTR, GSL, GM, GN = 0.6, 0.5, 0.7, 0.5
TDIM = GRID * GRID                    # 144
MDIM = 3 * 2 * NB_J                   # present+fast+slow x 2 joints = 96
NDIM = 6                              # per-joint delta one-hot(3): -,0,+
DIM = TDIM + MDIM + NDIM
K = 2048   # 512 gave ~4.7 cells/axis in the 4D (target x pose) space;
           # three emission forms failed identically there — partition test
STEP = np.deg2rad(3.0)
ZONE_R = 0.15
HOLD = 10
EP_CAP = 400
N_DEMOS = 250
EPOCHS = 3
ROUNDS = 3
ROLLOUTS = 60
EVAL_N = 100


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def tip(th):
    x = L1 * np.cos(th[0]) + L2 * np.cos(th[0] + th[1])
    y = L1 * np.sin(th[0]) + L2 * np.sin(th[0] + th[1])
    return np.array([x, y])


def target_code(p):
    gx = np.exp(-0.5 * ((p[0] - GC) / GSIG) ** 2)
    gy = np.exp(-0.5 * ((p[1] - GC) / GSIG) ** 2)
    v = np.outer(gy, gx).ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def joint_code(a):
    d = wrap(a - JC)
    v = np.exp(-0.5 * (d / JSIG) ** 2).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def angles_code(th):
    return np.concatenate([joint_code(th[0]), joint_code(th[1])])


def delta_code(lvls):
    v = np.zeros(6, np.float32)
    v[int(lvls[0]) + 1] = 1.0
    v[3 + int(lvls[1]) + 1] = 1.0
    return v


def decode_delta(half):
    return np.array([int(np.argmax(half[:3])) - 1,
                     int(np.argmax(half[3:6])) - 1])


def lvls_between(th, nxt):
    d = wrap(nxt - th)
    return np.clip(np.round(d / STEP), -1, 1).astype(int)


def expert_step(th, tgt):
    best, bd = th, np.linalg.norm(tip(th) - tgt)
    for d1 in (-STEP, 0.0, STEP):
        for d2 in (-STEP, 0.0, STEP):
            c = np.array([wrap(th[0] + d1), wrap(th[1] + d2)])
            dd = np.linalg.norm(tip(c) - tgt)
            if dd < bd - 1e-9:
                best, bd = c, dd
    return best


def rand_target(rng):
    r = rng.uniform(0.4, 1.9)
    a = rng.uniform(-np.pi, np.pi)
    return np.array([r * np.cos(a), r * np.sin(a)])


def rand_angles(rng):
    return np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)])


class Motor:
    def __init__(self, th):
        self.P = angles_code(th)
        self.Fa = self.P.copy()
        self.Sa = self.P.copy()

    def see(self, th):
        a = angles_code(th)
        self.P = a
        self.Fa = A_F * a + (1 - A_F) * self.Fa
        self.Sa = A_S * a + (1 - A_S) * self.Sa

    def state(self):
        return np.concatenate([self.P, GTR * self.Fa, GSL * self.Sa])


def build_z(tc, m, next_lvls=None):
    nh = (delta_code(next_lvls) if next_lvls is not None
          else np.zeros(NDIM, np.float32))
    return cn(np.concatenate([tc, GM * m.state(), GN * nh])).astype(np.float32)


def run_episode(policy, th0, tgt, cap=EP_CAP):
    th = th0.copy()
    m = Motor(th)
    hold = 0
    trace = []
    for t in range(cap):
        nxt = policy(th, tgt, m)
        trace.append((th.copy(), nxt.copy()))
        th = nxt
        m.see(th)
        if np.linalg.norm(tip(th) - tgt) < ZONE_R:
            hold += 1
            if hold >= HOLD:
                return True, t + 1, trace
        else:
            hold = 0
    return False, cap, trace


def make_policy(uni):
    def policy(th, tgt, m):
        q = build_z(target_code(tgt), m)
        row = uni.W[int(np.argmax(uni.W @ q))]
        lv = decode_delta(row[TDIM + MDIM:])
        return wrap(th + lv * STEP)
    return policy


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

    def expert_policy(th, tgt, m):
        return expert_step(th, tgt)

    zs = []
    n_ok = 0
    for _ in range(N_DEMOS):
        tgt, th0 = rand_target(rng), rand_angles(rng)
        ok, _, trace = run_episode(expert_policy, th0, tgt)
        if not ok:
            continue
        n_ok += 1
        tc = target_code(tgt)
        m = Motor(trace[0][0])
        for th, nxt in trace:
            zs.append(build_z(tc, m, next_lvls=lvls_between(th, nxt)))
            m.see(nxt)
    print(f"expert demos: {n_ok}/{N_DEMOS} reached, {len(zs)} samples",
          flush=True)

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
                q = build_z(tc, m)
                row = uni.W[int(np.argmax(uni.W @ q))]
                lv = decode_delta(row[TDIM + MDIM:])
                zs.append(build_z(
                    tc, m,
                    next_lvls=lvls_between(th, expert_step(th, tgt))))
                th = wrap(th + lv * STEP)
                m.see(th)
                if np.linalg.norm(tip(th) - tgt) < ZONE_R:
                    break

    np.savez(OUTPUT_DIR / "weights.npz", W=uni.W)
    print("weights.npz saved (arm_viewer-ready)", flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Two-axis arm reaching\n\nSee stdout for round-by-round.\n")
    print("Report written.")


if __name__ == "__main__":
    main()
