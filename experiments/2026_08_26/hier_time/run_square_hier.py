"""Bouncing square, HIERARCHICAL TIME (user's original design, at last).

L1 ticks every frame: query = [own leaky-summed input ; L2's downward
projection ; next-frame slot]. L2 samples MORE and outputs LESS: it
accumulates L1's unit activity over k=5 ticks and emits once per
window — the timescale separation is STRUCTURAL (temporal stride),
not a tuned gamma; a deeper stack would give clocks at dt, k*dt,
k^2*dt... for free. L2's retrieved row (which L1 units belong to this
ERA) holds steady between slow ticks and serves as the slow hand.
Position = fast local state x slow descending context.

Test: does the top-down era-context fully replace the LOCAL slow
trail as the second clock hand? References on this task: local
fast+slow with next-slot = 0.00; trail-only (no arrow) = 8.12.

Run:  .venv/bin/python experiments/2026_08_26/hier_time/run_square_hier.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_square"))

import numpy as np

from run_gpu_minibatch import Dict  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    FREE_RUN,
    H,
    PRIME,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "hier_time" / "results" / "square"
K1, K2 = 64, 16
GAMMA1 = 0.5                          # L1's own fast integration
KSTRIDE = 5                           # L2 emits once per 5 L1 ticks
GD, GN = 0.5, 0.5
DIM1 = D + K1 + D                     # [T1 ; down ; next]
EPOCHS = 3


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


class Rig:
    def __init__(self):
        self.b1 = Dict(K1, DIM1, 0.05)
        self.b2 = Dict(K2, 2 * K1, 0.05)   # [this era ; NEXT era] — L2's
                                           # own arrow, at its own timescale
        self.reset_state()

    def reset_state(self):
        self.T1 = np.zeros(D, np.float32)
        self.down = np.zeros(K1, np.float32)
        self.acc = np.zeros(K1, np.float32)
        self.prev = None
        self.tick = 0

    def l2_update(self, learning):
        if self.acc.sum() <= 0:        # empty window: no symbol, no update
            return
        cur = unit(self.acc.copy())
        if learning and self.prev is not None:
            z2 = cn1(np.concatenate([self.prev, 0.5 * cur]))[None, :]
            if self.b2.n_boot < self.b2.k:
                self.b2.bootstrap(z2)
            else:
                C = (z2 @ self.b2.W.T)[0]
                w = int(np.argmax(C))
                self.b2.update(z2, np.array([w]),
                               np.array([max(C[w], 0.0)], dtype=np.float32))
        if self.b2.n_boot > 0:
            q2 = cn1(np.concatenate([cur, np.zeros(K1, np.float32)]))
            w = int(np.argmax(self.b2.W @ q2))
            self.down = unit(np.maximum(self.b2.W[w, K1:], 0.0))
        self.prev = cur
        self.acc[:] = 0.0

    def step_state(self, xf, u1):
        """After L1's tick: integrate input, record activity, slow-tick L2."""
        self.T1 = xf + GAMMA1 * self.T1
        if u1 >= 0:
            self.acc[u1] += 1.0
        self.tick += 1
        return self.tick % KSTRIDE == 0

    def l1_key(self):
        return np.concatenate([unit(self.T1), GD * self.down])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rig = Rig()
    for _ in range(EPOCHS):
        rig.reset_state()
        for t in range(TRAIN_FRAMES):
            xf = frame_at(t)[0].ravel().astype(np.float32)
            rig.T1 = xf + GAMMA1 * rig.T1
            nxt = frame_at(t + 1)[0].ravel().astype(np.float32)
            z1 = cn1(np.concatenate([unit(rig.T1), GD * rig.down,
                                     GN * unit(nxt)]))[None, :]
            u1 = -1
            if rig.b1.n_boot < rig.b1.k:
                rig.b1.bootstrap(z1)
            else:
                C = (z1 @ rig.b1.W.T)[0]
                u1 = int(np.argmax(C))
                rig.b1.update(z1, np.array([u1]),
                              np.array([max(C[u1], 0.0)], dtype=np.float32))
            if u1 >= 0:
                rig.acc[u1] += 1.0
            rig.tick += 1
            if rig.tick % KSTRIDE == 0 and rig.b1.n_boot >= K1:
                rig.l2_update(learning=True)

    rig.reset_state()
    for t in range(PRIME):
        xf = frame_at(t)[0].ravel().astype(np.float32)
        rig.T1 = xf + GAMMA1 * rig.T1
        q = cn1(np.concatenate([unit(rig.T1), GD * rig.down,
                                np.zeros(D, np.float32)]))
        u1 = int(np.argmax(rig.b1.W @ q))
        rig.acc[u1] += 1.0
        rig.tick += 1
        if rig.tick % KSTRIDE == 0:
            rig.l2_update(learning=False)

    gen, true_x, gen_x = [], [], []
    for i in range(FREE_RUN):
        q = cn1(np.concatenate([unit(rig.T1), GD * rig.down,
                                np.zeros(D, np.float32)]))
        u1 = int(np.argmax(rig.b1.W @ q))
        pred = np.maximum(rig.b1.W[u1, D + K1:], 0.0)
        if pred.max() > 0:
            pred = pred / pred.max()
        nxt = pred.reshape(H, W)
        gen.append(nxt)
        true_x.append(frame_at(PRIME + i)[1])
        col = nxt.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        xf = nxt.ravel().astype(np.float32)
        rig.T1 = xf + GAMMA1 * rig.T1
        rig.acc[u1] += 1.0
        rig.tick += 1
        if rig.tick % KSTRIDE == 0:
            rig.l2_update(learning=False)

    save_gif(gen, OUTPUT_DIR / "generated_hier.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    print(f"HIERARCHICAL free-run {FREE_RUN}: mean |x error|={err.mean():.2f}, "
          f"max={err.max()}  (refs: local fast+slow 0.00, trail-only 8.12)",
          flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        f"# Square, hierarchical time\n\nmean {err.mean():.2f}, "
        f"max {err.max()}\n")


if __name__ == "__main__":
    main()
