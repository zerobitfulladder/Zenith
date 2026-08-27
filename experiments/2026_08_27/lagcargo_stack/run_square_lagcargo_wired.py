"""Bouncing square: FULL-STACK lag-cargo generation (wired feedback).

The complete unit, per the user's design: three same-rate layers,
decay-only timescales (0.5/0.9/0.975), top-1 speech upward, and now
the feedback path wired in — each layer's stored row is

    [ GC * cargo ; own lagged trail ; GD * lagged down-projection ]

learned JOINTLY, where the down-projection is the layer above's
retrieved-row context over this layer's units, refreshed after use
(one-tick lag). Playback runs the WHOLE stack every tick: L1 queries
cargo-empty with [trail ; down], emits its retrieved cargo, the
winner echoes upward through L2/L3 whose reads refresh the contexts.

Square = sanity gate: context is redundant here, so the wired rig
must HOLD 0.00 (wiring may not hurt the unambiguous case). The rig
then goes to the three-movies task, where context is load-bearing.

Run: .venv/bin/python experiments/2026_08_27/lagcargo_stack/run_square_lagcargo_wired.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import numpy as np

from run_square_completion import cn1, onehot  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    H,
    PRIME,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "lagcargo_stack" / "results" / "wired"
KS = (64, 32, 16)
GAMMAS = (0.5, 0.9, 0.975)
GC, GD = 0.5, 0.5
EPOCHS = 3
FREE = 100


class Layer:
    """Uniform unit: [cargo ; lagged trail ; lagged down] rows."""

    def __init__(self, k, in_dim, down_dim):
        self.k, self.in_dim, self.down_dim = k, in_dim, down_dim
        self.bank = Dict(k, 2 * in_dim + down_dim, 0.05)

    def reset(self):
        self.T = np.zeros(self.in_dim, np.float32)
        self.down = np.zeros(self.down_dim, np.float32)

    def row_vec(self, arrival):
        return cn1(np.concatenate([GC * cn1(arrival), cn1(self.T),
                                   GD * cn1(self.down)]))

    def query_vec(self):
        return cn1(np.concatenate([np.zeros(self.in_dim, np.float32),
                                   cn1(self.T), GD * cn1(self.down)]))

    def learn(self, arrival):
        z = self.row_vec(arrival)[None, :]
        if self.bank.n_boot < self.k:
            self.bank.bootstrap(z)
            return -1
        C = (z @ self.bank.W.T)[0]
        w = int(np.argmax(C))
        self.bank.update(z, np.array([w]),
                         np.array([max(C[w], 0.0)], dtype=np.float32))
        return w

    def read(self):
        if self.bank.n_boot < self.k:
            return -1
        return int(np.argmax(self.bank.W @ self.query_vec()))

    def perceive(self, arrival):
        """Match the full row (cargo included): 'which unit fired'."""
        if self.bank.n_boot < self.k:
            return -1
        return int(np.argmax(self.bank.W @ self.row_vec(arrival)))

    def cargo(self, w):
        return np.maximum(self.bank.W[w, : self.in_dim], 0.0)

    def keys_ctx(self, w):
        """The row's own-trail segment: context handed down."""
        return np.maximum(
            self.bank.W[w, self.in_dim: 2 * self.in_dim], 0.0)


def up_pass(stack, w1, learning):
    """Echo L1's winner upward; upper layers learn or perceive; their
    retrieved rows refresh the down contexts (consumed next tick)."""
    L1, L2, L3 = stack
    a2 = onehot(w1, KS[0])
    w2 = L2.learn(a2) if learning else L2.perceive(a2)
    L2.T = a2 + GAMMAS[1] * L2.T
    if w2 < 0:
        return
    L1.down = L2.keys_ctx(w2)
    a3 = onehot(w2, KS[1])
    w3 = L3.learn(a3) if learning else L3.perceive(a3)
    L3.T = a3 + GAMMAS[2] * L3.T
    if w3 >= 0:
        L2.down = L3.keys_ctx(w3)


def drive_tick(stack, x, learning):
    """Input-driven tick (training / priming)."""
    L1 = stack[0]
    w1 = L1.learn(x) if learning else L1.perceive(x)
    L1.T = x + GAMMAS[0] * L1.T
    if w1 >= 0:
        up_pass(stack, w1, learning)
    return w1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_dims = (D, KS[0], KS[1])
    down_dims = (KS[0], KS[1], 0)
    stack = [Layer(k, i, d) for k, i, d in zip(KS, in_dims, down_dims)]

    for _ in range(EPOCHS):
        for L in stack:
            L.reset()
        for t in range(TRAIN_FRAMES):
            drive_tick(stack, frame_at(t)[0].ravel().astype(np.float32),
                       learning=True)

    for L in stack:
        L.reset()
    for t in range(PRIME):
        drive_tick(stack, frame_at(t)[0].ravel().astype(np.float32),
                   learning=False)

    gen, gen_x, true_x = [], [], []
    for i in range(FREE):
        L1 = stack[0]
        w1 = L1.read()                       # cargo-empty successor query
        pred = L1.cargo(w1)
        if pred.max() > 0:
            pred = pred / pred.max()
        emission = pred.reshape(H, W)
        L1.T = emission.ravel().astype(np.float32) + GAMMAS[0] * L1.T
        up_pass(stack, w1, learning=False)   # echo the winner upward
        gen.append(emission)
        col = emission.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    report = (f"WIRED FULL-STACK free-run {FREE}: mean |x err| "
              f"{err.mean():.2f} (max {err.max()})  "
              f"(gate: must hold 0.00)\n"
              "first 33 (gen_x, true_x): "
              + " ".join(f"({g},{tn})" for g, tn
                         in zip(gen_x[:33], true_x[:33])))
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, wired full-stack lag-cargo generation\n\n" + report + "\n")


if __name__ == "__main__":
    main()
