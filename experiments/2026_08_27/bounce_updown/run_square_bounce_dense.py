"""Bouncing square: up-down bounce with DENSE inter-layer speech.

Same learning rule as the stacked lag-cargo rig ([cargo ; lagged
trail] per layer, delayed write), but the upward message is the WHOLE
graded activity profile (relu of the full-row match correlations)
instead of the winner's one-hot — the user's request, motivated by
dense communication winning at every static-image interface.

Inference: the up-down bounce (climb, ONE successor-read at TOP,
descend through cargos, emit, re-see). Descent tested both ways:
  argmax  — committed chain (one-hot descent)
  graded  — weighted reconstruction through the cargo basis,
            committed only at the pixel deblur
References (one-hot speech): TOP1 0.00 / TOP2 6.69 / TOP3 6.84.

Run: .venv/bin/python experiments/2026_08_27/bounce_updown/run_square_bounce_dense.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import numpy as np

from run_square_completion import cn1  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    H,
    PRIME,
    SQ,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "bounce_updown" / "results" / "dense"
KS = (64, 32, 16)
GAMMAS = (0.5, 0.9, 0.975)
GC = 0.5
EPOCHS = 3
FREE = 100
in_dims = (D, KS[0], KS[1])


def relu(v):
    return np.maximum(v, 0.0)


class Layer:
    def __init__(self, li):
        self.li = li
        self.k, self.in_dim = KS[li], in_dims[li]
        self.bank = Dict(self.k, 2 * self.in_dim, 0.05)

    def reset(self):
        self.T = np.zeros(self.in_dim, np.float32)

    def row_vec(self, arrival):
        return cn1(np.concatenate([GC * cn1(arrival), cn1(self.T)]))

    def learn(self, arrival):
        z = self.row_vec(arrival)[None, :]
        if self.bank.n_boot < self.k:
            self.bank.bootstrap(z)
            return None
        C = (z @ self.bank.W.T)[0]
        w = int(np.argmax(C))
        self.bank.update(z, np.array([w]),
                         np.array([max(C[w], 0.0)], dtype=np.float32))
        return relu(C)                       # DENSE message upward

    def perceive_msg(self, arrival):
        if self.bank.n_boot < self.k:
            return None
        return relu(self.bank.W @ self.row_vec(arrival))

    def successor(self):
        q = cn1(np.concatenate([np.zeros(self.in_dim, np.float32),
                                cn1(self.T)]))
        return int(np.argmax(self.bank.W @ q))

    def cargo(self, w):
        return relu(self.bank.W[w, : self.in_dim])


def deblur(pred):
    f = np.zeros(D, np.float32)
    idx = np.argsort(pred)[-(SQ * SQ):]
    idx = idx[pred[idx] > 1e-6]
    f[idx] = 1.0
    return f.reshape(H, W)


def climb(stack, x, learning):
    arrival = x
    for L in stack:
        msg = L.learn(arrival) if learning else L.perceive_msg(arrival)
        L.T = arrival + GAMMAS[L.li] * L.T
        if msg is None:
            return
        arrival = msg


def run_arm(stack, top, descent):
    for L in stack:
        L.reset()
    for t in range(PRIME):
        climb(stack, frame_at(t)[0].ravel().astype(np.float32), False)
    gen, gen_x, true_x = [], [], []
    for i in range(FREE):
        li = top - 1
        w = stack[li].successor()
        prof = stack[li].cargo(w)            # profile over layer li-1 units
        while li > 0:
            li -= 1
            if descent == "argmax":
                prof = stack[li].cargo(int(np.argmax(prof)))
            else:
                basis = relu(stack[li].bank.W[:, : stack[li].in_dim])
                prof = prof @ basis          # graded reconstruction
        frame = deblur(prof)                 # commit at the pixels
        gen.append(frame)
        col = frame.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(SQ), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])
        climb(stack, frame.ravel().astype(np.float32), False)
    err = np.abs(np.array(true_x) - np.array(gen_x))
    return err, gen


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stack = [Layer(li) for li in range(3)]
    for _ in range(EPOCHS):
        for L in stack:
            L.reset()
        for t in range(TRAIN_FRAMES):
            climb(stack, frame_at(t)[0].ravel().astype(np.float32), True)

    lines = []
    for top in (1, 2, 3):
        for descent in ("argmax", "graded"):
            err, gen = run_arm(stack, top, descent)
            save_gif(gen, OUTPUT_DIR / f"generated_top{top}_{descent}.gif")
            lines.append(f"TOP=L{top} {descent:6s}: mean |x err| "
                         f"{err.mean():.2f} (max {err.max()})")
    refs = "one-hot refs: TOP1 0.00 / TOP2 6.69 / TOP3 6.84"
    report = "\n".join(lines + [refs])
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, up-down bounce, DENSE inter-layer speech\n\n"
        + report + "\n")


if __name__ == "__main__":
    main()
