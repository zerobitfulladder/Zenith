"""MNIST carousel on the up-down bounce (user's task, night).

Input sequence: class 0,1,2,...,9, loop — and EVERY LAP USES A FRESH
EXEMPLAR of each class (a different 0, a different 1, ...). So the
fast scale never repeats (instance noise), while the class orbit is
the only stable structure and lives at the slow scale. This inverts
the square's situation: here the TOP genuinely knows something L1
cannot — what comes next in class terms — so the user's "TOP should
help" expectation gets its fair test.

Architecture = run_square_bounce_dense: three same-rate layers,
decay-only timescales (0.5/0.9/0.975), rows [GC*cargo ; lagged
trail] (delayed write), DENSE upward speech (relu full-row match
profile), inference = climb -> ONE successor-read at TOP -> graded
reconstruction descent -> emit -> re-see own emission -> bounce.

Metrics per arm (TOP = L1 / L2 / L3): advance accuracy (consecutive
emissions step the class by +1 mod 10, judged by nearest class-mean),
timeline accuracy (emission class == intended class at that tick),
GIF of the free run.

Run: .venv/bin/python experiments/2026_08_27/carousel/run_mnist_carousel_bounce.py
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
from run_temporal_square import save_gif  # noqa: E402
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "carousel" / "results" / "mnist_carousel_bounce"
SIDE = 28
D = SIDE * SIDE
KS = (128, 64, 16)
GAMMAS = (0.5, 0.9, 0.975)
GC = 0.5
EPOCHS = 2
TRAIN_TICKS = 3000            # 300 laps, fresh exemplars per lap
PRIME = 100
FREE = 100
in_dims = (D, KS[0], KS[1])

X = np.load(ROOT / "data/mnist/digits/train_images.npy")
y = np.load(ROOT / "data/mnist/digits/train_labels.npy")
BYCLASS = [X[y == c] for c in range(10)]
CLASS_MEAN = np.stack([b.mean(axis=0).ravel() for b in BYCLASS])
CM_N = np.stack([cn1(m) for m in CLASS_MEAN])


def frame_for(t):
    c, lap = t % 10, t // 10
    return BYCLASS[c][lap % len(BYCLASS[c])].ravel().astype(np.float32)


def classify(frame):
    return int(np.argmax(CM_N @ cn1(frame)))


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
        self.last_w = w
        return relu(C)

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


def climb(stack, x, learning):
    arrival = x
    for L in stack:
        msg = L.learn(arrival) if learning else L.perceive_msg(arrival)
        L.T = arrival + GAMMAS[L.li] * L.T
        if msg is None:
            return
        arrival = msg


def run_arm(stack, top):
    for L in stack:
        L.reset()
    for t in range(PRIME):
        climb(stack, frame_for(t), False)
    gen, cls = [], []
    for i in range(FREE):
        li = top - 1
        w = stack[li].successor()
        prof = stack[li].cargo(w)
        while li > 0:
            li -= 1
            basis = relu(stack[li].bank.W[:, : stack[li].in_dim])
            prof = prof @ basis
        if prof.max() > 0:
            prof = prof / prof.max()
        frame = prof.reshape(SIDE, SIDE)
        gen.append(frame)
        cls.append(classify(prof))
        climb(stack, prof.astype(np.float32), False)
    cls = np.array(cls)
    intended = (np.arange(PRIME, PRIME + FREE)) % 10
    adv = float(np.mean((np.diff(cls) % 10) == 1))
    timeline = float(np.mean(cls == intended))
    return adv, timeline, cls, gen


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stack = [Layer(li) for li in range(3)]
    homes = [dict(), dict(), dict()]
    for ep in range(EPOCHS):
        for L in stack:
            L.reset()
        for t in range(TRAIN_TICKS):
            for L in stack:
                L.last_w = -1
            climb(stack, frame_for(t), True)
            if ep == EPOCHS - 1:
                for li, L in enumerate(stack):
                    if L.last_w >= 0:
                        homes[li].setdefault(L.last_w, set()).add(t % 10)

    import json
    np.savez(OUTPUT_DIR / "weights.npz",
             **{f"W{li + 1}": stack[li].bank.W for li in range(3)})
    (OUTPUT_DIR / "homes.json").write_text(json.dumps(
        [{str(w): sorted(cs) for w, cs in homes[li].items()}
         for li in range(3)]))

    lines = []
    for top in (1, 2, 3):
        adv, timeline, cls, gen = run_arm(stack, top)
        save_gif(gen, OUTPUT_DIR / f"generated_top{top}.gif", ms=350)
        lines.append(
            f"TOP=L{top}: advance {adv:.2f}, timeline {timeline:.2f}, "
            f"first 30 classes: {''.join(map(str, cls[:30]))}")
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# MNIST carousel, up-down bounce, dense speech\n\n"
        "Fresh exemplar per class per lap; free-run judged by nearest "
        "class-mean.\n\n" + report + "\n")


if __name__ == "__main__":
    main()
