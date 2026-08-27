"""Bouncing square: UP-DOWN BOUNCE generation (user's vision, night).

Layers learn ONLY [cargo ; own lagged trail] (the pure stacked rig —
banks reused from run_square_lagcargo_stack). Inference is a bounce:
the input climbs the stack (perception, trails advance), ONE
successor-read happens at the chosen TOP level, and the answer
descends by pure reconstruction — each row's cargo names the next
unit of the layer below, down to L1 whose cargo is the next frame.
Emit, re-see own emission, climb, bounce again. No down segments, no
gates, no joint learning; the ambiguity resolution is positional:
the decision is made where the trail is slowest.

Arms: TOP = 1 (baseline, = flat lag-cargo), 2, 3. The question the
square answers: how far up can the successor decision move before
the slow trail's phase precision fails (L3 units own 7-17 phases)?

Run: .venv/bin/python experiments/2026_08_27/bounce_updown/run_square_bounce_updown.py
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
    W,
    frame_at,
    save_gif,
)

STACK_RES = ROOT / "experiments" / "2026_08_27" / "lagcargo_stack" / "results" / "stack"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "bounce_updown" / "results" / "onehot"
KS = (64, 32, 16)
GAMMAS = (0.5, 0.9, 0.975)
GC = 0.5
FREE = 100

z = np.load(STACK_RES / "weights.npz")
Ws = [z["W1"], z["W2"], z["W3"]]
in_dims = (D, KS[0], KS[1])


def perceive(li, arrival, T):
    v = cn1(np.concatenate([GC * cn1(arrival), cn1(T)]))
    return int(np.argmax(Ws[li] @ v))


def successor(li, T):
    q = cn1(np.concatenate([np.zeros(in_dims[li], np.float32), cn1(T)]))
    return int(np.argmax(Ws[li] @ q))


def cargo(li, w):
    return np.maximum(Ws[li][w, : in_dims[li]], 0.0)


def run_arm(top):
    trails = [np.zeros(d, np.float32) for d in in_dims]

    def climb(x):
        arrival = x
        for li in range(3):
            w = perceive(li, arrival, trails[li])
            trails[li] = arrival + GAMMAS[li] * trails[li]
            arrival = onehot(w, KS[li])

    for t in range(PRIME):
        climb(frame_at(t)[0].ravel().astype(np.float32))

    gen, gen_x, true_x = [], [], []
    for i in range(FREE):
        li = top - 1
        w = successor(li, trails[li])         # ONE arrow, at the top
        while li > 0:                          # pure reconstruction down
            w = int(np.argmax(cargo(li, w)))
            li -= 1
        pred = cargo(0, w)
        if pred.max() > 0:
            pred = pred / pred.max()
        frame = pred.reshape(H, W)
        gen.append(frame)
        col = frame.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])
        climb(frame.ravel().astype(np.float32))   # re-see own emission

    err = np.abs(np.array(true_x) - np.array(gen_x))
    return err, gen, gen_x, true_x


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for top in (1, 2, 3):
        err, gen, gen_x, true_x = run_arm(top)
        save_gif(gen, OUTPUT_DIR / f"generated_top{top}.gif")
        lines.append(
            f"TOP=L{top}: mean |x err| {err.mean():.2f} (max {err.max()})  "
            f"first 20: "
            + " ".join(f"({g},{tn})" for g, tn
                       in zip(gen_x[:20], true_x[:20])))
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, up-down bounce generation (successor read at TOP,\n"
        "# reconstruction down)\n\n" + report + "\n")


if __name__ == "__main__":
    main()
