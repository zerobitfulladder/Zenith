"""Bouncing square: the lag-cargo recipe stacked on the same-rate chassis.

Three layers, one rule (uniform unit): every layer ticks every input
tick; leaky trail (decay per layer 0.5 / 0.9 / 0.975 — timescale from
decay alone); delayed write at every layer — store
[GC * cn(what just arrived) ; cn(trail BEFORE it arrived)], then let
the trail absorb it. Upward speech: TOP-1 one-hot of the layer's
winner (direction-preserving, per the day's dense lesson). No
top-down wiring yet (the square needs none); upper layers here are
observers learning their own era-arrows.

Playback stays L1-driven (query [empty ; trail], emit retrieved
cargo) — expected to reproduce 0.00. Purpose of this run: verify the
stacked training and SAVE ALL THREE BANKS for the template galleries.

Run: .venv/bin/python experiments/2026_08_27/lagcargo_stack/run_square_lagcargo_stack.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import json

import numpy as np

from run_square_completion import cn1, onehot  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    H,
    PERIOD,
    PRIME,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "lagcargo_stack" / "results" / "stack"
KS = (64, 32, 16)
GAMMAS = (0.5, 0.9, 0.975)
GC = 0.5
EPOCHS = 3
FREE = 100


def learn(bank, v):
    z = cn1(v)[None, :]
    if bank.n_boot < bank.k:
        bank.bootstrap(z)
        return -1
    C = (z @ bank.W.T)[0]
    w = int(np.argmax(C))
    bank.update(z, np.array([w]),
                np.array([max(C[w], 0.0)], dtype=np.float32))
    return w


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_dims = (D, KS[0], KS[1])
    banks = [Dict(k, 2 * d, 0.05) for k, d in zip(KS, in_dims)]
    homes = [dict(), dict(), dict()]

    for ep in range(EPOCHS):
        trails = [np.zeros(d, np.float32) for d in in_dims]
        for t in range(TRAIN_FRAMES):
            arrival = frame_at(t)[0].ravel().astype(np.float32)
            for li in range(3):
                z = np.concatenate([GC * cn1(arrival), cn1(trails[li])])
                w = learn(banks[li], z)
                trails[li] = arrival + GAMMAS[li] * trails[li]
                if ep == EPOCHS - 1 and w >= 0:
                    homes[li].setdefault(w, set()).add(t % PERIOD)
                if w < 0:
                    break
                arrival = onehot(w, KS[li])     # top-1 speech upward

    np.savez(OUTPUT_DIR / "weights.npz",
             **{f"W{li + 1}": banks[li].W for li in range(3)})
    (OUTPUT_DIR / "homes.json").write_text(json.dumps(
        [{str(w): sorted(ph) for w, ph in homes[li].items()}
         for li in range(3)]))
    used = [len(homes[li]) for li in range(3)]
    print(f"used units: L1 {used[0]}/{KS[0]}, L2 {used[1]}/{KS[1]}, "
          f"L3 {used[2]}/{KS[2]}", flush=True)

    # ---- Free run, L1-driven (sanity: should stay 0.00) -------------------
    T1 = np.zeros(D, np.float32)
    for t in range(PRIME):
        T1 = frame_at(t)[0].ravel().astype(np.float32) + GAMMAS[0] * T1
    gen, gen_x, true_x = [], [], []
    for i in range(FREE):
        q = cn1(np.concatenate([np.zeros(D, np.float32), cn1(T1)]))
        w = int(np.argmax(banks[0].W @ q))
        pred = np.maximum(banks[0].W[w, :D], 0.0)
        if pred.max() > 0:
            pred = pred / pred.max()
        frame = pred.reshape(H, W)
        gen.append(frame)
        col = frame.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])
        T1 = frame.ravel().astype(np.float32) + GAMMAS[0] * T1
    save_gif(gen, OUTPUT_DIR / "generated.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    print(f"FREE-RUN {FREE}: mean |x err| {err.mean():.2f} (max {err.max()})",
          flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        f"# Square, stacked lag-cargo (3 layers, top-1 speech)\n\n"
        f"used units L1/L2/L3: {used}\n"
        f"free-run mean |x err| {err.mean():.2f}, max {err.max()}\n")


if __name__ == "__main__":
    main()
