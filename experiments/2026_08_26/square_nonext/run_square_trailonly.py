"""Bouncing square, TRAIL-ONLY (two hands): the leaky integration is the whole state.

User's maximal simplification: store nothing but cn(T) where
T = x + gamma*T. Advance = refractory retrieval (self-match excluded;
the FORWARD neighbor contains my whole trail plus one new frame — the
containment asymmetry is the arrow of time). Emission = the RESIDUAL:
retrieved trail minus its projection onto my trail — the one thing it
knows that I don't, i.e. the next frame (prediction-error, literally).

Reference: next-slot and rebalanced no-next both 0.00.

Run:  .venv/bin/python experiments/2026_08_26/square_nonext/run_square_trailonly.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "square_nonext" / "results" / "trailonly"
K = 64
GAMMA = 0.6      # fast trail
GAMMA_S = 0.9    # slow trail — the second clock hand; two rates = the arrow
GSL = 0.6
EPOCHS = 3


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dic = Dict(K, 2 * D, 0.05)
    for _ in range(EPOCHS):
        T = np.zeros(D, np.float32)
        Ts = np.zeros(D, np.float32)
        for t in range(TRAIN_FRAMES):
            xf = frame_at(t)[0].ravel().astype(np.float32)
            T = xf + GAMMA * T
            Ts = xf + GAMMA_S * Ts
            z = cn1(np.concatenate([T, GSL * Ts]))[None, :]
            if dic.n_boot < dic.k:
                dic.bootstrap(z)
            else:
                C = (z @ dic.W.T)[0]
                w = int(np.argmax(C))
                dic.update(z, np.array([w]),
                           np.array([max(C[w], 0.0)], dtype=np.float32))

    T = np.zeros(D, np.float32)
    Ts = np.zeros(D, np.float32)
    for t in range(PRIME):
        xf = frame_at(t)[0].ravel().astype(np.float32)
        T = xf + GAMMA * T
        Ts = xf + GAMMA_S * Ts

    gen, true_x, gen_x = [], [], []
    uprev = -1
    for i in range(FREE_RUN):
        q = cn1(np.concatenate([T, GSL * Ts]))
        sc = dic.W @ q
        if uprev >= 0:
            sc[uprev] = -1e9
        winner = int(np.argmax(sc))
        uprev = winner
        row = dic.W[winner]
        resid = row - float(row @ q) * q       # subtract what I already know
        frame = np.maximum(resid[:D], 0.0)
        if frame.max() > 0:
            frame = frame / frame.max()
        nxt = frame.reshape(H, W)
        gen.append(nxt)
        true_x.append(frame_at(PRIME + i)[1])
        col = nxt.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        T = frame + GAMMA * T
        Ts = frame + GAMMA_S * Ts
    save_gif(gen, OUTPUT_DIR / "generated_trailonly.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    print(f"TRAIL-ONLY (two hands) free-run {FREE_RUN}: mean |x error|={err.mean():.2f}, "
          f"max={err.max()}  (references: next-slot 0.00, no-next 0.00)",
          flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        f"# Square, trail-only\n\nmean {err.mean():.2f}, max {err.max()}\n")


if __name__ == "__main__":
    main()
