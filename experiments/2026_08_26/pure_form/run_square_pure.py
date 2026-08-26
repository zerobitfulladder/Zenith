"""Bouncing square, PURE form + neuronal fatigue — the debug bench.

Architecture exactly as specified: L1 stores cn([leaky-integrated
input ; G*down]); L2 = leaky integration of L1 activity at a slower
rate, era profiles keyed by themselves; no slots, no arrows, no
splits, everything queries with everything. Free-run adds ONE unit
property: FATIGUE — winners accumulate a decaying penalty trace.

Hypothesis under debug: fatigue-with-decay is THE ARROW. Plain
refractoriness coin-flips between +-1 neighbors (autocorrelation is
symmetric — measured 8.12 in the trail-only ladder); but the backward
neighbor was visited 2 steps ago and is still warm, the forward one is
fresh — a decaying been-here trace makes forward the only cool
direction. Prints winner/score/fatigue per step so the competition is
visible.

Run:  .venv/bin/python experiments/2026_08_26/pure_form/run_square_pure.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "pure_form" / "results" / "square"
K1T, K2T = 64, 16
GAMMA1, GAMMA2 = 0.5, 0.9
GD = 0.7
KSTRIDE = 5
DIM1 = D + K1T
BF, FD = 0.35, 0.6                   # fatigue add / decay per tick
EPOCHS = 3


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b1 = Dict(K1T, DIM1, 0.05)
    b2 = Dict(K2T, K1T, 0.05)

    for _ in range(EPOCHS):
        T1 = np.zeros(D, np.float32)
        down = np.zeros(K1T, np.float32)
        A2 = np.zeros(K1T, np.float32)
        tick = 0
        for t in range(TRAIN_FRAMES):
            xf = frame_at(t)[0].ravel().astype(np.float32)
            T1 = xf + GAMMA1 * T1
            z1 = cn1(np.concatenate([unit(T1), GD * down]))[None, :]
            u1 = -1
            if b1.n_boot < b1.k:
                b1.bootstrap(z1)
            else:
                C = (z1 @ b1.W.T)[0]
                u1 = int(np.argmax(C))
                b1.update(z1, np.array([u1]),
                          np.array([max(C[u1], 0.0)], dtype=np.float32))
            if u1 >= 0:
                a = np.zeros(K1T, np.float32)
                a[u1] = 1.0
                A2 = a + GAMMA2 * A2
            tick += 1
            if tick % KSTRIDE == 0 and A2.sum() > 0 and b1.n_boot >= K1T:
                z2 = cn1(A2)[None, :]
                if b2.n_boot < b2.k:
                    b2.bootstrap(z2)
                else:
                    C2 = (z2 @ b2.W.T)[0]
                    w2 = int(np.argmax(C2))
                    b2.update(z2, np.array([w2]),
                              np.array([max(C2[w2], 0.0)], dtype=np.float32))
                    down = unit(np.maximum(b2.W[w2], 0.0))

    # ---- Prime on real frames (fatigue warms too), then free-run ---------
    T1 = np.zeros(D, np.float32)
    down = np.zeros(K1T, np.float32)
    A2 = np.zeros(K1T, np.float32)
    fat = np.zeros(K1T, np.float32)
    tick = 0

    def l2_playback():
        nonlocal down
        if A2.sum() <= 0 or b2.n_boot == 0:
            return
        q2 = cn1(A2)
        w2 = int(np.argmax(b2.W @ q2))
        down = unit(np.maximum(b2.W[w2], 0.0))

    for t in range(PRIME):
        xf = frame_at(t)[0].ravel().astype(np.float32)
        T1 = xf + GAMMA1 * T1
        q = cn1(np.concatenate([unit(T1), GD * down]))
        sc = b1.W @ q - BF * fat
        u1 = int(np.argmax(sc))
        fat[:] = FD * fat
        fat[u1] += 1.0
        a = np.zeros(K1T, np.float32)
        a[u1] = 1.0
        A2 = a + GAMMA2 * A2
        tick += 1
        if tick % KSTRIDE == 0:
            l2_playback()

    gen, true_x, gen_x = [], [], []
    for i in range(FREE_RUN):
        q = cn1(np.concatenate([unit(T1), GD * down]))
        raw = b1.W @ q
        sc = raw - BF * fat
        u1 = int(np.argmax(sc))
        if i < 12:
            top3 = np.argsort(sc)[::-1][:3]
            print(f"  step {i}: win u{u1} "
                  f"(raw {raw[u1]:+.3f} fat {fat[u1]:.2f}) | top3 "
                  + " ".join(f"u{j}:{sc[j]:+.3f}" for j in top3), flush=True)
        fat[:] = FD * fat
        fat[u1] += 1.0
        row = b1.W[u1]
        Tmem = np.maximum(row[:D], 0.0)
        T1 = Tmem / (Tmem.max() + 1e-9) * (1.0 / (1.0 - GAMMA1))
        img = np.maximum(row[:D], 0.0).reshape(H, W)
        if img.max() > 0:
            img = img / img.max()
        gen.append(img)
        true_x.append(frame_at(PRIME + i)[1])
        col = img.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        a = np.zeros(K1T, np.float32)
        a[u1] = 1.0
        A2 = a + GAMMA2 * A2
        tick += 1
        if tick % KSTRIDE == 0:
            l2_playback()

    save_gif(gen, OUTPUT_DIR / "generated_pure.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    print(f"PURE+fatigue free-run {FREE_RUN}: mean |x error|={err.mean():.2f}, "
          f"max={err.max()}  (refs: slot forms 0.00; no-fatigue pure = frozen)",
          flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        f"# Square, pure form + fatigue\n\nmean {err.mean():.2f}, "
        f"max {err.max()}\n")


if __name__ == "__main__":
    main()
