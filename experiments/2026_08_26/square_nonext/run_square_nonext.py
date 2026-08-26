"""Bouncing square, USER'S NO-NEXT-SLOT mechanism — the clean comparison.

Association = [present frame ; trails-of-the-PAST] (trails updated
AFTER the vector is built, so the stored present sits one step ahead
of the stored trails). Playback queries with the present channel
empty: my trails — which include the frame I just emitted — match the
stored moment one step ahead, and its PRESENT is the emission. Time
from the integration lag; no dedicated next-half anywhere.

Reference: the next-slot version replayed 100 frames at 0.00 error.
If this matches, the two mechanisms are equivalent re-slottings of
the same associations and the user's form is validated (leaner by a
pathway). Ordering note: trails-include-present would freeze (the
matched moment re-emits itself) — the arm no-next variant had that
ordering, an added confound on its result.

Run:  .venv/bin/python experiments/2026_08_26/square_nonext/run_square_nonext.py
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

from run_gpu_minibatch import Dict  # noqa: E402  (numpy mode via GF_XP)
from run_temporal_square import (  # noqa: E402
    ALPHA_F,
    ALPHA_S,
    D,
    FREE_RUN,
    H,
    PRIME,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "square_nonext" / "results" / "nonext"
K = 64
GP, GTR, GSL = 0.4, 1.0, 0.5   # emission slot small, key dominant
DIM = 3 * D
EPOCHS = 3


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dic = Dict(K, DIM, 0.05)
    for _ in range(EPOCHS):
        F = np.zeros(D, np.float32)
        S = np.zeros(D, np.float32)
        for t in range(TRAIN_FRAMES):
            x, _ = frame_at(t)
            xf = x.ravel().astype(np.float32)
            z = cn1(np.concatenate([GP * xf, GTR * F, GSL * S]))[None, :]
            if dic.n_boot < dic.k:
                dic.bootstrap(z)
            else:
                C = (z @ dic.W.T)[0]
                w = int(np.argmax(C))
                dic.update(z, np.array([w]),
                           np.array([max(C[w], 0.0)], dtype=np.float32))
            F = ALPHA_F * xf + (1 - ALPHA_F) * F   # trails lag the present
            S = ALPHA_S * xf + (1 - ALPHA_S) * S

    F = np.zeros(D, np.float32)
    S = np.zeros(D, np.float32)
    for t in range(PRIME):
        xf = frame_at(t)[0].ravel().astype(np.float32)
        F = ALPHA_F * xf + (1 - ALPHA_F) * F
        S = ALPHA_S * xf + (1 - ALPHA_S) * S

    gen, true_x, gen_x = [], [], []
    for i in range(FREE_RUN):
        q = cn1(np.concatenate([np.zeros(D, np.float32), GTR * F, GSL * S]))
        winner = int(np.argmax(dic.W @ q))
        pred = np.maximum(dic.W[winner, :D], 0.0)
        if pred.max() > 0:
            pred = pred / pred.max()
        nxt = pred.reshape(H, W)
        gen.append(nxt)
        true_x.append(frame_at(PRIME + i)[1])
        col = nxt.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        xf = nxt.ravel().astype(np.float32)
        F = ALPHA_F * xf + (1 - ALPHA_F) * F
        S = ALPHA_S * xf + (1 - ALPHA_S) * S

    save_gif(gen, OUTPUT_DIR / "generated_nonext.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    print(f"NO-NEXT free-run {FREE_RUN} frames: mean |x error|={err.mean():.2f}, "
          f"max={err.max()}  (next-slot reference: 0.00/0)", flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        f"# Square, no-next-slot\n\nmean |x err| {err.mean():.2f}, "
        f"max {err.max()} (next-slot reference 0.00/0).\n")


if __name__ == "__main__":
    main()
