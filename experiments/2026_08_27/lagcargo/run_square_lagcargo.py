"""Bouncing square: same-rate chassis + OBSERVED cargo (delayed write).

The evening's conclusion made mechanism: identical row layout to
run_square_residcargo ([keys ; cargo]), but the cargo slot is filled
by OBSERVATION instead of inference. At tick t the keys (the trail as
it stood BEFORE the new frame arrived) are set aside; the write
completes at the same tick using the newly arrived frame as cargo.
So in every finished row: keys = older, cargo = newer — the offset
is the arrow, written at storage time. No next-pathway, no oracle:
the network stores only what reached its senses; storage simply lags
perception by one tick. This is the user's lag-advance form on the
user's same-rate chassis. Cargo at small gain: KEYS DOMINATE (law).

Playback: query [empty cargo ; current trail]; the retrieved row's
cargo — the observed successor of that past — is the emission.

Reference: lag-advance flat square = 0.00. Prediction: ~0.00 here.

Run: .venv/bin/python experiments/2026_08_27/lagcargo/run_square_lagcargo.py
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
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "lagcargo" / "results"
K = 64
G1 = 0.5
GC = 0.5                       # cargo gain: small minority of the row
EPOCHS = 3
FREE = 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = Dict(K, D + D, 0.05)

    for _ in range(EPOCHS):
        T1 = np.zeros(D, np.float32)
        for t in range(TRAIN_FRAMES):
            x = frame_at(t)[0].ravel().astype(np.float32)
            # keys = the trail BEFORE x arrived; cargo = x, just arrived.
            z = cn1(np.concatenate([GC * cn1(x), cn1(T1)]))[None, :]
            if bank.n_boot < bank.k:
                bank.bootstrap(z)
            else:
                C = (z @ bank.W.T)[0]
                w = int(np.argmax(C))
                bank.update(z, np.array([w]),
                            np.array([max(C[w], 0.0)], dtype=np.float32))
            T1 = x + G1 * T1              # trail updates AFTER the write

    np.savez(OUTPUT_DIR / "weights.npz", W=bank.W)

    T1 = np.zeros(D, np.float32)
    for t in range(PRIME):
        T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1

    gen, gen_x, true_x = [], [], []
    for i in range(FREE):
        q = cn1(np.concatenate([np.zeros(D, np.float32), cn1(T1)]))
        w = int(np.argmax(bank.W @ q))
        pred = np.maximum(bank.W[w, :D], 0.0)
        if pred.max() > 0:
            pred = pred / pred.max()
        frame = pred.reshape(H, W)
        gen.append(frame)
        col = frame.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(4), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])
        T1 = frame.ravel().astype(np.float32) + G1 * T1

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    np.savez(OUTPUT_DIR / "gen_frames.npz", gen=np.array(gen))
    err = np.abs(np.array(true_x) - np.array(gen_x))
    report = (f"FREE-RUN {FREE} emissions: mean |x err| {err.mean():.2f} "
              f"(max {err.max()})  (refs: lag-advance flat 0.00; tonight's "
              f"inferred-cargo 6.65)\n"
              "first 33 (gen_x, true_x): "
              + " ".join(f"({g},{tn})" for g, tn
                         in zip(gen_x[:33], true_x[:33])))
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, same-rate chassis + observed cargo (delayed write)\n\n"
        + report + "\n")


if __name__ == "__main__":
    main()
