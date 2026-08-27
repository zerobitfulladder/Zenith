"""Bouncing square: CONVOLUTIONAL lag-cargo unit (user's unification).

One unit type: receptive field bounds it in space (8x8 windows,
stride 1, SHARED dictionary), trail decay bounds it in time
(per-position leaky trails), the delayed write gives it the arrow
([GC * cargo=arriving window ; keys=that position's lagged trail]).
No separate spatial encoder — convolution IS the layer. Upper layers
(same unit over the code map, slower decay) come next; this run
validates the conv form of the unit alone on the square.

Laws respected: blank windows are not a symbol (positions with empty
window AND empty trail are skipped — zero-symbol law); commitment
happens once, at the pixels (overlap-add of per-position predicted
windows, then top-16 deblur).

Expected: free-run ~0.00; templates = local motion detectors
(mini comet keys, shifted cargo) — a V1-like sheet with arrows.

Run: .venv/bin/python experiments/2026_08_27/carousel/run_square_conv_temporal.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_square_completion import cn1  # noqa: E402
from run_temporal_square import (  # noqa: E402
    H,
    PRIME,
    SQ,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "carousel" / "results" / "square_conv_temporal"
WIN = 8                       # 8x8 receptive field, full height
NPOS = W - WIN + 1            # 17 horizontal positions, stride 1
WD = H * WIN                  # 64 dims per window
K = 24                        # shared dictionary
GAMMA = 0.5
GC = 0.5
EPS = 1e-6
EPOCHS = 3
FREE = 100


def windows(frame):
    return [frame[:, p:p + WIN].ravel().astype(np.float32)
            for p in range(NPOS)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = Dict(K, 2 * WD, 0.05)

    def learn(win, trail):
        z = cn1(np.concatenate([GC * cn1(win), cn1(trail)]))[None, :]
        if bank.n_boot < bank.k:
            bank.bootstrap(z)
            return
        C = (z @ bank.W.T)[0]
        w = int(np.argmax(C))
        bank.update(z, np.array([w]),
                    np.array([max(C[w], 0.0)], dtype=np.float32))

    for _ in range(EPOCHS):
        T = np.zeros((NPOS, WD), np.float32)
        for t in range(TRAIN_FRAMES):
            wins = windows(frame_at(t)[0])
            for p in range(NPOS):           # delayed write per position
                if wins[p].max() > EPS or T[p].max() > EPS:
                    learn(wins[p], T[p])
            for p in range(NPOS):
                T[p] = wins[p] + GAMMA * T[p]

    # ---- Free run ---------------------------------------------------------
    T = np.zeros((NPOS, WD), np.float32)
    for t in range(PRIME):
        wins = windows(frame_at(t)[0])
        for p in range(NPOS):
            T[p] = wins[p] + GAMMA * T[p]

    gen, gen_x, true_x = [], [], []
    emission = frame_at(PRIME - 1)[0]
    for i in range(FREE):
        wins = windows(emission)
        for p in range(NPOS):
            T[p] = wins[p] + GAMMA * T[p]
        canvas = np.zeros((H, W), np.float32)
        weight = np.zeros((H, W), np.float32)
        for p in range(NPOS):
            if T[p].max() <= EPS:
                continue                     # blank history: no symbol
            q = cn1(np.concatenate([np.zeros(WD, np.float32), cn1(T[p])]))
            w = int(np.argmax(bank.W @ q))
            pred = np.maximum(bank.W[w, :WD], 0.0).reshape(H, WIN)
            canvas[:, p:p + WIN] += pred
            weight[:, p:p + WIN] += 1.0
        canvas = np.where(weight > 0, canvas / np.maximum(weight, 1), 0.0)
        f = np.zeros(H * W, np.float32)      # commit once, at the pixels
        idx = np.argsort(canvas.ravel())[-(SQ * SQ):]
        idx = idx[canvas.ravel()[idx] > EPS]
        f[idx] = 1.0
        emission = f.reshape(H, W)
        gen.append(emission)
        col = emission.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(SQ), "valid"))))
        true_x.append(frame_at(PRIME + i)[1])

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    err = np.abs(np.array(true_x) - np.array(gen_x))
    report = (f"CONV lag-cargo free-run {FREE}: mean |x err| "
              f"{err.mean():.2f} (max {err.max()})  (whole-frame ref 0.00)\n"
              "first 33 (gen_x, true_x): "
              + " ".join(f"({g},{tn})" for g, tn
                         in zip(gen_x[:33], true_x[:33])))
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, convolutional lag-cargo unit\n\n" + report + "\n")

    fig, axes = plt.subplots(3, 8, figsize=(8 * 1.6, 3 * 2.0))
    for i in range(K):
        ax = axes[i // 8, i % 8]
        keys = np.maximum(bank.W[i, WD:], 0.0).reshape(H, WIN)
        cargo = np.maximum(bank.W[i, :WD], 0.0).reshape(H, WIN)
        stack = np.vstack([keys / max(keys.max(), 1e-9),
                           np.full((1, WIN), 0.5, np.float32),
                           cargo / max(cargo.max(), 1e-9)])
        ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"u{i}", fontsize=6)
        ax.axis("off")
    fig.suptitle("Shared 8x8 units: KEYS (window trail, top) / CARGO "
                 "(next window, bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates.png", dpi=140)
    print("wrote templates.png")


if __name__ == "__main__":
    main()
