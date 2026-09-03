"""What is template learning actually worth here?

`frozen_all` in tally.py never moves a template -- the 400 filters stay at their
random initialisation and only the table learns -- and it scores 0.9613 against
the fully trained champion's 0.9703. This draws both vocabularies side by side
against those two numbers.

Usage:  uv run python compare_init.py
"""

import numpy as np, cupy as cp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tally as T

OUT = Path(__file__).resolve().parent / "results"


def montage(ax, W, n=196):
    side = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((side * (T.PS + 1), side * (T.PS + 1)))
    for i in range(min(n, len(W))):
        r, c = divmod(i, side)
        w = W[i].reshape(T.PS, T.PS)
        v = np.abs(w).max() + 1e-9
        grid[r * (T.PS + 1):r * (T.PS + 1) + T.PS,
             c * (T.PS + 1):c * (T.PS + 1) + T.PS] = w / v
    ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    Xtr, ytr, Xte, yte = T.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)

    out = {}
    for arm in ("frozen_all", "base"):
        _, f, W, _, _ = T.run(arm, None, Xtr, ytr_g, Xte, yte_g)
        out[arm] = (W, f["recount"])
        print(f"  {arm:<11s} recount {f['recount']:.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, arm, ttl in ((axes[0], "frozen_all", "never learned — random 5x5 filters"),
                         (axes[1], "base", "trained — the champion's vocabulary")):
        W, acc = out[arm]
        montage(ax, W)
        ax.set_title(f"{ttl}\n{acc:.4f}", fontsize=11)
    gap = out["base"][1] - out["frozen_all"][1]
    fig.suptitle(f"196 of the 400 templates. Training the vocabulary is worth "
                 f"{gap*100:.1f} points.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "learned_vs_random.png", dpi=150)
    print(f"  wrote {OUT/'learned_vs_random.png'}")


if __name__ == "__main__":
    main()
