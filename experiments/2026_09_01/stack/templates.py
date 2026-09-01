"""Layer-1 templates from the co-adaptive stack, MNIST beside Fashion.

Both were trained the same way -- 180 templates from random starts, the table
biasing which patches each one wins as it fills. Sorted busiest first, so the
starved tail (Gaussian seeding plus the min-samples gate) is visible at the
bottom of each panel.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts")); sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E, single as SG

OUT = HERE / "results"
sets = [("MNIST", HERE.parent / "experts/results/coadapt.npz", "mnist"),
        ("Fashion", OUT / "full_stack_fashion_mnist.npz", "fashion_mnist")]
fig, axes = plt.subplots(1, 2, figsize=(13, 8))
for ax, (name, path, ds) in zip(axes, sets):
    z = np.load(path)
    W = z["W1" if "W1" in z.files else "W"].astype(np.float64)
    if W.ndim == 2:
        W = W[:, None, :]
    _, _, Xte, _ = E.load(ds)
    idx = SG.winners(W, Xte[:1500])
    cnt = np.bincount(idx[idx >= 0], minlength=len(W))
    order = np.argsort(-cnt)
    rows, cols = 15, 12
    tile = np.full((rows * 6 - 1, cols * 6 - 1), np.nan)
    for a, j in enumerate(order[:rows * cols]):
        r, c = divmod(a, cols)
        tile[r * 6:r * 6 + 5, c * 6:c * 6 + 5] = W[j, 0, :25].reshape(5, 5)
    v = np.nanmax(np.abs(tile))
    ax.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    dead = int((cnt < cnt.sum() * 0.0005).sum())
    ax.set_title(f"{name} -- {len(W)} layer-1 templates, busiest first\n"
                 f"{dead} of {len(W)} win under 0.05% of patches", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "l1_templates.png", dpi=130)
print("-> results/l1_templates.png")
