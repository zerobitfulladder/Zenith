"""The templates, laid out one hypercolumn per row.

    python show.py [mnist|fashion_mnist] [H_K ...]

Every minicolumn of a hypercolumn sits on one row, so a row is what that
hypercolumn can build patches out of. Red is positive, blue negative, and the
scale is per template.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
z = np.load(OUT / f"conv1_{DS}.npz")
keys = sys.argv[2:] or list(z.files)

for key in keys:
    W = z[key].astype(np.float64)
    H, K, _ = W.shape
    cols = min(K, 32)
    fig, axes = plt.subplots(H, cols, figsize=(cols * 0.42 + .8, H * 0.42 + .7),
                             squeeze=False)
    for h in range(H):
        for k in range(cols):
            a = axes[h][k]
            t = W[h, k].reshape(5, 5)
            m = np.abs(t).max() + 1e-12
            a.imshow(t, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
        axes[h][0].set_ylabel(f"h{h}", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle(f"{DS} — {key}: every minicolumn of every hypercolumn", fontsize=9)
    fig.subplots_adjust(left=.05, right=.995, top=1 - .5 / (H * 0.42 + .7),
                        bottom=.005, wspace=.08, hspace=.08)
    fig.savefig(OUT / f"tpl_{DS}_{key}.png", dpi=150); plt.close(fig)
    print(f"-> tpl_{DS}_{key}.png   ({H} hypercolumns x {K} minicolumns)")
