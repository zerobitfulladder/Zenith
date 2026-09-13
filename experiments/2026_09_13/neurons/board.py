import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
runs = [("hebb", 0.02), ("resid", 0.02), ("hebb", 0.05), ("resid", 0.05)]
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), facecolor=SURF, gridspec_kw=dict(height_ratios=[1.3, 0.7], hspace=0.25, wspace=0.12))
for k, (rule, rho) in enumerate(runs):
    tag = f"{rule}_r{rho:g}"
    r = json.load(open(OUT / f"{tag}.json")); Z = np.load(OUT / f"{tag}.npz")
    W, uses = Z["W"], Z["uses"]
    order = np.argsort(-uses)[:48]
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for j, t in enumerate(order):
        rr, c = divmod(j, 8)
        w = W[t].reshape(28, 28); grid[1 + rr * 29:29 + rr * 29, 1 + c * 29:29 + c * 29] = w / (np.abs(w).max() + 1e-8)
    ax = axes[0, k]; ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"); ax.axis("off")
    ax.set_title(f"{rule}, target rate {rho:g}: 48 most-used cells\nunexplained {r['unexplained']:.3f}  on/img {r['on_per_image']:.1f}  "
                 f"dead {r['dead']*100:.0f}%  size {r['support_median']:.0f}px  wholes/strokes/dots {r['wholes']}/{r['strokes']}/{r['dots']}",
                 fontsize=9, loc="left", color=INK)
    canvas = np.ones((2 * 30 + 1, 8 * 30 + 1))
    for i in range(8):
        x = Z["Xte"][i].reshape(28, 28); xm = x.max() + 1e-8
        canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - x / xm
        canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip(Z["xhat"][i].reshape(28, 28) / xm, 0, 1)
    ax = axes[1, k]; ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
    ax.set_title("input / rebuilt from the settled cells   (cells on: " + " ".join(str(int((Z["A8"][i] > 0).sum())) for i in range(8)) + ")",
                 fontsize=9, loc="left", color=INK2)
fig.suptitle("Whole-image cells, settling with inhibition, thresholds held by homeostasis: learn what you saw (hebb) vs learn the residual (resid)",
             x=0.02, ha="left", fontsize=12, color=INK)
fig.savefig(OUT / "board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
