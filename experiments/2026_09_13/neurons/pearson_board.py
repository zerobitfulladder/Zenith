import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
runs = sys.argv[1:] if len(sys.argv) > 1 else ["pearson_r0.02", "pearson_r0.05"]
name = sys.argv[0]
fig, axes = plt.subplots(2, len(runs), figsize=(7 * len(runs), 10.5), squeeze=False, facecolor=SURF, gridspec_kw=dict(height_ratios=[1.3, 0.7], hspace=0.3, wspace=0.1))
for k, tag in enumerate(runs):
    r = json.load(open(OUT / f"{tag}.json")); Z = np.load(OUT / f"{tag}.npz")
    W, uses = Z["W"], Z["uses"]
    order = np.argsort(-uses)[:48]
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for j, t in enumerate(order):
        rr, c = divmod(j, 8)
        w = W[t].reshape(28, 28); grid[1 + rr * 29:29 + rr * 29, 1 + c * 29:29 + c * 29] = w / (np.abs(w).max() + 1e-8)
    ax = axes[0, k]; ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"); ax.axis("off")
    ax.set_title(f"rate {r['rho']:g}, {r['passes']} passes, axon budget {'off' if not r.get('budget') else r['budget']}: 48 most-used cells\n"
                 f"unexplained {r['unexplained']:.3f} (median {r['unexplained_median']:.3f}, blow-ups {r['blowups']*100:.0f}%)  on/img {r['on_per_image']:.1f}  "
                 f"dead {r['dead']*100:.0f}%\nsize {r['support_median']:.0f}px  wholes/strokes/dots {r['wholes']}/{r['strokes']}/{r['dots']}  "
                 f"overlap {r['overlap_mean']:.2f} (nearest 5: {r['overlap_nearest5']:.2f})  smoothness {r.get('smoothness_median', float('nan')):.2f}  configs {r['distinct_configs']}",
                 fontsize=8.5, loc="left", color=INK)
    canvas = np.ones((2 * 30 + 1, 8 * 30 + 1))
    for i in range(8):
        x = Z["Xte"][i].reshape(28, 28); lo, hi = x.min(), x.max()
        canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - (x - lo) / (hi - lo + 1e-8)
        xh = Z["xhat"][i].reshape(28, 28)
        canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip((xh - lo) / (hi - lo + 1e-8), 0, 1)
    ax = axes[1, k]; ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
    ax.set_title("centred input / rebuilt from the settled cells, same scale   (cells on: " + " ".join(str(int((Z["A8"][i] > 0).sum())) for i in range(8)) + ")",
                 fontsize=8.5, loc="left", color=INK2)
fig.suptitle("Whole-image cells, settling with inhibition, Pearson drive, every active cell rotates toward what it saw, thresholds by homeostasis",
             x=0.02, ha="left", fontsize=11.5, color=INK)
out = OUT / ("pearson_board.png" if len(sys.argv) < 2 else "budget_board.png")
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", out)
