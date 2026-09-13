import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
tag = sys.argv[1]
r = json.load(open(OUT / f"{tag}.json")); Z = np.load(OUT / f"{tag}.npz")
W, uses = Z["W"], Z["uses"]
H = len(W); cols = 12 if H % 12 == 0 else 10; rows = int(np.ceil(H / cols))
order = np.argsort(-uses)
fig = plt.figure(figsize=(14, 1.45 * rows + 4), facecolor=SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[rows, 2.6], hspace=0.12, left=0.02, right=0.98, top=0.93, bottom=0.02)
ax = fig.add_subplot(gs[0]); ax.axis("off")
g = 30
grid = np.full((rows * g + 1, cols * g + 1), np.nan, np.float32)
for j, t in enumerate(order):
    rr, c = divmod(j, cols)
    w = W[t].reshape(28, 28); grid[1 + rr * g:29 + rr * g, 1 + c * g:29 + c * g] = w / (np.abs(w).max() + 1e-8)
nonneg = W.min() >= 0
ax.imshow(grid, cmap=("Reds" if nonneg else "RdBu_r"), vmin=(0 if nonneg else -1), vmax=1, interpolation="nearest")
for j, t in enumerate(order):
    rr, c = divmod(j, cols)
    ax.text(2 + c * g, 4 + rr * g, f"{int(uses[t])}", fontsize=6.5, color=INK, va="top",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.6))
ax.set_title(f"all {H} cells, sorted by use (number = images it was on for, of 2,000 held out)  |  "
             f"unexplained {r['unexplained']:.3f} (median {r['unexplained_median']:.3f})  on/img {r['on_per_image']:.1f}  dead {r['dead']*100:.0f}%  "
             f"size {r['support_median']:.0f}px  wholes/strokes/dots {r['wholes']}/{r['strokes']}/{r['dots']}\n"
             f"overlap {r['overlap_mean']:.2f} (nearest 5: {r['overlap_nearest5']:.2f})  smoothness {r['smoothness_median']:.2f}  "
             f"configs {r['distinct_configs']}  rate {r['rho']:g}, {r['passes']} passes" + (f", budget {r['budget']:g}, tilt eps {r.get('eps', 0):g}" if 'budget' in r else ", raw input, two budgets, share then scale"), fontsize=9, loc="left", color=INK)
ax = fig.add_subplot(gs[1]); ax.axis("off")
canvas = np.ones((2 * 30 + 1, 8 * 30 + 1))
for i in range(8):
    x = Z["Xte"][i].reshape(28, 28); lo, hi = x.min(), x.max()
    canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - (x - lo) / (hi - lo + 1e-8)
    xh = Z["xhat"][i].reshape(28, 28)
    if nonneg:
        xh = xh / (xh.max() + 1e-8) * hi
    canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip((xh - lo) / (hi - lo + 1e-8), 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.set_title(("input" if nonneg else "centred input") + " / rebuilt from the settled cells   (cells on: " + " ".join(str(int((Z["A8"][i] > 0).sum())) for i in range(8)) + ")",
             fontsize=9, loc="left", color=INK2)
fig.suptitle("Whole-image cells, settling, " + ("raw input, two budgets, share then scale" if nonneg else "Pearson, rotation, shared axon budget"), x=0.02, ha="left", fontsize=11.5, color=INK)
out = OUT / f"all_{tag}.png"
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=SURF); print("saved", out)
