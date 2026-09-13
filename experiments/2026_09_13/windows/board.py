import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer1 as L1

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
tag = sys.argv[1] if len(sys.argv) > 1 else "r0.04"
r = json.load(open(OUT / f"layer1_{tag}.json")); Z = np.load(OUT / f"layer1_{tag}.npz")
W, uses = Z["W"], Z["uses"].reshape(L1.NW, L1.K)
fig = plt.figure(figsize=(18, 11), facecolor=SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.4, 1], hspace=0.25, wspace=0.08, left=0.02, right=0.99, top=0.9, bottom=0.03)
ax = fig.add_subplot(gs[0, 0])
g = L1.WIN + 1
grid = np.full((L1.NW * g + 1, L1.K * g + 1), np.nan, np.float32)
for w in range(L1.NW):
    order = np.argsort(-uses[w])
    for j, t in enumerate(order):
        tile = W[w, t].reshape(L1.WIN, L1.WIN)
        grid[1 + w * g:1 + w * g + L1.WIN, 1 + j * g:1 + j * g + L1.WIN] = tile / (np.abs(tile).max() + 1e-8)
ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
ax.set_title(f"every cell: one row per window ({L1.NW} windows of {L1.WIN}x{L1.WIN}, stride {L1.STRIDE}), "
             f"{L1.K} cells per window sorted by use; red = above the patch mean, blue = below", fontsize=9.5, loc="left", color=INK)
ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel("windows, top-left to bottom-right", fontsize=8.5, color=INK2)
ax = fig.add_subplot(gs[0, 1])
F = Z["full"]; uf = Z["uses"]
order = np.argsort(-uf)[:48]
grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
for j, t in enumerate(order):
    rr, c = divmod(j, 8)
    tile = F[t].reshape(28, 28)
    grid[1 + rr * 29:29 + rr * 29, 1 + c * 29:29 + c * 29] = tile / (np.abs(tile).max() + 1e-8)
ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"); ax.axis("off")
ax.set_title("the 48 most-used cells, placed where they live in the image", fontsize=9.5, loc="left", color=INK)
ax = fig.add_subplot(gs[1, 0])
canvas = np.ones((2 * 30 + 1, 8 * 30 + 1))
for i in range(8):
    x = Z["Xte"][i]; xm = x.max() + 1e-8
    canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - x / xm
    canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip(Z["rec"][i] / xm, 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal"); ax.axis("off")
ax.set_title("eight held-out images: input / rebuilt from the settled cells   (cells on: " +
             " ".join(str(int((Z["A8"][i] > 0).sum())) for i in range(8)) + ")", fontsize=9.5, loc="left", color=INK)
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
lines = [f"{L1.H} cells = {L1.NW} windows x {L1.K}; target rate {r['rho']:g}, {r['passes']} pass(es) over 8k; held out 2k", "",
         f"cells on per image                 {r['cells_on_per_image']:.1f}",
         f"inked windows per image            {r['inked_windows_per_image']:.1f}",
         f"cells on per inked window          {r['cells_on_per_inked_window']:.2f}",
         f"inked windows with 2+ cells on     {r['windows_with_two_or_more_on']*100:.0f}%",
         f"inked windows left silent          {r['inked_windows_left_silent']*100:.0f}%",
         f"patch variance the best cell leaves {r['unexplained_patch_variance']:.3f}",
         f"pixel energy unexplained (rebuild) {r['pixel_unexplained']:.3f}",
         f"dead cells                         {r['dead_cells']*100:.0f}%",
         f"mean threshold                     {r['theta_mean']:.3f}",
         f"distinct configurations / 2k       {r['distinct_configs']}",
         f"time                               {r['t']:.0f} s"]
ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=9.5, va="top", color=INK, transform=ax.transAxes)
fig.suptitle("Layer 1 in terms of neurons: local windows, Pearson drive, settling with inhibition, every active cell rotates toward what it saw, thresholds by homeostasis",
             x=0.02, ha="left", fontsize=12, color=INK)
fig.savefig(OUT / f"board_{tag}.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
