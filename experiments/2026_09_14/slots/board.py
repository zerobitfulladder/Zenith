"""One board for the slots run: L1 cells, L2 node pictures (L1 settled under each node's expectation),
generations from the learned prior, and held-out rebuilds (input / L1 / top-down through the tables).

    python board.py <tag>
"""
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
W, uses1, uses2, nmem, named_by = Z["W"], Z["uses1"], Z["uses2"], Z["nmem"], Z["named_by"]
I, J = len(W), len(uses2)
g = 30


def grid_of(pics, cols, labels, cmap="Reds"):
    n = len(pics); rows = int(np.ceil(n / cols))
    grid = np.full((rows * g + 1, cols * g + 1), np.nan, np.float32)
    for q in range(n):
        rr, c = divmod(q, cols)
        w = pics[q].reshape(28, 28); grid[1 + rr * g:29 + rr * g, 1 + c * g:29 + c * g] = w / (np.abs(w).max() + 1e-8)
    return grid, rows


fig = plt.figure(figsize=(14, 34), facecolor=SURF)
gs = fig.add_gridspec(4, 1, height_ratios=[12, 10, 1.3, 3.4], hspace=0.10, left=0.02, right=0.98, top=0.965, bottom=0.01)

# --- L1 ---------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off")
o1 = np.argsort(-uses1)
grid, rows = grid_of(W[o1], 12, None)
ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
for q, t in enumerate(o1):
    rr, c = divmod(q, 12)
    ax.text(2 + c * g, 4 + rr * g, f"{int(uses1[t])}", fontsize=6.5, color=INK, va="top", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.6))
ax.set_title(f"L1: {I} whole-image cells, share then scale on 4-level pixels, frozen after 2 passes  (number = held-out images it was on for, of 2,000)  "
             f"L1 on/img {r['l1_on_per_image']:.1f}", fontsize=9, loc="left", color=INK)

# --- L2 ---------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1]); ax.axis("off")
o2 = np.argsort(-uses2)
pics = Z["a_j"] @ W
pics[(uses2 == 0) & (nmem == 0)] = 0                      # a node never on with no members: nothing to draw
grid, rows = grid_of(pics[o2], 10, None)
ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
for q, j in enumerate(o2):
    rr, c = divmod(q, 10)
    ax.text(2 + c * g, 4 + rr * g, f"{int(uses2[j])}", fontsize=6.5, color=INK, va="top", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.6))
    ax.text(27 + c * g, 27 + rr * g, f"{int(nmem[j])}/{int(named_by[j])}", fontsize=6, color=INK2, va="bottom", ha="right",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.6))
ax.set_title(f"L2: {J} nodes, 4 levels, parents of L1 through 4 named slots per child and one full table per child  |  "
             f"each picture = L1 settled under that node alone at level 3, painted\n"
             f"top-left = held-out images the node was on for;  bottom-right = members (children it raises by >0.3) / children naming it in a slot\n"
             f"held out: L2 on/img {r['l2_on_per_image']:.2f}, dead {r['l2_dead']*100:.0f}%, silent images {r['l2_silent_images']*100:.0f}%  |  "
             f"members mean {r['members_mean']:.1f}, <=1: {r['members_le1']*100:.0f}%, >=3: {r['members_ge3']*100:.0f}%  |  "
             f"slot-name purity {r['name_purity_mean']:.2f}, unchanged since pass 1: {r['names_stable_p1_to_end']*100:.0f}%\n"
             f"log-prob of the held-out L1 code per image: tables under settled L2 {r['loglik_table']:.1f} (prior {r['loglik_prior']:.1f})  |  "
             f"all L2 off {r['loglik_alloff_table']:.1f}  |  children independent (no L2) {r['loglik_indep']:.1f}",
             fontsize=8.5, loc="left", color=INK)

# --- generations ---------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[2]); ax.axis("off")
gen = Z["a_g"] @ W
canvas = np.ones((g + 1, 8 * g + 1))
for i in range(8):
    x = gen[i].reshape(28, 28); canvas[1:29, 1 + i * g:29 + i * g] = 1 - x / (x.max() + 1e-8)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("generations: L2 levels sampled from the learned prior, expected L1 levels through the tables, L1 settled under them, painted   (L2 nodes on: "
             + " ".join(str(int((Z["Tg"][i] > 0).sum())) for i in range(8)) + f";  prior P(on) per node {r['prior_on_mean']:.3f})", fontsize=8.5, loc="left", color=INK2)

# --- rebuilds ---------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[3]); ax.axis("off")
canvas = np.ones((3 * g + 1, 8 * g + 1))
rb = Z["a_rb"] @ W
for i in range(8):
    x = Z["Xte"][i].reshape(28, 28); hi = x.max() + 1e-8
    canvas[1:29, 1 + i * g:29 + i * g] = 1 - x / hi
    xh = Z["xhat_l1"][i].reshape(28, 28); canvas[31:59, 1 + i * g:29 + i * g] = 1 - np.clip(xh / (xh.max() + 1e-8), 0, 1)
    xt = rb[i].reshape(28, 28); canvas[61:89, 1 + i * g:29 + i * g] = 1 - np.clip(xt / (xt.max() + 1e-8), 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("held out: input / L1 rebuild from the settled cells / top-down rebuild (settled L2 -> expected L1 levels through the tables -> L1 settled under them)   (L1 on: "
             + " ".join(str(int((Z["A8"][i] > 0).sum())) for i in range(8)) + ";  L2 on: " + " ".join(str(int((Z["T8"][i] > 0).sum())) for i in range(8)) + ")",
             fontsize=8.5, loc="left", color=INK2)
fig.suptitle("Two layers, no labels: L1 share-then-scale on whole images, L2 a counted belief network with named parent slots", x=0.02, ha="left", fontsize=11.5, color=INK)
out = OUT / f"board_{tag}.png"
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=SURF); print("saved", out)
