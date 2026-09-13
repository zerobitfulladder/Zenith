"""Board for the EM reference and the self-priced arm."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = {"order": "#2a78d6", "hard": "#eb6834", "soft": "#1baf7a", "data": "#e87ba4"}
INITS = ["data", "order", "hard", "soft"]
em = {i: json.load(open(OUT / f"em_{i}.json")) for i in INITS if (OUT / f"em_{i}.json").exists()}
sp = {r: json.load(open(OUT / f"selfprice_{r}_s0.json")) for r in ("order", "soft") if (OUT / f"selfprice_{r}_s0.json").exists()}
online = {r: json.load(open(OUT / f"{r}_l0.02_s0.json")) for r in ("order", "hard", "soft")}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def tiles(ax, W, idx, title):
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for k, t in enumerate(idx[:48]):
        r, c = divmod(k, 8)
        w = W[t].reshape(28, 28)
        grid[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = w / (np.abs(w).max() + 1e-8)
    ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.axis("off")


fig = plt.figure(figsize=(17, 14), facecolor=SURF)
gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1, 1], hspace=0.4, wspace=0.2, top=0.95, left=0.04, right=0.99, bottom=0.04)

# row 1: EM cost per pass, on per pass, numbers
ax = fig.add_subplot(gs[0, 0])
for i, r in em.items():
    h = r["hist"]
    ax.plot([x["pass_"] for x in h], [x["cost"] for x in h], color=COL[i], lw=2, marker="o", ms=4, label=f"start: {i}")
ax.set_xlabel("EM pass (0 = the start vocabulary, before any M-step)", fontsize=8.5, color=INK2)
ax.set_title("EM: cost on the 4k training subset, price 0.02", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5)
style(ax); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[0, 1])
for i, r in em.items():
    h = r["hist"]
    ax.plot([x["pass_"] for x in h], [x["on"] for x in h], color=COL[i], lw=2, marker="o", ms=4, label=f"start: {i}")
ax.set_xlabel("EM pass", fontsize=8.5, color=INK2)
ax.set_title("EM: templates on per image", loc="left", fontsize=10, color=INK)
style(ax); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
axT = fig.add_subplot(gs[0, 2:4]); axT.axis("off")
lines = ["held out (2k), price 0.02          cost   unexpl.  on/img  px/tmpl  wholes/strokes/dots  tally", ""]
for i, r in em.items():
    lines.append(f"EM from {i:6s}                {r['cost']:.3f}   {r['unexplained']:.3f}   {r['on_per_image']:5.2f}    {r['support_median']:4.0f}     {r['wholes']:3d}/{r['strokes']:3d}/{r['dots']:3d}       {r['tally']:.3f}")
lines.append("")
for rr, r in online.items():
    s = np.array(r["support"])
    lines.append(f"online {rr:6s} (one pass)      {r['cost']:.3f}   {r['unexplained']:.3f}   {r['on_per_image']:5.2f}    {r['support_median']:4.0f}     {(s>60).sum():3d}/{((s>20)&(s<=60)).sum():3d}/{(s<=20).sum():3d}       {r['tally']:.3f}")
lines.append("")
lines.append("self-priced (names at -log2 usage, noise from the leftover), one pass:")
for rr, r in sp.items():
    lines.append(f"  {rr:6s}   price mean {r['price_mean']:.4f} [{r['price_min']:.4f}, {r['price_max']:.4f}]   on/img {r['on_per_image']:.2f}   unexpl. {r['unexplained']:.3f}   "
                 f"name bits/img {r['name_bits_per_image']:.1f}   px/tmpl {r['support_median']:.0f}   {r['wholes']}/{r['strokes']}/{r['dots']}   tally {r['tally']:.3f}")
axT.text(0, 1, "\n".join(lines), family="monospace", fontsize=8.2, va="top", color=INK, transform=axT.transAxes)

# row 2: EM templates per start
for j, i in enumerate(INITS):
    ax = fig.add_subplot(gs[1, j])
    p = OUT / f"weights_em_{i}.npz"
    if p.exists():
        Z = np.load(p)
        tiles(ax, Z["W"], np.argsort(-Z["fires"]), f"EM from {i}: the 48 most-used templates after {em[i]['passes']} passes")
    else:
        ax.axis("off")

# row 3: self-priced templates, trajectories
for j, rr in enumerate(("order", "soft")):
    ax = fig.add_subplot(gs[2, j])
    p = OUT / f"weights_selfprice_{rr}_s0.npz"
    if p.exists():
        Z = np.load(p)
        tiles(ax, Z["W"], np.argsort(-Z["fires"]), f"self-priced, {rr} rule: the 48 most-used templates")
    else:
        ax.axis("off")
ax = fig.add_subplot(gs[2, 2])
for rr, r in sp.items():
    t = r["traj"]
    ax.plot([x["step"] for x in t], [x["on"] for x in t], color=COL[rr], lw=2, label=f"{rr}: templates on")
ax2 = ax.twinx()
for rr, r in sp.items():
    t = r["traj"]
    ax2.plot([x["step"] for x in t], [x["price_mean"] for x in t], color=COL[rr], lw=1.5, ls=(0, (3, 2)), label=f"{rr}: mean price")
ax.set_xlabel("training batch (128 images each)", fontsize=8.5, color=INK2)
ax.set_ylabel("templates on per image", fontsize=8.5, color=INK2)
ax2.set_ylabel("mean price of a live template", fontsize=8.5, color=INK2)
ax.set_title("self-priced: does the code get cheaper with exposure?", loc="left", fontsize=10, color=INK)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
style(ax); ax2.spines[["top", "left"]].set_visible(False); ax2.spines["right"].set_color(GRID); ax2.tick_params(colors=INK2, labelsize=8)
ax = fig.add_subplot(gs[2, 3])
for rr in sp:
    p = OUT / f"weights_selfprice_{rr}_s0.npz"
    if p.exists():
        Z = np.load(p)
        live = Z["n"] > 0
        ax.scatter(Z["uses"][live], Z["price"][live], s=12, color=COL[rr], alpha=0.7, label=rr, edgecolor="none")
ax.set_xscale("log")
ax.set_xlabel("how often a template is used (share of images)", fontsize=8.5, color=INK2)
ax.set_ylabel("its price", fontsize=8.5, color=INK2)
ax.set_title("self-priced: frequent names are cheap", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5)
style(ax)
fig.suptitle("EM reference and self-pricing, one layer, ownership by strength", x=0.04, ha="left", fontsize=12, color=INK, y=0.985)
fig.savefig(OUT / "em_board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "em_board.png")
