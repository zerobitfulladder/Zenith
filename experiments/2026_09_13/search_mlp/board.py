"""One board: what the units of each layer look like in pixel space (templates
expanded down through the layers below, and the mean image each unit fired
on), the headline numbers, accuracy per layer per arm, selectivity per layer."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"
ARMS = ["backprop", "search-b1", "search-b4", "label-b4"]
COL = {"backprop": "#2a78d6", "search-b1": "#eb6834", "search-b4": "#1baf7a", "label-b4": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SHOW = "label-b4"        # whose units to draw

res = [json.load(open(p)) for p in sorted(OUT.glob("*_s*.json"))]
by = {a: [r for r in res if r["arm"] == a] for a in ARMS}
seeds = sorted({r["seed"] for r in res})
Z = np.load(OUT / f"weights_{SHOW}.npz")
W1, W2, W3 = Z["W1"], Z["W2"], Z["W3"]
P1 = W1                                    # layer 1 templates are pictures already
P2 = W2 @ W1                               # a layer-2 unit, expanded through layer 1
P3 = W3[:, :W2.shape[0]] @ P2              # a layer-3 unit, expanded through both
LAB = W3[:, W2.shape[0]:] if W3.shape[1] > W2.shape[0] else None
order = [np.argsort(-Z[f"N{i}"])[:48] for i in (1, 2, 3)]   # the 48 most-fired units


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def tiles(ax, P, idx, title, labels=None, cmap="RdBu_r"):
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for k, t in enumerate(idx):
        r, c = divmod(k, 8)
        w = P[t].reshape(28, 28)
        v = np.abs(w).max() + 1e-8
        grid[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = w / v
    if cmap == "RdBu_r":
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    else:
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    if labels is not None:
        for k, t in enumerate(idx):
            r, c = divmod(k, 8)
            l = LAB[t]
            share = np.linalg.norm(l) / (np.linalg.norm(W3[t]) + 1e-8)
            ax.text(1 + c * 29 + 1, 1 + r * 29 + 6, f"{int(l.argmax())}", fontsize=7, color=INK,
                    ha="left", va="center", fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.axis("off")


fig = plt.figure(figsize=(17, 17), facecolor=SURF)
gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 1.05], height_ratios=[1, 1, 0.95, 0.85],
                      hspace=0.38, wspace=0.22, top=0.95, left=0.04, right=0.99, bottom=0.04)

# rows 1-2: units in pixel space, for one arm, seed 0
tiles(fig.add_subplot(gs[0, 0]), P1, order[0], f"layer 1 ({SHOW}): the 48 most-fired templates")
tiles(fig.add_subplot(gs[0, 1]), P2, order[1], "layer 2: templates expanded down through layer 1")
tiles(fig.add_subplot(gs[0, 2]), P3, order[2],
      "layer 3: expanded through layers 2 and 1; digit = label the template carries", labels=LAB)
tiles(fig.add_subplot(gs[1, 0]), Z["FM1"], order[0], "layer 1: mean test image each unit fired on", cmap="gray_r")
tiles(fig.add_subplot(gs[1, 1]), Z["FM2"], order[1], "layer 2: mean test image each unit fired on", cmap="gray_r")
tiles(fig.add_subplot(gs[1, 2]), Z["FM3"], order[2], "layer 3: mean test image each unit fired on", cmap="gray_r")

# rows 1-2, col 4: the numbers
axT = fig.add_subplot(gs[0:2, 3])
axT.axis("off")


def ms(vals):
    v = np.array(vals)
    return f"{v.mean():.3f}" + (f" ±{v.std():.3f}" if len(v) > 1 else "      ")


lines = [f"12k train / 2k test, {len(seeds)} seeds, 784-256-128-64, 2 passes", ""]
lines.append(f"backprop head                  {ms([r['head'] for r in by['backprop']])}")
if by["label-b4"]:
    lines.append(f"label-b4  label stream read    {ms([r['label_read'] for r in by['label-b4']])}")
for a in ARMS:
    if by[a]:
        lines.append(f"{a:10s} tally, all layers   {ms([r['tally_all'] for r in by[a]])}")
lines.append("")
lines.append("per layer              L1      L2      L3")
for a in ARMS:
    if not by[a]:
        continue
    for key, name in [("tally", "tally"), ("probe", "probe"), ("picks", "units on"), ("dead", "dead"), ("sel_mean", "select.")]:
        row = [np.mean([r["layers"][i][key] for r in by[a]]) for i in range(3)]
        fmt = "{:7.1f}" if key == "picks" else "{:7.3f}"
        lines.append(f"{a:10s} {name:9s}" + " ".join(fmt.format(x) for x in row))
    lines.append(f"{a:10s} train s  {np.mean([r['t_train'] for r in by[a]]):7.1f}")
    lines.append("")
axT.text(0, 1, "\n".join(lines), family="monospace", fontsize=8.4, va="top", color=INK, transform=axT.transAxes)

# row 3: accuracy per layer -- tally (identity alone) and probe (capacity)
head = np.mean([r["head"] for r in by["backprop"]]) if by["backprop"] else None
for i in range(3):
    ax = fig.add_subplot(gs[2, i])
    groups = ["tally on which\nunits fired", "linear probe\non the code"]
    if i == 2 and by["label-b4"]:
        groups.append("label stream\n(top templates)")
    x = np.arange(len(groups))
    w = 0.8 / len(ARMS)
    for k, a in enumerate(ARMS):
        if not by[a]:
            continue
        vals = [[r["layers"][i]["tally"], r["layers"][i]["probe"]] for r in by[a]]
        if len(groups) == 3:
            vals = [v + [r.get("label_read", np.nan)] for v, r in zip(vals, by[a])]
        vals = np.array(vals, float)
        mu, lo, hi = np.nanmean(vals, 0), np.nanmin(vals, 0), np.nanmax(vals, 0)
        xs = x + (k - (len(ARMS) - 1) / 2) * w
        keep = ~np.isnan(mu)
        ax.bar(xs[keep], mu[keep], w * 0.9, color=COL[a], label=a, linewidth=0)
        ax.errorbar(xs[keep], mu[keep], yerr=[(mu - lo)[keep], (hi - mu)[keep]], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=2)
        for xx, m in zip(xs[keep], mu[keep]):
            ax.text(xx, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    if head is not None:
        ax.axhline(head, color=COL["backprop"], lw=1, ls=(0, (4, 3)))
        ax.text(x[-1] + 0.45, head + 0.012, "backprop head", fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(x, groups, fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(f"layer {i+1} ({[256,128,64][i]} units): held-out accuracy", loc="left", fontsize=10, color=INK)
    style(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if i == 0:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

# row 3, col 4: the headline, one bar per arm
ax = fig.add_subplot(gs[2, 3])
names, vals, cols = [], [], []
if by["backprop"]:
    names.append("backprop\nhead"); vals.append([r["head"] for r in by["backprop"]]); cols.append(COL["backprop"])
for a in ARMS[1:]:
    if by[a]:
        names.append(f"{a}\nL1 tally"); vals.append([r["layers"][0]["tally"] for r in by[a]]); cols.append(COL[a])
if by["label-b4"]:
    names.append("label-b4\nlabel stream"); vals.append([r["label_read"] for r in by["label-b4"]]); cols.append(COL["label-b4"])
mu = [np.mean(v) for v in vals]
ax.bar(np.arange(len(names)), mu, 0.7, color=cols, linewidth=0)
for k, (m, v) in enumerate(zip(mu, vals)):
    ax.errorbar(k, m, yerr=[[m - min(v)], [max(v) - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.text(k, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(np.arange(len(names)), names, fontsize=7.5, color=INK)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("the headline: label read, per arm", loc="left", fontsize=10, color=INK)
style(ax)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)

# row 4: selectivity per unit, per layer, per arm
for i in range(3):
    ax = fig.add_subplot(gs[3, i])
    bins = np.linspace(0.1, 1.0, 19)
    for a in ARMS:
        if not by[a]:
            continue
        sel = np.concatenate([r["layers"][i]["sel"] for r in by[a]])
        if len(sel):
            ax.hist(sel, bins=bins, histtype="step", lw=2, color=COL[a], label=a, density=True)
            ax.axvline(np.median(sel), color=COL[a], lw=1, ls=(0, (2, 2)))
    ax.set_xlabel("selectivity of a live unit: max over classes of P(class | unit fires)", fontsize=8.5, color=INK2)
    ax.set_title(f"layer {i+1}: how much one unit means one class", loc="left", fontsize=10, color=INK)
    ax.text(0.98, 0.62, "\n".join(f"{a}: {np.mean([r['layers'][i]['picks'] for r in by[a]]):.1f} on, "
                                  f"{np.mean([r['layers'][i]['dead'] for r in by[a]])*100:.0f}% dead"
                                  for a in ARMS if by[a]),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2, family="monospace")
    style(ax)
    ax.set_yticks([])
    if i == 0:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# row 4, col 4: how the label part sits in the layer-3 templates (label arm)
ax = fig.add_subplot(gs[3, 3])
if LAB is not None:
    share = np.linalg.norm(LAB, axis=1) / (np.linalg.norm(W3, axis=1) + 1e-8)
    live = Z["N3"] > 0
    ax.hist(share[live], bins=np.linspace(0, 1, 21), color=COL["label-b4"], linewidth=0)
    ax.set_xlabel("share of a layer-3 template's norm that is the label part", fontsize=8.5, color=INK2)
    ax.set_title(f"label-b4: {live.sum()} live top templates, label weight {LAB.max():.2f} max", loc="left",
                 fontsize=10, color=INK)
    style(ax)
    ax.set_yticks([])
else:
    ax.axis("off")

fig.suptitle("Search in place of backprop, same 784-256-128-64 ReLU shape: beam over the order, label as a "
             "second stream at the top", x=0.04, ha="left", fontsize=12, color=INK, y=0.985)
fig.savefig(OUT / "board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
