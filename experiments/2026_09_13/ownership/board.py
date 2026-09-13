"""One board: the templates of each rule, the configurations the hard rule finds
(input / who owns each pixel / reconstruction), supports and templates-on
distributions, the price sweep, and the completeness check."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
OUT = HERE / "results"
RULES = ["order", "hard", "soft"]
COL = {"order": "#2a78d6", "hard": "#eb6834", "soft": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
LAM = 0.02

res = [json.load(open(p)) for p in sorted(OUT.glob("*_l*_s*.json"))]
by = {r: [x for x in res if x["rule"] == r and x["lam"] == LAM] for r in RULES}
sweep = {x["lam"]: x for x in res if x["rule"] == "hard" and x["seed"] == 0}
astar = json.load(open(OUT / "astar_hard_l0.02.json")) if (OUT / "astar_hard_l0.02.json").exists() else None


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def tiles(ax, P, idx, title, cmap="RdBu_r", vmin=-1, vmax=1, rows=6, cols=8, norm=True):
    grid = np.full((rows * 29 + 1, cols * 29 + 1), np.nan, np.float32)
    for k, t in enumerate(idx[:rows * cols]):
        r, c = divmod(k, cols)
        w = P[t].reshape(28, 28)
        if norm:
            w = w / (np.abs(w).max() + 1e-8)
        grid[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = w
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.axis("off")


def ms(vals, f="{:.3f}"):
    v = np.array(vals, float)
    return f.format(v.mean()) + (f" ±{v.std():.3f}" if len(v) > 1 else "")


fig = plt.figure(figsize=(17, 15), facecolor=SURF)
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.05], height_ratios=[1, 1.05, 0.9],
                      hspace=0.35, wspace=0.2, top=0.95, left=0.04, right=0.99, bottom=0.05)

# row 1: templates per rule, the 48 most-used, seed 0
for j, r in enumerate(RULES):
    ax = fig.add_subplot(gs[0, j])
    p = OUT / f"weights_{r}_l{LAM:g}.npz"
    if p.exists():
        Z = np.load(p)
        order = np.argsort(-Z["fires"])[:48]
        tiles(ax, Z["W"], order, f"{r}: the 48 most-used templates (price {LAM}, seed 0)")
    else:
        ax.axis("off")

# row 1, col 4: the numbers
axT = fig.add_subplot(gs[0, 3])
axT.axis("off")
lines = [f"12k train / 2k test, one pass, 784 -> 256, price {LAM}", ""]
lines.append("                    order      hard       soft")
keys = [("support_median", "support px", "{:.0f}"), ("on_per_image", "on / image", "{:.2f}"),
        ("unexplained", "unexplained", "{:.3f}"), ("cost", "cost", "{:.3f}"), ("overlap", "overlap", "{:.3f}"),
        ("dead", "dead", "{:.2f}"), ("hired", "hired", "{:.0f}"), ("tally", "tally", "{:.3f}"),
        ("sel_mean", "selectivity", "{:.2f}"), ("train_rounds", "rounds", "{:.1f}"),
        ("train_cap_hits", "round-cap hits", "{:.0f}"), ("slot_cap_hits", "slot-cap hits", "{:.0f}"),
        ("t_train", "train s", "{:.0f}")]
for k, name, f in keys:
    row = []
    for r in RULES:
        v = [x[k] for x in by[r]]
        row.append(f.format(np.mean(v)) if v else "-")
    lines.append(f"{name:15s}" + "".join(f"{s:>11s}" for s in row))
lines.append("")
lines.append("hard, price sweep     0.01      0.02      0.04")
for k, name, f in [("support_median", "support px", "{:.0f}"), ("on_per_image", "on / image", "{:.2f}"),
                   ("unexplained", "unexplained", "{:.3f}"), ("tally", "tally", "{:.3f}"), ("dead", "dead", "{:.2f}")]:
    lines.append(f"{name:15s}" + "".join(f"{(f.format(sweep[l][k]) if l in sweep else '-'):>10s}" for l in (0.01, 0.02, 0.04)))
if astar:
    lines += ["", f"search check, hard {LAM}, {astar['n']} test images, mean cost:",
              f"  beam 1  {astar['beam1']:.4f}   beam 4  {astar['beam4']:.4f}   beam 16  {astar['beam16']:.4f}",
              f"  best-first from the beam-4 incumbent, {astar['cap']} expansions:",
              f"  found cheaper on {astar['astar_better']*100:.0f}% of images; proved optimal {astar['proved']*100:.0f}%",
              f"  (branching truncated at {astar['branch']} on every node: bound too loose to prove)",
              f"  bound violations {astar['violations']}"]
axT.text(0, 1, "\n".join(lines), family="monospace", fontsize=8.2, va="top", color=INK, transform=axT.transAxes)

# row 2: the configurations the hard rule finds on eight test images
ax = fig.add_subplot(gs[1, 0:3])
p = OUT / f"weights_hard_l{LAM:g}.npz"
if p.exists():
    Z = np.load(p)
    X, idx, owner, xhat = Z["X"], Z["idx"], Z["owner"], Z["xhat"]
    k = len(X)
    cell = 30
    canvas = np.full((3 * cell + 1, k * cell + 1, 3), 1.0, np.float32)
    qual = plt.get_cmap("tab20").colors
    for i in range(k):
        x = X[i].reshape(28, 28) / (X[i].max() + 1e-8)
        rec = np.clip(xhat[i].reshape(28, 28) / (X[i].max() + 1e-8), 0, 1)
        ow = owner[i].reshape(28, 28)
        img_in = np.repeat((1 - x)[..., None], 3, 2)
        img_rec = np.repeat((1 - rec)[..., None], 3, 2)
        img_ow = np.ones((28, 28, 3), np.float32)
        for s in np.unique(ow[ow >= 0]):
            img_ow[ow == s] = qual[int(s) % 20]
        img_ow = img_ow * (0.35 + 0.65 * x[..., None]) + (1 - (0.35 + 0.65 * x[..., None]))  # fade where no ink
        for rr, im in enumerate([img_in, img_ow, img_rec]):
            canvas[1 + rr * cell:1 + rr * cell + 28, 1 + i * cell:1 + i * cell + 28] = im
    ax.imshow(canvas, interpolation="nearest")
    n_on = [(idx[i] >= 0).sum() for i in range(k)]
    ax.set_title("hard rule: input / which template owns each pixel (one colour per template on) / reconstruction   "
                 + "   templates on: " + " ".join(str(n) for n in n_on), color=INK, fontsize=9.5, loc="left")
    for rr, name in enumerate(["input", "owners", "rebuilt"]):
        ax.text(-2, 1 + rr * cell + 14, name, ha="right", va="center", fontsize=8, color=INK2)
ax.axis("off")

# row 2, col 4: price sweep, hard rule
ax = fig.add_subplot(gs[1, 3])
ls = [l for l in (0.01, 0.02, 0.04) if l in sweep]
if ls:
    x = np.arange(len(ls))
    w = 0.38
    sup = [sweep[l]["support_median"] for l in ls]
    on = [sweep[l]["on_per_image"] for l in ls]
    ax.bar(x - w / 2, sup, w, color=COL["hard"], label="support (pixels per template)", linewidth=0)
    for xx, v in zip(x - w / 2, sup):
        ax.text(xx, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, on, w, color=COL["hard"], alpha=0.45, label="templates on per image", linewidth=0)
    for xx, v in zip(x + w / 2, on):
        ax2.text(xx, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"price {l}" for l in ls], fontsize=8.5, color=INK)
    ax.set_ylabel("pixels carrying 90% of a template", fontsize=8, color=INK2)
    ax2.set_ylabel("templates on per image", fontsize=8, color=INK2)
    ax.set_title("hard rule: what the price buys", loc="left", fontsize=10, color=INK)
    style(ax)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.tick_params(colors=INK2, labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
else:
    ax.axis("off")

# row 3: support per template, templates on per image, tally & selectivity
ax = fig.add_subplot(gs[2, 0])
bins = np.arange(0, 200, 8)
for r in RULES:
    s = np.concatenate([x["support"] for x in by[r]]) if by[r] else np.array([])
    if len(s):
        ax.hist(np.clip(s, 0, 199), bins=bins, histtype="step", lw=2, color=COL[r], label=r, density=True)
        ax.axvline(np.median(s), color=COL[r], lw=1, ls=(0, (2, 2)))
ax.set_xlabel("pixels carrying 90% of a live template's energy", fontsize=8.5, color=INK2)
ax.set_title("how big a template is", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
style(ax)
ax.set_yticks([])

ax = fig.add_subplot(gs[2, 1])
for r in RULES:
    if by[r]:
        h = np.mean([x["on_hist"] for x in by[r]], 0)
        h = h / h.sum()
        ax.plot(np.arange(len(h)), h, color=COL[r], lw=2, label=r, drawstyle="steps-mid")
ax.set_xlabel("templates on per test image (chosen by the search)", fontsize=8.5, color=INK2)
ax.set_title("how many templates explain an image", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
style(ax)
ax.set_yticks([])
ax.set_xlim(0, 20)

ax = fig.add_subplot(gs[2, 2])
x = np.arange(3)
w = 0.26
for k, r in enumerate(RULES):
    if not by[r]:
        continue
    vals = np.array([[z["tally"], z["sel_mean"], z["unexplained"]] for z in by[r]])
    mu, lo, hi = vals.mean(0), vals.min(0), vals.max(0)
    xs = x + (k - 1) * w
    ax.bar(xs, mu, w * 0.9, color=COL[r], label=r, linewidth=0)
    ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    for xx, m in zip(xs, mu):
        ax.text(xx, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, ["tally on which\ntemplates fired", "selectivity\nof a template", "unexplained\nenergy"], fontsize=8.5, color=INK)
ax.set_ylim(0, 1.25)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("identity and fit, held out", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
style(ax)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)

# row 3, col 4: completeness check, per image
ax = fig.add_subplot(gs[2, 3])
if astar:
    b = np.array([r["beam"] for r in astar["rows"]])
    a = np.array([r["astar"] for r in astar["rows"]])
    lo, hi = min(a.min(), b.min()) - 0.02, max(a.max(), b.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    b1 = np.array([r["beam1"] for r in astar["rows"]])
    b16 = np.array([r["beam16"] for r in astar["rows"]])
    ax.scatter(b16, b1, s=28, facecolor="none", edgecolor=COL["order"], linewidth=1.2, zorder=3, label="beam 1")
    ax.scatter(b16, b, s=28, color=COL["hard"], edgecolor=SURF, linewidth=0.8, zorder=4, label="beam 4")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("beam of 16: cost", fontsize=8.5, color=INK2)
    ax.set_ylabel("cost at a narrower beam", fontsize=8.5, color=INK2)
    ax.set_title(f"does a wider search find cheaper configurations? ({astar['n']} images)", loc="left", fontsize=10, color=INK)
    style(ax)
else:
    ax.axis("off")

fig.suptitle("Ownership by strength: one layer, the search over configurations, three learning rules",
             x=0.04, ha="left", fontsize=12, color=INK, y=0.985)
fig.savefig(OUT / "board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
