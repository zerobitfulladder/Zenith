"""One board for the tower: layer-1 templates, layer-2 templates expanded to pixels with the
label they carry, eight test images read by the tower, the label read five ways, and reuse."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tower as T

HERE = Path(__file__).parent
OUT = HERE / "results"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

res = json.load(open(OUT / "tower.json"))
Z = np.load(OUT / "weights.npz")
codes = np.load(OUT / "codes_test.npz")
W1, Wp1, n1 = Z["W1"], Z["Wp1"], Z["n1"]
W2, Wp2, n2 = Z["W2"], Z["Wp2"], Z["n2"]
H1 = W1.shape[0]
uses1 = (codes["C1"] > 0).sum(0)
uses2 = (codes["C2"] > 0).sum(0)
P2 = Wp2[:, :H1] @ Wp1                                # a top template painted through layer 1
LAB = W2[:, H1:]

# the tower, rebuilt from the saved weights, reads eight test images
tw = T.Tower(np.random.default_rng(0))
tw.L1.W, tw.L1.Wp, tw.L1.n, tw.L1.price = W1, Wp1, n1, Z["price1"]
tw.L2.W, tw.L2.Wp, tw.L2.n, tw.L2.price = W2, Wp2, n2, Z["price2"]
X8, y8 = Z["Xte"], Z["yte"]
st1, st2, _ = tw.read(X8, None, True)
pred8 = tw.label_top(st2)
exp8 = tw.expect(st2) @ Wp1                           # what the top expects, painted in pixels


def tiles(ax, P, order, title, labels=None, cmap="RdBu_r"):
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for k, t in enumerate(order[:48]):
        r, c = divmod(k, 8)
        w = P[t].reshape(28, 28)
        grid[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = w / (np.abs(w).max() + 1e-8)
    if cmap == "RdBu_r":
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    else:
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    if labels is not None:
        for k, t in enumerate(order[:48]):
            r, c = divmod(k, 8)
            share = np.linalg.norm(LAB[t]) / (np.linalg.norm(W2[t]) + 1e-8)
            ax.text(2 + c * 29, 7 + r * 29, f"{int(LAB[t].argmax())}" if share > 0.05 else "-", fontsize=7, color=INK,
                    fontweight="bold", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.axis("off")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


fig = plt.figure(figsize=(17, 15), facecolor=SURF)
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.15], height_ratios=[1, 1.05, 0.9],
                      hspace=0.35, wspace=0.2, top=0.95, left=0.03, right=0.99, bottom=0.04)

tiles(fig.add_subplot(gs[0, 0]), Wp1, np.argsort(-uses1), "layer 1: the 48 most-used templates")
tiles(fig.add_subplot(gs[0, 1]), P2, np.argsort(-uses2), "layer 2: 48 most-used groups through layer 1 (digit = label carried)", labels=True)
# layer-2 membership: which layer-1 units each of the 24 most-used groups contains
ax = fig.add_subplot(gs[0, 2])
top2 = np.argsort(-uses2)[:24]
mem = Wp2[top2, :H1] / np.maximum(Wp2[top2, :H1].max(1, keepdims=True), 1e-8)
cols = np.argsort(-uses1)[:64]
ax.imshow(mem[:, cols], cmap="Oranges", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_xlabel("the 64 most-used layer-1 units", fontsize=8.5, color=INK2)
ax.set_ylabel("24 most-used layer-2 groups", fontsize=8.5, color=INK2)
ax.set_title("what each group is made of (weight on each layer-1 unit)", loc="left", fontsize=9.5, color=INK)
ax.set_xticks([]); ax.set_yticks([])

axT = fig.add_subplot(gs[0, 3]); axT.axis("off")
fb, nf = res["modes"]["feedback"], res["modes"]["no_feedback"]
L = [f"8k train, one pass, 2k held out, price {T.LAM}, label weight {T.LABEL_W}", "",
     f"{'label read, held out':28s}{'feedback':>10s}{'no fb':>8s}",
     f"{'  at the top (L2 label part)':28s}{fb['label_top']:10.3f}{nf['label_top']:8.3f}",
     f"{'    when a top template is on':28s}{fb['label_top_when_covered']:10.3f}{nf['label_top_when_covered']:8.3f}",
     f"{'    coverage':28s}{fb['covered']:10.2f}{nf['covered']:8.2f}",
     f"{'  tally on L1 identities':28s}{fb['tally_L1']:10.3f}{nf['tally_L1']:8.3f}",
     f"{'  tally on L2 identities':28s}{fb['tally_L2']:10.3f}{nf['tally_L2']:8.3f}",
     f"{'  probe on L1 code':28s}{fb['probe_L1']:10.3f}{nf['probe_L1']:8.3f}",
     f"{'  probe on L2 code':28s}{fb['probe_L2']:10.3f}{nf['probe_L2']:8.3f}",
     f"{'  tower cost':28s}{fb['tower_cost']:10.3f}{nf['tower_cost']:8.3f}", "",
     f"{'reuse (feedback read)':28s}{'L1':>10s}{'L2':>8s}",
     f"{'  on per image':28s}{fb['reuse_L1']['on_per_image']:10.2f}{fb['reuse_L2']['on_per_image']:8.2f}",
     f"{'  live templates':28s}{fb['reuse_L1']['live']:10d}{fb['reuse_L2']['live']:8d}",
     f"{'  uses per template':28s}{fb['reuse_L1']['uses_mean']:10.1f}{fb['reuse_L2']['uses_mean']:8.1f}",
     f"{'  usage entropy, bits':28s}{fb['reuse_L1']['uses_entropy']:10.2f}{fb['reuse_L2']['uses_entropy']:8.2f}",
     f"{'  distinct configs / 2k':28s}{fb['reuse_L1']['distinct_configs']:10d}{fb['reuse_L2']['distinct_configs']:8d}", "",
     f"L2 groups: {res['L2']['live']} live, members median {res['L2']['members_median']:.0f} (mean {res['L2']['members_mean']:.1f}),",
     f"  wrappers (one L1 unit > 90% of norm) {res['L2']['wrappers']*100:.0f}%, label share median {res['L2']['label_share_median']:.2f},",
     f"  hired {res['L2']['hired']}, recycled for not paying {res['L2']['recycled']}",
     f"L1: support {res['L1']['support_median']:.0f} px, wholes/strokes/dots {res['L1']['wholes']}/{res['L1']['strokes']}/{res['L1']['dots']},",
     f"  hired {res['L1']['hired']}, recycled {res['L1']['recycled']}",
     f"train {res['t_train']:.0f} s"]
axT.text(0, 1, "\n".join(L), family="monospace", fontsize=8.3, va="top", color=INK, transform=axT.transAxes)

# row 2: eight test images through the tower
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
qual = plt.get_cmap("tab20").colors
rows = ["input", "L1 owners", "L1 rebuilt", "top expects", "label: top / true"]
canvas = np.ones((5 * 30 + 2, 8 * 30 + 2, 3))
for i in range(8):
    x = X8[i].reshape(28, 28); xm = x.max() + 1e-8
    img = 1 - np.repeat((x / xm)[..., None], 3, 2)
    own = np.ones((28, 28, 3))
    ow = st1["owner"][i]
    slots = [s for s in range(T.S) if st1["idx"][i][s] >= 0]
    for k, s in enumerate(slots):
        own[(ow == s).reshape(28, 28)] = qual[k % 20]
    fade = 0.25 + 0.75 * (x / xm)[..., None]
    own = own * fade + (1 - fade)
    reb = 1 - np.repeat(np.clip(st1["xhat"][i].reshape(28, 28) / xm, 0, 1)[..., None], 3, 2)
    ex = exp8[i].reshape(28, 28); ex = 1 - np.repeat(np.clip(ex / (ex.max() + 1e-8), 0, 1)[..., None], 3, 2)
    ex = ex * np.array([1.0, 0.85, 0.6]) + (1 - np.array([1.0, 0.85, 0.6])) * 0  # warm tint
    for r, im in enumerate([img, own, reb, ex]):
        canvas[1 + r * 30:1 + r * 30 + 28, 1 + i * 30:1 + i * 30 + 28] = im
ax.imshow(canvas, interpolation="nearest")
for r, name in enumerate(rows):
    ax.text(-4, 15 + r * 30, name, ha="right", va="center", fontsize=8.5, color=INK2)
for i in range(8):
    p = int(pred8[i]); t = int(y8[i])
    ax.text(15 + i * 30, 1 + 4 * 30 + 14, f"{p if p >= 0 else '-'} / {t}", ha="center", va="center", fontsize=11,
            color=(AQUA if p == t else ORANGE), fontweight="bold")
ax.set_title("eight held-out images read by the tower (feedback on): who owns each pixel at layer 1, the rebuild, "
             "what the top's groups expect below, and the label read from the top", loc="left", fontsize=10, color=INK)
ax.set_xlim(-30, 8 * 30 + 2); ax.set_ylim(5 * 30 + 2, -2)

# row 3: the label five ways, membership, usage
ax = fig.add_subplot(gs[2, 0:2])
names = ["label at\nthe top", "tally on\nL1 identities", "tally on\nL2 identities", "probe on\nL1 code", "probe on\nL2 code"]
keys = ["label_top", "tally_L1", "tally_L2", "probe_L1", "probe_L2"]
x = np.arange(len(names)); w = 0.38
for k, (mode, col, lab) in enumerate([("feedback", ORANGE, "feedback on"), ("no_feedback", BLUE, "feedback off")]):
    v = [res["modes"][mode][kk] for kk in keys]
    ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=lab, linewidth=0)
    for xx, vv in zip(x + (k - 0.5) * w, v):
        ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, names, fontsize=8.5, color=INK); ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("is the label explained at the top, or read off layer 1?  (held out)", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2); style(ax); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

ax = fig.add_subplot(gs[2, 2])
live2 = n2 > 0
memc = (Wp2[:, :H1] > 0.5 * Wp2[:, :H1].max(1, keepdims=True)).sum(1)[live2]
ax.hist(memc, bins=np.arange(0.5, max(memc.max(), 6) + 1.5, 1), color=ORANGE, linewidth=0)
ax.set_xlabel("layer-1 units carrying at least half a group's top weight", fontsize=8.5, color=INK2)
ax.set_title(f"how many parts a top group is made of ({live2.sum()} live)", loc="left", fontsize=10, color=INK)
style(ax); ax.set_yticks([])

ax = fig.add_subplot(gs[2, 3])
for u, col, lab in [(uses1[uses1 > 0], AQUA, "layer 1"), (uses2[uses2 > 0], ORANGE, "layer 2")]:
    ax.hist(u, bins=np.logspace(0, np.log10(max(uses1.max(), uses2.max()) + 1), 25), histtype="step", lw=2, color=col, label=lab)
ax.set_xscale("log"); ax.set_xlabel("times a template is on, over 2k held-out images", fontsize=8.5, color=INK2)
ax.set_title("reuse: how often each template is used", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); style(ax); ax.set_yticks([])

fig.suptitle("Two layers, one search over the tower, the label as an exact stream at the top", x=0.03, ha="left", fontsize=12, color=INK, y=0.985)
fig.savefig(OUT / "board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
