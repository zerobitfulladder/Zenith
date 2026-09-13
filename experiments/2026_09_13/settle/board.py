"""One board for the settling tower: what the groups are under each arm, how many parts
they have, the label read five ways, eight images through the carve tower."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import settle as S

HERE = Path(__file__).parent
OUT = HERE / "results"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ARMS = ["avg", "gated", "carve"]
COL = {"avg": "#2a78d6", "gated": "#eb6834", "carve": "#1baf7a"}
H1 = S.H1
res = {a: json.load(open(OUT / f"{a}.json")) for a in ARMS}
Z = {a: np.load(OUT / f"weights_{a}.npz") for a in ARMS}
C = {a: np.load(OUT / f"codes_{a}.npz") for a in ARMS}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def tiles(ax, P, order, title, LAB=None, W2=None, cmap="RdBu_r"):
    grid = np.full((6 * 29 + 1, 8 * 29 + 1), np.nan, np.float32)
    for k, t in enumerate(order[:48]):
        r, c = divmod(k, 8)
        w = P[t].reshape(28, 28)
        grid[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = w / (np.abs(w).max() + 1e-8)
    ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    if LAB is not None:
        for k, t in enumerate(order[:48]):
            r, c = divmod(k, 8)
            share = np.linalg.norm(LAB[t]) / (np.linalg.norm(W2[t]) + 1e-8)
            ax.text(2 + c * 29, 7 + r * 29, f"{int(LAB[t].argmax())}" if share > 0.05 else "-", fontsize=7, color=INK,
                    fontweight="bold", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.axis("off")


fig = plt.figure(figsize=(18, 16), facecolor=SURF)
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.2], height_ratios=[1, 0.9, 0.95],
                      hspace=0.32, wspace=0.18, top=0.95, left=0.03, right=0.99, bottom=0.04)

# row 1: layer 1 (carve arm) and the groups under each arm, painted through layer 1
W1 = Z["carve"]["W1"]
u1 = (C["carve"]["A1"] > 0).sum(0)
tiles(fig.add_subplot(gs[0, 0]), np.maximum(W1, 0), np.argsort(-u1), "layer 1: 48 most-used templates (after settling + learning)")
for k, a in enumerate(ARMS):
    W2p, W2 = Z[a]["W2p"], Z[a]["W2"]
    u2 = (C[a]["A2"] > 0).sum(0)
    P2 = W2p[:, :H1] @ np.maximum(Z[a]["W1"], 0)
    ax = fig.add_subplot(gs[0, k + 1]) if k < 2 else fig.add_subplot(gs[1, 0])
    tiles(ax, P2, np.argsort(-u2), f"layer 2, {a}: 48 most-used groups through layer 1 (digit = label carried)", LAB=W2[:, H1:], W2=W2)

# row 2: membership matrices and the numbers
for k, a in enumerate(ARMS):
    ax = fig.add_subplot(gs[1, k + 1]) if k < 2 else fig.add_subplot(gs[2, 3])
    W2p = Z[a]["W2p"]
    u2 = (C[a]["A2"] > 0).sum(0); u1a = (C[a]["A1"] > 0).sum(0)
    top2 = np.argsort(-u2)[:24]
    mem = W2p[top2, :H1] / np.maximum(W2p[top2, :H1].max(1, keepdims=True), 1e-8)
    cols = np.argsort(-u1a)[:64]
    ax.imshow(mem[:, cols], cmap="Oranges", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title(f"{a}: what the 24 most-used groups are made of", loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("the 64 most-used layer-1 units", fontsize=8, color=INK2); ax.set_ylabel("groups", fontsize=8, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])

axT = fig.add_subplot(gs[1, 3]); axT.axis("off")
L = [f"settling: {S.ITERS} steps, threshold {S.LAM1:.2f} (price {S.PRICE}), top-down drive {S.BETA:.2f}; 8k train, 2k held out", "",
     f"{'label read, held out (fb on / off)':36s}{'avg':>13s}{'gated':>13s}{'carve':>13s}"]
for key, name in [("label_top", "at the top"), ("tally_L1", "tally on L1 identities"), ("tally_L2", "tally on L2 identities"),
                  ("probe_L1", "probe on L1 code"), ("probe_L2", "probe on L2 code"), ("unexplained", "unexplained (pixels)")]:
    L.append(f"  {name:34s}" + "".join(f"{res[a]['modes']['feedback'][key]:6.3f}/{res[a]['modes']['no_feedback'][key]:5.3f} " for a in ARMS))
L.append("")
L.append(f"{'layer 2 groups':36s}{'avg':>13s}{'gated':>13s}{'carve':>13s}")
for key, name, fmt in [("live", "live", "{:13d}"), ("members_median", "members, median", "{:13.0f}"), ("members_mean", "members, mean", "{:13.1f}"),
                       ("wrappers", "wrappers (one unit > 90% of norm)", "{:13.0%}"), ("label_share_median", "label share, median", "{:13.2f}"), ("hired", "hired / carved", "{:13d}")]:
    L.append(f"  {name:34s}" + "".join(fmt.format(res[a]["L2"][key]) for a in ARMS))
L.append("")
L.append(f"{'reuse, feedback read':36s}{'avg':>13s}{'gated':>13s}{'carve':>13s}")
for key, name in [("on_per_image", "L2 on per image"), ("distinct_configs", "L2 distinct configs / 2k"), ("uses_entropy", "L2 usage entropy, bits")]:
    L.append(f"  {name:34s}" + "".join(f"{res[a]['modes']['feedback']['reuse_L2'][key]:13.2f}" for a in ARMS))
L.append(f"  {'L1 on per image (carve arm)':34s}{res['carve']['modes']['feedback']['reuse_L1']['on_per_image']:13.2f}")
L.append(f"  {'settle delta at the last step':34s}" + "".join(f"{res[a]['modes']['feedback']['settle_delta']:13.4f}" for a in ARMS))
axT.text(0, 1, "\n".join(L), family="monospace", fontsize=7.9, va="top", color=INK, transform=axT.transAxes)

# row 3: the label five ways, members, eight images through the carve tower
ax = fig.add_subplot(gs[2, 0])
names = ["label at\nthe top", "tally on\nL1", "tally on\nL2", "probe on\nL1", "probe on\nL2"]
keys = ["label_top", "tally_L1", "tally_L2", "probe_L1", "probe_L2"]
x = np.arange(len(names)); w = 0.26
for k, a in enumerate(ARMS):
    v = [res[a]["modes"]["feedback"][kk] for kk in keys]
    ax.bar(x + (k - 1) * w, v, w * 0.92, color=COL[a], label=a, linewidth=0)
    for xx, vv in zip(x + (k - 1) * w, v):
        ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(x, names, fontsize=8, color=INK); ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("is the label explained at the top? (feedback on, held out)", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3); style(ax); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

ax = fig.add_subplot(gs[2, 1])
for a in ARMS:
    W2p, n2 = Z[a]["W2p"], Z[a]["n2"]
    live = n2 > 0
    if live.any():
        ident = W2p[live, :H1]
        memc = (ident > 0.5 * ident.max(1, keepdims=True)).sum(1)
        ax.hist(memc, bins=np.arange(0.5, 12.5, 1), histtype="step", lw=2, color=COL[a], label=a)
ax.set_xlabel("layer-1 units carrying at least half a group's top weight", fontsize=8.5, color=INK2)
ax.set_title("how many parts a group is made of", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); style(ax); ax.set_yticks([])

ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
net = S.Net("carve", np.random.default_rng(0))
net.M1 = Z["carve"]["W1"].copy(); net.M2 = Z["carve"]["W2"].copy(); net.n2 = Z["carve"]["n2"]; net.refresh()
X8, y8 = Z["carve"]["Xte"], Z["carve"]["yte"]
st = net.settle(X8, None)
p8 = net.label_top(st)
exp8 = (st["a2"] @ net.W2p[:, :H1]) @ np.maximum(net.W1, 0)
canvas = np.ones((3 * 30 + 2, 8 * 30 + 2))
for i in range(8):
    x = X8[i].reshape(28, 28); xm = x.max() + 1e-8
    canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - x / xm
    canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip(st["xhat"][i].reshape(28, 28) / xm, 0, 1)
    e = exp8[i].reshape(28, 28)
    canvas[61:89, 1 + i * 30:29 + i * 30] = 1 - np.clip(e / (e.max() + 1e-8), 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
for r, name in enumerate(["input", "L1 rebuilt", "top expects"]):
    ax.text(-3, 15 + r * 30, name, ha="right", va="center", fontsize=8, color=INK2)
for i in range(8):
    p, t = int(p8[i]), int(y8[i])
    ax.text(15 + i * 30, 3 * 30 + 10, f"{p if p >= 0 else '-'}/{t}", ha="center", va="center", fontsize=10,
            color=(COL["carve"] if p == t else "#e34948"), fontweight="bold")
ax.set_xlim(-28, 8 * 30 + 2); ax.set_ylim(3 * 30 + 20, -2)
ax.set_title("carve tower, eight held-out images: label read at the top / true", loc="left", fontsize=10, color=INK)

fig.suptitle("Two layers settling together: explaining away by inhibition, the price as threshold, expectation as drive; "
             "three ways for layer-2 groups to exist", x=0.03, ha="left", fontsize=12, color=INK, y=0.985)
fig.savefig(OUT / "board.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "board.png")
