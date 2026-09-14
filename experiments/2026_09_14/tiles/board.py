"""One board for the non-overlapping tiles: the tables, and what stops double-counting buys.

    python board.py <tag>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
OLD = Path(__file__).parents[1] / "patches/results/tables_read.json"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
P, G, C = 4, 7, 10
NODES, CELLS = G * G, P * P
VMAX = 0.18                                     # a flat grid is 1/16 = 0.0625

tag = sys.argv[1] if len(sys.argv) > 1 else "tiles"
r = json.load(open(OUT / f"{tag}.json")); Z = np.load(OUT / f"{tag}.npz")
old = json.load(open(OLD)) if OLD.exists() else None
T, M, spread = Z["tables"], Z["mass"], Z["spread"]
gap, conf, pred, yte = Z["gap"], Z["conf"], Z["pred"], Z["yte"]
g = P + 1


def mosaic(Tc):
    Mo = np.full((G * g + 1, G * g + 1), np.nan, np.float32)
    for i in range(NODES):
        rr, cc = divmod(i, G)
        Mo[1 + rr * g:1 + rr * g + P, 1 + cc * g:1 + cc * g + P] = Tc[i].reshape(P, P)
    return Mo


fig = plt.figure(figsize=(15.5, 22), facecolor=SURF)
gs = fig.add_gridspec(4, 1, height_ratios=[6.0, 1.7, 3.4, 3.0], hspace=0.44,
                      left=0.04, right=0.98, top=0.918, bottom=0.02)
cap = lambda k, s, dy=0.011: fig.text(0.04, gs[k].get_position(fig).y1 + dy, s, fontsize=10.5, color=INK, va="bottom")

# --- A: the tables ---------------------------------------------------------------------------------
gsa = gs[0].subgridspec(2, 5, wspace=0.04, hspace=0.10)
for c in range(C):
    ax = fig.add_subplot(gsa[c // 5, c % 5]); ax.axis("off")
    ax.imshow(mosaic(T[c]), cmap="Reds", vmin=0, vmax=VMAX, interpolation="nearest")
    ax.set_title(f"C = {c}", fontsize=10.5, color=INK, pad=3)
cap(0, "A.  All 49 grids for one value of the parent, each drawn as a 4x4 picture in the window's own place.  The windows tile the picture exactly —\n"
       "     49 windows x 16 cells = 784 pixels, nothing shared, nothing left over.  Each grid sums to 1; a flat grid is 0.0625 a cell.")

# --- B: one window ---------------------------------------------------------------------------------
idx = np.arange(NODES); ok = M.sum(2).min(0) > 500
hi = int(idx[ok][np.argmax(spread[ok])])
gsb = gs[1].subgridspec(1, C, wspace=0.07)
for c in range(C):
    ax = fig.add_subplot(gsb[c])
    t = T[c, hi].reshape(P, P)
    ax.imshow(t, cmap="Reds", vmin=0, vmax=VMAX, interpolation="nearest")
    for a in range(P):
        for b in range(P):
            ax.text(b, a, f"{t[a, b]*100:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if t[a, b] > VMAX * 0.62 else INK2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"C = {c}", fontsize=9, color=INK, pad=2)
    if c == 0:
        ax.set_ylabel(f"window {hi}\n(row {hi//G}, col {hi%G})", fontsize=8, color=INK2)
cap(1, f"B.  Window {hi} in full — the window the label moves most — as percentages of the cell.")

# --- C: what stopping the double-counting bought -----------------------------------------------------
gsc = gs[2].subgridspec(1, 3, wspace=0.24, width_ratios=[1.5, 1, 1])
ax = fig.add_subplot(gsc[0])
ks = list(r["acc_by_gap"].keys()); x = np.arange(len(ks)); w = 0.38
v1 = [r["acc_by_gap"][k][0] for k in ks]; n1 = [r["acc_by_gap"][k][1] for k in ks]
ax.bar(x - w / 2, v1, w, color="#7a1f1f", label="4x4 tiles, no overlap (this build)")
for i, (a, n) in enumerate(zip(v1, n1)):
    ax.text(i - w / 2, a + 0.02, f"{a:.2f}\nn={n}", ha="center", fontsize=6.6, color=INK)
if old:
    v2 = [old["acc_by_gap_ink"].get(k, [np.nan, 0])[0] for k in ks]; n2 = [old["acc_by_gap_ink"].get(k, [0, 0])[1] for k in ks]
    ax.bar(x + w / 2, v2, w, color="#cfcbc4", label="5x5 stride 1, every pixel counted 25 times")
    for i, (a, n) in enumerate(zip(v2, n2)):
        if not np.isnan(a):
            ax.text(i + w / 2, a + 0.02, f"{a:.2f}\nn={n}", ha="center", fontsize=6.6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(ks, fontsize=8); ax.set_ylim(0, 1.2)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_xlabel("gap between the best two digits (nats)", fontsize=8.5)
ax.set_title("being unsure means something again\naccuracy now climbs with the gap, 0.39 to 1.00", fontsize=9.5, color=INK, loc="left")

ax = fig.add_subplot(gsc[1])
ax.hist(np.clip(gap, 0, 60), bins=60, color="#7a1f1f", label=f"tiles (median {r['median_gap']:.1f})")
ax.axvline(5, color=INK, ls="--", lw=1.0)
ax.set_xlabel("nats", fontsize=8.5)
ax.set_title(f"how torn it is\nmedian gap {r['median_gap']:.1f} nats, was {old['ink_median_gap_nats']:.0f}" if old else "how torn it is",
             fontsize=9.5, color=INK, loc="left")

ax = fig.add_subplot(gsc[2])
ax.hist(conf, bins=np.linspace(0.1, 1.0, 46), color="#7a1f1f")
ax.set_xlabel("top guess", fontsize=8.5)
ax.set_title(f"how sure it claims to be\nmean {r['mean_conf']:.3f}, was {old['ink_mean_conf']:.4f}" if old else "how sure it claims to be",
             fontsize=9.5, color=INK, loc="left")
for a in fig.axes[-3:]:
    a.tick_params(labelsize=7.5)
    for sp in a.spines.values(): sp.set_visible(False)
cap(2, "C.  The point of the build.  Each pixel now has exactly one parent, so each piece of evidence is counted once.", 0.030)

# --- D: genuinely torn ------------------------------------------------------------------------------
gsd = gs[3].subgridspec(2, 12, height_ratios=[1.35, 1], wspace=0.16, hspace=0.06)
Xa, ya, pa = Z["Xamb"], Z["yamb"], Z["post_amb"]
for k in range(12):
    ax = fig.add_subplot(gsd[0, k]); ax.axis("off")
    ax.imshow(Xa[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"true {ya[k]}", fontsize=8, color=INK, pad=2)
    ax = fig.add_subplot(gsd[1, k])
    top = np.argsort(-pa[k])[:2]
    ax.bar(range(C), pa[k], color=["#7a1f1f" if c in top else "#cfcbc4" for c in range(C)])
    ax.set_ylim(0, 1); ax.set_xticks(range(C)); ax.set_xticklabels(range(C), fontsize=5)
    ax.set_yticks([0, 0.5, 1] if k == 0 else []); ax.tick_params(labelsize=6, pad=1)
    for sp in ax.spines.values(): sp.set_visible(False)
cap(3, f"D.  The twelve most torn held-out pictures and what the machine believes about the parent's value.  {r['frac_gap_under_5']*100:.0f}% of pictures now\n"
       f"     sit within 5 nats of a second answer (it was 1%), so there is finally something for a resolution step to resolve.")

fig.suptitle(f"Non-overlapping tiles — {P}x{P} windows at stride {P}, {NODES} of them, {r['numbers_trained']:,} numbers, {r['n_train']:,} pictures in {r['train_seconds']}s\n"
             f"read of C {r['acc_ink']:.4f} (was {list(old['reads'].values())[0]:.4f} at stride 1) — 3.5 points paid for a machine that knows when it does not know: "
             f"mean top guess {r['mean_conf']:.3f} not {old['ink_mean_conf']:.4f}, median gap {r['median_gap']:.1f} nats not {old['ink_median_gap_nats']:.0f}" if old else "Non-overlapping tiles",
             fontsize=12, color=INK, y=0.986)
fig.savefig(OUT / f"{tag}.png", dpi=135, facecolor=SURF)
print(f"-> {OUT}/{tag}.png  window {hi}")
