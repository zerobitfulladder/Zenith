"""Board: a floor under every pixel, in training and in reading.

    python board_eigengrau.py <tag>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
r = json.load(open(OUT / f"{tag}_eigengrau.json")); Z = np.load(OUT / f"{tag}_eigengrau.npz")
Wb, Wa = Z["W_before"], Z["W_after"]
E, at, ab, cb, fr = Z["eps"], Z["acc_tr"], Z["acc_both"], Z["conf_both"], Z["frames"]
CH = r["chosen_eps"]
x = np.arange(len(E))

fig = plt.figure(figsize=(15, 12), facecolor=SURF)
gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 3.2], hspace=0.62, left=0.04, right=0.98, top=0.895, bottom=0.06)
cap = lambda k, s: fig.text(0.04, gs[k].get_position(fig).y1 + 0.013, s, fontsize=10.5, color=INK, va="bottom")

v = float(np.percentile(np.abs(Wa), 99.5))   # scaled to the AFTER picture, so the before frame clips: that is the point
for row, (W, ttl) in enumerate([
        (Wb, "A.  Before.  Each picture is one digit's scorecard — the single number every pixel is multiplied by.  The loud frame around the edge is\n"
             "     built from grids that saw a handful of stray specks in 50,000 pictures, scaled up to look as certain as the middle.  Both rows share one\n"
             "     colour scale, set by row B, so the frame here runs off the end of it."),
        (Wa, f"B.  After, with a floor of {CH:g} under every pixel in training.  A grid now starts flat and needs real ink to move off it.  The frame is gone;\n"
             f"     the digits underneath are unchanged.")]):
    gsx = gs[row].subgridspec(1, 10, wspace=0.04)
    for c in range(10):
        ax = fig.add_subplot(gsx[c]); ax.axis("off")
        ax.imshow(W[c], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f"{c}", fontsize=9.5, color=INK, pad=2)
    cap(row, ttl)

gsc = gs[2].subgridspec(1, 3, wspace=0.28)
lbl = [f"{e:g}" for e in E]
ax = fig.add_subplot(gsc[0])
ax.plot(x, fr, "-o", color="#7a1f1f", ms=5)
ax.axhline(1.0, color=INK2, ls=":", lw=1.0); ax.axvline(list(E).index(CH), color=INK, ls="--", lw=1.0)
for xx, yy in zip(x, fr): ax.annotate(f"{yy:.2f}", (xx, yy), fontsize=7, color=INK, textcoords="offset points", xytext=(0, 7), ha="center")
ax.set_title("the edge noise, gone\n(edge weight ÷ centre weight; 1.0 = even)", fontsize=9.5, color=INK, loc="left")
ax.set_ylabel("ratio", fontsize=8.5)

ax = fig.add_subplot(gsc[1])
ax.plot(x, at, "-o", color="#7a1f1f", ms=5, label="floor in training only")
ax.plot(x, ab, "--s", color="#c9a227", ms=4.5, label="floor in training AND reading")
ax.axvline(list(E).index(CH), color=INK, ls="--", lw=1.0)
for xx, yy in zip(x, ab): ax.annotate(f"{yy:.4f}", (xx, yy), fontsize=6.8, color=INK, textcoords="offset points", xytext=(0, -13), ha="center")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("held-out accuracy\nthe two lines sit on top of each other:\nthe reading half of the floor does nothing", fontsize=9.5, color=INK, loc="left")
ax.set_ylabel("correct", fontsize=8.5)

ax = fig.add_subplot(gsc[2])
ax.plot(x, cb, "-o", color="#7a1f1f", ms=5)
ax.axvline(list(E).index(CH), color=INK, ls="--", lw=1.0)
ax.set_ylim(0.99, 1.0)
for xx, yy in zip(x, cb): ax.annotate(f"{yy:.4f}", (xx, yy), fontsize=6.8, color=INK, textcoords="offset points", xytext=(0, 7), ha="center")
ax.set_title("how sure it claims to be\nflat at 0.998 throughout — this was never the cause", fontsize=9.5, color=INK, loc="left")
ax.set_ylabel("mean top guess", fontsize=8.5)

for q in range(3):
    a = fig.axes[-3 + q]
    a.set_xticks(x); a.set_xticklabels(lbl, fontsize=7.5)
    a.set_xlabel("floor under every pixel (black = this, not 0)", fontsize=8.5)
    a.tick_params(labelsize=7.5)
    for sp in a.spines.values(): sp.set_visible(False)
fig.text(0.04, gs[2].get_position(fig).y1 + 0.034, f"C.  The knob, swept.  Dashed line = the value kept ({CH:g}), the most we can add before it starts costing accuracy.", fontsize=10.5, color=INK, va="bottom")

fig.suptitle("A floor under every pixel — black is never truly black\n"
             f"in training it is exactly the 'imaginary ink' fix: {r['rows'][0]['imaginary_ink_per_cell']:.0f} vs "
             f"{[q['imaginary_ink_per_cell'] for q in r['rows'] if q['eps']==CH][0]:.0f} units of flat ink per grid.  "
             f"Accuracy {r['baseline_acc']:.4f} to {r['chosen_acc']:.4f};  the edge noise goes;  the over-confidence does not move",
             fontsize=12.5, color=INK, y=0.972)
fig.savefig(OUT / f"{tag}_eigengrau.png", dpi=135, facecolor=SURF)
print(f"-> {OUT}/{tag}_eigengrau.png")
