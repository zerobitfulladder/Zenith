"""Board: does a floor let a blank patch speak, and is what it says any good?

    python board_shapefloor.py <tag>
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
r = json.load(open(OUT / f"{tag}_shapefloor.json"))
E = [q["eps"] for q in r["rows"]]; x = np.arange(len(E))
acc = [q["acc"] for q in r["rows"]]; mute = [q["acc_blank_muted"] for q in r["rows"]]
ones = [q["guessed_1_share"] for q in r["rows"]]

fig = plt.figure(figsize=(13, 5.4), facecolor=SURF)
gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.07, right=0.97, top=0.72, bottom=0.14)

ax = fig.add_subplot(gs[0])
ax.plot(x, mute, "-o", color="#7a1f1f", ms=5, label="blank patches muted")
ax.plot(x, acc, "--s", color="#c9a227", ms=5, label="blank patches allowed to vote")
for xx, a, m in zip(x, acc, mute):
    ax.annotate(f"{a:.3f}", (xx, a), fontsize=7.5, color=INK, textcoords="offset points", xytext=(0, -14), ha="center")
    ax.annotate(f"{m:.3f}", (xx, m), fontsize=7.5, color=INK, textcoords="offset points", xytext=(0, 8), ha="center")
ax.legend(fontsize=9, frameon=False, loc="center right")
ax.set_ylim(0.5, 0.9)
ax.set_title("held-out accuracy, shape read\nthe gap IS the blank patches speaking — and every word costs", fontsize=10, color=INK, loc="left")
ax.set_ylabel("correct", fontsize=9)

ax = fig.add_subplot(gs[1])
ax.bar(x, ones, color="#7a1f1f")
ax.axhline(0.11, color=INK, ls="--", lw=1.2)
ax.text(len(E) - 0.5, 0.118, "true share of 1s: 11%", fontsize=8.5, color=INK, ha="right")
for xx, o in zip(x, ones):
    ax.annotate(f"{o*100:.0f}%", (xx, o), fontsize=8, color=INK, textcoords="offset points", xytext=(0, 4), ha="center")
ax.set_title("what the blanks actually say\nthey pile onto the emptiest digit", fontsize=10, color=INK, loc="left")
ax.set_ylabel("images guessed as '1'", fontsize=9)

for a in fig.axes:
    a.set_xticks(x); a.set_xticklabels([f"{e:g}" for e in E], fontsize=8)
    a.set_xlabel("floor under every pixel", fontsize=9); a.tick_params(labelsize=8)
    for sp in a.spines.values(): sp.set_visible(False)

fig.suptitle("Does Eigengrau let a blank patch speak?  Under the shape read, yes — and what it says is wrong\n"
             "42% of every picture's patches are blank.  Floored, they all become the same flat patch, so each one votes for whichever\n"
             "digit's grid is flattest there — and they outnumber the patches that actually saw a stroke",
             fontsize=11.5, color=INK, y=0.985)
fig.savefig(OUT / f"{tag}_shapefloor.png", dpi=140, facecolor=SURF)
print(f"-> {OUT}/{tag}_shapefloor.png")
