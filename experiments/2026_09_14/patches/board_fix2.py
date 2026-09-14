"""Board for fix 2: what happens when a grid speaks only as loudly as its evidence.

    python board_fix2.py <tag>
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
r = json.load(open(OUT / f"{tag}_fix2.json")); Z = np.load(OUT / f"{tag}_fix2.npz")
Wb, Wa = Z["W_before"], Z["W_after"]
A, accs, confs, frames = Z["alphas"], Z["accs"], Z["confs"], Z["frames"]
CH = r["chosen_alpha"]

fig = plt.figure(figsize=(15, 11.6), facecolor=SURF)
gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 3.0], hspace=0.44, left=0.04, right=0.98, top=0.90, bottom=0.06)
cap = lambda k, s: fig.text(0.04, gs[k].get_position(fig).y1 + 0.012, s, fontsize=10.5, color=INK, va="bottom")

for row, (W, ttl) in enumerate([(Wb, "before: a grid built from 2 stray specks looks as certain as one built from 83,000 units of ink"),
                                (Wa, f"after: every grid starts from flat and needs real ink to move off it (imaginary ink = {CH:g} per cell)")]):
    v = float(np.percentile(np.abs(Wb), 99.5))
    gsx = gs[row].subgridspec(1, 10, wspace=0.04)
    for c in range(10):
        ax = fig.add_subplot(gsx[c]); ax.axis("off")
        ax.imshow(W[c], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f"{c}", fontsize=9.5, color=INK, pad=2)
    cap(row, ("A.  " if row == 0 else "B.  ") + ttl)

gsc = gs[2].subgridspec(1, 3, wspace=0.26)
for q, (v, ttl, yl, good) in enumerate([
        (frames, "noise around the edge\n(edge weight ÷ centre weight, 1.0 = even)", "ratio", 1.0),
        (accs, "held-out accuracy\nunchanged until the grids are gagged", "correct", None),
        (confs, "how sure it claims to be\nthe over-confidence does NOT come from this problem", "mean top guess", None)]):
    ax = fig.add_subplot(gsc[q])
    ax.semilogx(A, v, "-o", color="#7a1f1f", ms=5)
    ax.axvline(CH, color=INK, ls="--", lw=1.0)
    if good is not None:
        ax.axhline(good, color=INK2, ls=":", lw=1.0)
    for x, y in zip(A, v):
        ax.annotate(f"{y:.3f}" if q != 0 else f"{y:.2f}", (x, y), fontsize=7, color=INK,
                    textcoords="offset points", xytext=(0, 7), ha="center")
    ax.set_xlabel("imaginary ink added to every cell", fontsize=8.5)
    ax.set_ylabel(yl, fontsize=8.5); ax.tick_params(labelsize=7.5)
    ax.set_title(ttl, fontsize=9.5, color=INK, loc="left")
    for sp in ax.spines.values(): sp.set_visible(False)
cap(2, f"C.  The knob, swept.  Dashed line = the value kept ({CH:g}): the most we can add before it starts costing accuracy.")

fig.suptitle("Fix 2 — a grid should speak only as loudly as the evidence behind it\n"
             f"{r['windows_under_100_ink']} of {r['windows_total']} grids were built from under 100 units of ink;  "
             f"the edge noise goes away, accuracy moves {r['baseline_acc']:.4f} to {r['chosen_acc']:.4f}, and the over-confidence does not budge",
             fontsize=12.5, color=INK, y=0.975)
fig.savefig(OUT / f"{tag}_fix2.png", dpi=135, facecolor=SURF)
print(f"-> {OUT}/{tag}_fix2.png")
