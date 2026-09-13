import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tower as T
OR = "_or" in (sys.argv[1] if len(sys.argv) > 1 else "")
CAT = "_cat" in (sys.argv[1] if len(sys.argv) > 1 else "")
ABS = len(sys.argv) > 2 and sys.argv[2] == "abs"      # draw tiles as they are: one shared scale per layer, grayscale

OUT = Path(__file__).parent / "results"
INK, INK2, SURF, GOOD, BAD = "#0b0b0b", "#52514e", "#fcfcfb", "#1baf7a", "#e34948"
tag = sys.argv[1] if len(sys.argv) > 1 else ""
r = json.load(open(OUT / f"tower{tag}.json")); Z = np.load(OUT / f"tower{tag}.npz")
W1, W2, u1, u2 = Z["W1"], Z["W2"], Z["uses1"], Z["uses2"]
H1 = W1.shape[0]
P2 = Z["top_pic"]; top_a1 = Z["top_a1"]
LAB = (W2 @ Z["pat"].T) if OR else ((W2[:, H1:] @ Z["pat"].T) if CAT else W2[:, H1:])


def tiles(ax, P, order, cols, title, labels=None):
    rows = int(np.ceil(len(order) / cols)); g = 30
    grid = np.full((rows * g + 1, cols * g + 1), np.nan, np.float32)
    vmax = float(P.max()) + 1e-8
    for j, t in enumerate(order):
        rr, c = divmod(j, cols)
        w = P[t].reshape(28, 28); grid[1 + rr * g:29 + rr * g, 1 + c * g:29 + c * g] = (w / vmax) if ABS else (w / (w.max() + 1e-8))
    ax.imshow(grid, cmap=("gray_r" if ABS else "Reds"), vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
    if ABS:
        title = title + f"  [as is: one scale for the whole layer, black = {vmax:.3f}]"
    for j, t in enumerate(order):
        rr, c = divmod(j, cols)
        txt = f"{int(labels[0][t])}" if labels is not None else f"{int(LAB[t].argmax())} ({LAB[t].max() / (W2[t].sum() + 1e-8):.2f})"
        ax.text(2 + c * g, 4 + rr * g, txt, fontsize=6.5, color=INK, va="top",
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.6))
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)


fig = plt.figure(figsize=(18, 17), facecolor=SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.45, 1], hspace=0.12, wspace=0.06, left=0.02, right=0.98, top=0.94, bottom=0.02)
ax = fig.add_subplot(gs[0, 0])
tiles(ax, W1, np.argsort(-u1), 12, f"layer 1: all {H1} cells, sorted by use (number = images on for)", labels=(u1,))
ax = fig.add_subplot(gs[0, 1])
tiles(ax, P2, np.argsort(-u2), 10, f"layer 2: all {W2.shape[0]} cells, each drawn by SETTLING layer 1 under that cell's expectation alone; digit = best-matching class pattern (share of weight on it)")
# held-out reads and generations
ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
canvas = np.ones((2 * 30 + 1, 8 * 30 + 1))
for i in range(8):
    x = Z["Xte"][i].reshape(28, 28); hi = x.max() + 1e-8
    canvas[1:29, 1 + i * 30:29 + i * 30] = 1 - x / hi
    xh = Z["xhat"][i].reshape(28, 28); canvas[31:59, 1 + i * 30:29 + i * 30] = 1 - np.clip(xh / (xh.max() + 1e-8), 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
for i in range(8):
    p, t = int(Z["P8"][i]), int(Z["yte"][i])
    ax.text(15 + i * 30, 2 * 30 + 8, f"{p if p >= 0 else '-'} / {t}", ha="center", va="center", fontsize=11, fontweight="bold", color=(GOOD if p == t else BAD))
ax.set_ylim(2 * 30 + 16, -2)
fb, nf = r["modes"]["feedback"], r["modes"]["no_feedback"]
ax.set_title(f"eight held-out images: input / layer-1 rebuild / label read at the top vs true   (L1 on: " +
             " ".join(str(int((Z["A1_8"][i] > 0).sum())) for i in range(8)) + "; L2 on: " + " ".join(str(int((Z["A2_8"][i] > 0).sum())) for i in range(8)) + ")\n"
             f"label at top {fb['label_top']:.3f} (feedback) / {nf['label_top']:.3f} (no feedback), covered {fb['covered']*100:.0f}%   unexplained {fb['unexplained']:.3f}   "
             f"L1 on {fb['L1_on']:.1f} dead {fb['L1_dead']*100:.0f}%   L2 on {fb['L2_on']:.1f} dead {fb['L2_dead']*100:.0f}%\n"
             f"L2 cells: members median {r['L2']['members_median']:.0f} (mean {r['L2']['members_mean']:.1f}), wrappers {r['L2']['wrappers']*100:.0f}%, "
             f"top-1 line share {r['L2']['top1_share_median']:.2f}, label share {r['L2']['label_share_median']:.2f}, L1 cells proposed when settled {r['L2']['proposal_cells_median']:.0f}   "
             f"label line value {r['label_w']:.2f}   {r['passes']} passes" + (f" + {r['passes2']} free with the label; label entropy per top cell {r['L2'].get('label_entropy_median_bits', 0):.2f} bits" if r.get('passes2') else ""),
             fontsize=9, loc="left", color=INK)
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
gen = Z["gen"]; a2g = Z["a2g"]
canvas = np.ones((30 + 1, 10 * 30 + 1))
for c in range(10):
    g = gen[c].reshape(28, 28); canvas[1:29, 1 + c * 30:29 + c * 30] = 1 - np.clip(g / (g.max() + 1e-8), 0, 1)
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
for c in range(10):
    ax.text(15 + c * 30, 30 + 8, f"{c}  ({int((a2g[c] > 0).sum())} top, {int((Z['a1g'][c] > 0).sum())} L1)", ha="center", va="center", fontsize=8, color=INK2)
ax.set_ylim(30 + 16, -2)
ax.set_title("generation: drive the top with the label line alone, settle the top, settle layer 1 under its expectation, paint the settled cells", fontsize=9.5, loc="left", color=INK)
fig.suptitle("Two layers, same settling, same share-then-scale learning, " + ("label OR-ed in as a random sparse pattern over layer 1's lines" if OR else ("label as a random sparse pattern on 100 lines of its own" if CAT else "label concatenated at the top")), x=0.02, ha="left", fontsize=12, color=INK)
fig.savefig(OUT / f"board{tag}{'_abs' if ABS else ''}.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / f"board{tag}{'_abs' if ABS else ''}.png")
