"""One board: every node's ten tables, drawn as 2-D pictures in the node's own place on the image.

    python board.py <tag>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
P, G, C = 5, 24, 10
NODES, CELLS = G * G, P * P
VMAX = 0.12                      # a cell of a flat table is 0.04; 0.12 lets the shape show, peaks clip
ZR, ZC, ZN = 8, 8, 8             # zoom block: 8x8 nodes starting at node row 8, col 8

tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
r = json.load(open(OUT / f"{tag}.json")); Z = np.load(OUT / f"{tag}.npz")
T, M, spread, ent = Z["tables"], Z["mass_raw"], Z["spread"], Z["ent"]
mass = M.sum(2)
ok = mass.min(0) > 500                                     # nodes that see real ink for every label
g = P + 1


def mosaic(Tc, r0=0, c0=0, n=G):
    Mo = np.full((n * g + 1, n * g + 1), np.nan, np.float32)
    for rr in range(n):
        for cc in range(n):
            Mo[1 + rr * g:1 + rr * g + P, 1 + cc * g:1 + cc * g + P] = Tc[(r0 + rr) * G + c0 + cc].reshape(P, P)
    return Mo


def draw_tables(fig, sub, node, title):
    gsx = sub.subgridspec(1, C, wspace=0.07)
    for c in range(C):
        ax = fig.add_subplot(gsx[c])
        t = T[c, node].reshape(P, P)
        ax.imshow(t, cmap="Reds", vmin=0, vmax=VMAX, interpolation="nearest")
        for a in range(P):
            for b in range(P):
                ax.text(b, a, f"{t[a, b]*100:.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if t[a, b] > VMAX * 0.62 else INK2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"C = {c}", fontsize=8.5, color=INK, pad=2)
        if c == 0:
            ax.set_ylabel(title, fontsize=8, color=INK2)


fig = plt.figure(figsize=(15.5, 23.5), facecolor=SURF)
gs = fig.add_gridspec(5, 1, height_ratios=[6.2, 6.2, 1.55, 1.55, 2.9], hspace=0.26,
                      left=0.035, right=0.985, top=0.942, bottom=0.015)
cap = lambda k, s: fig.text(0.035, gs[k].get_position(fig).y1 + 0.007, s, fontsize=10.5, color=INK, va="bottom")

# --- A: all 576 tables, per value of the parent ---------------------------------------------------
gsa = gs[0].subgridspec(2, 5, wspace=0.03, hspace=0.08)
for c in range(C):
    ax = fig.add_subplot(gsa[c // 5, c % 5]); ax.axis("off")
    ax.imshow(mosaic(T[c]), cmap="Reds", vmin=0, vmax=VMAX, interpolation="nearest")
    ax.set_title(f"C = {c}", fontsize=10.5, color=INK, pad=3)
cap(0, "A.  Every node's table for one value of the parent, drawn as a 5x5 picture and placed where that node sits on the image.  576 nodes per panel\n"
       "(24 x 24 windows, stride 1, no padding).  Each little picture sums to 1;  colour runs 0 to 0.12, so a flat table (0.04 a cell) is mid-pink and peaks clip.")

# --- A2: the same tables, zoomed ------------------------------------------------------------------
gsz = gs[1].subgridspec(2, 5, wspace=0.03, hspace=0.08)
for c in range(C):
    ax = fig.add_subplot(gsz[c // 5, c % 5]); ax.axis("off")
    ax.imshow(mosaic(T[c], ZR, ZC, ZN), cmap="Reds", vmin=0, vmax=VMAX, interpolation="nearest")
    ax.set_title(f"C = {c}", fontsize=10.5, color=INK, pad=3)
cap(1, f"B.  The same thing zoomed: the {ZN} x {ZN} block of nodes starting at row {ZR}, column {ZC} — the middle of the image, where the digits live.\n"
       f"Neighbouring windows overlap in 20 of their 25 pixels, so each table is its neighbour shifted by one pixel.")

# --- B: two nodes in full -------------------------------------------------------------------------
idx = np.arange(NODES)
hi = int(idx[ok][np.argmax(spread[ok])])
lo = int(idx[ok][np.argmin(spread[ok])])
draw_tables(fig, gs[2], hi, f"node {hi}\n(row {hi//G}, col {hi%G})")
cap(2, f"C.  One node's ten tables in full, as percentages of the cell.  Node {hi} — of the nodes that see real ink, the one the label moves most "
       f"(total variation {spread[hi]:.2f} from its own label-average).")
draw_tables(fig, gs[3], lo, f"node {lo}\n(row {lo//G}, col {lo%G})")
cap(3, f"     Node {lo} — the one the label moves least ({spread[lo]:.2f}).  Its ten tables are very nearly the same table, so this node says almost nothing about the parent.")

# --- C: maps, and what the tables reduce to ---------------------------------------------------------
gsc = gs[4].subgridspec(1, 4, wspace=0.22, width_ratios=[1, 1, 1, 2.1])
for q, (v, ttl, cm) in enumerate([
        (np.log10(np.maximum(mass.sum(0), 1)).reshape(G, G), "ink a node collected (log10)", "Blues"),
        (spread.reshape(G, G), "how far the label moves the table\n(total variation from the label-average)", "magma"),
        (ent.min(0).reshape(G, G), f"peakiest of a node's ten tables\n(bits; a flat table = {r['uniform_bits']})", "viridis_r")]):
    ax = fig.add_subplot(gsc[q])
    im = ax.imshow(v, cmap=cm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045)
gsd = gsc[3].subgridspec(2, 5, wspace=0.05, hspace=0.14)
for c in range(C):
    img = np.zeros((28, 28), np.float32); cnt = np.zeros((28, 28), np.float32)
    Mc = M[c].reshape(G, G, P, P)
    for rr in range(G):
        for cc in range(G):
            img[rr:rr + P, cc:cc + P] += Mc[rr, cc]; cnt[rr:rr + P, cc:cc + P] += 1
    img /= cnt
    ax = fig.add_subplot(gsd[c // 5, c % 5]); ax.axis("off")
    ax.imshow(img, cmap="Reds", vmin=0, vmax=img.max(), interpolation="nearest")
    ax.set_title(f"{c}", fontsize=8, color=INK, pad=1)
pos = gsc[3].get_position(fig)
fig.text(pos.x0, pos.y1 + 0.004, "what all 144,000 numbers reduce to: 10 images of 784 pixels",
         fontsize=9.5, color=INK, va="bottom")
cap(4, "D.  Where the ink is, where the label matters, how peaked the tables are — and the thing underneath them.")

fig.suptitle(f"Ten tables per patch node, filled by adding pixel values  —  {P}x{P} windows, stride 1, no padding, {NODES} nodes, "
             f"{C} tables of {CELLS} cells each = {r['numbers_trained']:,} numbers, {r['n_train']:,} images in {r['train_seconds']}s\n"
             f"nothing is shared between nodes, yet neighbouring nodes' tables differ by {r['neighbour_tv_on_shared']:.3f} on the 20 cells they overlap: "
             f"the 576 tables of one label are one class-ink image, cropped 576 ways", fontsize=12, color=INK, y=0.992)
fig.savefig(OUT / f"{tag}.png", dpi=135, facecolor=SURF)
print(f"-> {OUT}/{tag}.png   hi={hi} lo={lo} ok={int(ok.sum())}")
