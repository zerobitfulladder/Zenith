"""Board for the brute-force read: per-patch match scores, and what summing them does.

    python board_read.py <tag>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
G, C = 24, 10

tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
r = json.load(open(OUT / f"{tag}_read.json")); Z = np.load(OUT / f"{tag}_read.npz")
M = [Z["maps1"], Z["maps2"], Z["maps3"]]                     # each (nshow,576,10)
TOT = [Z["tot1"], Z["tot2"], Z["tot3"]]
Xs, ys, ps, W = Z["Xshow"], Z["yshow"], Z["post_show"], Z["W"]
gap1, yte, pred1 = Z["gap1"], Z["yte"], Z["pred1"]
NS = len(Xs)
SNAMES = ["ink   x · logT", "shape   (x/ink) · logT", "corr   patch vs table"]


def pref(s):                                                  # (576,10) -> (10,24,24) centred per node
    return (s - s.mean(1, keepdims=True)).T.reshape(C, G, G)


def show_maps(fig, sub, k, which, ttl):
    s = M[which][k]
    p = pref(s)
    v = float(np.percentile(np.abs(p), 99.5))
    gsx = sub.subgridspec(1, 12, wspace=0.06, width_ratios=[1.25, 1.25] + [1] * 10)
    ax = fig.add_subplot(gsx[0]); ax.axis("off")
    ax.imshow(Xs[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"true {ys[k]}", fontsize=8.5, color=INK, pad=2)
    ax = fig.add_subplot(gsx[1])
    t = TOT[which][k]; top = np.argsort(-t)[:2]
    ax.bar(range(C), ps[k] if which == 0 else (t - t.min()) / (t.max() - t.min() + 1e-9),
           color=["#7a1f1f" if c in top else "#cfcbc4" for c in range(C)])
    ax.set_xticks(range(C)); ax.set_xticklabels(range(C), fontsize=4.5); ax.set_yticks([])
    ax.set_title("posterior" if which == 0 else "score", fontsize=7.5, color=INK2, pad=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    for c in range(C):
        ax = fig.add_subplot(gsx[2 + c]); ax.axis("off")
        ax.imshow(p[c], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f"{c}", fontsize=8.5, color="#7a1f1f" if c == top[0] else INK, pad=2)
    fig.text(0.037, sub.get_position(fig).y1 + 0.004, ttl, fontsize=9.5, color=INK, va="bottom")


fig = plt.figure(figsize=(15.5, 20.5), facecolor=SURF)
gs = fig.add_gridspec(6, 1, height_ratios=[2.6, 1.45, 1.45, 1.45, 1.45, 2.5], hspace=0.48,
                      left=0.035, right=0.985, top=0.912, bottom=0.02)
cap = lambda k, s: fig.text(0.035, gs[k].get_position(fig).y1 + 0.011, s, fontsize=10.5, color=INK, va="bottom")

# --- A: what read 1 collapses to -------------------------------------------------------------------
gsa = gs[0].subgridspec(2, 5, wspace=0.04, hspace=0.12)
Wd = W - W.mean(0, keepdims=True)
v = float(np.percentile(np.abs(Wd), 99.5))
for c in range(C):
    ax = fig.add_subplot(gsa[c // 5, c % 5]); ax.axis("off")
    ax.imshow(Wd[c], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(f"C = {c}", fontsize=10, color=INK, pad=3)
cap(0, f"A.  Read 1 is a linear classifier — measured, not argued.  Summing the 576 per-patch log-likelihoods is identical to multiplying the raw pixels by one\n"
       f"weight image per label (drawn here; red = votes for, blue = against).  Biggest disagreement over 10,000 images: {r['collapse_max_abs_diff']:.3f} out of scores averaging\n"
       f"~8,400, which is float rounding; same prediction on {r['collapse_same_prediction']*100:.0f}% of them.  The loud frame is the border tables: no ink behind them, full confidence.")

# --- B..D: the per-patch score maps -----------------------------------------------------------------
show_maps(fig, gs[1], 0, 0, "B.  Where each label's evidence comes from.  Each map is the 576 per-patch scores for one value of C, minus that patch's average over the ten "
                            "values —\n     red = this patch prefers this label, blue = argues against it.  Score 1 (ink).  A confidently-read image:")
show_maps(fig, gs[2], 1, 0, "C.  The most torn held-out image, score 1 (ink):")
show_maps(fig, gs[3], 2, 0, "D.  The second most torn, score 1 (ink):")
show_maps(fig, gs[4], 1, 1, "E.  The same torn image under score 2 (shape) — every patch weighted equally however much ink it holds:")

# --- F: scoreboard ------------------------------------------------------------------------------------
gsf = gs[5].subgridspec(1, 3, wspace=0.26, width_ratios=[1.1, 1.5, 1.4])
ax = fig.add_subplot(gsf[0])
accs = list(r["reads"].values())
ax.barh(range(3), accs, color=["#7a1f1f", "#a85a3c", "#c9a227"])
for i, (a, nm) in enumerate(zip(accs, SNAMES)):
    ax.text(0.015, i, nm, va="center", fontsize=8.5, color="white")
    ax.text(a + 0.008, i, f"{a:.4f}", va="center", fontsize=9, color=INK)
ax.set_yticks([]); ax.set_xlim(0, 1.06); ax.invert_yaxis(); ax.tick_params(labelsize=8)
ax.axvline(0.9251, color=INK, ls="--", lw=1.0)
ax.text(0.918, 2.42, "3x3 whole-pattern tables 0.9251", fontsize=7.5, color=INK, ha="right", va="center")
ax.set_title("held-out read of C, 10,000 images", fontsize=9.5, color=INK, loc="left")
for sp in ax.spines.values(): sp.set_visible(False)

ax = fig.add_subplot(gsf[1])
ks = list(r["acc_by_gap_ink"].keys()); vs = [r["acc_by_gap_ink"][k][0] for k in ks]; ns = [r["acc_by_gap_ink"][k][1] for k in ks]
ax.bar(ks, vs, color="#7a1f1f")
for i, (a, n) in enumerate(zip(vs, ns)):
    ax.text(i, a + 0.02, f"{a:.2f}\nn={n}", ha="center", fontsize=7, color=INK)
ax.set_ylim(0, 1.15); ax.tick_params(labelsize=7.5)
ax.set_title("read 1: accuracy by the gap between the best two values of C (nats)\nthe gap no longer tells you much — it is flat until it is huge",
             fontsize=9.5, color=INK, loc="left")
for sp in ax.spines.values(): sp.set_visible(False)

ax = fig.add_subplot(gsf[2])
ax.hist(np.clip(gap1, 0, 600), bins=60, color="#7a1f1f")
ax.axvline(5, color=INK, ls="--", lw=1.0)
ax.set_title(f"read 1: the gap, in nats.  median {r['ink_median_gap_nats']:.0f}\n"
             f"mean top-1 posterior {r['ink_mean_conf']:.4f}, {r['ink_frac_conf_over_99']*100:.0f}% over 0.99",
             fontsize=9.5, color=INK, loc="left")
ax.set_xlabel("nats (clipped at 600)", fontsize=8); ax.tick_params(labelsize=7)
for sp in ax.spines.values(): sp.set_visible(False)
fig.text(0.035, gs[5].get_position(fig).y1 + 0.022, "F.  Summing works — and it lands below the 3x3 whole-pattern tables, and far more over-confident.", fontsize=10.5, color=INK, va="bottom")

fig.suptitle("Brute-force read: every patch scores every value of the parent, the scores are summed, the largest wins\n"
             "576 nodes x 10 values x 3 ways of scoring a patch, 10,000 held-out images, 3.5 s",
             fontsize=12.5, color=INK, y=0.992)
fig.savefig(OUT / f"{tag}_read.png", dpi=135, facecolor=SURF)
print(f"-> {OUT}/{tag}_read.png")
