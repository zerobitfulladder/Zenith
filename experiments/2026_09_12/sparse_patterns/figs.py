"""One board, from results/results.json."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "results.json").read_text())
SEEDS = list(R["seeds"].keys())
ARMS = ["bits", "patterns", "patterns+stats"]
CL = {"bits": "#b4453c", "patterns": "#3d6fb4", "patterns+stats": "#2c8c6a"}


def mean(arm, k):
    return float(np.mean([R["seeds"][s][arm][k] for s in SEEDS]))


def vmean(arm, k):
    v = [R["seeds"][s][arm]["vocab"][k] for s in SEEDS]
    return None if v[0] is None else float(np.mean(v))


fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("A sparse vector that autocompletes itself: bits, patterns, and where you carve",
             fontsize=13, y=0.98)

# ---- (a) vocabulary --------------------------------------------------------
a = ax[0, 0]
arms = ["patterns", "patterns+stats"]
x = np.arange(len(arms))
a.bar(x - 0.18, [vmean(m, "best_f1") for m in arms], 0.34,
      label="represented alone (F1 to an atom)", color="#3d6fb4")
a.bar(x + 0.18, [vmean(m, "coverage") for m in arms], 0.34,
      label="represented at all (coverage)", color="#9dbbdd")
for i, m in enumerate(arms):
    a.text(i, 1.0, f"{vmean(m,'n_patterns'):.0f} patterns, "
                   f"{vmean(m,'recovered'):.0f}/11 atoms alone",
           ha="center", fontsize=9, color="#333")
a.set_xticks(x); a.set_xticklabels(arms); a.set_ylim(0, 1.18)
a.set_title("1. vocabulary — 11 hidden atoms recovered from scratch", fontsize=10.5)
a.legend(fontsize=8.5, loc="upper right"); a.grid(axis="y", alpha=0.25)

# ---- (b) completion --------------------------------------------------------
b = ax[0, 1]
x = np.arange(len(ARMS))
b.bar(x - 0.18, [mean(m, "comp_recall") for m in ARMS], 0.34, label="recall",
      color=[CL[m] for m in ARMS])
b.bar(x + 0.18, [mean(m, "comp_precision") for m in ARMS], 0.34, label="precision",
      color=[CL[m] for m in ARMS], alpha=0.45)
b.set_xticks(x); b.set_xticklabels(ARMS, fontsize=9); b.set_ylim(0, 1.0)
b.set_title("2. completion — half the on-bits given, the rest asked for", fontsize=10.5)
b.legend(fontsize=8.5); b.grid(axis="y", alpha=0.25)
b.text(0.5, 0.93, "bit-counting recalls more and is wrong more often",
       transform=b.transAxes, ha="center", fontsize=8.5, style="italic", color="#555")

# ---- (c) the bit engine has no working price -------------------------------
c = ax[1, 0]
sw = R["bit_price_sweep"]
pr = sorted(float(k) for k in sw)
c.plot(pr, [sw[str(p)]["comp_recall"] for p in pr], "o-", color="#b4453c",
       label="completion recall")
c.plot(pr, [sw[str(p)]["tri_violation_cue2"] for p in pr], "s--", color="#333",
       label="produces the illegal triple")
c.axhline(0, color="#999", lw=0.6)
c.set_xlabel("price a bit must beat to be added"); c.set_ylim(-0.05, 1.08)
c.set_title("3. bit-counting: no price both completes and refuses", fontsize=10.5)
c.legend(fontsize=8.5); c.grid(alpha=0.25)
c.annotate("refusal only arrives\nwhen recall is already 0",
           xy=(13, 0.0), xytext=(6.5, 0.42), fontsize=8.5, color="#444",
           arrowprops=dict(arrowstyle="->", color="#888"))

# ---- (d) the carving -------------------------------------------------------
d = ax[1, 1]
carve = R["carving"]
cs = list(carve.keys())
prices = [3.0, 1.0, 0.0, -2.0]
keys = ["atoms", "atoms+stats", "unions", "unions+stats"]
cols = {"atoms": "#d38b84", "atoms+stats": "#b4453c",
        "unions": "#8fc4b1", "unions+stats": "#2c8c6a"}
xg = np.arange(len(prices))
for i, key in enumerate(keys):
    v = [np.mean([carve[s][key][str(p)]["tri_violation_cue1"] for s in cs]) for p in prices]
    w = [np.mean([carve[s][key][str(p)]["tri_willing_cue1"] for s in cs]) for p in prices]
    off = (i - 1.5) * 0.2
    bars = d.bar(xg + off, v, 0.19, color=cols[key], label=key,
                 edgecolor="#444", linewidth=0.5)
    for bar, ww, vv in zip(bars, w, v):
        if ww == 0:                       # refused everything: vacuous refusal
            bar.set_hatch("////")
            d.text(bar.get_x() + bar.get_width() / 2, 0.03, "mute",
                   ha="center", fontsize=6.5, rotation=90, color="#555")
        elif vv == 0:
            d.text(bar.get_x() + bar.get_width() / 2, 0.03, "clean",
                   ha="center", fontsize=6.5, rotation=90, color="#2c8c6a",
                   fontweight="bold")
d.set_xticks(xg); d.set_xticklabels([f"{p:g}" for p in prices])
d.set_xlabel("read price  (lower = more willing to speculate)")
d.set_ylabel("rate of producing the illegal triple")
d.set_ylim(0, 1.32)
d.set_title("4. the carving decides — same engine, vocabulary handed to it", fontsize=10.5)
d.legend(fontsize=8, ncol=2, loc="upper left"); d.grid(axis="y", alpha=0.25)
d.text(0.5, 0.80, "hatched = refuses everything, so refusing the triple means nothing.\n"
                  "only `unions+stats` speculates at every price and is never illegal.",
       transform=d.transAxes, ha="center", fontsize=8.5, style="italic", color="#333")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE / "results" / "board.png", dpi=135)
print("wrote results/board.png")
