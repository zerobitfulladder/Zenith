"""results/stack.png -- what the second floor buys."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "stack.json").read_text())
TAG = "learned floor 0"
SEEDS = list(R[TAG].keys())
KEYS = [("tri_willing_cue1", "speculates\n(cue A, adds B or C)"),
        ("tri_violation_cue1", "illegal\n(adds BOTH B and C)"),
        ("tri_violation_cue2", "illegal\n(cue A+B, adds C)"),
        ("excl_violation", "illegal\n(cue F1, adds F2)")]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                       gridspec_kw={"width_ratios": [1.15, 1]})
fig.suptitle("A second floor of the identical engine, run on floor 0's own output",
             fontsize=12.5, y=1.0)

# ---- left: one floor vs two ------------------------------------------------
a = ax[0]
x = np.arange(len(KEYS))
for i, (which, lab, col) in enumerate(
        [("one_floor", "one floor", "#b4453c"), ("two_floor", "two floors", "#2c8c6a")]):
    v = [np.mean([R[TAG][s][which][k] for s in SEEDS]) for k, _ in KEYS]
    a.bar(x + (i - 0.5) * 0.36, v, 0.34, label=lab, color=col,
          edgecolor="#333", linewidth=0.5)
    for xi, vi in zip(x, v):
        a.text(xi + (i - 0.5) * 0.36, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=8.5)
a.set_xticks(x); a.set_xticklabels([l for _, l in KEYS], fontsize=8.5)
a.set_ylim(0, 1.22); a.set_ylabel("rate over 200 probes x 5 seeds")
a.legend(fontsize=9, loc="upper right"); a.grid(axis="y", alpha=0.25)
a.set_title("floor-0 read price held at 2.5 — where one floor is mute", fontsize=10.5)
a.text(0.02, 0.86, "one floor says nothing at this price, so every\n"
                   "completion the stack makes comes from upstairs",
       transform=a.transAxes, fontsize=8.5, style="italic", color="#555")

# ---- right: what floor 1 minted -------------------------------------------
b = ax[1]
b.axis("off")
b.set_title("what floor 1 minted, unprompted", fontsize=10.5)
lines = []
for s in SEEDS:
    voc = R[TAG][s]["floor1_vocab"]
    keep = [v for v in voc if set(v.strip("{}").split(",")) <= {"A", "B", "C"}]
    rest = len(voc) - len(keep)
    lines.append(f"seed {s}:   " + "   ".join(sorted(keep))
                 + (f"    (+{rest} others)" if rest else ""))
b.text(0.0, 0.80, "\n\n".join(lines), fontsize=10, family="monospace",
       va="top", color="#222")
b.text(0.0, 0.22, "the three legal pairs, in every seed — which is exactly the\n"
                  "vocabulary that had to be hand-built in the carving test.\n"
                  "Floor 0 keeps the parts; the combinations live upstairs,\n"
                  "where the rule about them can finally be written down.",
       fontsize=9.5, va="top", color="#2c8c6a", style="italic")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(HERE / "results" / "stack.png", dpi=135)
print("wrote results/stack.png")
