"""THE board. One file, one picture: results/20_board.png.

Every experiment in this folder writes its numbers to a json; this reads all of
them and redraws the single board. Nothing else in this folder draws accuracy.
Run it after any new run.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
BASE, LABEL, DRAW, BOTH = "#1b6ca8", "#2f8f4e", "#6a3d9a", "#d98014"
FREE = "#0f8b8d"          # needs no teacher signal at all
STACK = "#b5651d"        # two rungs
rows = []


def add(name, acc, color):
    rows.append((name, float(acc), color))


j = lambda f: json.loads((OUT / f).read_text())

for k, v in j("baselines_fashion_mnist.json")["results"].items():
    add(f"baseline: {k}", v["acc"], BASE)

l1 = j("l1_metrics.json")["results"]
for arm in ("competitive", "conscience", "control"):
    add(f"label-picked {arm}: fit gate", l1[arm]["gate_fit"], LABEL)
add("label-picked control: conf gate", l1["control"]["gate_confidence"], LABEL)

dp = j("dopamine_metrics.json")["results"]
for k, lab in (("A", "label-picked + reluctance"),
               ("B", "label-picked + repulsion"),
               ("A+B", "label-picked + reluctance + repulsion")):
    add(lab, dp[k]["acc"], LABEL)

if (OUT / "draw_metrics.json").exists():
    dr = j("draw_metrics.json")["results"]
    for k, lab in (("draw", "draw-picked"),
                   ("draw+A", "draw-picked + reluctance"),
                   ("draw+B", "draw-picked + repulsion"),
                   ("draw+A+B", "draw-picked + reluctance + repulsion"),
                   ("draw+C", "draw-picked + conscience"),
                   ("draw+C+A", "draw-picked + conscience + reluctance"),
                   ("draw+C+B", "draw-picked + conscience + repulsion"),
                   ("draw+C+A+B", "draw-picked + conscience + reluctance + repulsion")):
        if k in dr:
            add(lab, dr[k]["acc"], DRAW)

if (OUT / "both_metrics.json").exists():
    bm = j("both_metrics.json")
    best = max(bm["lams"], key=lambda l: bm["results"][f"{l}+A"]["acc"])
    add(f"graded on BOTH (λ={best:g})", bm["results"][f"{best}"]["acc"], BOTH)
    add(f"graded on BOTH (λ={best:g}) + reluctance",
        bm["results"][f"{best}+A"]["acc"], BOTH)

if (OUT / "selfcal.json").exists():
    sc = j("selfcal.json")
    NAME = {"graded λ=4, no punishment": "graded on BOTH, no punishment",
            "graded λ=4 + sleep": "graded on BOTH + sleep"}
    for k, lab in NAME.items():
        if k not in sc:
            continue
        add(lab, sc[k]["raw"], FREE)
        add(lab + " + habitual calibration", sc[k]["minus habitual (all inputs)"], FREE)
        if "teacher-trained reluctance" in sc[k]:
            add(lab + " + reluctance (teacher)", sc[k]["teacher-trained reluctance"], BOTH)

cv = OUT.parent.parent / "conv" / "results" / "fashion_stack.json"
if cv.exists():
    st = json.loads(cv.read_text())["results"]
    add("TWO RUNGS: L1 patches -> L2 (teacher-free)", st["stack (L1 -> L2)"]["acc"], STACK)
    add("same L2 unit on raw pixels (control)",
        st["same unit on raw pixels"]["acc"], STACK)

rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9.6, 0.32 * len(rows) + 2.4))
ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
for i, r in enumerate(rows):
    ax.text(r[1] + .006, i, f"{r[1]:.4f}", va="center", fontsize=7.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.axvline(0.1, color="k", lw=.8, ls=":")
ax.text(0.105, -0.65, "chance", fontsize=7)
ax.set_xlim(0, 1.02); ax.grid(alpha=.25, axis="x")
ax.set_xlabel("test accuracy — Fashion-MNIST, 20000 train / 5000 test", fontsize=9)
ax.set_title("the board", fontsize=12)
fig.tight_layout()
fig.subplots_adjust(bottom=0.055 + 1.45 / fig.get_figheight())
lines = [
    (BASE, "blue — off-the-shelf classifier"),
    (LABEL, "green — training winner = the expert most confident in the TRUE LABEL"),
    (DRAW, "purple — training winner = the expert that DRAWS the image best"),
    (BOTH, "orange — training winner graded on BOTH: image error + λ · (1 − confidence in the true label)"),
    (FREE, "teal — NO teacher signal anywhere: the label is treated as a second sensory stream, never as a verdict"),
    (STACK, "brown — two rungs: 5x5 patch unit, identity-only message pooled to 6x6, then one unit over the map"),
    ("#444", "reluctance: a learned per-expert offset at the gate — read time, content frozen"),
    ("#444", "repulsion: the wrong gate winner is rotated AWAY from that image — training"),
    ("#444", "conscience: winning a lot makes winning next less likely — anti-starvation"),
]
for i, (c, t) in enumerate(lines):
    fig.text(0.035, 0.045 + (len(lines) - 1 - i) * 0.185 / fig.get_figheight(),
             t, fontsize=7.2, color=c)
fig.savefig(OUT / "20_board.png", dpi=140); plt.close(fig)
print(f"{len(rows)} entries -> {OUT}/20_board.png")
for n, a, _ in rows[::-1]:
    print(f"  {a:.4f}  {n}")
