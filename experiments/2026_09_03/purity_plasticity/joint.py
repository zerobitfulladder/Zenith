"""Does it beat the champion when there is nothing to forget?

All ten classes at once, no phases, the same number of updates as the split
(40 epochs on 12k images at batch 512 = 960 batches). Both step rules crossed
with the champion's belief bias and the summed bias from `combined.py`, so this
also separates the step rule's contribution from the competition's.

Usage:  uv run python joint.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import purity_split as P
import rf_sweep as R

OUT = P.OUT
P.EPOCHS = 2 if P.SMOKE else 40
#        tag                  rule      route     beta_L  beta_B
ARMS = [("cntn  B1.5",       "cntn",   "belief",  0.0,   1.5),      # the champion
        ("purity B1.5",      "purity", "belief",  0.0,   1.5),
        ("cntn  L0.25+B0.5", "cntn",   "both",    0.25,  0.5),
        ("purity L0.25+B0.5","purity", "both",    0.25,  0.5),
        ("cntn  L0.25+B1.5", "cntn",   "both",    0.25,  1.5),
        ("purity L0.25+B1.5","purity", "both",    0.25,  1.5),
        ("purity L0.25",     "purity", "label",   0.25,  0.0),
        ("purity none",      "purity", "none",    0.0,   0.0),
        ("cntn  none",       "cntn",   "none",    0.0,   0.0)]
if P.SMOKE:
    ARMS = ARMS[:1] + ARMS[3:4]


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(P.PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if P.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    phases = [(Xtr, ytr_g, "0-9")]
    print(f"joint 0-9, {P.EPOCHS} epochs, batch {P.BATCH if hasattr(P,'BATCH') else R.BATCH}, "
          f"seeds {P.SEEDS}\n")
    print(f"{'':<20}{'recount':>9}{'(seeds)':>22}{'online':>9}{'gap':>7}{'dead':>6}{'still':>7}"
          f"{'purity':>8}")
    res = {}
    for tag, rule, route, bl, bb in ARMS:
        P.BETA_L, P.BETA_B = bl, bb
        t0 = time.time()
        runs = []
        for seed in P.SEEDS:
            c, b, _ = P.run(rig, rule, route, phases, Xtr, ytr_g, Xte, yte_g, seed)
            runs.append({"seed": seed, "curve": c})
        res[tag] = {"rule": rule, "route": route, "beta_L": bl, "beta_B": bb, "runs": runs}
        fin = [r["curve"][-1] for r in runs]
        rec = np.array([f["recount"][2] for f in fin]); onl = np.mean([f["online"][2] for f in fin])
        print(f"  {tag:<18}{rec.mean():>9.4f}"
              + "  (" + "/".join(f"{x:.3f}" for x in rec) + ")"
              + f"{onl:>9.4f}{rec.mean() - onl:>7.4f}"
              f"{np.mean([f['n_dead'] for f in fin]):>6.0f}{np.mean([f['n_still'] for f in fin]):>7.0f}"
              f"{np.mean([f['purity_live'] for f in fin]):>8.3f}   ({time.time() - t0:.0f}s)",
              flush=True)
    (OUT / "joint.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap("tab10")
    for i, (tag, r) in enumerate(res.items()):
        ls = ":" if r["rule"] == "cntn" else "-"
        col = cmap(i // 2) if i < 6 else cmap(3 + i - 6)
        for ax, get in zip(axes, (lambda p: p["recount"][2], lambda p: p["online"][2],
                                  lambda p: p["purity_live"])):
            x, yv = P.mean_curve(r["runs"], get)
            ax.plot(x, yv, ls, color=col, lw=2.2 if i == 0 else 1.6, label=tag)
    for ax, t in zip(axes, ("recount accuracy", "online table", "mean row purity of live templates")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("training batches"); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.8, 0.95); axes[1].set_ylim(0.3, 0.95)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Joint 0-9, no phases, {P.EPOCHS} epochs, {len(P.SEEDS)} seeds.  "
                 f"Dotted = champion's cnt/n step, solid = purity step", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "joint.png", dpi=140); plt.close(fig)
    print(f"\nwrote {OUT / 'joint.png'}")


if __name__ == "__main__":
    main()
