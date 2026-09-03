"""The middle ground: 5, 9, 13, 17 pixel templates.

At 28x28 the summed bias + purity step beats the whole-digit champion by 3
points joint and 2 on the split. At 5x5 the vocabulary is generic and class
pressure is known to cost (whole_digit, §2). The user's bet: in between,
templates are specific enough for the tally to guide them, and a template that
wins a digit at a particular POSITION is itself a discriminative fact -- which
is what the per-cell table T[t, cell, y] records, and what the step rule now
reads (purity_split.purity, per-cell, usage-weighted).

Joint 0-9 at every size; the split at 9 and 13. Batch 128 (memory), 3 epochs
(per phase), 2 seeds, four arms:

    champion          cnt/n step, belief 1.5
    cntn  L0.25+B0.5  cnt/n step, summed bias
    purity L0.25+B0.5 purity step, summed bias
    purity B1.5       purity step, belief only  (the arm that must NOT be used alone)

Usage:  uv run python patch_sizes.py [--smoke]
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
SMOKE = P.SMOKE
SIZES = [9] if SMOKE else [5, 9, 13, 17]
SPLIT_SIZES = [9] if SMOKE else [9, 13]
SEEDS = [7] if SMOKE else [7, 8]
P.EPOCHS = 1 if SMOKE else 3
P.PROBE_EVERY = 10 if SMOKE else 40
ARMS = [("champion",          "cntn",   "belief", 0.0,  1.5),
        ("cntn L0.25+B0.5",   "cntn",   "both",   0.25, 0.5),
        ("purity L0.25+B0.5", "purity", "both",   0.25, 0.5),
        ("purity B1.5",       "purity", "belief", 0.0,  1.5)]
if SMOKE:
    ARMS = ARMS[::2]


def final(runs, key):
    return np.mean([r["curve"][-1][key] for r in runs], axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    joint = [(Xtr, ytr_g, "0-9")]
    split = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    res = {"joint": {}, "split": {}}
    print(f"sizes {SIZES}, split at {SPLIT_SIZES}, {P.EPOCHS} epochs, seeds {SEEDS}\n")

    print("JOINT 0-9")
    print(f"{'size':<6}{'arm':<20}{'recount':>9}{'online':>9}{'gap':>8}{'dead':>6}{'still':>7}{'purity':>8}")
    for ps in SIZES:
        R.BATCH = 512 if ps == 28 else 128
        rig = R.Rig(ps)
        for tag, rule, route, bl, bb in ARMS:
            P.BETA_L, P.BETA_B = bl, bb
            t0 = time.time()
            runs = [{"seed": s, "curve": P.run(rig, rule, route, joint, Xtr, ytr_g, Xte, yte_g, s)[0]}
                    for s in SEEDS]
            res["joint"][f"{ps}|{tag}"] = {"ps": ps, "arm": tag, "runs": runs}
            rec, onl = final(runs, "recount")[2], final(runs, "online")[2]
            print(f"{ps:<6}{tag:<20}{rec:>9.4f}{onl:>9.4f}{rec-onl:>8.4f}{final(runs,'n_dead'):>6.0f}"
                  f"{final(runs,'n_still'):>7.0f}{final(runs,'purity_live'):>8.3f}   ({time.time()-t0:.0f}s)",
                  flush=True)
        (OUT / "patch_sizes.json").write_text(json.dumps(res, indent=1))

    print("\nSPLIT 0-4 -> 5-9 -> 0-9")
    print(f"{'size':<6}{'arm':<20}{'after 5-9: old/new':>20}{'final all':>11}{'online':>9}"
          f"{'#com':>6}{'drift':>7}{'conv':>6}{'dead':>6}")
    for ps in SPLIT_SIZES:
        R.BATCH = 128
        rig = R.Rig(ps)
        for tag, rule, route, bl, bb in ARMS:
            P.BETA_L, P.BETA_B = bl, bb
            t0 = time.time()
            runs = [{"seed": s, "curve": P.run(rig, rule, route, split, Xtr, ytr_g, Xte, yte_g, s)[0]}
                    for s in SEEDS]
            res["split"][f"{ps}|{tag}"] = {"ps": ps, "arm": tag, "runs": runs}
            e = [P.end_points(r["curve"]) for r in runs]
            m = lambda k: np.mean([x[k] for x in e])
            endB = [[p for p in r["curve"] if p["phase"] == 1][-1] for r in runs]
            conv = np.mean([p.get("converted", np.nan) for p in endB])
            print(f"{ps:<6}{tag:<20}{m('B_old'):>10.4f}/{m('B_new'):.4f}{m('C_all'):>11.4f}"
                  f"{m('C_online'):>9.4f}{m('n_com'):>6.0f}{m('drift_B'):>7.3f}{conv:>6.2f}"
                  f"{final(runs,'n_dead'):>6.0f}   ({time.time()-t0:.0f}s)", flush=True)
        (OUT / "patch_sizes.json").write_text(json.dumps(res, indent=1))
    draw(res)


def draw(res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cols = {"champion": "k", "cntn L0.25+B0.5": "#e07b00", "purity L0.25+B0.5": "#c1121f",
            "purity B1.5": "#1b6ca8"}
    for tag, _, _, _, _ in ARMS:
        J = [res["joint"][f"{ps}|{tag}"]["runs"] for ps in SIZES]
        axes[0].plot(SIZES, [final(r, "recount")[2] for r in J], "-o", color=cols[tag], label=tag)
        axes[1].plot(SIZES, [final(r, "recount")[2] - final(r, "online")[2] for r in J], "-o",
                     color=cols[tag], label=tag)
        S = [res["split"].get(f"{ps}|{tag}") for ps in SPLIT_SIZES]
        if all(S):
            axes[2].plot(SPLIT_SIZES, [np.mean([P.end_points(r["curve"])["B_old"] for r in s["runs"]])
                                       for s in S], "-o", color=cols[tag], label=tag)
    for ax, t in zip(axes, ("joint 0-9, recount", "joint: recount - online (staleness)",
                            "split: old classes after the 5-9 phase")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("patch side (pixels)"); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7)
    axes[0].set_xticks(SIZES); axes[1].set_xticks(SIZES); axes[2].set_xticks(SPLIT_SIZES)
    fig.suptitle(f"Does the tally-guided rig pay at intermediate receptive fields?  "
                 f"{P.EPOCHS} epochs, {len(SEEDS)} seeds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "patch_sizes.png", dpi=140); plt.close(fig)
    print(f"\nwrote {OUT / 'patch_sizes.png'}")


if __name__ == "__main__":
    main()
