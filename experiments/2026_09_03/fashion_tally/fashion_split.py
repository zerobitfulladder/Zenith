"""The tally-guided rig on Fashion-MNIST: does it stop forgetting there?

On MNIST the purity step + summed bias (label 0.25 + belief 0.5) beat the
whole-digit champion by 2 points on the split and 3 joint, and at 5-17 px it
changed nothing -- because MNIST strokes are generic, drift is free, and nobody
forgets on patches. Fashion features are less generic (a sleeve edge is not a
boot edge), so patches may have something to forget here. Same rigs, same
protocol, Fashion classes 0-4 (tshirt trouser pullover dress coat) then 5-9
(sandal shirt sneaker bag boot) then all ten.

Arms:
    champion            cnt/n step, belief 1.5        the MNIST champion's rule
    cntn none           cnt/n step, geometry only     no class pressure at all
    cntn L0.25+B0.5     cnt/n step, summed bias       the competition change alone
    purity L0.25+B0.5   purity step, summed bias      the full proposal

Sizes: 9x9 (batch 128, 3 epochs/phase) and 28x28 (batch 512, 20 epochs/phase),
split and joint, 2 seeds.

Usage:  uv run python fashion_split.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "purity_plasticity"))
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import purity_split as P
import rf_sweep as R

OUT = HERE / "results"
SMOKE = P.SMOKE
SIZES = [9] if SMOKE else [9, 28]
SEEDS = [7] if SMOKE else [7, 8]
EPOCHS = {9: 3, 13: 3, 28: 20}
ARMS = [("champion",          "cntn",   "belief", 0.0,  1.5),
        ("cntn none",         "cntn",   "none",   0.0,  0.0),
        ("cntn L0.25+B0.5",   "cntn",   "both",   0.25, 0.5),
        ("purity L0.25+B0.5", "purity", "both",   0.25, 0.5)]
if SMOKE:
    ARMS = ARMS[::3]
NAMES = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]


def final(runs, key):
    return np.mean([r["curve"][-1][key] for r in runs], axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = R.E.load("fashion_mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    joint = [(Xtr, ytr_g, "0-9")]
    split = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    print(f"Fashion-MNIST, {len(ytr)} train / {len(yte)} test, sizes {SIZES}, seeds {SEEDS}")
    print(f"phase A = {NAMES[:5]}, phase B = {NAMES[5:]}\n")
    res = {"joint": {}, "split": {}}
    for ps in SIZES:
        R.BATCH = 512 if ps == 28 else 128
        P.EPOCHS = 1 if SMOKE else EPOCHS[ps]
        P.PROBE_EVERY = 10 if ps == 28 else 40
        rig = R.Rig(ps)
        print(f"== {ps}x{ps}, {rig.npos} positions, batch {R.BATCH}, {P.EPOCHS} epochs/phase")
        print(f"{'arm':<20}{'JOINT recount':>14}{'online':>8}{'gap':>7}{'dead':>6} | "
              f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}{'#com':>6}{'drift':>7}{'conv':>6}{'dead':>6}")
        for tag, rule, route, bl, bb in ARMS:
            P.BETA_L, P.BETA_B = bl, bb
            t0 = time.time()
            jr = [{"seed": s, "curve": P.run(rig, rule, route, joint, Xtr, ytr_g, Xte, yte_g, s)[0]}
                  for s in SEEDS]
            sr = [{"seed": s, "curve": P.run(rig, rule, route, split, Xtr, ytr_g, Xte, yte_g, s)[0]}
                  for s in SEEDS]
            res["joint"][f"{ps}|{tag}"] = {"ps": ps, "arm": tag, "runs": jr}
            res["split"][f"{ps}|{tag}"] = {"ps": ps, "arm": tag, "runs": sr}
            jrec, jonl = final(jr, "recount")[2], final(jr, "online")[2]
            e = [P.end_points(r["curve"]) for r in sr]
            m = lambda k: np.mean([x[k] for x in e])
            endB = [[p for p in r["curve"] if p["phase"] == 1][-1] for r in sr]
            conv = np.mean([p.get("converted", np.nan) for p in endB])
            print(f"{tag:<20}{jrec:>14.4f}{jonl:>8.4f}{jrec-jonl:>7.4f}{final(jr,'n_dead'):>6.0f} | "
                  f"{m('B_old'):>14.4f}/{m('B_new'):.4f}{m('C_all'):>8.4f}{m('C_online'):>8.4f}"
                  f"{m('n_com'):>6.0f}{m('drift_B'):>7.3f}{conv:>6.2f}{final(sr,'n_dead'):>6.0f}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "fashion.json").write_text(json.dumps(res, indent=1))
        print()
    draw(res)


COLS = {"champion": "k", "cntn none": "#777777", "cntn L0.25+B0.5": "#e07b00",
        "purity L0.25+B0.5": "#c1121f"}


def draw(res):
    fig, axes = plt.subplots(len(SIZES), 3, figsize=(15, 4 * len(SIZES)), squeeze=False)
    for i, ps in enumerate(SIZES):
        panels = [("old classes 0-4, recount", lambda p: p["recount"][0]),
                  ("new classes 5-9, recount", lambda p: p["recount"][1]),
                  ("all ten, online table", lambda p: p["online"][2])]
        first = next(v for k, v in res["split"].items() if v["ps"] == ps)
        bounds = first["runs"][0].get("bounds") or [0]
        for ax, (ttl, get) in zip(axes[i], panels):
            for k, v in res["split"].items():
                if v["ps"] != ps:
                    continue
                x, yv = P.mean_curve(v["runs"], get)
                ax.plot(x, yv, "-", color=COLS[v["arm"]], lw=2.2 if v["arm"] == "champion" else 1.6,
                        label=v["arm"])
            ph = [p["phase"] for p in first["runs"][0]["curve"]]
            st = [p["step"] for p in first["runs"][0]["curve"]]
            for b in (1, 2):
                if b in ph:
                    ax.axvline(st[ph.index(b)], color="k", lw=0.8, ls=":")
            ax.set_title(f"{ps}x{ps}: {ttl}", fontsize=10); ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[i][0].legend(fontsize=7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("training batches")
    fig.suptitle(f"Fashion-MNIST split 0-4 -> 5-9 -> 0-9, {len(SEEDS)} seeds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fashion_split.png", dpi=140); plt.close(fig)
    print(f"wrote {OUT / 'fashion_split.png'}")


if __name__ == "__main__":
    main()
