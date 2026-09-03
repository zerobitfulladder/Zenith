"""Dead templates chase the data. 28x28, joint and split.

A template that has never won moves toward the patch it is nearest to; only the
nearest dead template moves for a given patch; it is neither counted nor read;
the moment it wins a real competition it is an ordinary template. No threshold,
no smoothing bonus. Tested on the whole-digit rig, where the champion left 135
dead templates in joint training and the summed-bias arms 77-185.

Usage:  uv run python dead_chase.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import purity_split as P
import rf_sweep as R

OUT = P.OUT
SEEDS = [7] if P.SMOKE else [7, 8]
P.EPOCHS = 2 if P.SMOKE else 20
ARMS = [("champion",       "cntn",   "belief", 0.0,  1.5, None),
        ("champion+chase", "cntn",   "belief", 0.0,  1.5, "all"),
        ("champion+err",   "cntn",   "belief", 0.0,  1.5, "err"),
        ("best",           "purity", "both",   0.25, 0.5, None),
        ("best+chase",     "purity", "both",   0.25, 0.5, "all"),
        ("best+err",       "purity", "both",   0.25, 0.5, "err"),
        ("best+uncommitted", "purity", "both", 0.25, 0.5, "uncommitted")]
SPLIT_ARMS = ["best", "best+chase", "best+err", "best+uncommitted"]
if "--err-only" in sys.argv:
    ARMS = [a for a in ARMS if a[-1] == "err"]
if "--unc-only" in sys.argv:
    ARMS = [a for a in ARMS if a[-1] == "uncommitted"]
if P.SMOKE:
    ARMS = ARMS[-1:]


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(28); R.BATCH = 512
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if P.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    joint = [(Xtr, ytr_g, "0-9")]
    split = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    res = {}
    print(f"28x28, {P.EPOCHS} epochs/phase, seeds {SEEDS}, chase eta {P.CHASE_ETA}\n")
    print(f"{'arm':<16}{'JOINT recount':>14}{'online':>8}{'dead':>6}{'still':>7} | "
          f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}{'dead':>6}")
    for tag, rule, route, bl, bb, chase in ARMS:
        P.BETA_L, P.BETA_B, P.CHASE = bl, bb, chase is not None
        P.CHASE_MODE = chase or "all"
        t0 = time.time()
        J = [P.run(rig, rule, route, joint, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
        line = (f"{tag:<16}{np.mean([c[-1]['recount'][2] for c in J]):>14.4f}"
                f"{np.mean([c[-1]['online'][2] for c in J]):>8.4f}"
                f"{np.mean([c[-1]['n_dead'] for c in J]):>6.0f}{np.mean([c[-1]['n_still'] for c in J]):>7.0f} | ")
        res[tag] = {"joint": J}
        if tag in SPLIT_ARMS:
            S = [P.run(rig, rule, route, split, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
            e = [P.end_points(c) for c in S]
            m = lambda k: np.mean([x[k] for x in e])
            line += (f"{m('B_old'):>14.4f}/{m('B_new'):.4f}{m('C_all'):>8.4f}{m('C_online'):>8.4f}"
                     f"{np.mean([c[-1]['n_dead'] for c in S]):>6.0f}")
            res[tag]["split"] = S
        print(line + f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / ("dead_chase_err.json" if "--err-only" in sys.argv else "dead_chase_unc.json" if "--unc-only" in sys.argv else "dead_chase.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
