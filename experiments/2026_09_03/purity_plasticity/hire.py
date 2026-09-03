"""Hire on error, best effort. 28x28, joint and split.

When the tally's own prediction for an image is wrong AND the winner is
committed (row purity >= 0.5), the image is handed to the cheapest template:
the one whose row contributes least information to the read (dead = 0). Its
row is wiped so it learns the image at full step; one hire per class per batch;
hiring stops by itself once the class is read correctly. Falls back to
sacrificing a live template when none is free.

Usage:  uv run python hire.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import purity_split as P
import rf_sweep as R

OUT = P.OUT
SEEDS = [7] if P.SMOKE else [7, 8]
P.EPOCHS = 2 if P.SMOKE else 20
ARMS = [("best",           "purity", "both",   0.25, 0.5, False, True),
        ("best+hire",      "purity", "both",   0.25, 0.5, True,  True),
        ("champion",       "cntn",   "belief", 0.0,  1.5, False, False),
        ("champion+hire",  "cntn",   "belief", 0.0,  1.5, True,  False)]
if P.SMOKE:
    ARMS = ARMS[1:2]


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
    print(f"28x28, {P.EPOCHS} epochs/phase, seeds {SEEDS}\n")
    print(f"{'arm':<15}{'JOINT recount':>14}{'online':>8}{'dead':>6} | "
          f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}{'dead':>6}")
    for tag, rule, route, bl, bb, hire, do_split in ARMS:
        P.BETA_L, P.BETA_B, P.HIRE, P.CHASE = bl, bb, hire, False
        t0 = time.time()
        J = [P.run(rig, rule, route, joint, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
        line = (f"{tag:<15}{np.mean([c[-1]['recount'][2] for c in J]):>14.4f}"
                f"{np.mean([c[-1]['online'][2] for c in J]):>8.4f}"
                f"{np.mean([c[-1]['n_dead'] for c in J]):>6.0f} | ")
        res[tag] = {"joint": J}
        if do_split:
            S = [P.run(rig, rule, route, split, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
            e = [P.end_points(c) for c in S]
            m = lambda kk: np.mean([x[kk] for x in e])
            line += (f"{m('B_old'):>14.4f}/{m('B_new'):.4f}{m('C_all'):>8.4f}{m('C_online'):>8.4f}"
                     f"{np.mean([c[-1]['n_dead'] for c in S]):>6.0f}")
            res[tag]["split"] = S
        print(line + f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "hire.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
