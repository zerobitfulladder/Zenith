"""SoftHebb's two ingredients on our whole-digit rig, single layer, tally read.

    soft   losers move TOWARD the input by their softmax activity (winner-take-most)
    anti   losers move AWAY (the paper's soft anti-Hebbian term)

Both on top of the best rig (purity step, label 0.25 + belief 0.5, hire on
error); counting stays hard (top-1). Two temperatures. Joint and split.

Usage:  uv run python soft_push.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import purity_split as P
import rf_sweep as R

OUT = P.OUT
SEEDS = [7] if P.SMOKE else [7, 8]
P.EPOCHS = 2 if P.SMOKE else 20
ARMS = [("best",            None,   0.0),
        ("soft tau 0.05",   "hebb", 0.05),
        ("soft tau 0.2",    "hebb", 0.2),
        ("anti tau 0.05",   "anti", 0.05),
        ("anti tau 0.2",    "anti", 0.2)]
if P.SMOKE:
    ARMS = ARMS[3:4]


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
    P.BETA_L, P.BETA_B, P.HIRE, P.CHASE = 0.25, 0.5, True, False
    print(f"28x28 MNIST, best rig + SoftHebb terms, {P.EPOCHS} epochs/phase, seeds {SEEDS}\n")
    print(f"{'arm':<15}{'JOINT recount':>14}{'online':>8}{'dead':>6} | "
          f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}")
    res = {}
    for tag, soft, tau in ARMS:
        P.SOFT, P.TAU = soft, tau
        t0 = time.time()
        J = [P.run(rig, "purity", "both", joint, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
        S = [P.run(rig, "purity", "both", split, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
        e = [P.end_points(c) for c in S]
        m = lambda kk: np.mean([x[kk] for x in e])
        res[tag] = {"joint": J, "split": S}
        print(f"{tag:<15}{np.mean([c[-1]['recount'][2] for c in J]):>14.4f}"
              f"{np.mean([c[-1]['online'][2] for c in J]):>8.4f}{np.mean([c[-1]['n_dead'] for c in J]):>6.0f} | "
              f"{m('B_old'):>14.4f}/{m('B_new'):.4f}{m('C_all'):>8.4f}{m('C_online'):>8.4f}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "soft_push.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
