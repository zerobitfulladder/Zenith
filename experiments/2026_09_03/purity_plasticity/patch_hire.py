"""Hire on error at 9x9: can the tally's novelty branch lift the patch rig?

Reference: top-1, no pressure, cnt/n step = 0.9710 joint at 9x9 (../residual_chain).
On parts a misread image does not mean any patch was wrong, so the offending
patches are those whose winner is committed at that cell (purity >= 0.5) to
evidence against the label (T < 0); each is handed to the nearest template that
is uncommitted at that cell. No row is wiped. Joint 0-9, 12k/3k, 3 epochs.

Usage:  uv run python patch_hire.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import purity_split as P
import rf_sweep as R

OUT = P.OUT
def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
PS = arg("--ps", 9)
NTRAIN, NTEST = arg("--ntrain", 12000), arg("--ntest", 3000)
R.E.N_TRAIN, R.E.N_TEST = NTRAIN, NTEST
SEEDS = [7] if P.SMOKE else [7, 8]
P.EPOCHS = 1 if P.SMOKE else 3
P.PROBE_EVERY = 10 ** 6            # phase-end probe only; probing is expensive on patches
ARMS = [("top1 none",        "cntn", "none", 0.0,  0.0, False),
        ("none + hire",      "cntn", "none", 0.0,  0.0, True),
        ("summed + hire",    "cntn", "both", 0.25, 0.5, True)]
if P.SMOKE:
    ARMS = ARMS[1:2]


def main():
    OUT.mkdir(exist_ok=True)
    R.BATCH = 128; rig = R.Rig(PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if P.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    joint = [(Xtr, ytr_g, "0-9")]
    res = {}
    print(f"{PS}x{PS}, {rig.npos} positions, {P.EPOCHS} epochs, seeds {SEEDS}, train {len(ytr)} / test {len(yte)}\n")
    print(f"{'arm':<16}{'recount':>9}{'(seeds)':>18}{'online':>9}{'dead':>6}{'purity':>8}")
    for tag, rule, route, bl, bb, hire in ARMS:
        P.BETA_L, P.BETA_B, P.HIRE, P.CHASE = bl, bb, hire, False
        t0 = time.time()
        J = [P.run(rig, rule, route, joint, Xtr, ytr_g, Xte, yte_g, s)[0] for s in SEEDS]
        rec = np.array([c[-1]["recount"][2] for c in J])
        print(f"{tag:<16}{rec.mean():>9.4f}  (" + "/".join(f"{x:.4f}" for x in rec) + ")"
              f"{np.mean([c[-1]['online'][2] for c in J]):>9.4f}{np.mean([c[-1]['n_dead'] for c in J]):>6.0f}"
              f"{np.mean([c[-1]['purity_live'] for c in J]):>8.3f}   ({time.time() - t0:.0f}s)", flush=True)
        res[tag] = J
    (OUT / f"patch_hire_{PS}_{NTRAIN}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
