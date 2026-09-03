"""How many whole-digit templates does phase A actually use?

The forgetting test is only meaningful if phase A leaves NO free capacity. The
dense-experts result (`2026_08_31/continual_fixed`) does not forget
precisely BECAUSE 26 of 40 experts were still dead when phase B arrived, and its
README says so: "it works because there was slack". If whole-digit templates are
mostly dead after 0-4, phase B fills the empty ones, nothing is overwritten, and
the run reproduces that result by accident instead of testing churn.

The patch rig had 0/400 dead because each image donated 331 patch-wins -- 6041
images gave 2.0M wins over 400 templates. A whole digit donates one win, so 6041
images give 6041 wins, and winner-take-all concentrates them. This sweeps K to
find where phase A saturates, and reports phase-A accuracy alongside so the
saturated K is not also a crippled one.

Usage:  uv run python saturation.py
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

import whole as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def run_one(k, XA, yA, Xte, yte_g, epochs, seed=7):
    V.K = k
    rng = np.random.default_rng(seed)
    W, N, n, _ = V.train_cont(XA, yA, rng, epochs=epochs)
    iA, kA = V.code(W, XA)
    wins = cp.bincount(iA[kA], minlength=k)
    T, _ = V.rebuild(W, XA, yA)
    acc = V.evaluate(T, V.code(W, Xte), yte_g)["old"]
    dead = int((wins == 0).sum())
    return {"K": k, "dead": dead, "live": k - dead, "phaseA_old": acc,
            "wins_per_live": float(wins[wins > 0].mean()),
            "gini_top10pct": float(cp.sort(wins)[-max(1, k // 10):].sum() / wins.sum())}


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = V.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    XA, yA = Xtr[A], ytr_g[A]

    lines = [f"phase A = {int(A.sum())} images of 0-4, {V.EPOCHS} epochs, "
             f"beta {V.BETA}", "",
             f"{'K':>5} {'live':>6} {'dead':>6} {'wins/live':>10} "
             f"{'top10%share':>12} {'phaseA 0-4':>11}"]
    print("\n".join(lines), flush=True)
    rows = []
    for k in [25, 50, 100, 200, 400, 800]:
        t0 = time.time()
        r = run_one(k, XA, yA, Xte, yte_g, V.EPOCHS)
        rows.append(r)
        line = (f"{r['K']:>5} {r['live']:>6} {r['dead']:>6} "
                f"{r['wins_per_live']:>10.1f} {r['gini_top10pct']:>12.3f} "
                f"{r['phaseA_old']:>11.4f}   ({time.time()-t0:.0f}s)")
        print(line, flush=True); lines.append(line)

    sat = [r for r in rows if r["dead"] == 0]
    msg = ("\nsaturated (0 dead): " +
           (", ".join(f"K={r['K']} at {r['phaseA_old']:.4f}" for r in sat)
            if sat else "NONE -- every K leaves free capacity"))
    print(msg, flush=True); lines.append(msg)

    (OUT / "saturation.json").write_text(json.dumps(rows, indent=2))
    (OUT / "saturation.log").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
