"""Fix 2: a grid should speak only as loudly as the evidence behind it.

Before scaling a grid to add up to 1, add A units of imaginary, evenly-spread ink to every cell.
A window that collected 83,000 units of real ink does not notice.  A window that collected 2 units
goes flat, and a flat grid gives every digit the same score -- which is what "I know nothing"
should look like.  A is the one knob: how much real ink a window must collect before we listen.

Everything else is unchanged: the same counting, the same read (ink: sum of x * log grid).

    python fix2.py [tag]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import run as R
from read import weight_images

HERE = Path(__file__).parent
OUT = HERE / "results"
ALPHAS = [0.001, 1.0, 10.0, 100.0, 1000.0, 10000.0]        # imaginary ink per cell; 0.001 = what we had
BATCH = 2000


def read_totals(Xte, logT, logprior):
    tot = np.zeros((len(Xte), R.C))
    lt = logT.reshape(R.C, -1).T                            # (14400,10)
    for b in range(0, len(Xte), BATCH):
        Xw = R.windows(Xte[b:b + BATCH]).reshape(-1, R.NODES * R.CELLS).astype(np.float64)
        tot[b:b + BATCH] = Xw @ lt
    return tot + logprior


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
    t0 = time.time()
    M = np.load(OUT / f"{tag}.npz")["mass_raw"].astype(np.float64)        # (10,576,25) raw ink sums
    Xtr, ytr, Xte, yte = R.load()
    prior = np.bincount(ytr, minlength=R.C) / len(ytr)
    logprior = np.log(prior)

    mass = M.sum(2)                                                       # (10,576)
    edge = np.zeros((R.G, R.G), bool); edge[:3] = edge[-3:] = True; edge[:, :3] = edge[:, -3:] = True
    pix_edge = np.zeros((28, 28), bool); pix_edge[:4] = pix_edge[-4:] = True; pix_edge[:, :4] = pix_edge[:, -4:] = True

    rows, Ws = [], {}
    for A in ALPHAS:
        T = (M + A) / (M.sum(2, keepdims=True) + A * R.CELLS)
        logT = np.log(T)
        tot = read_totals(Xte, logT, logprior)
        pred = tot.argmax(1)
        z = tot - tot.max(1, keepdims=True); post = np.exp(z); post /= post.sum(1, keepdims=True)
        s = np.sort(tot, 1); gap = s[:, -1] - s[:, -2]
        conf = post.max(1)

        W = weight_images(logT); Wd = W - W.mean(0, keepdims=True); Ws[A] = Wd
        frame = float(np.abs(Wd)[:, pix_edge].mean() / np.abs(Wd)[:, ~pix_edge].mean())

        # how flat are the grids of the windows that saw almost nothing?
        flat_when_empty = float((T[mass < 100] * R.CELLS).std(1).mean()) if (mass < 100).any() else float("nan")
        acc_by_gap = {}
        for a, b in [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)]:
            m = (gap >= a) & (gap < b)
            if m.sum() > 20:
                acc_by_gap[f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == yte[m]).mean()), 3), int(m.sum())]
        rows.append(dict(alpha=A, acc=round(float((pred == yte).mean()), 4),
                         mean_conf=round(float(conf.mean()), 4),
                         frac_conf_over_99=round(float((conf > 0.99).mean()), 4),
                         median_gap=round(float(np.median(gap)), 1),
                         edge_to_centre_weight=round(frame, 3),
                         empty_grid_flatness=round(flat_when_empty, 4),
                         acc_by_gap=acc_by_gap))
        print(f"A={A:>9}: acc {rows[-1]['acc']:.4f}  conf {rows[-1]['mean_conf']:.4f}  "
              f"gap {rows[-1]['median_gap']:>7.1f}  edge/centre {frame:.2f}")

    # the choice rule: the most imaginary ink we can afford for free -- the largest A whose
    # accuracy is within 0.005 of the baseline.  Quieting a window that knows nothing should not
    # cost us anything; if it does, we have taken too much away.
    base = rows[0]["acc"]
    best = [r for r in rows if base - r["acc"] <= 0.005][-1]
    res = dict(tag=tag, knob="imaginary ink per cell, added before scaling the grid to 1",
               n_test=len(Xte), seconds=round(time.time() - t0, 2), chosen_alpha=best["alpha"], chosen_rule="largest A costing <=0.005 accuracy",
               baseline_acc=rows[0]["acc"], chosen_acc=best["acc"], rows=rows,
               windows_under_100_ink=int((mass < 100).sum()), windows_total=int(mass.size))
    json.dump(res, open(OUT / f"{tag}_fix2.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{tag}_fix2.npz",
                        W_before=Ws[ALPHAS[0]].astype(np.float32),
                        W_after=Ws[best["alpha"]].astype(np.float32),
                        alphas=np.array(ALPHAS), accs=np.array([r["acc"] for r in rows]),
                        confs=np.array([r["mean_conf"] for r in rows]),
                        frames=np.array([r["edge_to_centre_weight"] for r in rows]))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
