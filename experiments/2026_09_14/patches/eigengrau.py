"""Eigengrau: black is never truly black.  A floor E under every pixel, in training AND in reading.

In training, every picture carries E in every pixel, so every grid collects n_c * E units of flat
ink whatever the window saw -- which is fix 2 arriving by another road: a window that caught 2 real
specks ends up flat, a window that caught 83,000 units does not notice.
In reading, the picture being read carries the same floor, so no patch is ever silent.

The two halves are measured apart, because they do different things:
  * floor in training only  -> grids speak as loudly as their evidence      (fix 2)
  * floor in both           -> plus a standing per-digit offset, the same for every picture
                               (each digit is credited for how much blankness it normally has)

    python eigengrau.py [tag]
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
EPS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
BATCH = 2000


def totals(Xte, logT, logprior, floor):
    lt = logT.reshape(R.C, -1).T
    tot = np.zeros((len(Xte), R.C))
    for b in range(0, len(Xte), BATCH):
        Xw = R.windows(Xte[b:b + BATCH]).reshape(-1, R.NODES * R.CELLS).astype(np.float64)
        tot[b:b + BATCH] = Xw @ lt
    if floor:
        tot = tot + floor * lt.sum(0)                     # the floor's own contribution: one number per digit
    return tot + logprior


def stats(tot, yte):
    pred = tot.argmax(1)
    z = tot - tot.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = np.sort(tot, 1); gap = s[:, -1] - s[:, -2]
    out = dict(acc=round(float((pred == yte).mean()), 4),
               mean_conf=round(float(p.max(1).mean()), 4),
               frac_conf_over_99=round(float((p.max(1) > 0.99).mean()), 4),
               median_gap=round(float(np.median(gap)), 1))
    ab = {}
    for a, b in [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)]:
        m = (gap >= a) & (gap < b)
        if m.sum() > 20:
            ab[f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == yte[m]).mean()), 3), int(m.sum())]
    out["acc_by_gap"] = ab
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
    t0 = time.time()
    M0 = np.load(OUT / f"{tag}.npz")["mass_raw"].astype(np.float64)
    Xtr, ytr, Xte, yte = R.load()
    n_c = np.bincount(ytr, minlength=R.C).astype(np.float64)
    prior = n_c / n_c.sum(); logprior = np.log(prior)
    pix_edge = np.zeros((28, 28), bool); pix_edge[:4] = pix_edge[-4:] = True; pix_edge[:, :4] = pix_edge[:, -4:] = True

    rows, Ws = [], {}
    for E in EPS:
        M = M0 + n_c[:, None, None] * E                      # the floor, collected over training
        T = (M + (1e-3 if E == 0 else 0.0))
        T = T / T.sum(2, keepdims=True)
        logT = np.log(T)
        tr_only = stats(totals(Xte, logT, logprior, 0.0), yte)
        both = stats(totals(Xte, logT, logprior, E), yte)
        Wd = weight_images(logT); Wd = Wd - Wd.mean(0, keepdims=True); Ws[E] = Wd
        frame = float(np.abs(Wd)[:, pix_edge].mean() / np.abs(Wd)[:, ~pix_edge].mean())
        rows.append(dict(eps=E, imaginary_ink_per_cell=round(float(n_c.mean() * E), 2),
                         train_only=tr_only, both=both, edge_to_centre_weight=round(frame, 3)))
        print(f"E={E:<7}(= {rows[-1]['imaginary_ink_per_cell']:>8.1f} ink/cell)  "
              f"train-only acc {tr_only['acc']:.4f} conf {tr_only['mean_conf']:.4f}  |  "
              f"both acc {both['acc']:.4f} conf {both['mean_conf']:.4f} gap {both['median_gap']:>7.1f}  |  edge/centre {frame:.2f}")

    base = rows[0]["both"]["acc"]
    ok = [r for r in rows if base - r["both"]["acc"] <= 0.005]
    chosen = ok[-1]
    res = dict(tag=tag, n_test=len(Xte), seconds=round(time.time() - t0, 2),
               chosen_eps=chosen["eps"], chosen_rule="largest floor costing <=0.005 accuracy",
               baseline_acc=base, chosen_acc=chosen["both"]["acc"], rows=rows)
    json.dump(res, open(OUT / f"{tag}_eigengrau.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{tag}_eigengrau.npz",
                        W_before=Ws[0.0].astype(np.float32), W_after=Ws[chosen["eps"]].astype(np.float32),
                        eps=np.array(EPS),
                        acc_tr=np.array([r["train_only"]["acc"] for r in rows]),
                        acc_both=np.array([r["both"]["acc"] for r in rows]),
                        conf_both=np.array([r["both"]["mean_conf"] for r in rows]),
                        gap_both=np.array([r["both"]["median_gap"] for r in rows]),
                        frames=np.array([r["edge_to_centre_weight"] for r in rows]))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
