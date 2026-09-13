"""The EM reference: the best templates this objective can have with this search.

E-step   the search, over a fixed training subset, templates held still
M-step   each template becomes the coefficient-weighted average of the images
         on the pixels it owned; pixels it never owned are left as they were
         (the objective is indifferent there); unit norm after. Templates that
         owned nothing are re-seeded from the largest leftovers.
Repeat. Cost per pass is logged; it should not rise.

    python em.py <init> [passes] [n_train]     init in {data, order, hard, soft}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
import engine as E  # noqa: E402
from run import load, support, tally_fit, TALLY_N  # noqa: E402

LAM = 0.02


def main():
    init = sys.argv[1]
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    Xtr, ytr, Xte, yte = load(0)
    X = Xtr[:n_train]
    net = E.Ownership("hard", LAM, np.random.default_rng(0))
    if init == "data":
        rng = np.random.default_rng(1)
        net.M = X[rng.choice(len(X), E.H, replace=False)].copy()
    else:
        net.M = np.load(OUT / f"weights_{init}_l{LAM:g}.npz")["W"].copy()
    net.W = E.unit(net.M)
    net.Wp = np.maximum(net.W, 0)
    net.n[:] = 1
    hist = []
    t0 = time.time()
    for it in range(passes + 1):
        # E-step over the subset
        num = np.zeros((E.H, E.D), np.float64)
        den = np.zeros((E.H, E.D), np.float64)
        owned = np.zeros(E.H, bool)
        costs, counts, unex = [], [], []
        R_big, e_big = [], []
        for b in range(0, len(X), 128):
            xb = X[b:b + 128]
            st = net.search(xb)
            idx, ow, cf, xh = st["idx"], st["owner"], st["coef"], st["xhat"]
            tid = np.where(ow >= 0, idx[np.arange(len(xb))[:, None], np.maximum(ow, 0)], -1)        # (B,d) template per pixel
            a = np.where(ow >= 0, cf[np.arange(len(xb))[:, None], np.maximum(ow, 0)], 0.0)           # (B,d) its coefficient
            m = tid >= 0
            flat = tid[m] * E.D + np.nonzero(m)[1]
            np.add.at(num.ravel(), flat, (a[m] * xb[m]))
            np.add.at(den.ravel(), flat, (a[m] ** 2))
            owned[np.unique(tid[m])] = True
            costs.append(st["cost"]); counts.append(st["count"])
            unex.append(st["cost"] - LAM * st["count"])
            R = np.maximum(xb - xh, 0)
            e = (R ** 2).sum(1)
            top = np.argsort(-e)[:4]
            R_big.append(R[top]); e_big.append(e[top])
        cost = float(np.concatenate(costs).mean())
        hist.append(dict(pass_=it, cost=cost, on=float(np.concatenate(counts).mean()),
                         unexplained=float(np.concatenate(unex).mean()), owned=int(owned.sum())))
        print(f"  em[{init}] pass {it}: cost {cost:.4f}  unexplained {hist[-1]['unexplained']:.4f}  on/img {hist[-1]['on']:.2f}  "
              f"templates that owned something {owned.sum()}  {time.time()-t0:.0f}s", flush=True)
        if it == passes:
            break
        # M-step
        has = den > 0
        newM = net.M.astype(np.float64).copy()
        newM[has] = num[has] / den[has]
        net.M = newM.astype(np.float32)
        # re-seed the templates that owned nothing from the largest leftovers
        dead = np.flatnonzero(~owned)
        if len(dead):
            R_big = np.concatenate(R_big); e_big = np.concatenate(e_big)
            order = np.argsort(-e_big)
            for k, t in enumerate(dead[:len(order)]):
                net.M[t] = R_big[order[k]]
        net.W = E.unit(net.M)
        net.Wp = np.maximum(net.W, 0)
    # held-out evaluation
    Cte, ste = net.codes(Xte)
    Ctr, _ = net.codes(Xtr[:TALLY_N])
    supp, _ = support(net.Wp)
    live = owned
    F_te, F_tr = Cte > 0, Ctr > 0
    L = tally_fit(F_tr, ytr[:TALLY_N])
    tally = float(((F_te.astype(np.float64) @ L).argmax(1) == yte).mean())
    res = dict(init=init, passes=passes, n_train=n_train, hist=hist, t=time.time() - t0,
               on_per_image=float(ste["count"].mean()), unexplained=float((ste["cost"] - LAM * ste["count"]).mean()),
               cost=float(ste["cost"].mean()), support_median=float(np.median(supp[live])),
               support=[int(k) for k in supp[live]], wholes=int((supp[live] > 60).sum()),
               strokes=int(((supp[live] > 20) & (supp[live] <= 60)).sum()), dots=int((supp[live] <= 20).sum()),
               live=int(live.sum()), tally=tally)
    with open(OUT / f"em_{init}.json", "w") as f:
        json.dump(res, f)
    np.savez(OUT / f"weights_em_{init}.npz", W=net.W, Wp=net.Wp, fires=F_te.sum(0), owned=owned)
    print(f"em[{init}] test: cost {res['cost']:.4f}  unexplained {res['unexplained']:.3f}  on/img {res['on_per_image']:.2f}  "
          f"support {res['support_median']:.0f}px  wholes/strokes/dots {res['wholes']}/{res['strokes']}/{res['dots']}  tally {tally:.3f}  {res['t']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
