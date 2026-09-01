"""Let the class differences decide how codes are COMPARED. No new templates.

The symptom that has survived everything today: matching a code against a class
average scores 0.9100 while a learned rotation scores 0.9767. That 6.7-point gap
means the classes are stretched cigars, not balls -- some directions inside a
class vary hugely and mean nothing, and a plain cosine cannot know which to
ignore. A linear probe learns exactly that and pockets the difference.

The counted table already knows which directions matter:

    T[t, cell, class] = log P(template t | cell, class) / P(template t | cell)
    w[t, cell]        = how much T varies ACROSS classes at that spot

High where 7s have a horizontal stroke and 1s do not. Near zero for a generic
stroke every digit contains. Then scale each dimension of the pooled code by
w^p and compare in that stretched space.

Nothing is allocated, nothing is retrained, no gradients. The features stay
unsupervised -- these are the NO-label templates -- and the only place the class
enters is the weighting. If the gap closes, the stable class code is bought by
counting.
"""

import json
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, readouts as R

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = 4, 10, 1e-12
POWERS = [0.0, 0.5, 1.0, 2.0]


def pooled(idx, h):
    b = S.SIDE // GRID
    M = np.zeros((len(idx), S.SIDE * S.SIDE, h), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = 1.0
    return M.reshape(len(idx), GRID, b, GRID, b, h).max(axis=(2, 4)).reshape(
        len(idx), GRID * GRID, h)


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)


def ladder(Atr, ytr, Ate, yte):
    M = unit(np.stack([Atr[ytr == c].mean(0) for c in range(NL)]))
    nm = float(((unit(Ate) @ M.T).argmax(1) == yte).mean())
    nn = float((ytr[(unit(Ate) @ unit(Atr).T).argmax(1)] == yte).mean())
    net = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax",
                    hidden=0, epochs=30)
    lin = float((R.predict_net(net, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())
    w, b, g, _ = S.simstats(Ate[:600], yte[:600])
    return {"nearest_mean": nm, "1nn": nn, "linear": lin, "gap": lin - nm,
            "code_gap": g, "within": w, "between": b}


def main():
    W = np.load(OUT / "single_unlabelled.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()
    import single as SG
    itr, ite = SG.winners(W, Xtr), SG.winners(W, Xte)

    T = B.table(itr, ytr, h, GRID)                       # (h, ncell, NL)
    w = T.std(axis=2)                                     # (h, ncell) -- spread over classes
    print(f"{h} templates, {GRID}x{GRID} cells.  weight: min {w.min():.3f} "
          f"median {np.median(w):.3f} max {w.max():.3f}", flush=True)

    Ptr, Pte = pooled(itr, h), pooled(ite, h)             # (n, ncell, h)
    wm = w.T[None]                                        # (1, ncell, h)
    res = {}
    for p in POWERS:
        sc = wm ** p
        Atr = (Ptr * sc).reshape(len(Ptr), -1)
        Ate = (Pte * sc).reshape(len(Pte), -1)
        r = ladder(Atr, ytr, Ate, yte)
        res[f"p={p}"] = r
        print(f"  w^{p:<4} nearest-mean {r['nearest_mean']:.4f}  1-NN {r['1nn']:.4f}  "
              f"linear {r['linear']:.4f}  LADDER GAP {r['gap']:.4f}  "
              f"code gap {r['code_gap']:+.4f}", flush=True)

    res["reference"] = {"unweighted nearest_mean": 0.9100, "unweighted linear": 0.9767,
                        "unweighted ladder gap": 0.0667,
                        "shaped-by-feedback nearest_mean (H30)": 0.9237}
    (OUT / "weighting.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
