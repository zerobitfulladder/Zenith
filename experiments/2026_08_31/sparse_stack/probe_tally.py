"""Classify by counting, not by matching. Layer three replaced with a tally.

Layer three is a nearest-neighbour matcher over whole vectors: it consumes
coordinates and never asks which template fired. This asks the opposite
question -- what if the ONLY thing read is which templates fired?

    tally[p, i, y]  =  how often template i fired at position p
                       on a digit labelled y

Then a test code votes: every active (position, template) contributes what
that cell has historically said about each label, and the labels are summed.
No prototypes, no winner-take-all, no learning at layer three at all -- one
pass of counting and one matmul.

This is the first read in the repo that uses a template's IDENTITY rather
than its magnitude as a coordinate, so it is the first that a sparse code
can be better at than a dense one.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_30" / "stack3"))
from stack3 import ConvStack                                     # noqa: E402
from sparse_stack import SparseStack                             # noqa: E402
from run_sparse_stack import load, K1, K2, ETA1, EPOCHS, SEED, onehot  # noqa

ALPHA = 1.0            # Laplace smoothing on the counts
BOARD = {"dense, layer three": 0.9232, "wta, layer three": 0.5708,
         "sparse k=4, layer three": 0.7494,
         "sparse k=4 settled, layer three": 0.7946,
         "judge (logistic on pixels)": 0.9074}


def tally_read(Htr, ytr, Hte, yte):
    """Every build/score combination, all of them one matmul."""
    Y = onehot(ytr)
    out = {}
    for build in ("count", "mass"):
        B = (Htr > 0).astype(np.float64) if build == "count" else Htr
        T = B.T @ Y                                   # (cells, 10)
        P = (T + ALPHA) / (T.sum(1, keepdims=True) + 10 * ALPHA)
        for form, M in (("log", np.log(P)), ("linear", P)):
            for score in ("count", "mass"):
                Q = (Hte > 0).astype(np.float64) if score == "count" else Hte
                pred = (Q @ M).argmax(1)
                out[f"{build}/{form}/{score}"] = float((pred == yte).mean())
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    res = {}
    for tag, mk in (("dense", lambda r: ConvStack("dense", K1, K2, ETA1, r)),
                    ("wta", lambda r: ConvStack("wta", K1, K2, ETA1, r)),
                    ("sparse k=4", lambda r: SparseStack(K1, K2, 4, ETA1, r)),
                    ("sparse k=4 settled",
                     lambda r: SparseStack(K1, K2, 4, ETA1, r, read="settle"))):
        rng = np.random.default_rng(SEED)
        cs = mk(rng)
        cs.train(Xtr, EPOCHS, rng)
        Htr, Hte = cs.forward(Xtr), cs.forward(Xte)
        r = tally_read(Htr, ytr, Hte, yte)
        best = max(r, key=r.get)
        res[tag] = {"variants": r, "best": best, "best_accuracy": r[best],
                    "fraction_lit": float((Hte > 0).mean())}
        print(f"{tag:<20} best {r[best]:.4f}  ({best})   "
              f"pure count {r['count/log/count']:.4f}   "
              f"lit {res[tag]['fraction_lit']:.3f}   [{time.time()-t0:.0f}s]")

    print("\nagainst the board:")
    for k, v in BOARD.items():
        print(f"  {k:<34} {v:.4f}")
    (HERE / "results" / "tally.json").write_text(json.dumps(
        {"alpha": ALPHA, "results": res, "board": BOARD,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
