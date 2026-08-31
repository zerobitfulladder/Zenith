"""The whole stack on Fashion-MNIST, on the same split as the baselines.

Everything as it stands: L1 = 32 hypercolumns x 8 minicolumns on 5x5 patches,
unsupervised (rebuild error minus conscience). Message = identity only, pooled
to 6x6. L2 = one unit of 20 x 16 with the label as a second stream (λ=4) and
conscience. No repulsion, no reluctance, no sleep, no teacher.

The earlier MNIST run used 8000/2000; the Fashion baselines were measured on
20000 train / 5000 test, so this uses that split and the numbers are directly
comparable.
"""

import json, time
from pathlib import Path
import numpy as np
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c
c.N_TRAIN, c.N_TEST = 20000, 5000
from l2 import train as l2_train, report as l2_report, H2, K2, SEED   # noqa: E402
from l2_identity import identity_map                                  # noqa: E402

DS, H1, K1 = "fashion_mnist", 32, 8
BOARD = {"SVM rbf": 0.8926, "MLP 512-256": 0.8850, "MLP 256": 0.8824,
         "random forest 300": 0.8698, "logistic (lbfgs)": 0.8512,
         "linear SVM": 0.8506, "kNN k=3": 0.8366, "nearest centroid": 0.6950,
         "one layer, teacher-free": 0.6868, "one layer, with teacher": 0.8278}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = c.load(DS)
    print(f"{DS}: {len(Xtr)} train / {len(Xte)} test  (same split as the baselines)",
          flush=True)

    W1, wins1 = c.train(Xtr, H1, K1, np.random.default_rng(SEED + 1))
    r1, _ = c.evaluate(W1, Xte, yte, wins1)
    print(f"  L1: rebuild {r1['rebuild_err']:.4f}  dead {r1['dead']}/{H1}  "
          f"busiest {r1['busiest']*100:.1f}%  overlap {r1['overlap']:.4f}  "
          f"kept {r1['kept_patches']*100:.0f}% of patches", flush=True)
    np.savez_compressed(OUT / f"conv1_{DS}.npz",
                        **{f"H{H1}_K{K1}": W1.astype(np.float32)})

    Ftr, Fte = identity_map(W1, Xtr), identity_map(W1, Xte)
    print(f"  L2 input {Ftr.shape[1]} numbers "
          f"({float((Ftr != 0).mean())*100:.0f}% nonzero)", flush=True)

    res = {"L1": r1}
    for tag, A, B in (("stack (L1 -> L2)", Ftr, Fte),
                      ("same unit on raw pixels", Xtr.reshape(len(Xtr), -1),
                       Xte.reshape(len(Xte), -1))):
        W, wins = l2_train(A, ytr, np.random.default_rng(SEED + 1), tag)
        res[tag] = l2_report(W, wins, B, yte, tag)
        np.savez_compressed(OUT / f"l2_{DS}_{tag.split()[0]}.npz",
                            W=W.astype(np.float32), wins=wins)

    print("\nagainst the board:")
    rows = sorted(list(BOARD.items()) +
                  [(k, v["acc"]) for k, v in res.items() if k != "L1"],
                  key=lambda kv: -kv[1])
    for k, v in rows:
        mark = "  <-- ours" if k in res else ""
        print(f"  {k:<28} {v:.4f}{mark}")
    (OUT / "fashion_stack.json").write_text(json.dumps(
        {"results": res, "board": BOARD, "seconds": round(time.time() - t0, 1)},
        indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
