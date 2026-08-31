"""Finer pooling and more L2 hypercolumns, on Fashion, reusing the trained L1.

L1 is fixed (32 hypercolumns x 8 minicolumns, already trained). What varies is
how finely its map is pooled before L2 sees it, and how many hypercolumns L2
has to divide the work between.

    grid 6  -> 36 cells x 32 =  1152 inputs
    grid 8  -> 64 cells x 32 =  2048
    grid 12 -> 144 cells x 32 = 4608

Reference on this split: stack at grid 6 / H2 20 = 0.7190, the same unit on raw
pixels = 0.6466, logistic on pixels = 0.8512.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c
c.N_TRAIN, c.N_TEST = 20000, 5000
import l2                                                          # noqa: E402
from l2_identity import identity_map                               # noqa: E402

DS, L1KEY, SEED = "fashion_mnist", "H32_K8", 0
CONFIGS = [(6, 40), (8, 20), (8, 40), (8, 60), (12, 40)]


def main():
    t0 = time.time()
    W1 = np.load(OUT / f"conv1_{DS}.npz")[L1KEY].astype(np.float64)
    Xtr, ytr, Xte, yte = c.load(DS)
    res, cache = {}, {}
    for g, h2 in CONFIGS:
        if g not in cache:
            cache[g] = (identity_map(W1, Xtr, gridn=g), identity_map(W1, Xte, gridn=g))
        Ftr, Fte = cache[g]
        l2.H2 = h2
        tag = f"grid{g}_H{h2}"
        W, wins = l2.train(Ftr, ytr, np.random.default_rng(SEED + 1), tag)
        r = l2.report(W, wins, Fte, yte, tag)
        r["dims"] = int(Ftr.shape[1])
        live = np.array(r["wins"]).sum(1)
        r["really_live"] = int((live > live.sum() * 0.002).sum())
        res[tag] = r
        print(f"  -> {tag:<12} dims {Ftr.shape[1]:>4}  accuracy {r['acc']:.4f}  "
              f"really live {r['really_live']}/{h2}  purity {r['purity']:.3f}",
              flush=True)
        np.savez_compressed(OUT / f"l2_{DS}_{tag}.npz", W=W.astype(np.float32),
                            wins=wins)
    (OUT / "sweep2.json").write_text(json.dumps(
        {"results": res, "reference": {"grid6_H20": 0.7190, "raw pixels": 0.6466,
                                       "logistic": 0.8512},
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
