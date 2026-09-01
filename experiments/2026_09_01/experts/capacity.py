"""Same budget, split differently: many small experts or few large ones.

Every row holds H x K = 240 templates of 35 numbers -- 8,400 either way. What
changes is how the budget is spent:

    many experts, few templates each   a fine partition, a weak fit per expert,
                                       and a MORE informative winner
    few experts, many templates each   a coarse partition, a strong fit, and a
                                       winner that says less

There is a ceiling on K: the joint vector is 35 numbers, so an expert holding
K of them spans K/35 of the space. At K=16 that is nearly half, the errors
converge, and "which expert won" stops meaning much.

Readout held fixed at the 6x6 grid that won the last sweep (0.9507 at H=30, K=8).

A confound to keep in sight and why the table size is reported: the counted
table is H x 36 x 10, so more experts also buys a bigger readout. H=120 gets a
four-times larger table than H=30 for free. If the many-expert rows win, some
of that is the readout, not the vocabulary.

Prediction: many-small wins. Yesterday's plain k-means -- 64 templates, no
subspace at all, the K=1 extreme -- beat 30 experts of 8 under the identical
readout (0.9817 vs 0.9717).
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, blank as B

OUT = Path(__file__).resolve().parent / "results"
GRID, BUDGET = 6, 240
CONFIGS = [(120, 2), (60, 4), (30, 8), (15, 16)]


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(E.SEED + 1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    Vtr = E.join(U, L)
    print(f"{len(U):,} patches, budget {BUDGET} templates x {Vtr.shape[1]} numbers "
          f"= {BUDGET*Vtr.shape[1]:,}\n", flush=True)

    res = {}
    for h, k in CONFIGS:
        t1 = time.time()
        W = E.train(Vtr, h, k, 0.0, np.random.default_rng(E.SEED + 1))
        itr, ite = B.winners(W, Xtr), B.winners(W, Xte)
        T = B.table(itr, ytr, h, GRID)
        p = B.score(ite, T, GRID).argmax(1)
        acc = float((p == yte).mean())

        cnt = np.zeros(h); ld = np.zeros((h, 10))
        r, c = np.nonzero(ite >= 0)
        np.add.at(cnt, ite[r, c], 1.0); np.add.at(ld, (ite[r, c], yte[r]), 1.0)
        d = ld / np.maximum(ld.sum(1, keepdims=True), 1)
        live = cnt > 0
        row = {"H": h, "K": k, "acc": acc,
               "expert_params": int(h * k * Vtr.shape[1]),
               "table_params": int(h * GRID * GRID * 10),
               "class_purity": float(d.max(1)[live].mean()),
               "live": int(live.sum()),
               "subspace_fraction": round(k / Vtr.shape[1], 3),
               "seconds": round(time.time() - t1, 1)}
        res[f"H{h}_K{k}"] = row
        print(f"  H={h:<4} K={k:<3} live {row['live']:>3}/{h:<4} acc {acc:.4f}  "
              f"purity {row['class_purity']:.3f}  subspace {row['subspace_fraction']:.2f}  "
              f"table {row['table_params']:,}  ({time.time()-t1:.0f}s)", flush=True)

    res["reference"] = {"H30_K8_grid4": 0.9423, "H30_K8_grid6": 0.9507,
                        "kmeans 64 single templates (probe)": 0.9817}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "capacity.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
