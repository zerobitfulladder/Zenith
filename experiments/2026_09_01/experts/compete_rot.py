"""The rotation test on the rule we actually use: winner-take-all competition.

Every rotation measurement so far was on a dense or sparse CODING rule, and all
of them came back at chance except non-negative templates. But the design this
project kept is neither -- it is top-1 competition, where one template wins a
patch outright and only the winner learns.

That produced visibly crisp oriented gradients on the same data where sparse
coding produced static. Two possibilities, and they have opposite implications:

    the competition genuinely finds a preferred basis    -> two seeds agree
    the gradients merely reflect the dominant covariance -> two seeds find a
        of the patches, and any rotation of them            ROTATED set that
        looks equally gradient-like                         looks just as good

100 templates so the chance line from the other runs applies directly.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, rotation as R

OUT = Path(__file__).resolve().parent / "results"
N = 100


def main():
    Xtr, ytr, _, _ = E.load()
    rng = np.random.default_rng(1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    P = (U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), R.EPS))

    r = np.random.default_rng(0)
    A = r.standard_normal((N, 25)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    Bm = r.standard_normal((N, 25)); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
    chance = R.match(A, Bm)

    W1 = E.train(P, N, 1, 0.0, np.random.default_rng(10))[:, 0, :]
    W2 = E.train(P, N, 1, 0.0, np.random.default_rng(99))[:, 0, :]
    m = R.match(W1, W2)
    print(f"  chance {chance:.4f}")
    print(f"  competition, two seeds: match {m:.4f}  ({m-chance:+.4f})", flush=True)
    print(f"  for comparison: nonneg templates +0.306, sparse coding +0.011")

    order = np.argsort(-np.abs(W1).sum(1))
    tile = np.full((10 * 6 - 1, 10 * 6 - 1), np.nan)
    for i, j in enumerate(order[:100]):
        rr, cc = divmod(i, 10)
        tile[rr * 6:rr * 6 + 5, cc * 6:cc * 6 + 5] = W1[j].reshape(5, 5)
    plt.figure(figsize=(6, 6.4))
    v = np.nanmax(np.abs(tile))
    plt.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    plt.title(f"winner-take-all competition, {N} templates\n"
              f"match across two seeds {m:.3f} (chance {chance:.3f})", fontsize=10)
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(OUT / "rotation_compete.png", dpi=130); plt.close()
    (OUT / "compete_rot.json").write_text(json.dumps(
        {"chance": chance, "match": m, "over_chance": m - chance,
         "reference": {"nonneg templates": 0.7813, "sparse coding best": 0.4882}},
        indent=2))


if __name__ == "__main__":
    main()
