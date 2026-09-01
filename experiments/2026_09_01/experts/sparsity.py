"""Does rectification pin the basis once the code is actually SPARSE?

Condition B -- centred patches, signed templates, rectified coefficients, only
the positive ones learning from the residual -- landed at chance (+0.006). But
it ran at 29 of 100 active, which is a thinned dense code rather than a sparse
one, so the constraint may simply have been too weak to bite.

Two things measured across a sparsity sweep:

    match       do two runs from different seeds find the same templates
    anti-pairs  how many template pairs point in OPPOSITE directions

The anti-pair count tests the escape route. If the dictionary holds both w and
-w, rectification is free: whichever sign the patch has, one of the pair fires,
and together they reproduce the unrectified projection exactly. The relu then
constrains nothing and the underlying set can still be any rotation. Centred
MNIST patches are near-symmetric under sign flip, so the data invites it.

If match climbs with sparsity, rectification does pin the basis and my original
claim was right but under-powered. If it stays flat while anti-pairs stay high,
the escape is real and only constraining the TEMPLATES works.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, rotation as R

OUT = Path(__file__).resolve().parent / "results"
LAMS = [0.05, 0.15, 0.30, 0.50]


def anti(W, thr=-0.85):
    G = W @ W.T
    np.fill_diagonal(G, 0.0)
    return int((G < thr).sum() // 2), float(G.min())


def main():
    Xtr, ytr, _, _ = E.load()
    P = R.patches(Xtr, ytr, np.random.default_rng(1), True)
    r = np.random.default_rng(0)
    A = r.standard_normal((R.N, 25)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    Bm = r.standard_normal((R.N, 25)); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
    chance = R.match(A, Bm)
    print(f"  chance {chance:.4f}   (100 templates over 25 dims)\n", flush=True)

    res = {"chance": chance}
    for lam in LAMS:
        R.LAM = lam
        W1, a1 = R.train(P, 10, True, False)
        W2, _ = R.train(P, 99, True, False)
        m = R.match(W1, W2)
        act = float((a1 > 1e-6).sum(1).mean())
        np_, mn = anti(W1)
        L = np.linalg.norm(W1, 2) ** 2
        rec = float(np.linalg.norm(
            P[:2000] - R.codes(P[:2000].astype(np.float32), W1, True, L) @ W1, axis=1).mean())
        res[str(lam)] = {"match": m, "over_chance": m - chance, "active": act,
                         "anti_pairs": np_, "min_cosine": mn, "rebuild_error": rec}
        print(f"  lam {lam:<5} active {act:5.1f}/100  match {m:.4f} ({m-chance:+.4f})  "
              f"anti-pairs {np_:3d}  min cos {mn:+.3f}  error {rec:.4f}", flush=True)

    best = max(LAMS, key=lambda l: res[str(l)]["match"])
    R.LAM = best
    W1, _ = R.train(P, 10, True, False)
    order = np.argsort(-np.abs(W1).sum(1))
    tile = np.full((10 * 6 - 1, 10 * 6 - 1), np.nan)
    for i, j in enumerate(order[:100]):
        rr, cc = divmod(i, 10)
        tile[rr * 6:rr * 6 + 5, cc * 6:cc * 6 + 5] = W1[j].reshape(5, 5)
    plt.figure(figsize=(6, 6.4))
    v = np.nanmax(np.abs(tile))
    plt.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    plt.title(f"centred, rectified, lam={best}\nactive {res[str(best)]['active']:.1f}/100   "
              f"match {res[str(best)]['match']:.3f} (chance {chance:.3f})", fontsize=10)
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(OUT / "rotation_sparsity.png", dpi=130); plt.close()
    res["reference"] = {"A raw nonneg W and a": 0.7813}
    (OUT / "sparsity.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
