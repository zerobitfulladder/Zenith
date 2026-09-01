"""Two more conditions: does the label pin the basis, and was it W or the data?

D  centred patches WITH the label block, unconstrained coefficients.
   The question: joint training with labels changes WHICH subspace gets chosen
   -- does it also pin the basis inside it? Yesterday's dense_ensemble already
   argued not, on paper: reading the label out of a dense subspace is
       label_hat = W_lab^T W_img x = M x
   and M is exactly rotation-invariant, since (QW)_lab^T (QW)_img
   = W_lab^T Q^T Q W_img = M. The label constrains the SPAN, not the basis.
   Prediction: at chance.

E  raw non-negative patches, SIGNED templates, non-negative coefficients.
   Condition A changed two things at once -- non-negative templates AND raw
   data. This isolates them. If E is high, non-negative DATA was enough and the
   coefficient constraint starts biting once the dictionary cannot supply the
   opposite polarity. If E is at chance, the template constraint did the work.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, rotation as R

OUT = Path(__file__).resolve().parent / "results"


def main():
    Xtr, ytr, _, _ = E.load()
    rng = np.random.default_rng(1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    J = np.zeros((len(U), 35), np.float32)
    J[:, :25] = U
    J[np.arange(len(U)), 25 + L] = 1.0
    J /= np.linalg.norm(J, axis=1, keepdims=True) + R.EPS
    Pr = R.patches(Xtr, ytr, np.random.default_rng(1), False)

    r = np.random.default_rng(0)
    out = {}
    for d in (25, 35):
        A = r.standard_normal((R.N, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
        Bm = r.standard_normal((R.N, d)); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
        out[f"chance_{d}d"] = R.match(A, Bm)
    print(f"  chance: 25d {out['chance_25d']:.4f}   35d {out['chance_35d']:.4f}\n", flush=True)

    for name, P, na, nw, ch in (
            ("D centred + LABEL, unconstrained", J, False, False, out["chance_35d"]),
            ("E raw, SIGNED W, nonneg a", Pr, True, False, out["chance_25d"])):
        W1, a1 = R.train(P, 10, na, nw)
        W2, _ = R.train(P, 99, na, nw)
        m = R.match(W1, W2)
        act = float((a1 > 1e-6).sum(1).mean())
        L_ = np.linalg.norm(W1, 2) ** 2
        rec = float(np.linalg.norm(
            P[:2000] - R.codes(P[:2000].astype(np.float32), W1, na, L_) @ W1, axis=1).mean())
        out[name] = {"match": m, "chance": ch, "over_chance": m - ch,
                     "active_per_patch": act, "rebuild_error": rec}
        print(f"  {name:<34} match {m:.4f}  (chance {ch:.4f}, {m-ch:+.4f})   "
              f"active {act:.1f}   error {rec:.4f}", flush=True)

        order = np.argsort(-np.abs(W1[:, :25]).sum(1))
        tile = np.full((10 * 6 - 1, 10 * 6 - 1), np.nan)
        for i, j in enumerate(order[:100]):
            rr, cc = divmod(i, 10)
            tile[rr * 6:rr * 6 + 5, cc * 6:cc * 6 + 5] = W1[j, :25].reshape(5, 5)
        plt.figure(figsize=(6, 6.4))
        v = np.nanmax(np.abs(tile))
        plt.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        plt.title(f"{name}\nmatch {m:.3f} (chance {ch:.3f})", fontsize=10)
        plt.xticks([]); plt.yticks([]); plt.tight_layout()
        plt.savefig(OUT / f"rotation_{name[0]}.png", dpi=130); plt.close()

    out["reference"] = {"A raw nonneg W+a": 0.7813, "B centred nonneg a": 0.4819,
                        "C centred unconstrained": 0.4651}
    (OUT / "rotation2.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
