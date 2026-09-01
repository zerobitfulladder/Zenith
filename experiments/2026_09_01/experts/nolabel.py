"""Did the label in the templates ever touch the 0.9423?

In the run everything else was built on, the winner was chosen by ink alone
(lambda = 0) and the counting table reads only WHICH expert won. So the ten
label numbers inside every template were trained, sat there, and were never
read by the pipeline that produced the best honest number.

If that reasoning is right, deleting the label block entirely -- experts trained
on nothing but 25-dimensional patches, no supervision anywhere below the tally
-- should reproduce it.

Not a perfectly matched control, and the direction of the mismatch is worth
stating: with the label gone, all 8 templates fit 25 numbers instead of 35, so
these experts have MORE capacity for ink. If the score moves up, that is why,
and it is a point in favour of leaving the label out rather than evidence about
the tally.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E
import vote as V
import readouts as R

OUT = Path(__file__).resolve().parent / "results"
SIDE, GRID = V.SIDE, V.GRID
H, K, SEED = E.H, E.K, E.SEED


def winners(W, X, chunk=128):
    """Winning expert at every position, ink only. Also mean rebuild error."""
    idx = np.full((len(X), SIDE * SIDE), -1, np.int16)
    tot, n = 0.0, 0
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        B = Q.reshape(-1, E.PS * E.PS)
        S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
        Rb = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
        err = ((B[None] - Rb) ** 2).sum(-1)
        w = err.argmin(0)
        k = keep.reshape(-1)
        tot += float(err[w, np.arange(len(B))][k].sum()); n += int(k.sum())
        idx[a:a + len(Q)] = np.where(keep.reshape(len(Q), -1),
                                     w.reshape(len(Q), -1), -1)
    return idx, tot / max(n, 1)


def pooled(idx, h):
    b = SIDE // GRID
    M = np.zeros((len(idx), SIDE * SIDE, h), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = 1.0
    return M.reshape(len(idx), GRID, b, GRID, b, h).max(axis=(2, 4)).reshape(len(idx), -1)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(SEED + 1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    P = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), E.EPS)   # 25 dims, no label
    print(f"{len(P):,} patches, dim {P.shape[1]} (label block deleted)", flush=True)

    W = E.train(P, H, K, 0.0, np.random.default_rng(SEED + 1))
    itr, e_tr = winners(W, Xtr)
    ite, e_te = winners(W, Xte)

    T1 = V.llr_table(itr, ytr, H, True)
    T0 = V.llr_table(itr, ytr, H, False)
    res = {"dim": int(P.shape[1]), "H": H, "K": K,
           "ink_rebuild_error": e_te,
           "LLR by 4x4 cell": float((V.llr_score(ite, T1, True).argmax(1) == yte).mean()),
           "LLR bag": float((V.llr_score(ite, T0, False).argmax(1) == yte).mean())}
    n = R.fit_net(pooled(itr, H), np.arange(len(itr)), None, ytr, 10, "softmax",
                  hidden=0, epochs=30)
    res["dense expert map (probe)"] = float(
        (R.predict_net(n, pooled(ite, H), np.arange(len(ite)), None).argmax(1) == yte).mean())
    res["with label block, same pipeline"] = {
        "LLR by 4x4 cell": 0.9423, "LLR bag": 0.5460, "dense expert map (probe)": 0.9717}
    res["seconds"] = round(time.time() - t0, 1)

    print()
    for k in ("LLR by 4x4 cell", "LLR bag", "dense expert map (probe)"):
        print(f"  {k:<26} no label {res[k]:.4f}   with label "
              f"{res['with label block, same pipeline'][k]:.4f}   "
              f"{res[k] - res['with label block, same pipeline'][k]:+.4f}")
    print(f"\n  mean ink rebuild error {e_te:.4f}")
    (OUT / "nolabel.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "weights_nolabel.npz", W=W.astype(np.float32))
    print(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
