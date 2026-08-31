"""Layer two trained jointly with the label. Its winners vote. No layer three.

    L2 input at each position:  [ 3x3 window of L1 codes ; label * gain ]

Every position sees the same label, so a template becomes a (local shape,
label distribution) pair. At inference only the image half is written; the
pursuit runs against the templates restricted to those cells, and each
winner's label block is a vote. The 9 positions x k winners vote and the
argmax wins.

The question: is a LEARNED label distribution better than a COUNTED one?
The tally gets P(label | template, position) by counting after the fact on
label-blind templates. This gets it by letting the label shape which
templates exist at all. Same read, different origin.

Layer one is identical and label-blind in both, and is trained once and
shared across the rho sweep.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sparse_column"))
sys.path.insert(0, str(HERE.parent / "place_code"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_30" / "stack3"))
from stack3 import unit_rows, l2_windows, GRID, WIN, L2_GRID, EPS   # noqa
from sparse_column import SparseColumn                              # noqa
from sparse_stack import SparseStack                                # noqa
from run_sparse_stack import load, K1, K2, ETA1, EPOCHS, SEED, onehot  # noqa

KMAX, RHOS, ALPHA = 4, [0.15, 0.3, 0.6], 1.0
IMG = WIN * WIN * K1                    # 576 image cells at layer two
NPOS = L2_GRID * L2_GRID                # 9 positions
BOARD = {"L1+L2 codes + tally (label-blind L2)": 0.8672,
         "L1+L2 + layer three (matching)": 0.7494,
         "dense + layer three": 0.9232}


def join(U, L, rho):
    """[window ; label * gain], unit rows. No centring."""
    V = np.zeros((len(U), IMG + 10))
    V[:, :IMG] = U
    g = rho * np.linalg.norm(U, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
    V[:, IMG:] = L * g[:, None]
    return unit_rows(V)[0]


def tally_read(Htr, ytr, Hte, yte):
    """The counting baseline, for whatever code is handed in."""
    T = (Htr > 0).astype(np.float64).T @ onehot(ytr)
    P = np.log((T + ALPHA) / (T.sum(1, keepdims=True) + 10 * ALPHA))
    return float((((Hte > 0).astype(np.float64) @ P).argmax(1) == yte).mean())


def vote(col, Uim, n, floor=0.25):
    """Write the image half only; the winners' label blocks vote."""
    sub = col.W[:, :IMG]
    nrm = np.linalg.norm(sub, axis=1)
    keep = nrm > floor * nrm.max()                 # ignore templates with
    idx = np.nonzero(keep)[0]                      # almost nothing here
    Wr = sub[idx] / nrm[idx, None]
    probe = SparseColumn(len(idx), IMG, kmax=KMAX, rng=np.random.default_rng(0))
    probe.W, probe.n_boot = Wr, len(idx)
    C = probe.pursue(Uim)[0]                       # (n*9, kept)

    Lb = np.maximum(col.W[idx][:, IMG:], 0.0)
    Lb = Lb / np.maximum(Lb.sum(1, keepdims=True), EPS)     # distributions
    out = {}
    for tag, Q, M in (("mass", C, Lb),
                      ("presence", (C > 0).astype(np.float64), Lb),
                      ("presence/log", (C > 0).astype(np.float64), np.log(Lb + 1e-9))):
        out[tag] = (Q @ M).reshape(n, NPOS, 10).sum(1)
    return out, float(nrm.std() / nrm.mean()), int(keep.sum())


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    rng = np.random.default_rng(SEED)

    # ---- layer one: identical, label-blind, trained once ----------------
    cs = SparseStack(K1, K2, KMAX, ETA1, rng)
    for _ in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 256):
            U = cs._l1_in(Xtr[order[s:s + 256]])
            cs.l1.learn(U[np.linalg.norm(U, axis=1) > EPS])
        cs.l1.revive(cs._l1_in(Xtr[rng.choice(len(Xtr), 64, replace=False)]))
    print(f"layer one trained ({time.time()-t0:.0f}s), label-blind")

    def field(X, chunk=1000):
        return np.concatenate([cs.l1_field(X[i:i + chunk]).reshape(len(X[i:i + chunk]), -1)
                               for i in range(0, len(X), chunk)])

    def windows(X, chunk=1000):
        out = []
        for i in range(0, len(X), chunk):
            F = cs.l1_field(X[i:i + chunk])
            out.append(unit_rows(l2_windows(F))[0])
        return np.concatenate(out)

    # ---- free control: does layer two earn its place? -------------------
    res = {"L1 codes + tally": tally_read(field(Xtr), ytr, field(Xte), yte)}
    print(f"  L1 codes + tally (no layer two)   {res['L1 codes + tally']:.4f}")

    Utr, Ute = windows(Xtr), windows(Xte)
    Ltr = np.repeat(onehot(ytr), NPOS, axis=0)
    print(f"  layer-two windows ready ({time.time()-t0:.0f}s)")

    # ---- the experiment -------------------------------------------------
    res["labeled_l2"] = {}
    for rho in RHOS:
        r2 = np.random.default_rng(SEED + 1)
        col = SparseColumn(K2, IMG + 10, kmax=KMAX, eta=ETA1, rng=r2)
        J = join(Utr, Ltr, rho)
        for _ in range(EPOCHS):
            o = r2.permutation(len(J))
            for s in range(0, len(o), 256):
                col.learn(J[o[s:s + 256]])
        V, spread, kept = vote(col, Ute, len(Xte))
        acc = {t: float((v.argmax(1) == yte).mean()) for t, v in V.items()}
        # what the label-trained L2 codes look like to a plain tally
        Ctr = vote(col, Utr, len(Xtr))[0]
        res["labeled_l2"][str(rho)] = {
            "accuracy": acc, "best": max(acc, key=acc.get),
            "best_accuracy": acc[max(acc, key=acc.get)],
            "image_half_norm_spread": round(spread, 4), "templates_kept": kept}
        print(f"  rho={rho:<5} vote {acc}  "
              f"(best {acc[max(acc, key=acc.get)]:.4f})  "
              f"img-half norm spread {spread:.3f}  kept {kept}/{K2}  "
              f"[{time.time()-t0:.0f}s]")

    print("\nagainst the board:")
    for k, v in BOARD.items():
        print(f"  {k:<40} {v:.4f}")
    (HERE / "results" / "labeled_l2.json").write_text(json.dumps(
        {"kmax": KMAX, "rhos": RHOS, "results": res, "board": BOARD,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
