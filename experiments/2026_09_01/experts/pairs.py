"""Two extensions to the counted table: overlapping class codes, and pairs.

    CLASS CODE   the table is currently indexed by a one-hot label, so Shirt and
                 Pullover are as unrelated as Shirt and Sandal, and evidence for
                 "upper-body garment" gets split four ways before a hard winner
                 is picked. Instead: cluster the CLASS MEANS -- what the classes
                 actually look like, not what they are called -- into a tree, and
                 give each class a code of its leaf plus all its ancestors.
                 Related classes then share units and share evidence.

                 Derived from the data. Nothing is hand-assigned, so if the
                 upper-body group does not fall out on its own, the premise is
                 wrong and we will see it in the tree.

    PAIRS        the table counts single features. A linear probe implicitly
                 uses how features CO-VARY, which is the residual we measured
                 (3.3 points on MNIST after the weighting closed the rest).
                 So count pairs too -- template i at a position, template j at a
                 nearby offset -- which is also exactly the lateral compatibility
                 table from the cortex discussion, unmeasured until now.

                 6 offsets x 180 x 180 x 10 = 1.9M counts. Cheap, and estimated
                 from ~2,000 pair observations per image rather than the ~4
                 samples-per-dimension a full covariance would get.

2 x 2: one-hot vs tree, singles vs singles+pairs.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, single as SG

OUT = Path(__file__).resolve().parent / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
GRID, NL, ALPHA, EPS = 6, 10, 1.0, 1e-12
OFFSETS = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 1), (1, -1)]
SIDE = S.SIDE


# ---------------------------------------------------------------- class tree
def class_tree(F, y):
    """Average-linkage tree over the class means. Returns codes (NL, U)."""
    M = np.stack([F[y == c].mean(0) for c in range(NL)])
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), EPS)
    sim = M @ M.T
    groups = {c: [c] for c in range(NL)}
    merges = []
    while len(groups) > 1:
        ks = list(groups)
        best, bi, bj = -2, None, None
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                s = np.mean([sim[x, z] for x in groups[ks[a]] for z in groups[ks[b]]])
                if s > best:
                    best, bi, bj = s, ks[a], ks[b]
        merges.append((groups[bi] + groups[bj], best))
        groups[NL + len(merges) - 1] = groups.pop(bi) + groups.pop(bj)
    codes = np.zeros((NL, NL + len(merges)), np.float32)
    codes[np.arange(NL), np.arange(NL)] = 1.0                    # leaf units
    for i, (members, _) in enumerate(merges):
        codes[members, NL + i] = 1.0                             # ancestor units
    return codes, merges


# ---------------------------------------------------------------- tables
def single_table(idx, U, k, g):
    cell = B.cellmap(g)
    N = np.zeros((k, g * g, U.shape[1]))
    r, c = np.nonzero(idx >= 0)
    np.add.at(N, (idx[r, c], cell[c]), U[r])
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * k)
    pm = ((N.sum(2, keepdims=True) + ALPHA * U.shape[1])
          / (N.sum((0, 2), keepdims=True) + ALPHA * k * U.shape[1]))
    return (np.log(pc) - np.log(pm)).astype(np.float32)


def pair_index(idx, k):
    """(rows, flat pair index) for every (template i, template j, offset)."""
    n = len(idx)
    G = idx.reshape(n, SIDE, SIDE)
    rows, flat = [], []
    for o, (dy, dx) in enumerate(OFFSETS):
        A = G[:, max(0, -dy):SIDE - max(0, dy), max(0, -dx):SIDE - max(0, dx)]
        Bm = G[:, max(0, dy):SIDE - max(0, -dy), max(0, dx):SIDE - max(0, -dx)]
        m = (A >= 0) & (Bm >= 0)
        r, _, _ = np.nonzero(m)
        rows.append(r)
        flat.append(((A[m].astype(np.int64) * k + Bm[m]) * len(OFFSETS)) + o)
    return np.concatenate(rows), np.concatenate(flat)


def pair_table(rows, flat, U, k):
    P = k * k * len(OFFSETS)
    N = np.zeros((P, U.shape[1]))
    for u in range(U.shape[1]):
        w = U[rows, u]
        nz = w > 0
        N[:, u] = np.bincount(flat[nz], weights=w[nz], minlength=P)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * P)
    pm = ((N.sum(1, keepdims=True) + ALPHA * U.shape[1])
          / (N.sum() + ALPHA * P * U.shape[1]))
    return (np.log(pc) - np.log(pm)).astype(np.float32)


def score(idx, Ts, Tp, k, g, n):
    cell = B.cellmap(g)
    out = np.zeros((n, Ts.shape[2]), np.float32)
    Tc = Ts[:, cell, :]
    P = np.arange(SIDE * SIDE)
    for a in range(0, n, 500):
        b = idx[a:a + 500]; v = b >= 0
        out[a:a + len(b)] = (Tc[np.where(v, b, 0), P[None, :]] * v[..., None]).sum(1)
    if Tp is not None:
        rows, flat = pair_index(idx, k)
        for u in range(Tp.shape[1]):
            out[:, u] += np.bincount(rows, weights=Tp[flat, u], minlength=n)
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    W = np.load(OUT / ("coadapt.npz" if DS == "mnist" else "../../stack/results/"
                       "full_stack_fashion_mnist.npz"))
    W1 = (W["W"] if "W" in W.files else W["W1"]).astype(np.float64)
    k = len(W1)
    itr, ite = SG.winners(W1, Xtr), SG.winners(W1, Xte)
    Ftr = np.zeros((len(itr), k), np.float32)                    # bag, for the tree
    r, c = np.nonzero(itr >= 0)
    np.add.at(Ftr, (r, itr[r, c]), 1.0)

    onehot = np.eye(NL, dtype=np.float32)
    tree, merges = class_tree(Ftr, ytr)
    names = (["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
              "Sneaker", "Bag", "Ankle boot"] if "fashion" in DS
             else [str(i) for i in range(NL)])
    print(f"{DS}: {k} L1 templates.  class tree, merges in order:")
    for members, s in merges[:6]:
        print(f"    {[names[m] for m in members]}   similarity {s:.3f}")
    print(f"  code length {tree.shape[1]} units, {tree.sum(1).mean():.1f} active per class\n",
          flush=True)

    res = {}
    # pairs are ~6x more numerous than singles and maximally redundant (adjacent
    # patches share 4 of 5 pixels), so they need a weight rather than a free vote.
    Ts1 = single_table(itr, onehot[ytr], k, GRID)
    rows, flat = pair_index(itr, k)
    Tp1 = pair_table(rows, flat, onehot[ytr], k)
    ev_s = score(ite, Ts1, None, k, GRID, len(ite))
    ev_p = score(ite, np.zeros_like(Ts1), Tp1, k, GRID, len(ite))
    print("  PAIR WEIGHT sweep (one-hot classes)", flush=True)
    for wp in (0.0, 0.02, 0.05, 0.15, 0.5, 1.0):
        acc = float(((ev_s + wp * ev_p).argmax(1) == yte).mean())
        res[f"pairw={wp}"] = acc
        print(f"    pair weight {wp:<5} {acc:.4f}", flush=True)

    # the tree: ancestors as a PRIOR on top of the leaf, not averaged with it,
    # because a shared unit contributes identically to every class sharing it
    Tst = single_table(itr, tree[ytr], k, GRID)
    ev_t = score(ite, Tst, None, k, GRID, len(ite))
    leaf = ev_t[:, :NL]
    anc = tree[:, NL:]
    print("  ANCESTOR WEIGHT sweep (tree classes)", flush=True)
    for wa in (0.0, 0.1, 0.3, 1.0):
        cls = leaf + wa * (ev_t[:, NL:] @ anc.T) / np.maximum(anc.sum(1), 1)[None, :]
        acc = float((cls.argmax(1) == yte).mean())
        res[f"ancw={wa}"] = acc
        print(f"    ancestor weight {wa:<5} {acc:.4f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    res["tree"] = [[names[m] for m in mm] for mm, _ in merges]
    (OUT / f"pairs_{DS}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
