"""Gathering what the experts said at every position into one class call.

Each expert wins some patches and, for each, rebuilds the label block -- its
OPINION about the class of the digit that patch came from. Per patch that
opinion is nearly worthless (0.1162 against 0.10 chance). There are ~576 of them
per image. The question is entirely how to add them up.

Six rules on the same trained experts, all read with the label blank:

    1 sum of rebuilds     what experts.py did. Every patch contributes its
                          opinion, so five hundred near-flat votes drown the
                          few diagnostic ones.
    2 LLR bag             log P(expert h | class c) / P(expert h), summed.
                          A patch that occurs equally in every class scores
                          log 1 = 0 and stays silent. Position-free.
    3 LLR by cell         the same counts conditioned on a coarse 4x4 position,
                          so "this stroke HERE" can mean something.
    4 LLR top-k           only the k most diagnostic patches vote at all.
    5 dense expert map    pooled one-hot over experts, 4x4 x H, linear readout.
                          The experts' answers gathered densely and resolved
                          together rather than by a fixed formula.
    6 dense opinion map   pooled label-opinions, 4x4 x 10, linear readout.

Plus a reference: the same dense readout over yesterday's 64 k-means templates,
which asks whether 30 competing experts are a better vocabulary than 64
competing templates or a worse one.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E
import readouts as R

OUT = HERE / "results"
GRID, ALPHA = 4, 1.0                        # 4x4 cells of 6x6 positions; Laplace
SIDE = 28 - E.PS + 1                        # 24 positions a side
NL = 10


def encode(W, X, chunk=64):
    """Per image: winning expert at every position, its opinion, pooled maps."""
    n, h = len(X), len(W)
    idx = np.full((n, SIDE * SIDE), -1, np.int16)
    sumreb = np.zeros((n, NL), np.float32)
    emap = np.zeros((n, GRID, GRID, h), np.float32)
    omap = np.zeros((n, GRID, GRID, NL), np.float32)
    b = SIDE // GRID
    for a in range(0, n, chunk):
        Q, keep = E.patches(X[a:a + chunk])
        B = E.join(Q.reshape(-1, E.PS * E.PS))
        S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
        Rb = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
        ink = ((B[None] - Rb)[:, :, :E.PS * E.PS] ** 2).sum(-1)
        win = ink.argmin(0)
        op = Rb[win, np.arange(len(B)), E.PS * E.PS:]
        m = len(Q)
        win = win.reshape(m, -1); op = op.reshape(m, -1, NL)
        k = keep.reshape(m, -1)
        idx[a:a + m] = np.where(k, win, -1)
        sumreb[a:a + m] = (op * k[..., None]).sum(1)
        oh = np.zeros((m, SIDE * SIDE, h), np.float32)
        r, c = np.nonzero(k)
        oh[r, c, win[r, c]] = 1.0
        emap[a:a + m] = oh.reshape(m, SIDE, SIDE, h).reshape(
            m, GRID, b, GRID, b, h).max(axis=(2, 4))
        omap[a:a + m] = (op * k[..., None]).reshape(m, SIDE, SIDE, NL).reshape(
            m, GRID, b, GRID, b, NL).sum(axis=(2, 4))
    return idx, sumreb, emap.reshape(n, -1), omap.reshape(n, -1)


def llr_table(idx, y, h, by_cell=False):
    """log P(expert | class) - log P(expert). Silent where the expert is generic."""
    b = SIDE // GRID
    cell = ((np.arange(SIDE * SIDE) // SIDE) // b) * GRID + ((np.arange(SIDE * SIDE) % SIDE) // b)
    shape = (h, GRID * GRID, NL) if by_cell else (h, 1, NL)
    N = np.zeros(shape)
    r, c = np.nonzero(idx >= 0)
    np.add.at(N, (idx[r, c], cell[c] if by_cell else 0, y[r]), 1.0)
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * h)          # P(h | cell, c)
    pm = (N.sum(2, keepdims=True) + ALPHA * NL) / \
         (N.sum((0, 2), keepdims=True) + ALPHA * h * NL)              # P(h | cell)
    return np.log(pc) - np.log(pm)


def llr_score(idx, T, by_cell, topk=0):
    b = SIDE // GRID
    cell = ((np.arange(SIDE * SIDE) // SIDE) // b) * GRID + ((np.arange(SIDE * SIDE) % SIDE) // b)
    out = np.zeros((len(idx), NL))
    for i in range(len(idx)):
        p = np.nonzero(idx[i] >= 0)[0]
        if not len(p):
            continue
        v = T[idx[i, p], cell[p] if by_cell else 0]                   # (m, 10)
        if topk and len(v) > topk:
            v = v[np.argsort(-(v.max(1) - v.min(1)))[:topk]]
        out[i] = v.sum(0)
    return out


def report(yt, yp, name):
    cm = np.zeros((NL, NL), int)
    np.add.at(cm, (yt, yp), 1)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    print(f"\n  {name}   accuracy {tp.sum()/cm.sum():.4f}")
    print("     class  prec   rec    f1     n")
    for c in range(NL):
        print(f"       {c}   {pr[c]:.3f}  {rc[c]:.3f}  {f1[c]:.3f}  {cm[c].sum():5d}")
    print(f"     macro  {pr.mean():.3f}  {rc.mean():.3f}  {f1.mean():.3f}")
    print("     confusion (row = true)")
    print("        " + "".join(f"{c:>6}" for c in range(NL)))
    for c in range(NL):
        print(f"      {c} " + "".join(f"{v:>6}" for v in cm[c]))
    return {"acc": float(tp.sum() / cm.sum()), "precision": pr.tolist(),
            "recall": rc.tolist(), "f1": f1.tolist(), "confusion": cm.tolist()}


def linear(A, ya, B, yb):
    n = R.fit_net(A, np.arange(len(A)), None, ya, NL, "softmax", hidden=0, epochs=30)
    return R.predict_net(n, B, np.arange(len(B)), None).argmax(1)


def main():
    t0 = time.time()
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()
    print(f"{h} experts x {W.shape[1]} templates, {len(Xtr)} train / {len(Xte)} test",
          flush=True)
    itr, str_, etr, otr = encode(W, Xtr)
    ite, ste, ete, ote = encode(W, Xte)
    print(f"  encoded ({time.time()-t0:.0f}s)", flush=True)

    res, preds = {}, {}
    preds["1 sum of rebuilds"] = ste.argmax(1)
    T0 = llr_table(itr, ytr, h, False)
    preds["2 LLR bag"] = llr_score(ite, T0, False).argmax(1)
    T1 = llr_table(itr, ytr, h, True)
    preds["3 LLR by 4x4 cell"] = llr_score(ite, T1, True).argmax(1)
    preds["4 LLR by cell, top-40"] = llr_score(ite, T1, True, topk=40).argmax(1)
    preds["5 dense expert map"] = linear(etr, ytr, ete, yte)
    preds["6 dense opinion map"] = linear(otr, ytr, ote, yte)

    W1 = np.load(HERE.parents[1] / "2026_08_31/kmeans/results/km_mnist.npz")["64"]
    import scenes as S
    def kmap(X, chunk=128):
        b = SIDE // GRID
        out = np.zeros((len(X), GRID, GRID, 64), np.float32)
        for a in range(0, len(X), chunk):
            Q, keep = E.patches(X[a:a + chunk])
            wn = np.einsum('ipd,kd->ipk', Q, W1.astype(np.float64)).argmax(-1)
            M = np.zeros((len(Q), SIDE * SIDE, 64), np.float32)
            r, c = np.nonzero(keep)
            M[r, c, wn[r, c]] = 1.0
            out[a:a + len(Q)] = M.reshape(len(Q), SIDE, SIDE, 64).reshape(
                len(Q), GRID, b, GRID, b, 64).max(axis=(2, 4))
        return out.reshape(len(X), -1)
    preds["ref: dense k-means-64 map"] = linear(kmap(Xtr), ytr, kmap(Xte), yte)

    print("\n" + "=" * 46)
    for k, p in preds.items():
        print(f"  {k:<28} {float((p == yte).mean()):.4f}")
    print("=" * 46)
    best = max(preds, key=lambda k: (preds[k] == yte).mean())
    for k in ("1 sum of rebuilds", "3 LLR by 4x4 cell", best):
        if k not in res:
            res[k] = report(yte, preds[k], k)
    res["all"] = {k: float((p == yte).mean()) for k, p in preds.items()}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "vote.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
