"""One hypercolumn per patch. 180 templates. They compete; one learns.

The multi-template expert is dropped. Today's sweep said why: at fixed budget,
120 experts of 2 beat 15 of 16 by 4.8 points, and the subspace fraction tracked
it monotonically. A template that spans a subspace has no identity; a template
that IS a direction does. So take that to its end -- K = 1, and every template
competes on its own.

Two versions, identical apart from what the template contains:

    labelled     the template is [ patch 25 ; label 10 ], the winner is chosen
                 by INK alone (never by the label -- reading has no label), and
                 the winner rotates toward the whole thing, label included
    unlabelled   the template is the patch, 25 numbers, nothing else

The label cannot pick the winner in either case, so the two differ only in
whether the label bends the ink directions the templates settle on. Earlier at
H=30 that was worth about a point, in the label's favour, on a single seed.

Same readout both times: the counted table at the 6x6 grid. Plus the two ends
of the readout ladder, because the interesting question is no longer accuracy
but whether the code is a blob a class average can find.
"""

import json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, blank as B, settle as S, readouts as R

OUT = Path(__file__).resolve().parent / "results"
H, GRID, NL, EPS = 180, 6, 10, 1e-12


def winners(W, X, chunk=200):
    """Winner per position, for K=1 templates of either width.

    With one template per unit the rebuild is just S * w, so the ink error is
    ||b||^2 - 2 S (b.w_ink) + S^2 ||w_ink||^2 -- no need to build the rebuild
    array, which at 180 templates would be gigabytes.
    """
    d = W.shape[2]
    Wf = W[:, 0, :]
    Wi = Wf[:, :25]
    wn = (Wi ** 2).sum(1)
    idx = np.full((len(X), S.SIDE * S.SIDE), -1, np.int16)
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        P = Q.reshape(-1, E.PS * E.PS)
        Bv = E.join(P) if d > 25 else P / np.maximum(
            np.linalg.norm(P, axis=1, keepdims=True), EPS)
        Sc = Bv @ Wf.T
        Si = Bv[:, :25] @ Wi.T
        err = (Bv[:, :25] ** 2).sum(1, keepdims=True) - 2 * Sc * Si + Sc ** 2 * wn[None, :]
        w = err.argmin(1).reshape(len(Q), -1)
        idx[a:a + len(Q)] = np.where(keep.reshape(len(Q), -1), w, -1)
    return idx


def draw(W, cnt, ld, tag, title):
    order = np.argsort(-cnt)
    rows, cols = 12, 15
    tile = np.full((rows * 6 - 1, cols * 6 - 1), np.nan)
    for a, j in enumerate(order[:rows * cols]):
        r, c = divmod(a, cols)
        tile[r * 6:r * 6 + 5, c * 6:c * 6 + 5] = W[j, 0, :25].reshape(5, 5)
    d = ld[order] / np.maximum(ld[order].sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [3, 1]})
    v = np.nanmax(np.abs(tile))
    ax[0].imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax[0].set_title(f"{title} -- {H} templates, busiest first", fontsize=11)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(d, cmap="magma", aspect="auto", interpolation="nearest")
    ax[1].set_title("classes each template won", fontsize=10)
    ax[1].set_xticks(range(NL)); ax[1].set_yticks([])
    plt.tight_layout(); plt.savefig(OUT / f"single_{tag}.png", dpi=120); plt.close()


def run(Vtr, Xtr, ytr, Xte, yte, tag, title):
    t0 = time.time()
    cache = OUT / f"single_{tag}.npz"
    if cache.exists():
        W = np.load(cache)["W"].astype(np.float64)
        print(f"  {title}: reusing cached weights", flush=True)
    else:
        W = E.train(Vtr, H, 1, 0.0, np.random.default_rng(E.SEED + 1))
    itr, ite = winners(W, Xtr), winners(W, Xte)
    T = B.table(itr, ytr, H, GRID)
    p = B.score(ite, T, GRID).argmax(1)
    acc = float((p == yte).mean())

    cnt = np.zeros(H); ld = np.zeros((H, NL))
    r, c = np.nonzero(ite >= 0)
    np.add.at(cnt, ite[r, c], 1.0); np.add.at(ld, (ite[r, c], yte[r]), 1.0)
    live = cnt > 0
    d = ld / np.maximum(ld.sum(1, keepdims=True), 1)

    def pooled(idx):
        b = S.SIDE // 4
        M = np.zeros((len(idx), S.SIDE * S.SIDE, H), np.float32)
        rr, cc = np.nonzero(idx >= 0); M[rr, cc, idx[rr, cc]] = 1.0
        return M.reshape(len(idx), 4, b, 4, b, H).max(axis=(2, 4)).reshape(len(idx), -1)
    Atr, Ate = pooled(itr), pooled(ite)
    M = Atr[:, None] if False else np.stack([Atr[ytr == c].mean(0) for c in range(NL)])
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), EPS)
    An = Ate / np.maximum(np.linalg.norm(Ate, axis=1, keepdims=True), EPS)
    nm = float(((An @ M.T).argmax(1) == yte).mean())
    net = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax", hidden=0, epochs=30)
    lin = float((R.predict_net(net, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())
    w, bb, g, _ = S.simstats(Ate[:600], yte[:600])

    draw(W, cnt, ld, tag, title)
    row = {"acc_counted_table": acc, "nearest_mean": nm, "linear": lin,
           "ladder_gap": lin - nm, "code_gap": g, "class_purity": float(d.max(1)[live].mean()),
           "live": int(live.sum()), "seconds": round(time.time() - t0, 1)}
    if Vtr.shape[1] > 25:
        row["label_block_energy"] = float((W[:, 0, 25:] ** 2).sum(-1).mean())
    print(f"  {title:<12} table {acc:.4f}   nearest-mean {nm:.4f}   linear {lin:.4f}   "
          f"gap {lin-nm:.4f}   purity {row['class_purity']:.3f}   live {row['live']}/{H}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(OUT / f"single_{tag}.npz", W=W.astype(np.float32), counts=cnt,
                        lab_dist=ld)
    return row


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(E.SEED + 1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    print(f"{len(U):,} patches, one hypercolumn, {H} competing templates\n", flush=True)
    res = {}
    res["labelled"] = run(E.join(U, L), Xtr, ytr, Xte, yte, "labelled", "WITH label")
    P = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), EPS)
    res["unlabelled"] = run(P.astype(np.float64), Xtr, ytr, Xte, yte,
                            "unlabelled", "NO label")
    res["reference"] = {"H120_K2": 0.9613, "H30_K8_grid6": 0.9507,
                        "shaped nearest-mean (H30)": 0.9237,
                        "shaped linear (H30)": 0.9687}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "single.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
