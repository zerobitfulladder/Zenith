"""The whole MNIST pipeline, moved to Fashion-MNIST unchanged.

Worth running because Fashion is where this project's competition machinery
broke before: the fit gate went 0.94 -> 0.61 and the competition reversed sign,
with the right expert in the top 3 for 82% of images but first for only 60%.
Silhouette was not class.

Nothing is tuned for the new data. Same 5x5 patches, same 30 experts of 8
templates, same ink-only competition (lambda = 0), same counted table
log P(expert | cell, class) / P(expert | cell), same 4x4 cells.

References on Fashion from 2026-08-31: one layer teacher-free 0.6868, one layer
with a teacher 0.8278, the conv stack 0.7950, logistic on raw pixels 0.8512.
And on MNIST this pipeline gets 0.9423 counted, 0.9717 probed.

The interesting question is not whether the number drops -- it will, Fashion is
harder for everything. It is WHERE it drops. If position stops paying (the bag
and the by-cell numbers converge), that says Fashion classes differ by texture
rather than by arrangement, which is the opposite of digits.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E, vote as V, settle as S, readouts as R

OUT = HERE / "results"
DS = "fashion_mnist"
NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SIDE, GRID, NL = V.SIDE, V.GRID, 10


def winners(W, X):
    return np.concatenate([np.where(kp, f.argmax(1), -1) for f, kp in
                           [S.fits(W, X[a:a + 1000]) for a in range(0, len(X), 1000)]])


def report(yt, yp, tag):
    cm = np.zeros((NL, NL), int); np.add.at(cm, (yt, yp), 1)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    print(f"\n  {tag}   accuracy {tp.sum()/cm.sum():.4f}")
    print("     class          prec   rec    f1     n")
    for c in range(NL):
        print(f"   {NAMES[c]:>12}   {pr[c]:.3f}  {rc[c]:.3f}  {f1[c]:.3f}  {cm[c].sum():5d}")
    print(f"     macro          {pr.mean():.3f}  {rc.mean():.3f}  {f1.mean():.3f}")
    off = cm - np.diag(tp)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:6], off.shape))[0]
    print("     worst confusions:  " + "   ".join(
        f"{NAMES[a]}->{NAMES[b]} {off[a,b]}" for a, b in top))
    return {"acc": float(tp.sum() / cm.sum()), "f1": f1.tolist(),
            "confusion": cm.tolist()}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    rng = np.random.default_rng(E.SEED + 1)
    U, L = E.sample(Xtr, ytr, rng, E.PER_IMG)
    print(f"{DS}: {len(U):,} patches, {len(Xtr)} train / {len(Xte)} test", flush=True)

    W = E.train(E.join(U, L), E.H, E.K, 0.0, np.random.default_rng(E.SEED + 1))
    itr, ite = winners(W, Xtr), winners(W, Xte)
    T1, T0 = V.llr_table(itr, ytr, E.H, True), V.llr_table(itr, ytr, E.H, False)
    p_cell = V.llr_score(ite, T1, True).argmax(1)
    p_bag = V.llr_score(ite, T0, False).argmax(1)

    res = {"dataset": DS}
    res["LLR by cell"] = report(yte, p_cell, "LLR by 4x4 cell")
    res["LLR bag"] = {"acc": float((p_bag == yte).mean())}

    def pooled(idx):
        b = SIDE // GRID
        M = np.zeros((len(idx), SIDE * SIDE, E.H), np.float32)
        r, c = np.nonzero(idx >= 0); M[r, c, idx[r, c]] = 1.0
        return M.reshape(len(idx), GRID, b, GRID, b, E.H).max(axis=(2, 4)).reshape(len(idx), -1)
    n = R.fit_net(pooled(itr), np.arange(len(itr)), None, ytr, NL, "softmax",
                  hidden=0, epochs=30)
    res["dense probe"] = {"acc": float(
        (R.predict_net(n, pooled(ite), np.arange(len(ite)), None).argmax(1) == yte).mean())}
    w, b, g, _ = S.simstats(pooled(ite)[:600], yte[:600])
    res["code"] = {"within": w, "between": b, "gap": g}

    print(f"\n  LLR bag (no position)   {res['LLR bag']['acc']:.4f}")
    print(f"  LLR by 4x4 cell         {res['LLR by cell']['acc']:.4f}")
    print(f"  dense probe             {res['dense probe']['acc']:.4f}")
    print(f"  code gap                {g:+.4f}")

    # misclassified gallery
    bad = np.nonzero(p_cell != yte)[0][:24]
    fig, ax = plt.subplots(3, 8, figsize=(13, 5.6))
    for a, i in zip(ax.ravel(), bad):
        a.imshow(Xte[i], cmap="gray_r"); a.axis("off")
        a.set_title(f"{NAMES[yte[i]]}\n-> {NAMES[p_cell[i]]}", fontsize=7, color="tab:red")
    for a in ax.ravel()[len(bad):]:
        a.axis("off")
    plt.suptitle(f"{DS}: misclassified by the counted table  "
                 f"(accuracy {res['LLR by cell']['acc']:.4f})", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "misclassified.png", dpi=125); plt.close()

    res["reference_2026_08_31"] = {"one layer teacher-free": 0.6868,
                                   "one layer with teacher": 0.8278,
                                   "conv stack": 0.7950, "logistic on pixels": 0.8512}
    res["reference_mnist_same_pipeline"] = {"LLR bag": 0.5460, "LLR by cell": 0.9423,
                                            "dense probe": 0.9717, "code gap": 0.1189}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "fashion.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "weights.npz", W=W.astype(np.float32))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
