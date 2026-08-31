"""Two rungs, both plain k-means, identity only.

L1  the 5x5 patch codebook already trained by km.py, slid at stride 1.
    At each of the 24x24 positions it emits a one-hot over its K1 templates,
    carrying that patch's contrast as the magnitude. Flat patches emit nothing.

L2  one layer over the WHOLE map -- 24 x 24 x K1 numbers -- joined with the
    label at matched energy, trained by the same online spherical k-means.

    image in, label blank  -> nearest L2 template -> read its label half
    label in, image blank  -> nearest L2 template -> its map half, painted back
                              through the L1 templates

The map is stored as indices plus magnitudes and densified per batch; storing it
dense would be 2.9 GB for the training set.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km, joint

OUT = Path(__file__).resolve().parent / "results"
K1, K2, EPOCHS, ETA_MIN, SEED = "64", 100, 12, 0.01, 0
DS = "mnist"
SIDE = 28 - km.PS + 1                      # 24 positions a side
EPS = 1e-12


def l1_map(W1, X, chunk=256):
    """Winner index and magnitude at every position. -1 where the patch is flat."""
    X = X.reshape(-1, 28, 28)
    idx = np.full((len(X), SIDE * SIDE), -1, np.int16)
    mag = np.zeros((len(X), SIDE * SIDE), np.float32)
    for a in range(0, len(X), chunk):
        Q, keep, norm = km.prep(km.patches(X[a:a + chunk]))
        s = np.einsum("ipd,kd->ipk", Q, W1)
        w = s.argmax(-1)
        idx[a:a + len(Q)] = np.where(keep, w, -1)
        mag[a:a + len(Q)] = np.where(keep, norm, 0.0)
    return idx, mag


def densify(idx, mag, k1, y=None, rho=1.0):
    """(n, 24*24*k1 [+10]) centred and normalised."""
    n, p = idx.shape
    V = np.zeros((n, p * k1 + 10))
    r, c = np.nonzero(idx >= 0)
    V[r, c * k1 + idx[r, c]] = mag[r, c]
    if y is not None:
        e = np.linalg.norm(V[:, :p * k1], axis=1)
        L = np.zeros((n, 10)); L[np.arange(n), y] = 1.0
        V[:, p * k1:] = L * (rho * e)[:, None]
    return joint.cn(V)


def paint(M, W1, k1):
    """An L2 template's map half -> a 28x28 picture, through the L1 codebook."""
    M = M.reshape(SIDE * SIDE, k1)
    sel, mag = M.argmax(1), M.max(1)
    canvas, cnt = np.zeros((28, 28)), np.zeros((28, 28))
    for p in range(SIDE * SIDE):
        if mag[p] <= 0:
            continue
        r, c = divmod(p, SIDE)
        canvas[r:r + km.PS, c:c + km.PS] += W1[sel[p]].reshape(km.PS, km.PS) * mag[p]
        cnt[r:r + km.PS, c:c + km.PS] += 1
    return canvas / np.maximum(cnt, 1)


def main():
    t0 = time.time()
    W1 = np.load(OUT / f"km_{DS}.npz")[K1].astype(np.float64)
    k1 = len(W1)
    Xtr, ytr, Xte, yte = joint.load(DS)
    itr, mtr = l1_map(W1, Xtr)
    ite, mte = l1_map(W1, Xte)
    live = float((itr >= 0).mean())
    print(f"{DS}: L1 K={k1}, map {SIDE}x{SIDE}x{k1} = {SIDE*SIDE*k1} numbers, "
          f"{live*100:.0f}% of positions active", flush=True)

    rng = np.random.default_rng(SEED + 1)
    seed = rng.choice(len(Xtr), 3000, replace=False)
    W = joint.kmeanspp(densify(itr[seed], mtr[seed], k1, ytr[seed]), K2, rng)
    n = np.zeros(K2)
    order = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        rng.shuffle(order)
        for s in range(0, len(order), 512):
            b = order[s:s + 512]
            B = densify(itr[b], mtr[b], k1, ytr[b])
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], ETA_MIN) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
        print(f"  epoch {ep+1}/{EPOCHS}", flush=True)

    P = SIDE * SIDE * k1
    lab = W[:, P:].argmax(1)
    acc, tw = [], []
    for s in range(0, len(Xte), 1000):
        B = densify(ite[s:s + 1000], mte[s:s + 1000], k1)
        w = (B @ W.T).argmax(1)
        tw.append(w); acc.append(lab[w] == yte[s:s + 1000])
    acc = float(np.concatenate(acc).mean())
    pur = []
    for s in range(0, len(Xtr), 2000):
        B = densify(itr[s:s + 2000], mtr[s:s + 2000], k1, ytr[s:s + 2000])
        pur.append((B @ W.T).argmax(1))
    pur = np.concatenate(pur)
    purity = np.mean([np.bincount(ytr[pur == j], minlength=10).max() /
                      max((pur == j).sum(), 1) for j in range(K2) if (pur == j).any()])
    print(f"\n  classification (image in, label blank) {acc:.4f}")
    print(f"  dead templates {int((n == 0).sum())}/{K2}   purity {purity:.3f}")
    print(f"  templates per class {np.bincount(lab, minlength=10).tolist()}")

    # ---- label in, image out ------------------------------------------------
    V = np.zeros((10, P + 10)); V[np.arange(10), P + np.arange(10)] = 1.0
    Sg = joint.cn(V) @ W.T
    fig, axes = plt.subplots(10, 8, figsize=(8.4, 10.6))
    for d in range(10):
        for j, t in enumerate(np.argsort(Sg[d])[::-1][:8]):
            a = axes[d][j]
            a.imshow(paint(W[t, :P], W1, k1), cmap="gray")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{lab[t]}·{int(n[t])}", fontsize=5, pad=1.2)
        axes[d][0].set_ylabel(f"asked {d}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.suptitle("two rungs: label in -> L2 names which L1 template belongs "
                 "where -> painted back", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.95, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / "stack_generate.png", dpi=140); plt.close(fig)

    # ---- every L2 template, painted into pixels; and the L1 codebook -------
    o = np.argsort(n)[::-1]
    fig, axes = plt.subplots(10, 10, figsize=(10, 10.4))
    for i, a in enumerate(np.ravel(axes)):
        a.imshow(paint(W[o[i], :P], W1, k1), cmap="gray")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{lab[o[i]]}·{int(n[o[i]])}", fontsize=5, pad=1.2)
    fig.suptitle(f"all {K2} L2 templates painted back through L1, most used first",
                 fontsize=9)
    fig.subplots_adjust(left=.005, right=.995, top=.955, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / "stack_l2_templates.png", dpi=140); plt.close(fig)

    cols = 16; rows = int(np.ceil(k1 / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * .55, rows * .58))
    for i, a in enumerate(np.ravel(axes)):
        a.set_xticks([]); a.set_yticks([])
        if i >= k1:
            a.axis("off"); continue
        t = W1[i].reshape(km.PS, km.PS)
        m = np.abs(t).max() + EPS
        a.imshow(t, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
    fig.suptitle(f"the L1 codebook these are built from ({k1} templates, 5x5)",
                 fontsize=9)
    fig.subplots_adjust(left=.005, right=.995, top=1 - .5 / (rows * .58),
                        bottom=.005, wspace=.06, hspace=.06)
    fig.savefig(OUT / "stack_l1_templates.png", dpi=160); plt.close(fig)

    np.savez_compressed(OUT / "stack.npz", W=W.astype(np.float32), n=n, lab=lab)
    (OUT / "stack.json").write_text(json.dumps(
        {"K1": k1, "K2": K2, "acc": acc, "purity": float(purity),
         "dead": int((n == 0).sum()), "active_positions": live,
         "params": int(K2 * (P + 10)), "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> stack_generate.png")


if __name__ == "__main__":
    main()
