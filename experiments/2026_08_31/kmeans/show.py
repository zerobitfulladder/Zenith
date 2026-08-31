"""The templates, and what an image looks like when every patch is replaced by
the identity of its nearest template -- the reconstruction that identity-only
coding actually permits.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km

OUT = Path(__file__).resolve().parent / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
z = np.load(OUT / f"km_{DS}.npz")
res = json.loads((OUT / f"km_{DS}.json").read_text())["results"]

for K in ("64", "256"):
    W = z[K].astype(np.float64)
    use = np.array(res[K]["use"])
    o = np.argsort(use)[::-1]
    n = min(len(W), 128)
    cols = 16; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * .55, rows * .58))
    for i, a in enumerate(np.ravel(axes)):
        a.set_xticks([]); a.set_yticks([])
        if i >= n:
            a.axis("off"); continue
        t = W[o[i]].reshape(km.PS, km.PS)
        m = np.abs(t).max() + 1e-12
        a.imshow(t, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
    fig.suptitle(f"{DS} — spherical k-means, K={K}, most used first"
                 f"   (correlation {res[K]['correlation']:.3f})", fontsize=9)
    fig.subplots_adjust(left=.005, right=.995, top=1 - .5 / (rows * .58),
                        bottom=.005, wspace=.06, hspace=.06)
    fig.savefig(OUT / f"templates_{DS}_K{K}.png", dpi=160); plt.close(fig)
    print(f"-> templates_{DS}_K{K}.png")

# ---- identity-only reconstruction -----------------------------------------
Xtr, ytr, Xte, yte = km.load(DS)
sel = [int(np.nonzero(yte == d)[0][0]) for d in range(10)]
imgs = Xte[sel]
fig, axes = plt.subplots(4, 10, figsize=(10.5, 4.5))
for j, im in enumerate(imgs):
    axes[0][j].imshow(im, cmap="gray"); axes[0][j].set_title(str(yte[sel[j]]), fontsize=7)
for r, K in enumerate(("16", "64", "256")):
    W = z[K].astype(np.float64)
    for j, im in enumerate(imgs):
        P = km.patches(im[None])
        Q, keep, norm = km.prep(P)
        win = (Q[0] @ W.T).argmax(1)
        canvas, wsum = np.zeros((28, 28)), np.zeros((28, 28))
        side = 28 - km.PS + 1
        for p in range(Q.shape[1]):
            if not keep[0, p]:
                continue
            rr, cc = divmod(p, side)
            # identity only: the template, rescaled by the patch's own contrast
            canvas[rr:rr + km.PS, cc:cc + km.PS] += W[win[p]].reshape(km.PS, km.PS) * norm[0, p]
            wsum[rr:rr + km.PS, cc:cc + km.PS] += 1
        axes[r + 1][j].imshow(canvas / np.maximum(wsum, 1), cmap="gray")
for a in np.ravel(axes):
    a.set_xticks([]); a.set_yticks([])
for r, lab in enumerate(("original", "K=16", "K=64", "K=256")):
    axes[r][0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
fig.suptitle("rebuilt from identities alone: every patch replaced by its "
             "nearest template", fontsize=10)
fig.subplots_adjust(left=.07, right=.995, top=.88, bottom=.01, wspace=.05, hspace=.08)
fig.savefig(OUT / f"rebuild_{DS}.png", dpi=150); plt.close(fig)
print("-> rebuild_" + DS + ".png")

d = [(int(k), 2 - 2 * v["correlation"]) for k, v in sorted(res.items(), key=lambda kv: int(kv[0]))]
print("\ndistortion vs K (identity-only):")
for k, v in d:
    print(f"  K={k:<4} squared error {v:.4f}")
for (k1, v1), (k2, v2) in zip(d, d[1:]):
    de = 2 * np.log(k2 / k1) / np.log(v1 / v2)
    print(f"  K {k1}->{k2}: distortion falls {v1/v2:.2f}x  =>  effective dimension {de:.1f}")
