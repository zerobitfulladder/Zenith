"""The honest reconstruction test: non-overlapping patches.

With stride 1 there are 576 overlapping codes for a 784-pixel image -- more data
than the image, and every pixel is estimated ~25 times over. Here the image is
tiled by disjoint 5x5 patches, 25 of them covering 25x25 pixels, so the whole
code is 25 indices plus 25 magnitudes. That is a real compression ratio and a
real test of what identity alone can carry.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km

OUT = Path(__file__).resolve().parent / "results"
DS = "mnist"
z = np.load(OUT / f"km_{DS}.npz")
Xtr, ytr, Xte, yte = km.load(DS)
S = km.PS
G = 28 // S                      # 5 tiles across, covering 25x25 of the 28x28


def tiles(X):
    T = X[:, :G * S, :G * S].reshape(len(X), G, S, G, S).transpose(0, 1, 3, 2, 4)
    return T.reshape(len(X), G * G, S * S)


def code_and_rebuild(W, X):
    P = tiles(X).astype(np.float64)
    mu = P.mean(-1, keepdims=True)
    C = P - mu
    n = np.linalg.norm(C, axis=-1)
    Q = C / np.maximum(n, km.EPS)[..., None]
    win = np.einsum("ipd,kd->ipk", Q, W).argmax(-1)
    R = W[win] * n[..., None] + mu                     # identity + contrast + mean
    err = np.linalg.norm(R - P, axis=-1) / np.maximum(np.linalg.norm(P, axis=-1), km.EPS)
    img = R.reshape(len(X), G, G, S, S).transpose(0, 1, 3, 2, 4).reshape(len(X), G * S, G * S)
    return win, img, float(err.mean())


res = {}
print(f"tiled {G}x{G} = {G*G} disjoint patches per image, "
      f"{G*G} indices + {G*G} magnitudes vs 784 pixels")
for K in ("16", "64", "256"):
    W = z[K].astype(np.float64)
    win, img, err = code_and_rebuild(W, Xte)
    bits = G * G * np.log2(int(K))
    res[K] = {"rel_error": err, "index_bits": float(bits),
              "bits_per_pixel_index_only": float(bits / (G * S) ** 2)}
    print(f"  K={K:<4} relative error {err:.4f}   "
          f"{bits:.0f} bits of index for {(G*S)**2} pixels "
          f"({bits/(G*S)**2:.2f} bits/pixel, before the magnitudes)")

sel = [int(np.nonzero(yte == d)[0][0]) for d in range(10)]
fig, axes = plt.subplots(4, 10, figsize=(10.5, 4.5))
for j, i in enumerate(sel):
    axes[0][j].imshow(Xte[i, :G * S, :G * S], cmap="gray")
    axes[0][j].set_title(str(yte[i]), fontsize=7)
for r, K in enumerate(("16", "64", "256")):
    _, img, _ = code_and_rebuild(z[K].astype(np.float64), Xte[sel])
    for j in range(10):
        axes[r + 1][j].imshow(img[j], cmap="gray")
for a in np.ravel(axes):
    a.set_xticks([]); a.set_yticks([])
for r, lab in enumerate(("original", "K=16", "K=64", "K=256")):
    axes[r][0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
fig.suptitle("non-overlapping tiles: 25 indices per image, no averaging to hide "
             "behind", fontsize=10)
fig.subplots_adjust(left=.07, right=.995, top=.88, bottom=.01, wspace=.05, hspace=.08)
fig.savefig(OUT / "rebuild_tiled.png", dpi=150)
(OUT / "tiled.json").write_text(json.dumps(res, indent=2))
print("-> rebuild_tiled.png")
