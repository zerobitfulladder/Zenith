"""Paint an L2 hypercolumn back into pixels.

An L2 template holds, for each of 36 cells, how much it expects each of the 32
L1 hypercolumns to fire there. To see that as a picture:

  1. give every L1 hypercolumn a face -- the mean 5x5 patch it actually wins
  2. for each L2 hypercolumn compute, per (cell, L1 channel), the RMS across its
     16 minicolumns. RMS because the minicolumns span a subspace with no
     preferred basis, so any single template is arbitrary but their joint
     magnitude is not.
  3. stamp each L1 face into the cell it belongs to, weighted by that number,
     and sum into a 28x28 canvas.

Two versions: what it expects, and what it expects MORE than the average L2
hypercolumn does -- the second is what distinguishes it from its neighbours.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c
from common import EPS

G, PS, GRIDPOS = 6, 5, 24
L1KEY = "H32_K8"


def l1_faces(W1, X, n=4000):
    """The mean patch each L1 hypercolumn wins -- its face."""
    H = len(W1)
    acc, cnt = np.zeros((H, PS * PS)), np.zeros(H)
    for a in range(0, n, c.CHUNK_IMG):
        Q, keep = c.prep(c.grid(X[a:a + c.CHUNK_IMG]))
        kf = keep.reshape(-1)
        flat = Q.reshape(-1, PS * PS)[kf]
        e, _ = c.errors(W1, flat, want_S=False)
        win = e.argmin(1)
        np.add.at(acc, win, flat)
        np.add.at(cnt, win, 1.0)
    return acc / np.maximum(cnt, 1)[:, None], cnt


def paint(E, faces, topk=3):
    """E is (36, 32) -> a 28x28 canvas.

    Only the top-k channels per cell are stamped. Summing all 32 averages every
    orientation together and every cell comes out a blob -- the whole point is
    WHICH edge this hypercolumn expects here, not how much of anything.
    """
    canvas = np.zeros((28, 28))
    step = GRIDPOS // G
    for cell in range(G * G):
        r, col = divmod(cell, G)
        top, left = r * step + step // 2, col * step + step // 2
        w = E[cell].copy()
        if topk < len(w):
            w[np.argsort(w)[:-topk]] = 0.0
        w = np.maximum(w, 0.0)
        canvas[top:top + PS, left:left + PS] += (w[:, None] * faces).sum(0).reshape(PS, PS)
    return canvas


def main():
    W1 = np.load(OUT / f"conv1_mnist.npz")[L1KEY].astype(np.float64)
    z = np.load(OUT / "l2_identity.npz")
    W, wins = z["W"].astype(np.float64), z["wins"]
    H2, K2, _ = W.shape
    H1 = len(W1)
    X, y, _, _ = c.load("mnist")
    faces, cnt = l1_faces(W1, X)
    print(f"L1 faces from {int(cnt.sum())} patches, "
          f"least-used hypercolumn saw {int(cnt.min())}")

    claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
    live = wins.sum(1) > wins.sum() * 0.002
    E = np.sqrt((W[:, :, :G * G * H1].reshape(H2, K2, G * G, H1) ** 2).mean(1))
    Emean = E[live].mean(0)

    order = [h for h in np.argsort(claim) if live[h]]
    fig, axes = plt.subplots(2, len(order), figsize=(len(order) * 1.15, 2.9))
    for j, h in enumerate(order):
        for row, M, ttl in ((0, E[h] - Emean, "distinctive"),
                            (1, E[h] - Emean, "distinctive, top-1")):
            img = paint(M, faces, topk=3 if row == 0 else 1)
            m = np.abs(img).max() + EPS
            a = axes[row][j]
            a.imshow(img, cmap="bwr", vmin=-m, vmax=m, interpolation="bilinear")
            a.set_xticks([]); a.set_yticks([])
        axes[0][j].set_title(f"h{h} says {claim[h]}", fontsize=6)
    axes[0][0].set_ylabel("top-3 edges\nit favours", fontsize=7)
    axes[1][0].set_ylabel("top-1 edge\nit favours", fontsize=7)
    fig.suptitle("what each L2 hypercolumn expects, painted back into pixels",
                 fontsize=10)
    fig.subplots_adjust(left=.045, right=.995, top=.84, bottom=.01,
                        wspace=.06, hspace=.06)
    fig.savefig(OUT / "l2_in_pixels.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(8.4, 4.4))
    o = np.argsort(cnt)[::-1]
    for i, a in enumerate(np.ravel(axes)):
        t = faces[o[i]].reshape(PS, PS)
        m = np.abs(t).max() + EPS
        a.imshow(t, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"L1 h{o[i]}", fontsize=5.5, pad=1.5)
    fig.suptitle("the face of each L1 hypercolumn — the mean patch it wins",
                 fontsize=10)
    fig.subplots_adjust(left=.005, right=.995, top=.90, bottom=.005,
                        wspace=.05, hspace=.25)
    fig.savefig(OUT / "l1_faces.png", dpi=150); plt.close(fig)
    print("-> l2_in_pixels.png, l1_faces.png")


if __name__ == "__main__":
    main()
