"""Generation through both rungs: a label goes in, a digit comes out.

    label -> L2 completes its code block -> for each of the 36 cells, which L1
    hypercolumn it expects and how strongly -> L1 supplies what that
    hypercolumn looks like -> a 28x28 canvas

L2 alone cannot draw anything: it only ever sees identities, so it has no idea
what a patch looks like. L1 holds that. Two ways for L1 to answer:

    face     the mean patch that hypercolumn wins -- its prototype
    sample   drawn from that hypercolumn's own coefficient mean and covariance,
             so the same identity can be rendered many ways
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
from common import EPS, center_norm

G, PS, GRIDPOS, L1KEY = 6, 5, 24, "H32_K8"


def l1_stats(W1, X, n=6000):
    """Each L1 hypercolumn's face, and the mean/covariance of its coefficients."""
    H, K, _ = W1.shape
    acc, cnt = np.zeros((H, PS * PS)), np.zeros(H)
    s1, s2 = np.zeros((H, K)), np.zeros((H, K, K))
    for a in range(0, n, c.CHUNK_IMG):
        Q, keep = c.prep(c.grid(X[a:a + c.CHUNK_IMG]))
        kf = keep.reshape(-1)
        flat = Q.reshape(-1, PS * PS)[kf]
        e, S = c.errors(W1, flat)
        win = e.argmin(1)
        code = S[np.arange(len(win)), win]
        np.add.at(acc, win, flat); np.add.at(cnt, win, 1.0)
        np.add.at(s1, win, code)
        for h in np.unique(win):
            m = win == h
            s2[h] += code[m].T @ code[m]
    n_ = np.maximum(cnt, 1)[:, None]
    mu = s1 / n_
    cov = s2 / np.maximum(cnt, 1)[:, None, None] - mu[:, :, None] * mu[:, None, :]
    return acc / n_, mu, cov, cnt


def paint(E, W1, mu, cov, faces, mode, rng, topk=2):
    canvas = np.zeros((28, 28))
    step = GRIDPOS // G
    for cell in range(G * G):
        r, col = divmod(cell, G)
        top, left = r * step + step // 2, col * step + step // 2
        w = np.maximum(E[cell], 0.0)
        if topk < len(w):
            w[np.argsort(w)[:-topk]] = 0.0
        if w.sum() <= 0:
            continue
        for h in np.nonzero(w)[0]:
            if mode == "face":
                patch = faces[h]
            else:
                A = np.linalg.cholesky(cov[h] + 1e-8 * np.eye(cov.shape[1]))
                patch = (mu[h] + 0.8 * (A @ rng.standard_normal(len(mu[h])))) @ W1[h]
            canvas[top:top + PS, left:left + PS] += w[h] * patch.reshape(PS, PS)
    return canvas


def main():
    W1 = np.load(OUT / f"conv1_mnist.npz")[L1KEY].astype(np.float64)
    z = np.load(OUT / "l2_identity.npz")
    W2, wins = z["W"].astype(np.float64), z["wins"]
    H2, K2, D2 = W2.shape
    H1 = len(W1)
    X, y, _, _ = c.load("mnist")
    faces, mu, cov, cnt = l1_stats(W1, X)
    claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
    live = wins.sum(1) > wins.sum() * 0.002

    rng = np.random.default_rng(0)
    rows = ["face", "sample", "sample", "sample"]
    fig, axes = plt.subplots(len(rows) + 1, 10, figsize=(10.5, 1.15 * (len(rows) + 1)))
    for d in range(10):
        cand = np.nonzero((claim == d) & live)[0]
        h = int(cand[wins[cand].sum(1).argmax()])
        V = np.zeros(D2); V[G * G * H1 + d] = 1.0
        q = center_norm(V[None])[0]
        E = ((W2[h] @ q) @ W2[h])[:G * G * H1].reshape(G * G, H1)
        axes[0][d].imshow(np.linalg.norm(E, axis=1).reshape(G, G), cmap="magma",
                          interpolation="nearest")
        axes[0][d].set_title(f"{d}  (h{h})", fontsize=7)
        for r, mode in enumerate(rows):
            img = paint(E.copy(), W1, mu, cov, faces, mode, rng)
            m = np.abs(img).max() + EPS
            axes[r + 1][d].imshow(img, cmap="bwr", vmin=-m, vmax=m,
                                  interpolation="bilinear")
        for a in axes[:, d]:
            a.set_xticks([]); a.set_yticks([])
    axes[0][0].set_ylabel("where\n(6x6)", fontsize=6, rotation=0, ha="right", va="center")
    axes[1][0].set_ylabel("faces", fontsize=6, rotation=0, ha="right", va="center")
    for r in range(2, len(rows) + 1):
        axes[r][0].set_ylabel("sampled", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("label in, digit out: L2 says which edge belongs where, "
                 "L1 says what an edge looks like", fontsize=10)
    fig.subplots_adjust(left=.075, right=.995, top=.88, bottom=.01,
                        wspace=.05, hspace=.08)
    fig.savefig(OUT / "l2_imagine.png", dpi=150); plt.close(fig)
    print("-> l2_imagine.png")


if __name__ == "__main__":
    main()
