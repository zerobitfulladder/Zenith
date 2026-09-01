"""What does one image actually look like as a sparse vector, and is it stable?

At every one of the 24x24 patch positions exactly one expert wins. Give it a
number -- how much of that patch it explains, 1 minus its rebuild error -- and
give every losing expert 0. Concatenate: 576 positions x 30 experts = 17,280
numbers, of which at most 576 are nonzero.

The question is whether two 4s produce the same pattern and a 4 and a 9 produce
different ones. Shown two ways, because the answer differs:

    full        the raw 17,280, position by position
    pooled      the same thing maxed over a 4x4 grid of areas -> 480 numbers,
                which is the form that actually classified at 0.94

Yesterday's law predicts the split: one-hot identities are orthogonal, so at
full resolution a one-pixel shift replaces a nonzero entry with a different
nonzero entry and the overlap collapses. Pooling is what lets two instances of
the same digit overlap at all.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E

OUT = Path(__file__).resolve().parent / "results"
SIDE = 28 - E.PS + 1
GRID = 4
CLASSES = [0, 1, 4, 9]
NPER = 5


def sparse(W, img):
    """(576, 30) -- the winner's fit at each position, zeros for the losers."""
    h = len(W)
    Q, keep = E.patches(img[None])
    B = E.join(Q.reshape(-1, E.PS * E.PS))
    S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
    R = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
    err = ((B[None] - R)[:, :, :E.PS * E.PS] ** 2).sum(-1)
    win = err.argmin(0)
    fit = np.clip(1.0 - err[win, np.arange(len(B))], 0.0, 1.0)
    k = keep.reshape(-1)
    V = np.zeros((SIDE * SIDE, h), np.float32)
    V[np.arange(len(B))[k], win[k]] = fit[k]
    return V, np.where(k, win, -1).reshape(SIDE, SIDE)


def pool(V, h):
    b = SIDE // GRID
    return V.reshape(SIDE, SIDE, h).reshape(GRID, b, GRID, b, h).max(axis=(1, 3)).ravel()


def cos(A):
    n = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return n @ n.T


def main():
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    _, _, Xte, yte = E.load()
    imgs, maps, full, pooled, lab = [], [], [], [], []
    for c in CLASSES:
        for i in np.nonzero(yte == c)[0][:NPER]:
            V, m = sparse(W, Xte[i])
            imgs.append(Xte[i]); maps.append(m)
            full.append(V.ravel()); pooled.append(pool(V, h)); lab.append(c)
    full = np.array(full); pooled = np.array(pooled); lab = np.array(lab)
    n = len(lab)

    Cf, Cp = cos(full), cos(pooled)
    same = lab[:, None] == lab[None, :]
    off = ~np.eye(n, dtype=bool)
    stats = {
        "nonzero_per_image": float((full > 0).sum(1).mean()),
        "dims_full": int(full.shape[1]), "dims_pooled": int(pooled.shape[1]),
        "full_within": float(Cf[same & off].mean()), "full_between": float(Cf[~same].mean()),
        "pooled_within": float(Cp[same & off].mean()), "pooled_between": float(Cp[~same].mean()),
    }
    stats["full_gap"] = stats["full_within"] - stats["full_between"]
    stats["pooled_gap"] = stats["pooled_within"] - stats["pooled_between"]
    for k, v in stats.items():
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 2.6, 2.6, 3.0], hspace=0.35)

    top = gs[0].subgridspec(2, n, hspace=0.05, wspace=0.05)
    for i in range(n):
        a = fig.add_subplot(top[0, i]); a.imshow(imgs[i], cmap="gray_r"); a.axis("off")
        if i % NPER == 0:
            a.set_title(f"class {lab[i]}", fontsize=9, color="tab:blue")
        b = fig.add_subplot(top[1, i])
        b.imshow(np.ma.masked_less(maps[i], 0), cmap="tab20", vmin=0, vmax=h - 1,
                 interpolation="nearest"); b.axis("off")
    fig.text(0.005, 0.895, "image", fontsize=8, rotation=90)
    fig.text(0.005, 0.80, "winner\nexpert", fontsize=8, rotation=90)

    for ax, M, name in ((fig.add_subplot(gs[1]), full, f"FULL  {full.shape[1]:,} numbers, "
                         f"{stats['nonzero_per_image']:.0f} nonzero"),
                        (fig.add_subplot(gs[2]), pooled,
                         f"POOLED to 4x4  {pooled.shape[1]} numbers")):
        ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=7)
        for y in range(NPER, n, NPER):
            ax.axhline(y - 0.5, color="cyan", lw=1.0)
        ax.set_xticks([])

    bot = gs[3].subgridspec(1, 2, wspace=0.25)
    for j, (C, t, g) in enumerate(((Cf, "similarity, FULL", stats["full_gap"]),
                                   (Cp, "similarity, POOLED", stats["pooled_gap"]))):
        a = fig.add_subplot(bot[0, j])
        im = a.imshow(C, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"{t}   within-minus-between = {g:+.3f}", fontsize=10)
        a.set_xticks(range(n)); a.set_xticklabels(lab, fontsize=6)
        a.set_yticks(range(n)); a.set_yticklabels(lab, fontsize=6)
        for k in range(NPER, n, NPER):
            a.axhline(k - .5, color="w", lw=.8); a.axvline(k - .5, color="w", lw=.8)
        plt.colorbar(im, ax=a, fraction=0.046)

    plt.savefig(OUT / "sparse_patterns.png", dpi=125, bbox_inches="tight")
    (OUT / "patterns.json").write_text(json.dumps(stats, indent=2))
    print("\n-> results/sparse_patterns.png")


if __name__ == "__main__":
    main()
