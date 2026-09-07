"""What one patch's firing looks like -- is only a small number of templates
used for each piece?

Uses the templates learned in rebuild.py (raw patches, unit length, flat fee
per template used). With a flat fee the unused templates are switched off
EXACTLY, so "used" needs no threshold: it is simply a coefficient that is not
zero.

Usage:  uv run python per_patch.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparsenet as S
import unitnorm as U
import rebuild as RB

OUT = RB.OUT
LAM, K = 0.06, 64


def main():
    Phi = np.load(OUT / "phi_raw_unit.npy")
    P, _ = U.prep("raw")
    rng = np.random.default_rng(5)
    X = P[rng.integers(0, len(P), 8000)]
    A = S.settle(Phi, X, LAM, "abs")
    on = A != 0
    cnt = on.sum(1)
    order = np.argsort(-np.abs(A).sum(0))
    print(f"  {cnt.mean():.1f} templates used per patch on average "
          f"(median {np.median(cnt):.0f}, range {cnt.min()}-{cnt.max()}) out of {K}")

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 0.85, 0.9], hspace=0.55,
                          wspace=0.45)

    ax = fig.add_subplot(gs[0, :4])
    ax.imshow(np.abs(A[:200][:, order]), aspect="auto", cmap="magma",
              interpolation="nearest")
    ax.set_xlabel("template"); ax.set_ylabel("patch")
    ax.set_title("200 patches, one row each -- bright = that template was used.\n"
                 "Almost every row is nearly all black: that is the sparseness.",
                 fontsize=9)

    ax = fig.add_subplot(gs[0, 4:])
    ax.hist(cnt, bins=np.arange(0, K + 2) - 0.5, color="0.25")
    ax.axvline(np.median(cnt), color="crimson", lw=1.2,
               label=f"median {np.median(cnt):.0f} of {K}")
    ax.set_xlabel("templates used by one patch"); ax.set_ylabel("patches")
    ax.legend(fontsize=8)
    ax.set_title(f"how many of the {K} templates each patch uses", fontsize=9)

    pick = rng.permutation(np.where(cnt > 2)[0])[:6]
    for j, i in enumerate(pick):
        a1 = fig.add_subplot(gs[1, j])
        a1.imshow(X[i].reshape(S.PS, S.PS), cmap="gray"); a1.axis("off")
        a1.set_title(f"patch -- {cnt[i]} used", fontsize=8)
        a2 = fig.add_subplot(gs[2, j])
        a2.bar(np.arange(K), A[i][order], width=1.0, color="0.25")
        a2.axhline(0, color="crimson", lw=0.5)
        a2.set_ylim(-np.abs(A[pick]).max(), np.abs(A[pick]).max())
        a2.tick_params(labelsize=5, length=1.5)
        a2.set_xlabel("template", fontsize=7)
        if j: a2.set_yticklabels([])
        else: a2.set_ylabel("coefficient", fontsize=7)
    fig.suptitle(f"the code for a single patch: {cnt.mean():.1f} of {K} templates "
                 f"used on average, the rest exactly zero", fontsize=12)
    fig.savefig(OUT / "per_patch.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # --- every patch of one digit
    d = S.ROOT / "data"
    I = np.load(d / "mnist/digits/train_images.npy").astype(np.float64).reshape(-1, 28, 28)
    if I.max() > 1.5: I /= 255.0
    img = I[np.random.default_rng(3).permutation(len(I))[:8]][0]
    Q = sliding_window_view(img, (S.PS, S.PS)).reshape(-1, S.DIM)
    C = Q - Q.mean(1, keepdims=True)
    n = np.linalg.norm(C, axis=1, keepdims=True)
    ok = n[:, 0] > RB.FLOOR
    Ad = S.settle(Phi, C[ok] / n[ok], LAM, "abs")
    cd = (Ad != 0).sum(1)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8),
                           gridspec_kw=dict(width_ratios=[0.6, 1.5, 1]))
    ax[0].imshow(img, cmap="gray"); ax[0].axis("off")
    ax[0].set_title(f"one digit\n{ok.sum()} pieces with ink", fontsize=9)
    ax[1].imshow(np.abs(Ad[:, order]), aspect="auto", cmap="magma",
                 interpolation="nearest")
    ax[1].set_xlabel("template"); ax[1].set_ylabel("piece of the digit")
    ax[1].set_title("every piece of that one digit, one row each", fontsize=9)
    ax[2].hist(cd, bins=np.arange(0, K + 2) - 0.5, color="0.25")
    ax[2].axvline(np.median(cd), color="crimson", lw=1.2,
                  label=f"median {np.median(cd):.0f} of {K}")
    ax[2].set_xlabel("templates used by one piece"); ax[2].set_ylabel("pieces")
    ax[2].legend(fontsize=8)
    ax[2].set_title(f"{(Ad != 0).any(0).sum()} of {K} templates appear somewhere "
                    f"in this digit", fontsize=9)
    fig.suptitle("sparse per piece, but the whole digit touches nearly every template",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "one_digit_patches.png", dpi=140)
    plt.close(fig)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
