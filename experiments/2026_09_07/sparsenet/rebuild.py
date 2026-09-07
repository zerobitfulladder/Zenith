"""Rebuild whole digits out of the learned templates, to see what the code
actually keeps and what it throws away.

A digit is cut into 8x8 pieces at every position. Each piece is coded -- a
handful of templates, each with a number saying how much of it to use -- and
then rebuilt from those templates alone. Overlapping pieces are averaged back
into a whole picture.

Trained on RAW patches here, not the paper's filtered ones, so the rebuilt
picture is a digit you can look at rather than an edge map.

Usage:  uv run python rebuild.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparsenet as S
import unitnorm as U

OUT = S.OUT / "rebuild"; OUT.mkdir(exist_ok=True)
SFORM, K = "abs", 64
LAMS = [0.03, 0.06, 0.12]
FLOOR = 0.05          # a piece this flat is left as its own average grey


def code_image(img, Phi, lam):
    """Every 8x8 piece of one image, coded and rebuilt, averaged back together."""
    P = sliding_window_view(img, (S.PS, S.PS)).reshape(-1, S.DIM)
    m = P.mean(1, keepdims=True)
    C = P - m
    n = np.linalg.norm(C, axis=1, keepdims=True)
    ok = (n[:, 0] > FLOOR)
    a = np.zeros((len(P), K))
    if ok.any():
        a[ok] = S.settle(Phi, C[ok] / n[ok], lam, SFORM)
    R = a @ Phi * n + m                      # put the strength and grey back
    side = 28 - S.PS + 1
    out = np.zeros((28, 28)); cnt = np.zeros((28, 28))
    for j, (r, c) in enumerate(np.ndindex(side, side)):
        out[r:r + S.PS, c:c + S.PS] += R[j].reshape(S.PS, S.PS)
        cnt[r:r + S.PS, c:c + S.PS] += 1
    return out / cnt, a, ok


def main():
    P, _ = U.prep("raw")
    print("choosing how strictly to charge for using a template:")
    best = None
    for lam in LAMS:
        Phi, _ = U.train(P, lam, 0.0, 0.0, np.random.default_rng(U.SEED), U.UPDATES)
        m = S.measure(Phi, P, lam, np.random.default_rng(1), SFORM, n=4000); m.pop("A")
        print(f"   lam {lam:5.3f}: {m['active']:4.1f} of {K} templates per piece, "
              f"{100 * m['resid']:4.1f}% of the piece left over")
        if best is None or abs(m["resid"] - 0.10) < abs(best[2] - 0.10):
            best = (lam, Phi, m["resid"])
    lam, Phi, _ = best
    print(f"   -> lam {lam}\n")
    np.save(OUT / "phi_raw_unit.npy", Phi)

    d = S.ROOT / "data"
    X = np.load(d / "mnist/digits/train_images.npy").astype(np.float64).reshape(-1, 28, 28)
    if X.max() > 1.5:
        X /= 255.0
    rng = np.random.default_rng(3)
    imgs = X[rng.permutation(len(X))[:8]]

    # --- whole digits, rebuilt from templates alone
    shots = []
    for img in imgs:
        r, a, ok = code_image(img, Phi, lam)
        w = np.abs(a[ok]).sum(0)                  # how much each template is used here
        shots.append((img, r, a[ok], w))
    order = np.argsort(-sum(w for _, _, _, w in shots))   # same order in every panel
    top = max(w.max() for _, _, _, w in shots)

    fig, ax = plt.subplots(4, 8, figsize=(13.5, 7.4),
                           gridspec_kw=dict(height_ratios=[1, 1, 1, 0.9]))
    used = []
    for j, (img, r, a, w) in enumerate(shots):
        u = (np.abs(a) > 0).sum(1).mean(); used.append(u)
        err = np.abs(img - r).mean() / (np.abs(img).mean() + 1e-12)
        for i, pic in enumerate((img, r, img - r)):
            ax[i, j].imshow(pic, cmap="gray", vmin=-1 if i == 2 else 0, vmax=1)
            ax[i, j].axis("off")
        ax[0, j].set_title(f"{u:.1f} templates\nper piece", fontsize=7)
        ax[2, j].set_title(f"{100 * err:.0f}% off", fontsize=7)
        ax[3, j].bar(np.arange(K), w[order], width=1.0, color="0.25")
        ax[3, j].set_ylim(0, top * 1.05); ax[3, j].set_xlim(-0.5, K - 0.5)
        ax[3, j].tick_params(labelsize=5, length=1.5)
        ax[3, j].set_title(f"{(w > 0).sum()} of {K} templates used", fontsize=7)
        if j:
            ax[3, j].set_yticklabels([])
        else:
            ax[3, j].set_ylabel("total use", fontsize=7)
        ax[3, j].set_xlabel("template", fontsize=7)
    for i, t in enumerate(("original", "rebuilt from templates", "what was lost")):
        ax[i, 0].text(-0.35, 0.5, t, rotation=90, va="center", ha="center",
                      transform=ax[i, 0].transAxes, fontsize=9)
    fig.suptitle(f"whole digits rebuilt from {K} learned templates -- each 8x8 piece "
                 f"written with {np.mean(used):.1f} templates on average\n"
                 f"bars: how much each template is used across the whole digit "
                 f"(templates in the same order in every panel)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "digits.png", dpi=140); plt.close(fig)

    # --- one piece, taken apart
    img = imgs[0]
    Pp = sliding_window_view(img, (S.PS, S.PS)).reshape(-1, S.DIM)
    mm = Pp.mean(1, keepdims=True); C = Pp - mm
    nn = np.linalg.norm(C, axis=1, keepdims=True)
    a = np.zeros((len(Pp), K)); ok = nn[:, 0] > FLOOR
    a[ok] = S.settle(Phi, C[ok] / nn[ok], lam, SFORM)
    cnt = (np.abs(a) > 0).sum(1)
    pick = np.argsort(np.abs(cnt - 4))[0]                 # a piece using about four
    coef = a[pick]; live = np.argsort(-np.abs(coef))[:(np.abs(coef) > 0).sum()]
    n = len(live)
    fig, ax = plt.subplots(2, n + 2, figsize=(1.5 * (n + 2), 3.4))
    ax[0, 0].imshow((C[pick] / nn[pick]).reshape(S.PS, S.PS), cmap="gray")
    ax[0, 0].set_title("the piece", fontsize=8)
    run = np.zeros(S.DIM)
    for i, t in enumerate(live):
        ax[0, i + 1].imshow(Phi[t].reshape(S.PS, S.PS), cmap="gray")
        ax[0, i + 1].set_title(f"template {t}\nx {coef[t]:+.2f}", fontsize=8)
        run = run + coef[t] * Phi[t]
        ax[1, i + 1].imshow(run.reshape(S.PS, S.PS), cmap="gray")
        ax[1, i + 1].set_title(f"first {i + 1} added", fontsize=8)
    ax[0, n + 1].imshow(run.reshape(S.PS, S.PS), cmap="gray")
    ax[0, n + 1].set_title("rebuilt", fontsize=8)
    for A in ax.ravel():
        A.axis("off")
    fig.suptitle(f"one 8x8 piece of a digit, written as {n} templates added together",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "one_piece.png", dpi=140); plt.close(fig)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
