"""Pictures: what the experts imagine, and what they see when they are wrong.

Three figures per arm (none / B):
  50_imagine_*    give an expert ONLY a label and let it complete the image
  51_rebuild_*    a test image, rebuilt by the expert of its own class and by
                  the expert that actually wins the fit gate
  52_templates_*  the raw templates of a few experts
"""

from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG = 784


def rebuild(W, Q):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return e, R


def imagine(W, labels):
    """Blank image, label written in -- let each expert complete the picture."""
    V = np.zeros((len(labels), N_IMG + 10))
    V[np.arange(len(labels)), N_IMG + np.asarray(labels)] = 1.0
    Q = center_norm(V)
    _, R = rebuild(W, Q)
    return R[:, :, :N_IMG]


def show(a, v, cmap="gray"):
    m = np.abs(v).max() + EPS
    a.imshow(v.reshape(28, 28), cmap=cmap, vmin=-m if cmap == "bwr" else None,
             vmax=m if cmap == "bwr" else None)
    a.set_xticks([]); a.set_yticks([])


def main():
    z = np.load(OUT / "dopamine_weights.npz")
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    for tag in ("none", "B"):
        W = z[f"W_{tag}"].astype(np.float64)
        wins = z[f"wins_{tag}"]
        alive = wins.sum(1) > 0
        claim = np.where(alive, wins.argmax(1), -1)
        idx = np.where(alive)[0]

        # --- imagine: every live expert completes its own claimed class -----
        G = imagine(W, [max(claim[h], 0) for h in idx])
        n = len(idx)
        cols = 10; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.25))
        for a in np.ravel(axes):
            a.axis("off")
        for j, h in enumerate(idx):
            a = np.ravel(axes)[j]; a.axis("on"); show(a, G[j, h], "bwr")
            a.set_title(f"e{h} · {CLASSES[claim[h]][:9]}", fontsize=5.5, pad=1.5)
        fig.suptitle(f"[{tag}] each expert imagines its own class "
                     f"(label in, image out)", fontsize=10)
        fig.subplots_adjust(left=.005, right=.995, top=.93, bottom=.005,
                            wspace=.05, hspace=.28)
        fig.savefig(OUT / f"50_imagine_{tag}.png", dpi=140); plt.close(fig)

        # --- rebuild: own-class expert vs the one that wins the gate --------
        E, R = rebuild(W, join(Xte[:2000]))
        E = np.where(alive[None], E, np.inf)
        win = E.argmin(1)
        pick = []
        for c in range(10):
            cand = np.where((yte[:2000] == c) & (claim[win] != yte[:2000]))[0]
            if not len(cand):
                cand = np.where(yte[:2000] == c)[0]
            pick.append(cand[0])
        fig, axes = plt.subplots(3, 10, figsize=(11.5, 4.1))
        for j, i in enumerate(pick):
            own = np.where(claim == yte[i])[0]
            oi = own[E[i, own].argmin()] if len(own) else win[i]
            show(axes[0, j], Xte[i]); axes[0, j].set_title(
                CLASSES[yte[i]][:10], fontsize=6, pad=2)
            show(axes[1, j], R[i, oi, :N_IMG], "bwr"); axes[1, j].set_title(
                f"own e{oi}  err {E[i,oi]:.2f}", fontsize=5.5, pad=2)
            show(axes[2, j], R[i, win[i], :N_IMG], "bwr"); axes[2, j].set_title(
                f"gate e{win[i]}={CLASSES[claim[win[i]]][:8]}\nerr {E[i,win[i]]:.2f}",
                fontsize=5.5, pad=2)
        for r, lab in enumerate(("image", "rebuilt by own class",
                                 "rebuilt by gate winner")):
            axes[r, 0].set_ylabel(lab, fontsize=6)
        fig.suptitle(f"[{tag}] lower error wins the gate — and it is often "
                     f"the wrong class", fontsize=10)
        fig.subplots_adjust(left=.06, right=.995, top=.86, bottom=.01,
                            wspace=.06, hspace=.35)
        fig.savefig(OUT / f"51_rebuild_{tag}.png", dpi=140); plt.close(fig)

        # --- the templates themselves ---------------------------------------
        sel = idx[:8]
        fig, axes = plt.subplots(len(sel), 12, figsize=(9, .82 * len(sel)))
        for r, h in enumerate(sel):
            for c in range(12):
                show(axes[r, c], W[h, c, :N_IMG], "bwr")
            axes[r, 0].set_ylabel(f"e{h}\n{CLASSES[claim[h]][:8]}", fontsize=5,
                                  rotation=0, ha="right", va="center")
        fig.suptitle(f"[{tag}] 12 templates of 8 experts", fontsize=10)
        fig.subplots_adjust(left=.09, right=.995, top=.92, bottom=.005,
                            wspace=.05, hspace=.05)
        fig.savefig(OUT / f"52_templates_{tag}.png", dpi=140); plt.close(fig)
        print(f"[{tag}] wrote 50_imagine, 51_rebuild, 52_templates", flush=True)


if __name__ == "__main__":
    main()
