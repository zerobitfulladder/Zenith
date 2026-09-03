"""Draw the templates before and after phase B, at both betas.

A whole-digit template is 784 numbers, so it can just be looked at. That makes
the whole 2x2 visible: train on 0-4, then train on 5-9, then ask whether the
pictures that used to be 0-4 are still 0-4.

Outputs
  results/templates_drift.png    the 12 most-moved templates per beta, A -> B
  results/templates_all_*.png    all 400, after A and after B, each beta
  results/templates_patch.png    the 5x5 patch templates, for contrast

Usage:  uv run python pictures.py
"""

import sys
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import whole as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SEED = 7
NSHOW = 12


def show(ax, w, title=None, color="black"):
    a = float(np.abs(w).max())
    ax.imshow(w.reshape(28, 28), cmap="RdBu_r", vmin=-a, vmax=a)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, color=color, pad=2)


def run_phase(beta):
    """Phase A on 0-4, then phase B on 5-9, at this beta. Returns the templates
    and the table's top class for each, before and after."""
    V.BETA = beta
    Xtr, ytr, Xte, yte = V.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    ytr_g = cp.asarray(ytr)
    A = ytr_g < 5
    XA, yA, XB, yB = Xtr[A], ytr_g[A], Xtr[~A], ytr_g[~A]

    rng = np.random.default_rng(SEED)
    W, N, n, _ = V.train_cont(XA, yA, rng)
    W_A = W.copy()
    T_A, _ = V.rebuild(W_A, XA, yA)
    iA, kA = V.code(W_A, XA)
    wins_A = cp.bincount(iA[kA], minlength=V.K)
    cls_A = T_A[:, 0, :].argmax(1)

    W, N, n, _ = V.train_cont(XB, yB, rng, W, N, n)
    T_B, _ = V.rebuild(W, Xtr, ytr_g)            # oracle: full recount
    cls_B = T_B[:, 0, :].argmax(1)
    drift = 1.0 - cp.abs((W_A * W).sum(1))

    g = lambda x: cp.asnumpy(x)
    return dict(beta=beta, W_A=g(W_A), W_B=g(W), cls_A=g(cls_A), cls_B=g(cls_B),
                drift=g(drift), wins_A=g(wins_A))


def montage(W, path, title):
    k = len(W)
    side = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(side, side, figsize=(side * 0.42, side * 0.42))
    for i, ax in enumerate(axes.ravel()):
        if i < k:
            show(ax, W[i])
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.955)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    runs = {}
    for beta in (1.5, 0.0):
        print(f"beta {beta} ...", flush=True)
        runs[beta] = run_phase(beta)

    fig, axes = plt.subplots(4, NSHOW, figsize=(NSHOW * 1.05, 5.0))
    for r, beta in enumerate((1.5, 0.0)):
        d = runs[beta]
        live = np.where(d["wins_A"] > 0)[0]
        pick = live[np.argsort(-d["drift"][live])][:NSHOW]
        for c, i in enumerate(pick):
            moved = d["cls_A"][i] != d["cls_B"][i]
            show(axes[2 * r, c], d["W_A"][i], f"{d['cls_A'][i]}")
            show(axes[2 * r + 1, c], d["W_B"][i],
                 f"{d['cls_B'][i]}" + (" X" if moved else ""),
                 color="crimson" if moved else "black")
        n_moved = int((d["cls_A"][live] != d["cls_B"][live]).sum())
        axes[2 * r, 0].set_ylabel(f"beta {beta}\nafter 0-4", fontsize=9)
        axes[2 * r + 1, 0].set_ylabel(
            f"after 5-9\n{n_moved}/{len(live)} changed class", fontsize=9)
    fig.suptitle("The 12 most-moved whole-digit templates: trained on 0-4 (top row of each "
                 "pair), then on 5-9 (bottom).\nNumber above each = the class its table row "
                 "points at.", fontsize=10)
    fig.subplots_adjust(hspace=0.35, wspace=0.06, top=0.86)
    fig.savefig(OUT / "templates_drift.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote templates_drift.png", flush=True)

    for beta in (1.5, 0.0):
        d = runs[beta]
        montage(d["W_A"], OUT / f"templates_all_beta{beta:g}_afterA.png",
                f"all 400 templates after 0-4 only  (beta {beta})")
        montage(d["W_B"], OUT / f"templates_all_beta{beta:g}_afterB.png",
                f"the same 400 after 5-9  (beta {beta})")

    npz = HERE.parents[1] / "2026_09_01/stack/results/gpu_stack_mnist.npz"
    if npz.exists():
        W1 = cp.asnumpy(cp.load(npz)["W1"])
        side = int(np.ceil(np.sqrt(len(W1))))
        fig, axes = plt.subplots(side, side, figsize=(side * 0.3, side * 0.3))
        for i, ax in enumerate(axes.ravel()):
            if i < len(W1):
                a = float(np.abs(W1[i]).max())
                ax.imshow(W1[i].reshape(5, 5), cmap="RdBu_r", vmin=-a, vmax=a)
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        fig.suptitle("the 400 patch templates (5x5 px) — nothing here is a digit", fontsize=11)
        fig.subplots_adjust(wspace=0.06, hspace=0.06, top=0.95)
        fig.savefig(OUT / "templates_patch.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  wrote templates_patch.png", flush=True)

    extras(runs)
    for beta in (1.5, 0.0):
        d = runs[beta]
        live = d["wins_A"] > 0
        print(f"beta {beta}: {int((d['cls_A'][live] != d['cls_B'][live]).sum())}"
              f"/{int(live.sum())} live templates changed the class their table "
              f"row points at;  mean drift {d['drift'][live].mean():.4f}")



def extras(runs):
    """The drift figure shows the 12 MOST-moved templates, which is the tail by
    construction. This adds a uniformly random sample of live templates, so the
    typical fate is visible too, plus the drift distribution behind both."""
    rs = np.random.default_rng(0)
    fig, axes = plt.subplots(4, NSHOW, figsize=(NSHOW * 1.05, 5.0))
    for r, beta in enumerate((1.5, 0.0)):
        d = runs[beta]
        live = np.where(d["wins_A"] > 0)[0]
        pick = np.sort(rs.choice(live, NSHOW, replace=False))
        for c, i in enumerate(pick):
            moved = d["cls_A"][i] != d["cls_B"][i]
            show(axes[2 * r, c], d["W_A"][i], f"{d['cls_A'][i]}")
            show(axes[2 * r + 1, c], d["W_B"][i],
                 f"{d['cls_B'][i]}" + (" X" if moved else ""),
                 color="crimson" if moved else "black")
        n_moved = int((d["cls_A"][live] != d["cls_B"][live]).sum())
        axes[2 * r, 0].set_ylabel(f"beta {beta}\nafter 0-4", fontsize=9)
        axes[2 * r + 1, 0].set_ylabel(
            f"after 5-9\n{n_moved}/{len(live)} = {n_moved/len(live):.0%} changed",
            fontsize=9)
    fig.suptitle("RANDOM sample of live templates (not the tail): trained on 0-4, then on 5-9.",
                 fontsize=10)
    fig.subplots_adjust(hspace=0.35, wspace=0.06, top=0.90)
    fig.savefig(OUT / "templates_random.png", dpi=145, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for beta, col in ((1.5, "#2a6f97"), (0.0, "#c1121f")):
        d = runs[beta]
        live = d["wins_A"] > 0
        ax.hist(d["drift"][live], bins=40, range=(0, 0.6), alpha=0.55, color=col,
                label=f"beta {beta}  (mean {d['drift'][live].mean():.3f})")
    ax.set_xlabel("drift of a template during phase B   (1 - |cos| between before and after)")
    ax.set_ylabel("templates"); ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "drift_hist.png", dpi=145); plt.close(fig)
    print("  wrote templates_random.png, drift_hist.png", flush=True)

if __name__ == "__main__":
    main()
