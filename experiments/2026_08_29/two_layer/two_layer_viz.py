"""Pictures for the two-layer rig."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

import sys                                   # noqa: E402
from pathlib import Path                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regress"))
import pop_regress as P                      # noqa: E402

TRUE_C = "#111111"
L1_C = "#5b7fa6"
L2_C = "#d1495b"
MLP_C = "#3f7d20"


def save_shapes(l2, path, n=12):
    """The busiest layer-two templates, drawn as the curve pieces they are."""
    order = np.argsort(-l2.hc.wins)[:n]
    imgs = [l2.shape_image(l2.hc.row(int(i))) for i in order]
    v = float(np.max(np.abs(np.array(imgs)))) or 1.0
    off = l2.offsets()
    fig, axes = plt.subplots(3, 4, figsize=(15, 7.5))
    for ax, i, im in zip(axes.flat, order, imgs):
        ax.imshow(im, aspect="auto", origin="lower", cmap="bwr",
                  vmin=-v, vmax=v, interpolation="nearest",
                  extent=[off[0], off[-1], l2.code.centers[0],
                          l2.code.centers[-1]])
        ax.set_title(f"#{int(i)}  wins {int(l2.hc.wins[i])}", fontsize=8)
        ax.set_xlabel("x offset within the window", fontsize=7)
        ax.set_ylabel("y, relative to the first tap", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Layer two's templates ARE pieces of curve — each one a "
                 "shape the wave makes, stored whole and reusable wherever "
                 "that shape happens", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_window_codes(windows, path):
    """The SAME window of curve, written down four ways."""
    fig, axes = plt.subplots(1, len(windows),
                             figsize=(3.9 * len(windows), 4.6))
    for ax, (name, img, note) in zip(np.atleast_1d(axes), windows):
        v = float(np.max(np.abs(img))) or 1.0
        ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                  vmin=0, vmax=v, interpolation="nearest")
        ax.set_title(f"{name}\n{note}", fontsize=9)
        ax.set_xlabel("tap (position along x)", fontsize=8)
        ax.set_ylabel("cells of one tap", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("One window of curve, said four ways — this is the choice "
                 "of what a layer sends upward", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_bridge(res, xtr, ytr, mlp_fn, gap, path, title=""):
    """The hole, and what each interface put in it."""
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    fine = np.linspace(gap[0] - 2.2, gap[1] + 2.2, 400)
    for ax, (name, br, err) in zip(axes, res):
        ax.axvspan(*gap, color="#cccccc", alpha=0.5, lw=0, zorder=0)
        m = (xtr > fine[0]) & (xtr < fine[-1])
        ax.scatter(xtr[m], ytr[m], s=8, color=L1_C, alpha=0.3, zorder=1)
        ax.plot(fine, P.target(fine), color=TRUE_C, lw=2.2,
                label="f(x) (truth)")
        ax.plot(fine, mlp_fn(fine), color=MLP_C, lw=1.4, ls=":", label="MLP")
        kn = br["known"]
        ax.plot(br["x"][kn], br["field"][kn], "o", ms=4, color=L1_C,
                label="layer 1, taps it has data for")
        ax.plot(br["x"][~kn], br["field"][~kn], "x", ms=6, color="#999999",
                label="layer 1 inside the hole")
        ax.plot(br["x"][~kn], br["y"][~kn], "o-", ms=5, color=L2_C, lw=1.8,
                label="layer 2's completion")
        ax.set_xlim(fine[0], fine[-1])
        ax.set_xlabel("x")
        ax.set_title(f"{name}\nRMSE in the hole {err:.3f}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(title or "Bridging the hole with a second layer — same "
                 "layer one, same windows, same learning; only what a tap "
                 "SAYS differs", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_interface_bars(rows, path, title):
    names = [r["label"] for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) + 2.6))
    cols = [L2_C if r["kind"] == "l2" else "#9a9a9a" for r in rows]
    ax.barh(y, [r["rmse_hole"] for r in rows], color=cols)
    hi = max(r["rmse_hole"] for r in rows)
    for i, r in enumerate(rows):
        ax.text(min(r["rmse_hole"], hi) + hi * 0.01, i,
                f"{r['rmse_hole']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE inside the held-out hole")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
