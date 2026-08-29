"""Pictures for the convolutional stack."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

import sys                                   # noqa: E402
from pathlib import Path                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regress"))
import pop_regress as P                      # noqa: E402
import conv_stack as C                       # noqa: E402

TRUE_C = "#111111"
DATA_C = "#5b7fa6"
PRED_C = "#c8455a"
WALK_C = "#e08e00"
MLP_C = "#3f7d20"


def _slope_order(l1):
    sh = np.array([l1.template_patch(j) for j in range(l1.hc.n_boot)])
    return np.argsort(sh[:, -1] - sh[:, 0]), sh


def save_vocabulary(l1, path):
    """Every shared template, drawn as the little shape it stands for."""
    order, sh = _slope_order(l1)
    n = len(order)
    cols = 16
    rows = int(np.ceil(n / cols))
    xs = l1.centers[0] * 0 + np.arange(l1.T)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.15),
                             sharex=True, sharey=True)
    lim = float(np.abs(sh).max()) * 1.1
    for ax, j in zip(np.atleast_1d(axes).flat, order):
        ax.plot(xs, sh[j], "-o", ms=2.2, lw=1.4, color=PRED_C)
        ax.axhline(0, color="#cccccc", lw=.6)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{j}  {int(l1.hc.wins[j])}", fontsize=5.5, pad=1.5)
    for ax in np.atleast_1d(axes).flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Layer one's whole vocabulary — {n} shared shapes, each a "
                 f"5-sample patch 0.8 wide, written relative to its own mean. "
                 f"No location attached: any of these can occur anywhere.\n"
                 f"(sorted by slope; the number after # is how often it won)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=125)
    plt.close(fig)


def save_code_field(l1, g, codes, ok, gap, path):
    """What layer one says, everywhere — the same rows recur."""
    order, _ = _slope_order(l1)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    F = np.full((l1.hc.n_boot, len(g)), np.nan)
    for i, c in enumerate(codes):
        if c is None:
            continue
        k, m = c
        F[rank[k], i] = m
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.1]})
    axes[0].plot(g, P.target(g), color=TRUE_C, lw=1.6)
    axes[0].axvspan(*gap, color="#cccccc", alpha=.55, lw=0)
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=.25)
    axes[0].set_title("the curve, and underneath it what layer one says at "
                      "every x", fontsize=11)
    im = axes[1].imshow(np.ma.masked_invalid(F), aspect="auto", origin="lower",
                        cmap="magma", extent=[g[0], g[-1], 0, l1.hc.n_boot],
                        interpolation="nearest")
    axes[1].axvspan(*gap, color="#888888", alpha=.55, lw=0)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("shared template (sorted by slope)")
    fig.colorbar(im, ax=axes[1], label="match strength", pad=.01)
    axes[1].set_title("Each column is the graded code emitted there. The same "
                      "rows light up wherever the curve does the same thing — "
                      "that is content-overlap, and it is what the "
                      "place-bound layer one could not do.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_l2_templates(l2, l1, g, path, n=8):
    """Each layer-two template, decoded back into the stretch of curve."""
    order = np.argsort(-l2.hc.wins)[:n]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, j in zip(np.atleast_1d(axes).flat, order):
        sh = l2.template_shapes(int(j), l1)
        shapes = {p: sh[p] for p in range(l2.T)}
        span = (l2.T - 1) + (l1.T - 1) * l1.step
        sol = C.stitch(shapes, l1, span + 8,
                       anchor_vals=[0.0], anchor_idx=[span // 2])
        q = np.array(sorted(sol))
        y = np.array([sol[k] for k in q])
        ax.plot((q - q.mean()) * 0.1, y - y.mean(), lw=2, color=PRED_C)
        ax.axhline(0, color="#cccccc", lw=.7)
        ax.set_title(f"#{int(j)}   wins {int(l2.hc.wins[j])}", fontsize=9)
        ax.set_xlabel("x offset in the window", fontsize=8)
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes).flat[0].set_ylabel("y (relative)", fontsize=8)
    fig.suptitle("Layer two's templates, decoded back into curve — each is a "
                 "long-range stretch built out of layer one's shared shapes",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_completed_codes(l2, l1, g, codes, ok, filled, winner, centre, path):
    """The codes layer two invented where layer one had none."""
    ti = l2.taps_at(centre)
    row = l2.hc.row(winner)
    order, _ = _slope_order(l1)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    given = np.full((l1.hc.n_boot, len(ti)), np.nan)
    made = np.full((l1.hc.n_boot, len(ti)), np.nan)
    for n, q in enumerate(ti):
        if ok[q]:
            k, m = codes[q]
            given[rank[k], n] = m
        else:
            made[rank[np.arange(l1.hc.n_boot)], n] = row[l2.pos[n]][
                np.argsort(order)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ext = [g[ti[0]], g[ti[-1]], 0, l1.hc.n_boot]
    ax.imshow(np.ma.masked_invalid(given), aspect="auto", origin="lower",
              cmap="magma", extent=ext, interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(made), aspect="auto", origin="lower",
              cmap="viridis", extent=ext, interpolation="nearest", alpha=.95)
    ax.set_xlabel("x")
    ax.set_ylabel("shared template (sorted by slope)")
    ax.set_title("The window layer two matched: warm columns are codes layer "
                 "one supplied, green columns are the codes layer two "
                 "invented where layer one had none", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_regeneration(g, val, known, pred, walk, xtr, ytr, gap, mlp_fn, path):
    miss = ~known
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, (lo, hi) in zip(axes, [(g[0], g[-1]), (gap[0]-2.2, gap[1]+2.2)]):
        ax.axvspan(*gap, color="#cccccc", alpha=.5, lw=0, zorder=0)
        m = (xtr > lo) & (xtr < hi)
        ax.scatter(xtr[m], ytr[m], s=6, color=DATA_C, alpha=.3, zorder=1)
        fine = np.linspace(lo, hi, 700)
        ax.plot(fine, P.target(fine), color=TRUE_C, lw=2, label="f(x) (truth)")
        ax.plot(fine, mlp_fn(fine), color=MLP_C, lw=1.2, ls=":", label="MLP")
        ax.plot(g[miss], walk[miss], "s--", ms=4, color=WALK_C, lw=1.3,
                label="layer 1 alone, walking")
        ax.plot(g[miss], pred[miss], "o-", ms=5, color=PRED_C, lw=2,
                label="layer 1 + layer 2 (regenerated)")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("x")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("y")
    axes[0].set_title("the whole curve", fontsize=11)
    axes[1].set_title("the hole, close up — no training pair in the grey band",
                      fontsize=11)
    axes[1].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Regenerating the missing stretch from a shared shape "
                 "vocabulary and a window of codes over it", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
