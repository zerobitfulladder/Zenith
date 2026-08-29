"""Pictures for the population-coded regression rig."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

import pop_regress as P                      # noqa: E402

TRAIN_C = "#5b7fa6"
TRUE_C = "#111111"
MODEL_C = "#d1495b"
SOFT_C = "#e08e00"


def _gap(ax, gap=P.GAP):
    ax.axvspan(*gap, color="#cccccc", alpha=0.5, lw=0, zorder=0)


def save_dataset(xtr, ytr, grid, path, gap=P.GAP):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _gap(ax, gap)
    ax.plot(grid, P.target(grid), color=TRUE_C, lw=2, label="f(x) (truth)")
    ax.scatter(xtr, ytr, s=9, color=TRAIN_C, alpha=0.75,
               label=f"training pairs (n={len(xtr)})")
    ax.text(np.mean(gap), P.Y_HI - 0.4, "held out\n(never trained)",
            ha="center", va="top", fontsize=9, color="#555555")
    ax.set_xlim(P.X_LO, P.X_HI)
    ax.set_ylim(P.Y_LO, P.Y_HI)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("The task: learn f(x) from noisy pairs, with a slice of x "
                 "removed from training", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_encoding(enc, pairs, path):
    """The fall-off, then a few pairs as codes, then the raw array."""
    side = int(np.ceil(np.sqrt(enc.size)))
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(3, len(pairs), height_ratios=[1.15, 0.72, 1.5],
                          hspace=0.5, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    for v, c in ((2.0, "#d1495b"), (2.6, "#e08e00"), (-3.0, "#3f7d20")):
        cells, w = enc.bump("x", v)
        full = np.zeros(enc.nb)
        full[cells] = w
        ax.plot(enc.centers["x"], full, color=c, lw=2, label=f"x = {v:+.1f}")
        ax.fill_between(enc.centers["x"], full, color=c, alpha=0.15)
    ax.set_xlim(P.X_LO, P.X_HI)
    ax.set_xlabel("value")
    ax.set_ylabel("cell brightness")
    ax.set_title("How a number is written down: the value's own cell is "
                 "brightest, neighbours fade with distance.\n2.0 and 2.6 "
                 "share most of their cells, -3.0 shares none — that overlap "
                 "is the only thing letting an unseen number mean anything.",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    for j, (x, y) in enumerate(pairs):
        dense = enc.encode(x, y)
        a = fig.add_subplot(gs[1, j])
        a.imshow(enc.profiles(dense), aspect="auto", cmap="viridis",
                 vmin=0, vmax=1, interpolation="nearest")
        a.set_yticks([0, 1])
        a.set_yticklabels(["x cells", "y cells"], fontsize=8)
        a.set_xticks([])
        a.set_title(f"x = {x:+.2f}   y = {y:+.2f}", fontsize=9)

        pad = np.zeros(side * side, dtype=np.float32)
        pad[:enc.size] = dense
        b = fig.add_subplot(gs[2, j])
        b.imshow(pad.reshape(side, side), cmap="magma", vmin=0, vmax=1,
                 interpolation="nearest")
        b.set_xticks([])
        b.set_yticks([])
        on = int(np.count_nonzero(dense))
        b.set_title(f"both bumps OR'ed into one {enc.size}-cell array\n"
                    f"{on} cells on ({100 * on / enc.size:.1f}%)", fontsize=8)
    fig.suptitle("Encoding: one sparse vector per pair, saying “at this x, "
                 "the answer was this y”", fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_templates(hc, enc, path, n=16):
    order = np.argsort(-hc.wins)[:n]
    rows = [hc.row(int(i)) for i in order]
    prof = [enc.profiles(r) for r in rows]
    # scale to the rows actually shown: the other 8000 cells are the tiny
    # negative offset that keeps a template centred, and letting them set
    # the scale saturates every picture.
    v = float(np.max(np.abs(np.array(prof)))) or 1.0
    fig, axes = plt.subplots(4, 4, figsize=(16, 8))
    for ax, i, r in zip(axes.flat, order, rows):
        ax.imshow(enc.profiles(r), aspect="auto", cmap="bwr",
                  vmin=-v, vmax=v, interpolation="nearest")
        yv, _ = enc.sharpen(r[enc.pos["y"]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["x", "y"], fontsize=8)
        ax.set_xticks([])
        ax.set_title(f"#{int(i)}  wins {int(hc.wins[i])}   "
                     f"x{enc.decode_x(r):+.2f} → y{yv:+.2f}", fontsize=8)
    fig.suptitle("The 16 busiest minicolumns — each became one bump on the x "
                 "row bound to one bump on the y row: a local piece of the "
                 "curve, stored as a single template", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_template_map(hc, enc, xtr, ytr, grid, path, gap=P.GAP):
    xs = np.array([enc.decode_x(hc.row(i)) for i in range(hc.n_boot)])
    ys = np.array([enc.sharpen(hc.row(i)[enc.pos["y"]])[1]
                   for i in range(hc.n_boot)])
    ws = hc.wins[:hc.n_boot]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _gap(ax, gap)
    ax.scatter(xtr, ytr, s=8, color=TRAIN_C, alpha=0.35, label="training pairs")
    ax.plot(grid, P.target(grid), color=TRUE_C, lw=1.6, label="f(x) (truth)")
    sc = ax.scatter(xs, ys, s=np.clip(ws * 0.35, 6, 80), c=ws, cmap="autumn_r",
                    edgecolor="k", linewidth=0.3, zorder=3,
                    label="minicolumns (what each one stands for)")
    fig.colorbar(sc, ax=ax, label="times it won during training")
    ax.set_xlim(P.X_LO, P.X_HI)
    ax.set_ylim(P.Y_LO, P.Y_HI)
    ax.set_xlabel("x the minicolumn stands for")
    ax.set_ylabel("y it answers")
    ax.set_title(f"All {hc.n_boot} minicolumns, decoded — the population has "
                 f"tiled the curve, and nothing at all sits in the held-out "
                 f"slice", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_predictions(grid, pred, base, xtr, ytr, path, gap=P.GAP,
                     left_title="One hypercolumn, top-1 winner, y read off "
                                "the winner"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax in axes:
        _gap(ax, gap)
        ax.scatter(xtr, ytr, s=7, color=TRAIN_C, alpha=0.3, zorder=1)
        ax.plot(grid, P.target(grid), color=TRUE_C, lw=2.2,
                label="f(x) (truth)")
        ax.set_xlim(P.X_LO, P.X_HI)
        ax.set_ylim(P.Y_LO, P.Y_HI)
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)
    axes[0].plot(grid, pred["peak"], color=MODEL_C, lw=1.6,
                 label="brightest output cell (top-1)")
    axes[0].plot(grid, pred["centroid"], color=SOFT_C, lw=1.3, ls="--",
                 label="same output, read at sub-cell precision")
    axes[0].set_ylabel("y")
    axes[0].set_title(left_title, fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8.5)
    for name, yhat in base.items():
        axes[1].plot(grid, yhat, lw=1.5, label=name)
    axes[1].set_title("Off-the-shelf regressors on exactly the same pairs",
                      fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_readout_field(grid, pred, enc, path, gap=P.GAP,
                       probes=(-4.3, -0.5, 2.0, 4.6), note=""):
    fig = plt.figure(figsize=(15, 8.8))
    gs = fig.add_gridspec(2, len(probes), height_ratios=[1.55, 1.0],
                          hspace=0.36, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    f = pred["field"]
    v = float(np.percentile(np.abs(f), 99.7)) or 1.0
    ax.imshow(f, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=v,
              extent=[P.X_LO, P.X_HI, P.Y_LO, P.Y_HI], interpolation="nearest")
    ax.plot(grid, P.target(grid), color="#00e5ff", lw=1.8, label="f(x) (truth)")
    ax.plot(grid, pred["peak"], color="w", lw=1.0, ls="--",
            label="brightest cell (what generation emits)")
    for g in gap:
        ax.axvline(g, color="w", lw=0.9, alpha=0.55)
    ax.set_xlabel("x  (the query)")
    ax.set_ylabel("y  (the answer's cells)")
    ax.set_title("The answer as it actually comes out — blurry. Every column "
                 "is the y row read back for that x; brightness is how "
                 "strongly each y is asserted." + note, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.35,
              labelcolor="w", facecolor="#222222")

    cen = enc.centers["y"]
    for j, px in enumerate(probes):
        i = int(np.argmin(np.abs(grid - px)))
        a = fig.add_subplot(gs[1, j])
        prof = pred["field"][:, i]
        a.fill_between(cen, np.clip(prof, 0, None), color=MODEL_C, alpha=0.35)
        a.plot(cen, prof, color=MODEL_C, lw=1.5)
        a.axvline(P.target(grid[i]), color=TRUE_C, lw=1.6, label="true y")
        a.axvline(pred["peak"][i], color=SOFT_C, lw=1.4, ls="--",
                  label="brightest")
        inside = gap[0] <= grid[i] <= gap[1]
        a.set_title(f"x = {grid[i]:+.2f}" + ("   (held-out slice)" if inside
                                             else ""), fontsize=9)
        a.set_xlabel("y")
        a.set_xlim(P.Y_LO, P.Y_HI)
        a.grid(alpha=0.25)
        if j == 0:
            a.set_ylabel("cell value")
            a.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_width_sweep(rows, sparse, path):
    hw = [r["halfw"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(hw, [r["rmse_support"] for r in rows], "o-", color=TRAIN_C,
                 label="800 training pairs (dense)")
    axes[0].plot([r["halfw"] for r in sparse],
                 [r["rmse_support"] for r in sparse], "s-", color=MODEL_C,
                 label="60 training pairs (sparse)")
    axes[0].set_ylabel("RMSE on unseen x inside the training range")
    axes[0].set_title("With 60 examples, a one-cell code cannot answer an x\n"
                      "it has not literally seen. Overlap fixes that.",
                      fontsize=10)
    axes[1].plot(hw, [r["rmse_gap"] for r in rows], "o-", color="#3f7d20")
    axes[1].set_ylabel("RMSE inside the held-out slice")
    axes[1].set_title("And the same holds, with 800 examples, for x values\n"
                      "in the hole — sharper codes say nothing sensible.",
                      fontsize=10)
    for ax in axes[:2]:
        ax.set_yscale("log")
    axes[2].plot(hw, [r["distinct_winners"] for r in rows], "o-",
                 color=TRAIN_C)
    axes[2].set_ylabel("distinct minicolumns used")
    axes[2].set_title("Past ~16 cells the blur is too wide: different x\n"
                      "values stop being distinguishable at all", fontsize=10)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bump half-width (cells, out of 96)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_selection(rows, path):
    """Selection rule x capacity — the partial-query norm problem."""
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7))
    styles = {"dot": ("o-", MODEL_C,
                      "plain dot product (the drone's rule)"),
              "masked": ("s-", "#3f7d20",
                         "divided by the template's norm over the "
                         "queried cells")}
    for mode, (st, c, lab) in styles.items():
        sel = [r for r in rows if r["mode"] == mode]
        axes[0].plot([r["k"] for r in sel], [r["rmse_support"] for r in sel],
                     st, color=c, label=lab)
        axes[1].plot([r["k"] for r in sel], [r["distinct_winners"] for r in sel],
                     st, color=c, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE on unseen x inside the training range")
    axes[0].set_title("Adding minicolumns makes the plain rule WORSE",
                      fontsize=10)
    axes[1].set_ylabel("distinct minicolumns actually used")
    axes[1].set_title("...because a few of them monopolise the queries",
                      fontsize=10)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("minicolumns in the hypercolumn")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def save_gap_sweep(rows, path):
    w = [r["width"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(w, [r["rmse_model"] for r in rows], "o-", color=MODEL_C,
            label="hypercolumn")
    ax.plot(w, [r["rmse_edge"] for r in rows], "^--", color="#777777",
            label="hold the value at the nearer rim of the hole")
    ax.plot(w, [r["rmse_mlp"] for r in rows], "s-", color="#3f7d20",
            label="MLP (64-64, tanh)")
    ax.set_xlabel("width of the hole cut out of training (in x)")
    ax.set_ylabel("RMSE inside the hole")
    ax.set_title("How far can it bridge? The hypercolumn tracks the "
                 "rim-holding line: it answers with its nearest stored piece "
                 "of curve,\nit does not continue the curve through the hole. "
                 "A fitted function does.", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
