"""Pictures of what the single layer learns.

A template is an 8192-long vector, but it is readable: each channel
owns 128 known positions, so gathering the template's values at those
positions gives that channel's profile — what value of that quantity
the minicolumn stands for. Ten such rows describe a whole minicolumn.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

import sl_drone as W                      # noqa: E402


def profiles(vec, enc):
    """(10, NB): one row per channel, the template's own cell profile."""
    return np.stack([vec[enc.pos[c]] for c in W.ALL_CH])


def decode_all(vec, enc):
    """Peak cell per channel -> the value that channel stands for."""
    return {c: float(W.CENTERS[c][int(np.argmax(vec[enc.pos[c]]))])
            for c in W.ALL_CH}


def save_templates(hc, enc, path, tick, n=16):
    order = np.argsort(-hc.wins)[:n]
    rows = {int(i): hc.row(int(i)) for i in order}
    P = [profiles(rows[int(i)], enc) for i in order]
    v = float(np.percentile(np.abs(np.array(P)), 99.5)) or 1.0
    fig, axes = plt.subplots(4, 4, figsize=(17, 10))
    for ax, i, prof in zip(axes.flat, order, P):
        ax.imshow(prof, aspect="auto", cmap="bwr", vmin=-v, vmax=v,
                  interpolation="nearest")
        d = decode_all(rows[int(i)], enc)
        lv = enc.read_motors(rows[int(i)])
        ax.set_title(f"#{i}  wins {int(hc.wins[i])}   "
                     f"dx{d['dx']:+.1f} dy{d['dy']:+.1f} "
                     f"tilt{np.rad2deg(d['tilt']):+.0f}deg  ->  "
                     f"L{W.LEVELS[lv[0]]:.1f} R{W.LEVELS[lv[1]]:.1f}",
                     fontsize=7)
        ax.set_yticks(range(len(W.ALL_CH)))
        ax.set_yticklabels(W.ALL_CH, fontsize=6)
        ax.set_xticks([])
        for y in (7.5,):                       # sensory | motor divider
            ax.axhline(y, color="k", lw=1.2)
    fig.suptitle(f"16 busiest minicolumns @ tick {tick} — rows are "
                 f"channels (last two are the motors), columns are the "
                 f"{W.NB} cells of each channel", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def save_sample(enc, sens, motors, path, tick, winner=None):
    """One encoded moment: what goes in, and what the query looks like."""
    full = enc.encode(sens, motors)
    query = enc.encode(sens, None)
    rows = 3 if winner is None else 4
    fig, axes = plt.subplots(rows, 1, figsize=(14, 2.6 * rows))
    for ax, vec, name in ((axes[0], full, "TRAINING vector "
                           "(state OR'ed with the oracle's commands)"),
                          (axes[1], query, "QUERY at inference "
                           "(motor rows left empty)")):
        ax.imshow(profiles(vec, enc), aspect="auto", cmap="viridis",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(W.ALL_CH)))
        ax.set_yticklabels(W.ALL_CH, fontsize=7)
        ax.set_xticks([])
        ax.set_title(name, fontsize=9)
        ax.axhline(7.5, color="w", lw=1.2)
    side = int(np.ceil(np.sqrt(enc.size)))
    pad = np.zeros(side * side, dtype=np.float32)
    pad[:enc.size] = full
    axes[2].imshow(pad.reshape(side, side), cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(f"the same vector as the raw {enc.size}-wide sparse "
                      f"array ({int(np.count_nonzero(full))} bits on, "
                      f"{100 * np.count_nonzero(full) / enc.size:.2f}%)",
                      fontsize=9)
    if winner is not None:
        axes[3].imshow(profiles(winner, enc), aspect="auto", cmap="bwr",
                       interpolation="nearest")
        axes[3].set_yticks(range(len(W.ALL_CH)))
        axes[3].set_yticklabels(W.ALL_CH, fontsize=7)
        axes[3].set_xticks([])
        lv = enc.read_motors(winner)
        axes[3].set_title(f"the minicolumn this query woke — its motor "
                          f"rows read out as L{W.LEVELS[lv[0]]:.1f} "
                          f"R{W.LEVELS[lv[1]]:.1f}", fontsize=9)
        axes[3].axhline(7.5, color="k", lw=1.2)
    vals = "  ".join(f"{k}{v:+.2f}" for k, v in sens.items())
    fig.suptitle(f"a moment @ tick {tick}:  {vals}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
