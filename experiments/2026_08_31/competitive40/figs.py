"""Regenerate the template figures from the saved weights."""
import sys, numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "competitive"))
from compete40 import H, K, OUT
from compete import load, EPS
Xtr, *_ = load(); n_img = Xtr.shape[1]
z = np.load(OUT / "weights.npz")
for mode in ("competitive", "conscience"):
    W, wins = z[mode].astype(np.float64), z[mode + "_wins"]
    tot = wins.sum(1); live = np.argsort(-tot)[: (tot > tot.sum() * 0.002).sum()]
    dom = wins.argmax(1); pur = wins.max(1) / np.maximum(tot, 1)
    print(f"{mode}: {len(live)} live of {H}; classes claimed "
          f"{sorted(set(dom[live].tolist()))}")
    for c in range(10):
        who = [int(h) for h in live if dom[h] == c]
        print(f"    digit {c}: {len(who)} experts {who}")
    n = len(live)
    fig, axes = plt.subplots(n, 12, figsize=(8.5, 0.72 * n))
    axes = np.atleast_2d(axes)
    for r, h in enumerate(live):
        for j in range(12):
            t = W[h][j, :n_img].reshape(28, 28); m = np.abs(t).max() + EPS
            axes[r, j].imshow(t, cmap="bwr", vmin=-m, vmax=m)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(f"h{h}\n\"{dom[h]}\" {pur[h]:.2f}", fontsize=5.5,
                              rotation=0, ha="right", va="center")
    fig.suptitle(f"{mode} — the {n} live hypercolumns, 12 templates each "
                 f"(row label: which digit it claimed, and how purely)", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.94, bottom=.005,
                        wspace=.05, hspace=.05)
    fig.savefig(OUT / f"02_templates_{mode}.png", dpi=135); plt.close(fig)
print(f"-> {OUT}")
