"""What an L2 template looks like.

Each L2 template is 576 positions x 8 channels + 10 label slots. There is no
image to show directly, so per position we plot the NORM over the 8 channels --
"how much this template expects to happen here" -- as a 24x24 map. The label
part of the same template is printed beside it as the class it votes for.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
K1, NPOS = 8, 576
z = np.load(OUT / "l2_weights.npz")
meta = json.loads((OUT / "l2.json").read_text())

for tag, wkey, kkey in (("codes", "W_codes", "wins_codes"),
                        ("pixels", "W_pixels", "wins_pixels")):
    W, wins = z[wkey].astype(np.float64), z[kkey]
    H, K, D = W.shape
    alive = np.nonzero(wins.sum(1) > 0)[0]
    claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
    show = alive[:12]
    ncol = min(K, 8)
    fig, axes = plt.subplots(len(show), ncol,
                             figsize=(ncol * 1.05, len(show) * 1.15), squeeze=False)
    for r, h in enumerate(show):
        for j in range(ncol):
            t = W[h, j]
            if tag == "codes":
                img = np.linalg.norm(t[:NPOS * K1].reshape(NPOS, K1), axis=1)
                img = img.reshape(24, 24)
                cmap, kw = "magma", {}
            else:
                img = t[:784].reshape(28, 28)
                m = np.abs(img).max() + 1e-12
                cmap, kw = "bwr", {"vmin": -m, "vmax": m}
            a = axes[r][j]
            a.imshow(img, cmap=cmap, interpolation="nearest", **kw)
            a.set_xticks([]); a.set_yticks([])
        axes[r][0].set_ylabel(f"h{h}\nsays {claim[h]}\n{wins[h].sum()} won",
                              fontsize=5.5, rotation=0, ha="right", va="center")
    fig.suptitle(f"L2 templates — {tag}"
                 + ("  (norm over the 8 L1 channels at each position)"
                    if tag == "codes" else ""), fontsize=9)
    fig.subplots_adjust(left=.11, right=.995, top=.94, bottom=.005,
                        wspace=.06, hspace=.10)
    fig.savefig(OUT / f"l2_templates_{tag}.png", dpi=150); plt.close(fig)
    print(f"-> l2_templates_{tag}.png  ({len(alive)} live of {H})")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, (tag, kkey) in zip(ax, (("codes", "wins_codes"), ("pixels", "wins_pixels"))):
    w = z[kkey]
    im = a.imshow(w / np.maximum(w.sum(1, keepdims=True), 1), cmap="magma",
                  aspect="auto")
    a.set_xlabel("true digit"); a.set_ylabel("L2 hypercolumn")
    a.set_xticks(range(10)); a.set_title(f"what each won — {tag}", fontsize=9)
    fig.colorbar(im, ax=a, fraction=.046)
fig.tight_layout(); fig.savefig(OUT / "l2_wins.png", dpi=140); plt.close(fig)
print("-> l2_wins.png")
