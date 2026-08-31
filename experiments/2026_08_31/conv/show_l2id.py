"""L2 templates on the identity message.

Each template is 36 cells x 32 L1-hypercolumn channels. Two views per template:
the 6x6 map of its norm over channels (WHERE it expects activity), and which of
the 32 L1 hypercolumns it leans on (summed over cells).
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
G, H1 = 6, 32
z = np.load(OUT / "l2_identity.npz")
W, wins = z["W"].astype(np.float64), z["wins"]
H, K, D = W.shape
claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
order = np.argsort(claim)
ncol = 8
fig, axes = plt.subplots(H, ncol + 1, figsize=(ncol * .95 + 2.2, H * .95),
                         squeeze=False)
for r, h in enumerate(order):
    T = W[h, :, :G * G * H1].reshape(K, G * G, H1)
    for j in range(ncol):
        a = axes[r][j]
        a.imshow(np.linalg.norm(T[j], axis=1).reshape(G, G), cmap="magma",
                 interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
    a = axes[r][ncol]
    prof = np.linalg.norm(T, axis=1).mean(0)
    a.bar(range(H1), prof, color="#2f8f4e", width=1.0)
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(-.5, H1 - .5)
    axes[r][0].set_ylabel(f"h{h} says {claim[h]}\n{wins[h].sum()} won",
                          fontsize=5.5, rotation=0, ha="right", va="center")
axes[0][ncol].set_title("which L1 hypercolumns", fontsize=6)
fig.suptitle("L2 templates on the identity message — 8 minicolumns per row, "
             "as 6x6 maps of where they expect activity", fontsize=9)
fig.subplots_adjust(left=.10, right=.995, top=.955, bottom=.005,
                    wspace=.07, hspace=.10)
fig.savefig(OUT / "l2_templates_identity.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
im = ax.imshow(wins / np.maximum(wins.sum(1, keepdims=True), 1), cmap="magma",
               aspect="auto")
ax.set_xlabel("true digit"); ax.set_ylabel("L2 hypercolumn")
ax.set_xticks(range(10)); ax.set_title("what each L2 hypercolumn won", fontsize=9)
fig.colorbar(im, ax=ax, fraction=.046)
fig.tight_layout(); fig.savefig(OUT / "l2_wins_identity.png", dpi=140)
print("-> l2_templates_identity.png, l2_wins_identity.png")
print("claims:", {int(h): int(claim[h]) for h in order})
