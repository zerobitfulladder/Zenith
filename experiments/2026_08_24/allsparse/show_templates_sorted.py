"""All 512 L1 templates (8x8), greedy-sorted by similarity so clone
families appear as adjacent runs. Loads W1 from the saved all-sparse
weights (L1 is identical across arms — it trains on pixels only)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "experiments" / "2026_08_24" / "allsparse" / "results"

W = np.load(DIR / "weights_K64_100_dense.npz")["W1"]          # (512, 64)
C = W @ W.T

# Greedy similarity chain: start at the template with the highest total
# similarity, repeatedly append the nearest unused template.
n = len(W)
used = np.zeros(n, dtype=bool)
cur = int(np.abs(C).sum(axis=1).argmax())
order = [cur]
used[cur] = True
for _ in range(n - 1):
    sims = C[cur].copy()
    sims[used] = -np.inf
    cur = int(sims.argmax())
    order.append(cur)
    used[cur] = True

cols, rows = 22, 24
fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.55, rows * 0.62))
for ax in np.ravel(axes):
    ax.axis("off")
for i, t in enumerate(order):
    ax = np.ravel(axes)[i]
    p = W[t].reshape(8, 8)
    m = np.abs(p).max() + 1e-9
    ax.imshow(p, cmap="gray", vmin=-m, vmax=m)
fig.suptitle("All 512 L1 templates (8x8), similarity-sorted — "
             "clone families appear as adjacent runs", fontsize=11)
fig.tight_layout()
fig.savefig(DIR / "templates_8x8_K512_sorted.png", dpi=140)
print("written:", DIR / "templates_8x8_K512_sorted.png")

# Chain-neighbor similarity profile: how tight are consecutive pairs?
adj = np.array([C[order[i], order[i + 1]] for i in range(n - 1)])
print(f"adjacent-pair cosine: median={np.median(adj):.3f} "
      f"p90={np.percentile(adj, 90):.3f} "
      f"share>0.8={float((adj > 0.8).mean()):.2f} "
      f"share>0.9={float((adj > 0.9).mean()):.2f}")
