"""What the data looks like: eight examples of each of the ten classes."""

from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, CLASSES

OUT = Path(__file__).resolve().parent / "results"
Xtr, ytr, _, _ = load("fashion_mnist")
fig, axes = plt.subplots(10, 8, figsize=(6.4, 8.2))
for c in range(10):
    idx = np.where(ytr == c)[0][:8]
    for j, a in enumerate(axes[c]):
        a.imshow(Xtr[idx[j]].reshape(28, 28), cmap="gray"); a.set_xticks([]); a.set_yticks([])
    axes[c, 0].set_ylabel(CLASSES[c], fontsize=6, rotation=0, ha="right", va="center")
fig.suptitle("Fashion-MNIST", fontsize=10)
fig.subplots_adjust(left=.16, right=.99, top=.96, bottom=.005, wspace=.05, hspace=.05)
fig.savefig(OUT / "00_samples.png", dpi=130)
print("->", OUT / "00_samples.png")
