"""L2 gallery for the conv carousel rig: each code-level unit's KEYS
(context code trail) and CARGO (arriving code map), both rendered to
pixels through L1's cargo windows via overlap-add. Titles: the class
the rendered cargo/keys most resemble (by class-mean correlation).

Run: .venv/bin/python experiments/2026_08_27/carousel/show_conv_carousel_templates.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_square_completion import cn1  # noqa: E402

RES = ROOT / "experiments" / "2026_08_27" / "carousel" / "results" / "mnist_carousel_conv"
SIDE, WIN = 28, 8
NP1 = SIDE - WIN + 1
NPOS = NP1 * NP1
WD = WIN * WIN
K1, K2 = 64, 32
CODE_D = NPOS * K1

z = np.load(RES / "weights.npz")
W1, W2 = z["W1"], z["W2"]
cargo_basis = np.maximum(W1[:, :WD], 0.0)

X = np.load(ROOT / "data/mnist/digits/train_images.npy")
y = np.load(ROOT / "data/mnist/digits/train_labels.npy")
CM_N = np.stack([cn1(X[y == c].mean(axis=0).ravel()) for c in range(10)])


def render(prof):
    canvas = np.zeros((SIDE, SIDE), np.float32)
    weight = np.zeros((SIDE, SIDE), np.float32)
    mass = prof.sum(axis=1)
    thr = 0.1 * mass.max() if mass.max() > 0 else 1
    pred = prof @ cargo_basis
    for p in range(NPOS):
        if mass[p] <= thr:
            continue
        r, c = divmod(p, NP1)
        canvas[r:r + WIN, c:c + WIN] += pred[p].reshape(WIN, WIN)
        weight[r:r + WIN, c:c + WIN] += 1.0
    canvas /= np.maximum(weight, 1.0)
    return canvas / canvas.max() if canvas.max() > 0 else canvas


fig, axes = plt.subplots(4, 8, figsize=(8 * 1.5, 4 * 2.6))
for i in range(K2):
    ax = axes[i // 8, i % 8]
    keys = render(np.maximum(W2[i, CODE_D:], 0.0).reshape(NPOS, K1))
    cargo = render(np.maximum(W2[i, :CODE_D], 0.0).reshape(NPOS, K1))
    kc = int(np.argmax(CM_N @ cn1(keys.ravel())))
    cc = int(np.argmax(CM_N @ cn1(cargo.ravel())))
    ax.imshow(np.vstack([keys, np.full((2, SIDE), 0.5, np.float32), cargo]),
              cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f"u{i} ctx~{kc} -> {cc}", fontsize=6)
    ax.axis("off")
fig.suptitle("L2 code-level units: KEYS (context, top) / CARGO (arriving "
             "code, bottom), rendered through L1 windows", fontsize=10)
fig.tight_layout()
fig.savefig(RES / "templates_L2.png", dpi=135)
print("wrote", RES / "templates_L2.png")
