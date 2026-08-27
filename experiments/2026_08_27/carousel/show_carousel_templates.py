"""Per-layer galleries for the MNIST carousel bounce rig.

Each unit: KEYS (the lagged trail it answers to, top) over CARGO
(what it emits, bottom), projected to pixels; labeled with the digit
classes at whose ticks it won during the final training epoch.

Run: .venv/bin/python experiments/2026_08_27/carousel/show_carousel_templates.py
"""

import json
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

RES = ROOT / "experiments" / "2026_08_27" / "carousel" / "results" / "mnist_carousel_bounce"
SIDE = 28
D = SIDE * SIDE
KS = (128, 64, 16)
in_dims = (D, KS[0], KS[1])
z = np.load(RES / "weights.npz")
Ws = [z["W1"], z["W2"], z["W3"]]
homes = json.loads((RES / "homes.json").read_text())


def relu(v):
    return np.maximum(v, 0.0)


l1_cargo = relu(Ws[0][:, :D])                       # (K1, D)
l1_keys = relu(Ws[0][:, D:])
l2_cargo_px = relu(Ws[1][:, :KS[0]]) @ l1_cargo     # (K2, D)
l2_keys_px = relu(Ws[1][:, KS[0]:]) @ l1_cargo


def unit_imgs(li, w):
    if li == 0:
        return l1_keys[w], l1_cargo[w]
    if li == 1:
        return l2_keys_px[w], l2_cargo_px[w]
    keys = relu(Ws[2][w, KS[1]:]) @ l2_cargo_px
    cargo = relu(Ws[2][w, :KS[1]]) @ l2_cargo_px
    return keys, cargo


def mean_class(li, w):
    cs = homes[li].get(str(w))
    if not cs:
        return 99.0
    a = np.array(cs) * 2 * np.pi / 10
    return float(np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi))


for li, name in enumerate(["L1", "L2", "L3"]):
    k = KS[li]
    ncols = 16 if k > 64 else 8
    nrows = int(np.ceil(k / ncols))
    order = sorted(range(k), key=lambda w: mean_class(li, w))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.15, nrows * 2.15))
    axes = np.atleast_2d(axes)
    for i, w in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        keys_img, cargo_img = unit_imgs(li, w)
        stack = np.vstack([keys_img.reshape(SIDE, SIDE)
                           / max(keys_img.max(), 1e-9),
                           np.full((2, SIDE), 0.5, np.float32),
                           cargo_img.reshape(SIDE, SIDE)
                           / max(cargo_img.max(), 1e-9)])
        ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
        cs = homes[li].get(str(w), [])
        ax.set_title(f"u{w} [{','.join(map(str, cs)) or 'unused'}]",
                     fontsize=5)
        ax.axis("off")
    for i in range(k, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(f"{name}: KEYS (lagged context, top) / CARGO (emission, "
                 f"bottom); [digit classes where the unit wins]", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / f"templates_{name}.png", dpi=125)
    plt.close(fig)
    print(f"wrote templates_{name}.png")
