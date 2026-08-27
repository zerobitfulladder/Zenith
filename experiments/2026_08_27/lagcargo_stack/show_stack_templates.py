"""Per-layer template galleries for the stacked lag-cargo rig.

Every unit at every layer is the same object — [cargo ; keys] — shown
as keys (top) over cargo (bottom), projected down to pixel space:
L2/L3 profiles over lower units are rendered through the lower
layers' cargo frames so we can SEE what history a unit answers to and
what successor it emits. Sorted and labeled by final-epoch home phase.

Run: .venv/bin/python experiments/2026_08_27/lagcargo_stack/show_stack_templates.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_square"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_temporal_square import D, H, PERIOD, W  # noqa: E402

RES = ROOT / "experiments" / "2026_08_27" / "lagcargo_stack" / "results" / "stack"
KS = (64, 32, 16)
z = np.load(RES / "weights.npz")
Ws = [z["W1"], z["W2"], z["W3"]]
homes = json.loads((RES / "homes.json").read_text())
in_dims = (D, KS[0], KS[1])


def relu(v):
    return np.maximum(v, 0.0)


# pixel basis per layer: L1 cargo frames; then chain upward
basis = [relu(Ws[0][:, :D])]                      # (K1, D)
basis.append(relu(Ws[1][:, :KS[0]]) @ basis[0])   # L2 cargo -> pixels
# for projecting L3 profiles over L2 units, use L2 KEYS images (era blurs)
l2_keys_img = relu(Ws[1][:, KS[0]:]) @ basis[0]


def unit_imgs(li, w):
    Wb, ind = Ws[li], in_dims[li]
    cargo, keys = relu(Wb[w, :ind]), relu(Wb[w, ind:])
    if li == 0:
        return keys.reshape(H, W), cargo.reshape(H, W)
    if li == 1:
        return (keys @ basis[0]).reshape(H, W), \
               (cargo @ basis[0]).reshape(H, W)
    return (keys @ l2_keys_img).reshape(H, W), \
           (cargo @ l2_keys_img).reshape(H, W)


def circ_mean(li, w):
    ph = homes[li].get(str(w))
    if not ph:
        return 999.0
    a = np.array(ph) * 2 * np.pi / PERIOD
    return float(np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi))


for li, name in enumerate(["L1", "L2", "L3"]):
    k = KS[li]
    ncols = 8
    nrows = int(np.ceil(k / ncols))
    order = sorted(range(k), key=lambda w: circ_mean(li, w))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 1.55))
    axes = np.atleast_2d(axes)
    for i, w in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        keys_img, cargo_img = unit_imgs(li, w)
        stack = np.vstack([keys_img / max(keys_img.max(), 1e-9),
                           np.full((1, W), 0.5, np.float32),
                           cargo_img / max(cargo_img.max(), 1e-9)])
        ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
        ph = homes[li].get(str(w), [])
        lbl = ",".join(map(str, ph)) if len(ph) <= 3 else \
            f"{len(ph)} ph {ph[0]}..{ph[-1]}"
        ax.set_title(f"u{w} [{lbl or 'unused'}]", fontsize=5.5)
        ax.axis("off")
    for i in range(k, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(f"{name}: KEYS (what history it answers to, top) over "
                 f"CARGO (the successor it names, bottom), pixel-projected",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / f"templates_{name}.png", dpi=130)
    plt.close(fig)
    print(f"wrote templates_{name}.png")
