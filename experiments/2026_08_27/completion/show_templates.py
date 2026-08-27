"""Template gallery for the completion-arrow rig.

L1 rows: the stored frame-trail segment, shown as an 8x24 image.
L2/L3 rows: projected down to pixel space (row's unit-profile times
the layer below's frame parts) so we can SEE what stretch of the
timeline each unit covers. Each panel is titled with the row's home
phases from the final training epoch (phase = t mod 40).

Run: .venv/bin/python experiments/2026_08_27/completion/show_templates.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "experiments" / "2026_08_27" / "completion" / "results"
H, W, D, K1, K2 = 8, 24, 192, 64, 32

z = np.load(RES / "weights.npz")
W1, W2, W3 = z["W1"], z["W2"], z["W3"]
homes = json.loads((RES / "homes.json").read_text())


def relu(v):
    return np.maximum(v, 0.0)


def phase_label(li, w):
    ph = homes[li].get(str(w))
    if not ph:
        return "unused"
    return ",".join(str(p) for p in ph) if len(ph) <= 4 else \
        f"{len(ph)} ph: {ph[0]}..{ph[-1]}"


def circ_mean_phase(li, w):
    ph = homes[li].get(str(w))
    if not ph:
        return 999.0
    a = np.array(ph) * 2 * np.pi / 40
    return float(np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi))


L1_imgs = relu(W1[:, :D])
L2_imgs = relu(W2[:, :K1]) @ L1_imgs          # (K2, D)
L3_imgs = relu(W3[:, :K2]) @ relu(W2[:, :K1]) @ L1_imgs


def gallery(imgs, li, name, ncols=8):
    k = imgs.shape[0]
    order = sorted(range(k), key=lambda w: circ_mean_phase(li, w))
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 1.1))
    axes = np.atleast_2d(axes)
    for i, w in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        img = imgs[w].reshape(H, W)
        m = img.max()
        ax.imshow(img / m if m > 0 else img, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"u{w} [{phase_label(li, w)}]", fontsize=5)
        ax.axis("off")
    for i in range(k, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(f"{name} templates, sorted by home phase "
                 f"(brightness = weight, projected to pixels)", fontsize=9)
    fig.tight_layout()
    out = RES / f"templates_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


gallery(L1_imgs, 0, "L1")
gallery(L2_imgs, 1, "L2")
gallery(L3_imgs, 2, "L3")

for li, (name, k) in enumerate([("L1", K1), ("L2", K2), ("L3", W3.shape[0])]):
    spans = []
    for w in range(k):
        ph = homes[li].get(str(w))
        if ph and len(ph) > 1:
            p = np.array(ph)
            d = np.diff(np.sort(p))
            spread = 40 - d.max() if len(d) else 0   # circular span
            spans.append(spread)
        elif ph:
            spans.append(0)
    if spans:
        print(f"{name}: {len(spans)} used rows, mean phase-span "
              f"{np.mean(spans):.1f} frames, max {np.max(spans)} "
              f"(era should span ~{[3, 9, 27][li]})")
