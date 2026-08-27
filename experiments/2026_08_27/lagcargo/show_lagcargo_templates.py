"""Template gallery for the lag-cargo bank.

Each unit is one stored arrow: keys (the lagged trail it answers to,
top image) over cargo (the observed successor frame it emits, bottom
image). Units sorted and labeled by the phase at which they win a
cargo-empty query (teacher-forced pass over three periods).

Run: .venv/bin/python experiments/2026_08_27/lagcargo/show_lagcargo_templates.py
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
from run_temporal_square import D, H, PERIOD, W, frame_at  # noqa: E402

RES = ROOT / "experiments" / "2026_08_27" / "lagcargo" / "results"
G1 = 0.5
Wb = np.load(RES / "weights.npz")["W"]
K = Wb.shape[0]

homes = {}
T1 = np.zeros(D, np.float32)
for t in range(PERIOD + 3 * PERIOD):
    T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
    if t >= PERIOD:
        q = cn1(np.concatenate([np.zeros(D, np.float32), cn1(T1)]))
        w = int(np.argmax(Wb @ q))
        homes.setdefault(w, set()).add(t % PERIOD)


def circ_mean(w):
    ph = homes.get(w)
    if not ph:
        return 999.0
    a = np.array(sorted(ph)) * 2 * np.pi / PERIOD
    return float(np.angle(np.mean(np.exp(1j * a))) % (2 * np.pi))


order = sorted(range(K), key=circ_mean)
fig, axes = plt.subplots(8, 8, figsize=(8 * 2.2, 8 * 1.55))
for i, w in enumerate(order):
    ax = axes[i // 8, i % 8]
    keys = np.maximum(Wb[w, D:], 0.0).reshape(H, W)
    cargo = np.maximum(Wb[w, :D], 0.0).reshape(H, W)
    stack = np.vstack([keys / max(keys.max(), 1e-9),
                       np.full((1, W), 0.5, np.float32),
                       cargo / max(cargo.max(), 1e-9)])
    ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
    ph = sorted(homes.get(w, []))
    lbl = ",".join(map(str, ph)) if len(ph) <= 3 else \
        f"{len(ph)} ph {ph[0]}..{ph[-1]}"
    ax.set_title(f"u{w} [{lbl or 'unused'}]", fontsize=5.5)
    ax.axis("off")
fig.suptitle("Lag-cargo units: KEYS (lagged trail, top) over CARGO "
             "(observed next frame, bottom), sorted by winning phase",
             fontsize=10)
fig.tight_layout()
fig.savefig(RES / "templates.png", dpi=130)
print("wrote", RES / "templates.png",
      f"| used units: {len(homes)}/{K}")
