"""Draw the pupil's templates from pupil.npz.

L1: the 9x9 stroke templates (both channels when there are two: frame, and
frame-difference). L2: an object template is a 400x16 expectation of which
strokes appear in which of the 4x4 cells; drawn as a 4x4 mosaic where each
cell shows the stroke template the object expects most there, scaled by how
much it expects it. Plus the tally: which commands each object template stands
for (the marginal left/right level it reads).

Usage:  uv run python templates.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
npz = np.load(OUT / "pupil.npz"); cfg = json.load(open(OUT / "pupil.json"))
W1, W2, TL, TR = npz["W1"], npz["W2"], npz["TL"], npz["TR"]
K1, K2, ps, C, g = int(cfg["k1"]), int(cfg["k2"]), int(cfg["ps"]), int(cfg["frames"]), int(cfg["grid"])
gg = g * g
NLEV = TL.shape[1]

# ---- L1
fig, axes = plt.subplots(C, 1, figsize=(12, 6.2 * C), squeeze=False)
for ch in range(C):
    n = 20
    tile = np.zeros((n * (ps + 1), n * (ps + 1)))
    for t in range(min(K1, n * n)):
        r, c = divmod(t, n)
        w = W1[t].reshape(C, ps, ps)[ch]; v = np.abs(w).max() + 1e-9
        tile[r * (ps + 1):r * (ps + 1) + ps, c * (ps + 1):c * (ps + 1) + ps] = w / v
    axes[ch, 0].imshow(tile, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[ch, 0].set_xticks([]); axes[ch, 0].set_yticks([])
    axes[ch, 0].set_title(f"L1: all {K1} stroke templates, {ps}x{ps}, "
                          f"channel {'frame' if ch == 0 else 'frame minus previous'}", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / f"templates_L1_{C}ch.png", dpi=130); plt.close(fig)

# ---- L2: mosaic of the most-expected stroke per cell, for the 36 objects with the strongest tally
E = W2.reshape(K2, K1, gg)
strength = np.abs(TL).max(1) + np.abs(TR).max(1)          # how decided the object's command is
pick = np.argsort(-strength)[:36]
fig, axes = plt.subplots(6, 6, figsize=(13, 13.5))
for i, t in enumerate(pick):
    ax = axes[i // 6, i % 6]
    mosaic = np.zeros((g * (ps + 1), g * (ps + 1)))
    for c in range(gg):
        k = int(E[t, :, c].argmax()); a = E[t, k, c]
        w = W1[k].reshape(C, ps, ps)[0]; v = np.abs(w).max() + 1e-9
        r, cc = divmod(c, g)
        mosaic[r * (ps + 1):r * (ps + 1) + ps, cc * (ps + 1):cc * (ps + 1) + ps] = (w / v) * min(1.0, a / (np.abs(E[t]).max() + 1e-9))
    ax.imshow(mosaic, cmap="RdBu_r", vmin=-1, vmax=1)
    l, r = int(TL[t].argmax()), int(TR[t].argmax())
    ax.set_title(f"obj {t}: reads L{l} R{r}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"L2: 36 object templates with the most decided commands, drawn as the stroke each expects most in each "
             f"{g}x{g} cell (top of the image = top of the view).  Levels 0-{NLEV-1}, hover = {NLEV//2}", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / f"templates_L2_{C}ch.png", dpi=120); plt.close(fig)
print(f"wrote templates_L1_{C}ch.png, templates_L2_{C}ch.png  (frames={C})")
