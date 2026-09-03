"""Where did the decision come from? An exact importance map from the tally read.

The read is a sum: score(y) = sum over positions p of T[win_p, cell_p, y].
So each position's contribution to the decision margin, T[.., y*] - T[.., y2]
(winning class minus runner-up), is exact, and painting it over the 9x9 window
each patch covers gives a pixel map: red argued for the decision, blue against.
No gradient, no approximation. 9x9 rig, MNIST 12k/3k, no pressure, 3 epochs.

Usage:  uv run python importance.py
"""
import sys
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "two_rung_fashion"))
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import two_rung as TR
import rf_sweep as R

R.GRID, R.BATCH = 6, 128
rig = R.Rig(9)
Xtr, ytr, Xte, yte = R.E.load("mnist")
Xtr_g = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
Xte_g = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
W = TR.train_l1(rig, Xtr_g, 7)
itr, ktr = rig.code(W, Xtr_g); ite, kte = rig.code(W, Xte_g)
T = rig.table_from(rig.build_table(itr, ytr_g, ktr))
print("9x9 table accuracy:", rig.accuracy(T, (ite, kte), yte_g))

# per-position evidence for every class on the test set
C = cp.tile(rig.cells, len(ite)).reshape(len(ite), rig.npos)
E = T[ite, C] * kte[..., None]                       # (n, npos, NL)
score = E.sum(1); pred = score.argmax(1)
order = cp.argsort(-score, axis=1); y2 = order[:, 1]
margin = E[cp.arange(len(ite)), :, pred] - E[cp.arange(len(ite)), :, y2]   # (n, npos)

# paint each patch's share over its 9x9 window
side = rig.side
def paint(v):
    img = np.zeros((28, 28)); cnt = np.zeros((28, 28))
    v = cp.asnumpy(v).reshape(side, side)
    for i in range(side):
        for j in range(side):
            img[i:i + 9, j:j + 9] += v[i, j]; cnt[i:i + 9, j:j + 9] += 1
    return img / np.maximum(cnt, 1)

pick = [int(np.where(yte == d)[0][0]) for d in range(10)]
wrong = [int(i) for i in cp.asnumpy(cp.where(pred != yte_g)[0])[:4]]
sel = pick + wrong
fig, axes = plt.subplots(3, len(sel), figsize=(1.6 * len(sel), 5.2))
for c, i in enumerate(sel):
    img = Xte[i].reshape(28, 28)
    M = paint(margin[i]); v = np.abs(M).max() + 1e-9
    Ew = paint(E[i, :, pred[i]]); vw = np.abs(Ew).max() + 1e-9
    axes[0, c].imshow(img, cmap="gray"); axes[0, c].set_title(f"true {yte[i]}  read {int(pred[i])}", fontsize=8)
    axes[1, c].imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[2, c].imshow(Ew, cmap="RdBu_r", vmin=-vw, vmax=vw)
    for r in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
axes[1, 0].set_ylabel("for the decision\n(minus runner-up)", fontsize=8)
axes[2, 0].set_ylabel(f"evidence for\nthe read class", fontsize=8)
fig.suptitle("Exact importance from the tally read, 9x9 strokes, 6x6 cells. Red argued for, blue against. "
             "Last four are misreads.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(HERE / "results" / "importance.png", dpi=140)
print("wrote results/importance.png")
