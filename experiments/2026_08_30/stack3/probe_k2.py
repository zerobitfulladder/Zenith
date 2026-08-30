"""With layer one fixed at k1=16, what is the right k2?

Same law: the rule's fixed point is W^T W = I on the data, and
trace(W^T W) = k for unit templates, so k must match the number of
dimensions the input actually uses. Layer two's input is 3x3x16 = 144
cells; the question is how many of them carry anything.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stack3 import ConvStack, unit_rows, l2_windows, SIDE, EPS

ROOT = HERE.parents[2]
X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte = X[:20000], X[20000:20256]
hf = lambda A: float(np.mean(np.abs(np.diff(A.reshape(-1, SIDE, SIDE), axis=2))))

rng = np.random.default_rng(0)
base = ConvStack("dense", 16, 8, 0.5, rng)
base.train(Xtr, 2, rng)
Wv = unit_rows(l2_windows(base.l1_field(Xtr[:3000])))[0]
Wv = Wv[np.linalg.norm(Wv, axis=1) > EPS]
sv = np.linalg.svd(Wv[:6000], compute_uv=False)
print(f"layer-two input: 144 cells, really occupies "
      f"{sv.sum()**2/(sv**2).sum():.0f} dimensions\n")
print(f"{'k2':>4} | {'stack error':>11} {'dither':>8} {'W^T W spread':>13}")
for k2 in (32, 64, 96, 128, 144):
    rng = np.random.default_rng(0)
    cs = ConvStack("dense", 16, k2, 0.5, rng)
    cs.train(Xtr, 2, rng)
    rec = cs.backward(cs.forward(Xte), len(Xte))
    img = unit_rows(Xte.reshape(len(Xte), -1))[0]
    err = float(np.mean(np.linalg.norm(
        img - unit_rows(rec.reshape(len(Xte), -1))[0], axis=1)))
    lam = np.linalg.eigvalsh(cs.l2.W.T @ cs.l2.W)
    lam = lam[lam > 1e-9]
    print(f"{k2:>4} | {err:>11.3f} {hf(rec):>8.4f} "
          f"{lam.max()/lam.min():>13.1f}")
print(f"\noriginal images, high-frequency content: {hf(Xte):.4f}")
