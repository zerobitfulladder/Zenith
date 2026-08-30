"""Is the dither layer one being absurdly overcomplete?

A 4x4 patch has 16 dimensions. Part three gave layer one 64 templates for
it. Rotation-only learning cannot spread 64 unit vectors in 16 dimensions
(let alone the ~10 the data actually uses), so they pile onto the same
directions and W^T W stops being anything like the identity -- the rebuild
amplifies a few directions and drops the rest, patch by patch, and that
mismatch across a 7x7 grid of independently-rebuilt patches is the dither.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stack3 import ConvStack, unit_rows, patches_of, SIDE, EPS

ROOT = HERE.parents[2]
X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte = X[:20000], X[20000:20256]

hf = lambda A: float(np.mean(np.abs(np.diff(A.reshape(-1, SIDE, SIDE), axis=2))))
print(f"original images, high-frequency content: {hf(Xte):.4f}\n")
print(f"{'k1':>4} {'k2':>5} | {'L1 alone':>9} {'W^T W spread':>13} | "
      f"{'whole stack':>11} {'its dither':>10}")

for k1 in (8, 16, 32, 64):
    rng = np.random.default_rng(0)
    cs = ConvStack("dense", k1, 128, 0.5, rng)
    cs.train(Xtr, 2, rng)

    # layer one on its own: patch -> code -> patch
    U, nrm = unit_rows(patches_of(Xte))
    keep = nrm > EPS
    R = cs.l1.decode(cs.l1.encode(U, nrm))[keep]
    T = (U * nrm[:, None])[keep]
    a = (R * T).sum(1) / np.maximum((R * R).sum(1), EPS)   # best single scale
    l1err = float(np.mean(np.linalg.norm(T - a[:, None] * R, axis=1)
                          / np.linalg.norm(T, axis=1)))
    lam = np.linalg.eigvalsh(cs.l1.W.T @ cs.l1.W)
    spread = float(lam.max() / max(lam[lam > 1e-9].min(), 1e-12))

    rec = cs.backward(cs.forward(Xte), len(Xte))
    img = unit_rows(Xte.reshape(len(Xte), -1))[0]
    err = float(np.mean(np.linalg.norm(
        img - unit_rows(rec.reshape(len(Xte), -1))[0], axis=1)))
    print(f"{k1:>4} {128:>5} | {l1err:>9.3f} {spread:>13.1f} | "
          f"{err:>11.3f} {hf(rec):>10.4f}")
