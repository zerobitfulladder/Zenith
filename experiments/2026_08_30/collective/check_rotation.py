"""Why nothing here can pick out a meaningful set of templates.

The rebuild is recon = W^T W x. Swap W for R W, where R is any rotation of the
144 templates among themselves:

    (RW)^T (RW) = W^T R^T R W = W^T W

The rebuild is bit-for-bit identical. So the error the layer is minimising is
blind to which basis of the subspace the templates form -- every scrambling of
them is exactly as good. Nothing in the objective can prefer digit-shaped
templates, and that is why the learned ones are speckle.
"""
from pathlib import Path

import numpy as np

from collective import center_norm_batch
from run_collective import load_mnist, rebuild_errors

W = np.load(Path(__file__).resolve().parent / "results/W_weighted.npy")
_, _, Xte, _ = load_mnist(30000, 3000, 0)
Th, _ = center_norm_batch(Xte)

Wr = np.linalg.qr(np.random.default_rng(1).standard_normal((144, 144)))[0] @ W
e0, e1 = rebuild_errors(W, Th)[0], rebuild_errors(Wr, Th)[0]
print(f"learned templates       error {e0:.6f}")
print(f"scrambled among selves  error {e1:.6f}   (difference {abs(e0 - e1):.2e})")
print(f"but the templates moved: max |W - RW| = {np.abs(W - Wr).max():.3f}")
