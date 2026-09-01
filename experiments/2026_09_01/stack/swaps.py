"""36% of layer-1 winners change under feedback and nothing happens. Why?

Two candidates:
  the swaps are between NEAR-DUPLICATE templates, so the code changes and the
  information does not
  the swaps are real and their gains and losses cancel

Measured: how similar the old and new templates are, against the similarity of
random pairs; and how much the class evidence at that position actually moves.
"""
import sys
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E, gpu_stack as G, gpu_merge as M, gpu_bidir as BD

DS, SIDE, K1 = "mnist", G.SIDE, G.K1
Xtr, ytr, Xte, yte = E.load(DS)
Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
ytr_g = cp.asarray(ytr)
rng = np.random.default_rng(7)
W1, _ = G.l1_train(Xtr, ytr_g, rng)
itr, mtr, ktr = G.l1_code(W1, Xtr); ite, mte, kte = G.l1_code(W1, Xte)
T1, _ = G.build_table(itr, G.C1, ytr_g, K1, G.GRID1, ktr)
W2, _ = BD.train_l2(itr, mtr, ytr_g, np.random.default_rng(3))
jte = BD.l2_map(W2, ite, mte)
Td = BD.count_down(itr, BD.l2_map(W2, itr, mtr))

Wf = W1.astype(cp.float32)
n = 512
Q, _, _ = G.patches(Xte[:n])
err = (1.0 - (Q.reshape(-1, 25) @ Wf.T) ** 2).reshape(n, SIDE * SIDE, K1)
old = err.argmin(2)
new = (err - 0.3 * BD.feedback_bias(Td, jte[:n], n)).argmin(2)
k = kte[:n]
ch = (old != new) & k
print(f"\n  winners changed: {float(ch.mean()) * 100:.1f}% of kept positions")

# how alike are the swapped-in and swapped-out templates?
G1 = Wf @ Wf.T
cos_sw = float(cp.abs(G1[old[ch], new[ch]]).mean())
off = ~cp.eye(K1, dtype=bool)
print(f"  |cosine| old-vs-new template      {cos_sw:.3f}")
print(f"  |cosine| random template pair     {float(cp.abs(G1[off]).mean()):.3f}")
print(f"  |cosine| a template with itself    1.000")

# how much does the class evidence at that position actually move?
cells = cp.tile(G.C1, n).reshape(n, -1)
eo, en = T1[old, cells], T1[new, cells]
d = cp.linalg.norm(eo - en, axis=2)[ch]
mag = cp.linalg.norm(eo, axis=2)[ch]
print(f"  class-evidence change / its size  {float((d / cp.maximum(mag, 1e-9)).mean()):.3f}")
print(f"  argmax class unchanged at         "
      f"{float((eo[ch].argmax(1) == en[ch].argmax(1)).mean()) * 100:.1f}% of swaps")
