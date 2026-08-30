# 2026-08-30 — templates that explain together, and what can be built on them

## `collective/` — templates that explain the digit together

Are minicolumns a codebook, where each template has to account for the input
alone, or a dictionary, where they add up? One layer, one hypercolumn, 144
templates, MNIST, each template rotated toward what the others left over.

It works as a rebuild: error 0.228 against the codebook's 0.585 and the best
possible 0.200. But the 144 learned templates are speckle — orthogonal
(largest similarity 0.004), spanning MNIST's main subspace in an arbitrary
basis. The rebuild `WᵀW x` does not change under any rotation of the templates
among themselves (scrambling them leaves the error at 0.228429 exactly), so
the error cannot prefer digit-shaped templates. Crisp rebuilds are not
evidence that the templates learned anything.

Full writeup: [`collective/README.md`](collective/README.md).

---

## `dense_stack/` — can anything be built on a layer whose templates mean nothing?

The 2026-08-29 convolutional stack with layer one swapped for the dense rule,
against untrained, PCA and winner-take-all layer ones. Across the hole it never
saw, dense reaches 0.112 RMSE, beating the MLP (0.144) and winner-take-all at
the same budget (0.173); the published sparse K1=64 stack is still best at
0.086. Layer two reads whole vectors, and their geometry survives any rotation
of the templates, so a dense layer is a channel, not a vocabulary. What
learning buys is an even spread of energy across the code (3× against PCA's
1393×), not the subspace.

Full writeup: [`dense_stack/README.md`](dense_stack/README.md).

---

## `stack3/` — three layers, a label on the top, read from both ends

Layer one on 4x4 patches, layer two on 3x3 windows of its codes, layer three
on all layer-two codes with the label joined on, winner-take-all. The label
block's energy (rho) is the knob: usable band 0.3–0.6. With dense layers one
and two, image in / label out reaches 0.9284 against logistic regression's
0.9074 on the pixels; with winner-take-all below, 0.5708. From the label alone
the stack draws digits the judge agrees with 10/10 (dense, graded read).
Identity appears only at layer three, where something has to be named.

Full writeup: [`stack3/README.md`](stack3/README.md).

---

## `td_tracks/` — flying the drone from a (situation, thrusts) memory

A sensory hypercolumn (dx, dy, angle), a motor hypercolumn (two thrusts), and
a layer two storing the pairs, trained on the PID oracle and flown with the
motor half left empty. Neither read reached the target (success 0.0 against
the oracle's 1.0). No writeup was kept.

Full writeup: [`td_tracks/README.md`](td_tracks/README.md).
