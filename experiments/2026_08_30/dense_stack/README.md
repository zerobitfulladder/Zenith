# Part two: can anything be built on a layer whose templates mean nothing?

The layer above says the dense templates are arbitrary. The obvious next
question is whether that disqualifies them from being a *layer one* —
whether anything can be composed on top of units that have no identity.

So: the 2026-08-29 convolutional stack, unchanged, with layer one swapped
out. Same 1600 noisy pairs from a wave, same 1.6-wide hole cut out of
training entirely, same layer two, same least-squares decode. Only layer
one differs, four ways:

* **dense** — 144 templates, all scoring, all learning from the leftover
  (today's `weighted` rule)
* **dense-random** — the same 144, never trained
* **pca** — 144 fixed directions from one SVD of the patches, no learning
* **wta** — 144 templates, winner-take-all: the published 08-29 layer one
  at the same budget

## It works

| across the hole | RMSE |
|---|---|
| *published 08-29 wta stack, K1=64* | *0.086* |
| **dense L1 (144) + L2 over its codes** | **0.112** |
| MLP (64-64, tanh) | 0.144 |
| wta L1 (144) + L2 | 0.173 |
| dense-random L1 + L2 | 0.209 |
| pca L1 + L2 | 0.255 |
| conv L1 alone, walking | 0.428 |
| hold the value at the nearer rim | 0.743 |

The meaningless layer carries a working stack. It beats the fitted
network and comfortably beats winner-take-all at the same template
budget. In `regeneration.png` the red line tracks the truth across a hole
it never saw.

(Two honest notes. The best result on this task is still the *sparse*
one — the published K1=64 stack at 0.086 — and winner-take-all gets worse
as K1 grows, which 08-29's own vocabulary table already predicted, so
0.173 at K1=144 is not news about codebooks. And layer two picked
template #65 in every arm, so the arms really are being compared on layer
one alone.)

## But not by composition

Layer two never reads a template's name. It reads the whole 144-vector.
Every inner product between two codes is `x₁ᵀWᵀW x₂`, and that is
invariant to any rotation of the templates among themselves — **the exact
same invariance that destroys template identity guarantees that the
geometry survives it.** Two similar patches produce two similar codes no
matter which arbitrary basis the layer landed in.

Measured on the patches (`probe_dense.py`):

| | rebuild error | does patch similarity survive into the code? | spread of energy across the 144 entries |
|---|---|---|---|
| trained | 0.026 | 0.9998 | 3× |
| untrained | 0.933 | 0.9858 | 3× |
| pca | 0.0000 | 0.9999 | 1393× |

An **untrained** random layer already preserves patch similarity at
0.9858, and already gets 0.209 — better than layer one walking on its own
(0.428) and far better than the trivial answer (0.743). Most of what
layer one contributes is *having a faithful vector code at all*, which is
a property of random projection, not of learning.

So the templates are not parts, and nothing above is assembling them.
`l1_vocabulary_dense.png` is jagged zigzags — basis functions — where
08-29's winner-take-all layer learned "steep descents through flats to
steep rises". **The dense layer is a channel, not a vocabulary.**

## What the learning is actually worth — and it isn't the subspace

Training buys 0.209 → 0.112, and the reason is not that it found the
right subspace. `pca` finds that subspace exactly: rebuild error
**0.0000**, geometry **0.9999**. It is also the **worst arm in the
table**, at 0.255.

The difference is the last column. PCA's code energy is spread **1393×**
— the strongest entry carries 0.679, the median carries 0.0005. Every bit
of information is crammed into a handful of coordinates and the other
~140 are numerically dead. Layer two matches by correlation over cells,
so a code like that hands it about five usable numbers per tap.

The geodesic rule cannot do that, because every template is pinned to
unit length and can only rotate. It has no way to let one direction hog
the energy, so it settles into an equal-norm tight frame and the code
comes out flat — spread **3×**, all 144 entries carrying real signal.
That is whitening, and it is the whole of what layer one's learning is
worth here: **not the subspace, but the even spread of energy across it.**

## What this settles

Their guess was right about composition and wrong about usefulness. Dense
templates are perfectly good at feeding a layer that consumes whole
vectors — better than a codebook at the same budget, and better still for
being whitened rather than PCA'd. Nothing above them is composing parts,
because there are no parts; layer two is doing all the structural work on
what is essentially a well-conditioned rotation of its input.

The cost lands the moment anything upward wants to *name* a unit. A
gallery, a top-1 read, skeleton learning, a partial cue addressed by
identity — all of those ask "which template", and this layer has no
answer. Dense below is a decision to put every eggs in layer two's
basket, and to give up reading the stack from the inside.

## Files

| | |
|---|---|
| `dense_conv.py` | conv_stack's layer one with the column swapped; dense-aware decode (takes the hypercolumn from [`../collective/collective.py`](../collective/collective.py)) |
| `run_dense_stack.py` | the four arms, the pictures, `results/metrics.json` |
| `probe_dense.py` | rebuild, geometry preservation, code energy spread |
| `results/regeneration.png` | the hole, filled, every arm |
| `results/l1_vocabulary_*.png` | layer one's templates as 5-sample patches |
| `results/l1_code_field_*.png` | what layer one says at every x |

    python experiments/2026_08_30/dense_stack/run_dense_stack.py     # four arms, ~2 min
    python experiments/2026_08_30/dense_stack/probe_dense.py         # the mechanism numbers
