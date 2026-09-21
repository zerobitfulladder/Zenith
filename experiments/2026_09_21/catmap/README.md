# 2026-09-21 — catmap: is the torus wrap enough nonlinearity to classify?

**[VISION]** Learn on the n-torus instead of flat space. The space wraps, so a network of
linear layers should be able to learn nonlinear things by backprop with no ReLU, because the
wrap *is* the nonlinearity. Arnold's cat map shows the layer: an invertible linear map of the
torus. Stack them, train for classification, and the whole network is a bijection.

This folder answers only that question. Generation is dropped. The sibling folder
[`../nice_flow/`](../nice_flow/README.md) is an earlier build that put a ReLU MLP inside every layer and trained for
likelihood; it is NICE on a torus and does not test the claim.

Script: `catmap.py`. Sweep: `sweep.sh`. Every run's per-epoch history is in `results/runs/*.json`.

**The one-line result.** A stack of linear shears with wraps between them classifies MNIST
at **97.3%**, far past logistic regression at 91.4%. The same stack with the wraps removed,
which is a *single linear map*, scores **97.1%**. The wraps between layers do nothing that a
linear map cannot do. What supplies the nonlinearity is the **cosine readout against the
class chords**, which is a cleanup, and it is there in both arms.

**The follow-up (§III.1).** Make the shear periodic, so that every gradient in the network is
circular: the shift reads the untouched half through sine and cosine instead of as numbers.
That scores **98.3%**, above the ReLU-MLP ceiling, trains without any runaway, and is
*exactly the same function with the wraps removed*. In a torus-native network the wrap is
not an operation at all. The nonlinearity is the torus's own trigonometry, never the wrap.

---

## I. The construction

Every pixel is an angle. A layer picks half the angles (checkerboard, alternating), leaves
them alone, and shifts the other half by a **linear** map of the untouched half, then wraps:

```
theta_B  <-  theta_B + W theta_A + b     (mod 2pi)
theta_A  <-  theta_A
```

Inverse is subtraction. Arnold's cat map is this on two angles with W = 1; here W is real
and learned, 392×392 per layer. There is no ReLU, sigmoid, softmax or cos/sin embedding
anywhere in the stack. Angles live in [−π, π); pixel x maps to span·(x − ½).

Readout: mean cosine of all 784 output angles against ten fixed random chords, scaled by a
learned κ, softmax, cross-entropy. Label on the whole torus, no residual split.

Arms, everything else identical (Adam 1e-4, batch 256, 15 epochs, identity init):

| arm | what it is |
|---|---|
| **wrap** | the claim. Wrap after every layer |
| **no wrap** | control. Same weights, no wrap between layers. The L shears collapse to one linear map before the readout |
| **MLP** | ceiling. The shift is `W2 relu(W1 θ_A + b1) + b2`: the linear arm plus a ReLU. This is NICE |
| **logreg** | linear readout on raw pixels |

Every run records, per layer, the fraction of shifted angles that actually crossed ±π, and
the mean shift in turns. If that were near zero the wrap arm would be linear in practice and
the comparison would be void.

---

## II. Result [MEASURED]

Span = π (data on half the circle, no seam). Test accuracy after 15 epochs, seed 0.

| depth | wrap | no wrap | MLP |
|---|---|---|---|
| 2 | collapsed (ep 8, peak 0.912) | 0.960 | 0.430 (layer 1 ran away, 220 turns) |
| 4 | 0.968 | 0.966 | 0.980 |
| 8 | **0.973** | **0.972** | 0.980 |
| 16 | 0.973 | 0.973 | 0.979 |
| logistic regression | | 0.914 | |

Three seeds at depth 8: wrap 0.9731 / 0.9726 / 0.9721, no wrap 0.9717 / 0.9721 / 0.9702.
The gap is 0.0013, one seed standard deviation.

Random shuffle instead of checkerboard, depth 8: wrap 0.971. Same story.

[`accuracy.png`](results/accuracy.png), [`wrapfrac.png`](results/wrapfrac.png).

---

## III. What it means

- **[NEGATIVE] The wraps between layers do not classify.** Wrap and no-wrap are equal at
  every depth that trains, to within seed noise. The no-wrap arm is one linear map followed
  by the readout, so anything the wrap arm knows, a linear map knows too.
- **[MEASURED] The readout is the nonlinearity.** One linear map plus the cosine cleanup
  beats logistic regression by 5.7 points. `mean_i cos((Mθ)_i − chord_yi)` is a one-hidden-
  layer network with 784 cosine units and class-specific phases. That is where the torus does
  its work: at the cleanup, not in the stack. This is also what Nanda et al. found the
  modular-addition network doing, and what the choir has always done.
- **[MEASURED] The wraps fire, and the trained network depends on them.** At depth 8 the
  last two layers wrap 16–18% of their angles; earlier layers wrap almost none. Switch the
  wraps off in the trained wrap network and accuracy drops from 0.973 to 0.68–0.73. So the
  network found a solution that *uses* wraps. It is just not a better solution.
- **[MEASURED] The ReLU is worth 0.7 points.** MLP 0.980 against 0.973, at 5× the
  parameters. The linear-on-the-torus network gets within 0.7 of NICE's coupling with no
  nonlinearity in the stack. That is the honest positive: the torus readout carries almost
  all of it.
- **Depth helps the no-wrap arm too**, 0.960 → 0.973. Since that arm is linear at every
  depth, this is the *parameterisation* of the linear map improving (two alternating shears
  cannot express every matrix; eight can), not nonlinearity. **[CONFOUND]** noted so the
  depth curve is not read as evidence for the wrap.

### III.1 The gradient was blind to the wrap, and what happens when it is not [MEASURED]

**[Lavender]** The derivative of a wrap is 1, so in the backward pass every shear looks
linear. The loss as a function of the weights is a staircase of linear pieces with jumps, and
a gradient only sees the slope of the piece it stands on. It can never point *towards* a
jump. So the wraps that occurred in §II were accidents of weight growth, not something the
objective asked for. The experiment above tested "does the wrap help when the optimiser is
blind to it", not "can an optimiser that knows the space is a circle use it".

The smallest change that makes every gradient circular is to let the shear read the untouched
half periodically:

```
theta_B  <-  theta_B + A cos(theta_A) + B sin(theta_A) + b     (mod 2pi)
```

Still exactly invertible, since it reads only the untouched half. `--shift trig`. One run,
depth 8, span π, same settings as everything else:

| arm, depth 8, span π | params | test | wraps switched off after training |
|---|---|---|---|
| wrap, linear shear | 1.23M | 0.973 | 0.68–0.73 |
| no wrap | 1.23M | 0.972 | 0.972 |
| ReLU MLP shift | 6.43M | 0.980 | 0.970 |
| **periodic shear** | 2.46M | **0.983** | **0.983** |

Three things in one row:

- **Best of the family**, above the ReLU MLP at a third of its parameters.
- **No runaway.** Mean shift stays at 0.1 turns in every layer for all 15 epochs; the
  first-layer explosion of §V never starts. A periodic shear cannot produce a jump, so there
  is nothing for the optimiser to walk into.
- **The wrap is a no-op.** Switching the wraps off changes nothing, to four digits. Once
  every function downstream of an angle is periodic, wrapping it is the identity. This is the
  structural answer to the original question: the wrap *cannot* be the nonlinearity, because
  a torus-native network is invariant to it. The nonlinearity is sine and cosine of the other
  angles, which is Nanda's quadrature embedding and the choir's agreement.

It is a nonlinearity, so the "no nonlinearity" phrasing of the vision is gone. What survives
is stronger: the only nonlinearity is the torus's own trigonometry, and that beats a ReLU.

---

## IV. The seam [NEGATIVE]

Span = 2π puts black and white at the same angle. With wraps, **every checkerboard run
collapsed** within 2–5 epochs (depths 2, 4, 8, 16, linear and MLP alike), peaking at
0.87–0.91 first. The no-wrap arm at span 2π is *not* a seam arm: without a wrap, −π and +π
are different numbers, so the seam does not exist and it trains normally (0.971).

The seam is a real cost, but here it shows up as **training instability**, not as a
9-point accuracy tax as in [`../nice_flow/`](../nice_flow/README.md). One shuffle-split run at span 2π survived
(0.963). One run, not to be leaned on.

---

## V. The collapse, and the learning rate [MEASURED]

Every collapse has the same anatomy: **the first layer runs away.** Its mean shift goes from
0.2 turns to several (up to 220 for the depth-2 MLP) within an epoch, every angle it touches
wraps, and the loss returns to chance. The other layers stay at 0.1 turns throughout.
[`results/diag/sweep_lr1e-3_collapsed.log`](results/diag/sweep_lr1e-3_collapsed.log) is the first sweep: at Adam 1e-3, **every arm collapsed**,
including the no-wrap control and the MLP, between epochs 4 and 12. Gradient clipping at 1
did not save it. At 1e-4 all arms train; at 3e-4 the wrap arm degrades late. Depth 2 collapses
even at 1e-4.

`--init 1` (start with shifts of ~1 rad instead of at the identity) never leaves chance.
Starting near the identity is required; wrapping from the first step is fatal.

The backward pass through a wrap is the identity, so the optimiser steps as if the layer were
linear while the forward pass is a sawtooth. That is the mechanism I would look at first; it
is not measured.

The script now stops a run when test accuracy falls 25 points below its own best and records
the epoch.

---

## VI. Invertibility [MEASURED]

Exact in float64: roundtrip error 1e-12 on trained networks. In float32, 3e-6.

Two things learnt the hard way:

- **The [0, 2π) domain is wrong for this.** Black pixels are angle 0 exactly, which is *on*
  the seam. The inverse returns them as 2π − ε, and the next real-weighted shear multiplies
  that 2π. Deterministic, happens in float64, half the coordinates wrong. Centring the domain
  at [−π, π) and the data at span·(x − ½) fixed it.
- **With large weights the inverse is chaotic in float32.** Random shears with shifts of
  ~3 turns: 99% of coordinates wrong after 8 layers in float32, 1e-9 in float64. A rounding
  error that crosses the seam costs a whole turn times a weight. That is the cat map being an
  Anosov map. Trained networks keep their weights small enough that it does not bite.

---

## VII. Open

- **Fourier phases as the input.** Pixel intensity is not an angle; the seam and the span
  knob exist only to manage that. A Fourier phase is a genuine angle, translation becomes a
  per-frequency torus shift, and a real image has ~392 independent phases. Dead angles where
  the magnitude is near zero need either a low-frequency cutoff or the magnitude as a gate on
  the shift, `s = W(m ⊙ θ_A)`, which keeps invertibility since m is untouched side data.
- **A per-layer cleanup.** If the readout's cleanup is the nonlinearity, the torus-native
  per-layer nonlinearity is a per-voice cleanup, not a wrap. That is the choir's competition
  inside a block.
- **Why the first layer runs away**, and whether a periodic parameterisation of the shear
  (integer-constrained, or shift by sin of the other angles) removes it.
- Generation, dropped here on purpose.
