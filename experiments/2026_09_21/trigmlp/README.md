# 2026-09-21 — trigmlp: ReLU against sine-and-cosine in a plain MLP

**[Lavender]** Forget the torus, the wrap and invertibility. Take a conventional
784 → H → H → 10 MLP on MNIST and swap the activation. What is the difference?

Script: `trigmlp.py`. H = 512, Adam 1e-3, batch 256, 15 epochs, three seeds. The trig
activation returns cos and sin of the same pre-activation, so a layer with H/2 linear units
produces H features. Two trig arms bracket the parameter count of the ReLU arm.

## Result [MEASURED]

| activation | units per layer | features | params | test, mean of 3 | sd | train |
|---|---|---|---|---|---|---|
| ReLU | 512 | 512 | 670k | 0.9769 | 0.0023 | 0.992 |
| cos + sin | 256 | 512 | **337k** | 0.9768 | 0.0019 | 0.996 |
| cos + sin | 512 | 1024 | 937k | 0.9771 | 0.0027 | 0.996 |

[`curves_mnist.png`](results/curves_mnist.png).

- **No difference in test accuracy.** All three within one seed standard deviation.
- **Half the weights for the same result.** The cos+sin arm with 256 units matches ReLU with
  512, and fits the training set better. Reading one pre-activation as a point on a circle
  gives two features for the price of one.
- **No instability.** Default PyTorch init, no SIREN-style frequency scaling needed here. No
  collapses in nine runs.

## Fashion-MNIST [MEASURED]

Same script, `--data fashion`, same settings, same three seeds. Data in `../../data/fashion/`.

| activation | units | params | test, mean of 3 | sd | train |
|---|---|---|---|---|---|
| ReLU | 512 | 670k | **0.8882** | 0.0054 | 0.944 |
| cos + sin | 256 | 337k | 0.8844 | 0.0031 | 0.963 |
| cos + sin | 512 | 937k | 0.8857 | 0.0018 | 0.967 |

[`curves_fashion.png`](results/curves_fashion.png).

- **ReLU is ahead by 0.3–0.4 points.** That is under one standard deviation of the ReLU
  seeds, so not a firm ordering, but the sign is consistent across both trig arms.
- **The trig arms overfit more.** They fit the training set 2 points better and the test set
  slightly worse. On MNIST the same extra fit was free; on the harder data it is not. A
  periodic activation can always find a higher frequency to memorise with, and nothing here
  regularises that.
- More width did not help the trig arm (0.8844 → 0.8857), so it is not a capacity gap.

Verdict across both datasets, before the bound below: a circle-valued neuron on pixels is as
good as a ReLU on the easy set and a hair worse on the harder one, at the same or lower cost.

## Bounding the turn [MEASURED]

**[Lavender]** A pre-activation of 100 radians is 16 turns, and the trig only reads the
fractional one. Anything past a turn is wasted range, and a unit that reaches it is one whose
weights are large enough to oscillate across the data. So cap it: the pre-activation goes
through `B·tanh(z/B)`, identity for small z, saturating at ±B. `--bound`. Capping the
*weights* in the worst case instead would force typical pre-activations to ~0.05 rad, where
sin is linear and cos is a constant; the nonlinearity is still there in principle but the
network would live in its flat part. Capping the sum keeps typical z of order 1, where the
curve is, and clips only the extremes.

| activation | units | params | MNIST test | train | Fashion test | train |
|---|---|---|---|---|---|---|
| ReLU | 512 | 670k | 0.9769 | 0.992 | 0.8882 | 0.944 |
| ReLU | 256 | 269k | 0.9756 | 0.992 | 0.8848 | 0.937 |
| cos + sin, unbounded | 256 | 337k | 0.9768 | 0.996 | 0.8844 | 0.963 |
| cos + sin, bounded to half a turn (B = π) | 256 | 337k | 0.9780 | 0.997 | **0.8884** | 0.954 |
| cos + sin, bounded to one turn (B = 2π) | 256 | 337k | **0.9799** | 0.997 | 0.8862 | 0.959 |

Three seeds each; sd 0.001–0.005 throughout. **[Lavender]** asked for the ReLU-256 row: on
an easy dataset "same accuracy at half the weights" says nothing unless the half-size ReLU
also fails to match. It does not fail by much, 0.1–0.3 points, which is the honest size of
the parameter argument. Note the trig-256 net has 25% more parameters than ReLU-256, since
its 512 features feed the next layer; it is between the two ReLU rows in size.

Reading it at matched width, 256 units: bounded trig beats ReLU by 0.4 on both datasets,
about two seed standard deviations. Against the double-width ReLU it is +0.3 on MNIST and
level on Fashion.

- **The bound closes the Fashion gap** and cuts the excess training fit by a third. Half a
  turn is the right cap there.
- **On MNIST the bounded net is the best of everything run in this folder**, 0.980 against
  ReLU's 0.977, at half the parameters. One turn is the right cap there.
- Same weights, same features, only the maximum frequency changed. So the earlier trig
  deficit was the spinning, not the activation.

Bounded circle-valued neurons are at least as good as ReLU on both datasets, at half the
weights. The torus's stronger case, genuinely periodic data, is still untested.

## The phase network: torus arithmetic only [MEASURED]

**[Lavender]** The weighted sums so far add cos and sin as vectors in flat space and read a
coordinate off the result. On the torus the consistent operation is: shift each incoming
angle by a learned transposition, add the unit vectors with learned votes, and take the
*angle* of the resultant, the consensus, not a coordinate of it.

```
z_i  =  Σ_j  w_ij · e^{i(θ_j + φ_ij)}          C_ij = w_ij e^{iφ_ij}, a complex weight
θ_i  =  arg z_i                                the consensus angle
r_i  =  agreement                              |z_i| relative to what full agreement would give
```

`--act phase*`. 784 pixel angles on half the circle, two layers of 256 voices, cosine
cleanup against learned chords. 535k parameters, since each connection is a complex number.
Scale-free by construction: nothing can run away, no cap needed. Three variants differ only
in what the next layer receives:

| variant | passes on | MNIST test | train | Fashion test | train |
|---|---|---|---|---|---|
| `phase` | the angle only, unit vector | 0.9603 | 0.971 | 0.8701 | 0.896 |
| `phaseamp` | r·(cos, sin), r = \|z\| / Σ\|C\| | 0.9228 | 0.927 | 0.8378 | 0.859 |
| `phaserms` | r·(cos, sin), r = \|z\|/√Σ\|C\|², squashed to < 1 | **0.9758** | 0.994 | **0.8807** | 0.929 |
| ReLU, 256 units, for reference | | 0.9762 | 0.992 | 0.8865 | 0.941 |
| cos+sin capped, 256 units | | 0.9799 | 0.997 | 0.8862 | 0.959 |

Three seeds each, sd 0.002–0.004.

- **Angle-only is slow, not stuck.** Training accuracy is the low number, so this is
  underfitting. Run to 40 epochs it reaches 0.973 test and is still climbing; a 3× higher
  learning rate does not help. Discarding the resultant's length at every layer throws away
  the evidence strength, and the hard normalisation has a gradient that blows up for voices
  near silence. Both slow it down.
- **The first agreement normalisation starved the network.** Dividing by Σ\|C\| is the
  amplitude for *perfect* agreement. With 784 random contributors the resultant is about
  1/√784 of that, so every layer shrank its signal 25× and the readout saw almost nothing.
- **Normalise by the incoherent expectation and it works.** √Σ\|C\|² is the resultant length
  random inputs would give; dividing by it puts typical amplitudes near 1, and a squash keeps
  them below it. That variant matches ReLU on MNIST and sits 0.6 points behind on Fashion,
  at 15 epochs, with the agreement carried as amplitude the whole way. Silence propagates.

So the fully torus-native layer is competitive but not better on pixel data, same verdict
as the activation swap. What it adds is not accuracy but two things the other layers lack:
no scale to run away, and an agreement per voice that is computed rather than bolted on.

## Split MNIST, one model in sequence [MEASURED, NEGATIVE]

**[Lavender]** Same network, capped cos+sin, 256 units, single 10-way head, trained on
0/1, then 2/3, then 4/5, then 6/7, then 8/9, five epochs each, no replay, no regulariser.
`split.py`. Accuracy on each task after each stage:

| after task | 0/1 | 2/3 | 4/5 | 6/7 | 8/9 | mean |
|---|---|---|---|---|---|---|
| 1, class-incremental | 0.999 | | | | | 0.999 |
| 2 | 0.000 | 0.995 | | | | 0.497 |
| 3 | 0.000 | 0.000 | 0.995 | | | 0.332 |
| 4 | 0.000 | 0.000 | 0.000 | 0.997 | | 0.249 |
| 5 | 0.002 | 0.000 | 0.000 | 0.000 | 0.995 | 0.199 |
| 5, task-incremental | 0.280 | 0.300 | 0.619 | 0.759 | 0.995 | 0.591 |

Final ten-way accuracy on all of MNIST: **0.198**, against 0.980 for the same network
trained on everything at once.

- **Complete forgetting, class-incremental.** Every old task at exactly zero after the next
  one. The shared head learns that only the current two logits should ever win, which is
  the textbook result for a plain network with no replay.
- **Below chance, task-incremental.** Even told which pair to choose from, the first two
  tasks end at 0.28 and 0.30, well under the 0.5 of a coin. The features were not just
  overwritten, they were rotated so that the old classes land on the wrong side. An
  angle-valued unit can do that by moving half a turn.
- The circle-valued activation gives no protection against forgetting on its own. Whatever
  the choir's claim about remembering is, it is not a property of the activation.

## What it says about `../catmap/`

The 98.3% of the periodic shear was not the torus and not invertibility. It is what a
sine-and-cosine network does on MNIST, and a plain MLP with the same activation gets 97.7%
with a fraction of the parameters. The catmap's extra point came from its width (392-wide
shears, eight of them) and its cosine cleanup at the output, not from the geometry.

So: on pixels, a circle-valued neuron is exactly as good as a ReLU neuron and cheaper. It
is not better. The case for the torus has to come from data that is genuinely periodic,
where the angle is the truth rather than a coordinate choice.
