# 2026-08-31 — competition, and what each readout is for

A long day. The through-line: **every representation has a readout it is good
at, and scoring it on the wrong one measures nothing.** That turned up four
separate times, and the last of them produced the best number this project
has.

The other thread: **competition — only the winner learns — is what creates
identity.** Not sparsity, not overcompleteness, not nonnegativity, not labels.
That single line turns the same dense rule that produced 08-30's speckle into
something that finds parts, specialises by class, and resists forgetting.

---

## The headline

| | |
|---|---|
| **[competitive40/](competitive40/)** — 40 hypercolumns x 36 templates, winner-only learning, fit gate | **0.9392** |
| *08-30's dense + layer three, the previous best* | *0.9232* |
| *logistic regression on raw pixels* | *0.9074* |
| **[continual_fixed/](continual_fixed/)** — the same thing trained 0-4 then 5-9 | **0.31 pts** lost |
| *an MLP on the same split* | *loses **everything*** (0.9765 -> 0.0000) |

---

## `place_code/` — the encoder: one bump per channel

**[place_code/](place_code/)** — one bump per channel, everything nonnegative,
value carried by *position*. Replaces mean-centred signed input. The finding
that mattered: **`nb` controls selectivity, not fidelity.** At nb=2 (which is
exactly the ON/OFF split) the code already reconstructs at 0.04 RMSE but every
cell is lit for every value, so nothing is ever silent and nothing can carry
identity. Use nb>=8. Also: centring is not norm-constancy — halfw belongs near
1.5, and at 1.0 the code length swings by sqrt(2).

Full writeup: [`place_code/README.md`](place_code/README.md).

---

## `sparse_column/` — k-of-K matching pursuit as the hypercolumn rule

**[sparse_column/](sparse_column/)** — k-of-K matching pursuit as the
hypercolumn rule. Two findings:

* **Nonnegative template weights destroy the pursuit.** Every nonneg unit
  vector sits in one orthant, mean pairwise cosine 0.55 against 0.016 signed,
  and the pursuit stalls after the second pick. Nonnegativity belongs on the
  coefficients and the message, never the weights.
* **Learn with the pursuit, read with the settling.** Crossing the two rules
  shows they buy opposite things — stability is the encoder's (89% vs 79% of
  the support survives a nudge), the dictionary is the learner's (coherence
  0.036 vs 0.399). Under settling every active template rotates toward the
  *same* residual, which is 08-30's shared-leftover failure in miniature.

Full writeup: [`sparse_column/README.md`](sparse_column/README.md).

---

## `sparse_stack/` — the readout decides

**[sparse_stack/](sparse_stack/)** — the sparse column as layer one of the
08-30 stack. Sparse classified 13 points *worse* than dense (0.7494 vs
0.9232), drew better, and rebuilt much better (0.475 vs 0.733). Then replacing
layer three with a **tally** — count `[position, template, label]`, every
active cell votes, nothing learned — **inverted the whole board**:

| | matching (layer three) | counting (tally) |
|---|---|---|
| dense | **0.9232** | 0.7060 |
| sparse k=4 | 0.7494 | **0.8672** |

A lookup table beat 1024 learned prototypes by 12 points on the same codes.
Pure binary presence won for every sparse arm — **the magnitudes add nothing
to a counter; which templates fired is the whole signal.**

Also recorded there: an attempt to give sparse codes the dense arm's geometry
by scoring `(c1 W)·(c2 W)` **lost** 3.3 points. `c @ W` adds no information.

Full writeup: [`sparse_stack/README.md`](sparse_stack/README.md).

---

## `patch_ensemble/` and `patch_competitive/` — competition creates identity, on patches

**[patch_ensemble/](patch_ensemble/)** vs **[patch_competitive/](patch_competitive/)** —
the cleanest control of the day. Same rule, same 8x8 patches, one line
different:

| | all experts learn | only the winner learns |
|---|---|---|
| pairwise subspace overlap | **1.0000** | **0.3436** |
| templates | 100 panels of speckle | edges, corners, curves |
| in/out error gap | — | **47.6 pts** |

`patch_competitive/results/01_claimed.png` is a legible edge-and-curve
vocabulary learned with **no labels at all**. Honest caveat: it tiles a
continuum (many diagonals at gradually rotating angles) rather than
discovering discrete parts.

Full writeup: [`patch_ensemble/README.md`](patch_ensemble/README.md).
Full writeup: [`patch_competitive/README.md`](patch_competitive/README.md).

---

## `dense_ensemble/` — 30 hypercolumns voting

**[dense_ensemble/](dense_ensemble/)** — 30 hypercolumns trained jointly and
voting. Gains +2.9 points and saturates by 10, because they agree with each
other 83% of the time. Gating by self-confidence never beats averaging.

Full writeup: [`dense_ensemble/README.md`](dense_ensemble/README.md).

---

## `competitive/`, `competitive40/`, `competitive200/` — only the most confident hypercolumn learns

**[competitive/](competitive/)**, **[competitive40/](competitive40/)**,
**[competitive200/](competitive200/)** — only the hypercolumn most confident
in the *true* label learns.

| | fit gate | confidence gate | dead |
|---|---|---|---|
| 10 x 144 | 0.8698 | 0.1658 | 2/10 |
| **40 x 36** | **0.9390** | 0.2944 | 25/40 |
| 200 x 9 | 0.8994 | 0.1852 | 130/200 |

* **Partition beats capacity.** Identical 1440 templates: 0.8698 as 10x144,
  0.9390 as 40x36. Seven points from redistributing the same parameters.
* **There is an optimum**, and the gap between an expert's error on its own
  inputs and on everyone else's is what peaks there (13.8 / **17.9** / 15.6
  points). Too wide and everything fits; too narrow and an expert cannot model
  even its own thing.
* **Match the gate to the experts**: specialists need a *fit* gate,
  generalists a *confidence* gate, and backwards is catastrophic — 0.94 vs
  0.29 on identical weights.
* **08-30's speckle dissolves.** Every live hypercolumn's templates look like
  the digit it claimed. The rotation invariance is still there; the subspace
  being rotated is now one class's, so every basis of it looks like that
  class. **Meaning came from restricting the data, not from fixing the
  rotation.**
* And a correction: the oracle of 1.0000 is **trivial** — every specialist
  says its own digit for 99.9% of inputs, so "at least one is right" is
  guaranteed. The 6 points I claimed were sitting in the gate do not exist.

Full writeup: [`competitive/README.md`](competitive/README.md).
Full writeup: [`competitive40/README.md`](competitive40/README.md).
Full writeup: [`competitive200/README.md`](competitive200/README.md).

---

## `continual_fixed/` and `continual/` — forgetting

**[continual_fixed/](continual_fixed/)** — the same architecture on 0-4 then
5-9. **100% of the new-class samples went to experts that were dead after
phase one; 0 of 14 specialists changed their claimed digit.** Specialisation
protects itself: a trained "3" scores negative for label 7, so it *loses* that
competition to an uncommitted expert and is never overwritten.
(**[continual/](continual/)** is the parked earlier attempt, where swapping
the competition rule dropped phase-one accuracy to 0.52 and made the
forgetting numbers meaningless.)

Full writeup: [`continual_fixed/README.md`](continual_fixed/README.md).
Full writeup: [`continual/README.md`](continual/README.md).

---

## `drone_rl/`, `drone_tracks/`, `drone_imitate/`, `drone_stack/` — the drone

**The drone — [drone_rl/](drone_rl/), [drone_tracks/](drone_tracks/),
[drone_imitate/](drone_imitate/), [drone_stack/](drone_stack/)** — four RL
failures, one flying policy, and a control that deflates it.

| | success |
|---|---|
| teacher (PID) | 0.970 |
| per-hypercolumn **linear map** | 0.905 |
| **one global linear map, no architecture** | **0.875** |
| tally / one action per hypercolumn, any N up to 4096 | **0.000** |

The architecture flies, and the hypercolumns are worth 2.5 points of it. The
task is the problem: a PD teacher is **linear in the state**, so one linear
map represents it and a partition has nothing to do. My fault for choosing it.

But the measurement underneath is worth keeping. Handicapping the *teacher* to
a quantised state:

    bins/axis   4 -> 0.090    16 -> 0.355    32 -> 0.850    64 -> 0.950
    cells      4e3           1.7e7          1.1e9          6.9e10

**A billion cells to fly** — and that is the optimal controller, so it is a
floor. The measured within-cell spread falls as `N^-0.168`, i.e. effective
dimension **6.0**, exactly the prototype law: halving the error costs 64x the
experts. And random command error of sd 0.4 barely hurts (0.920) while
*systematic* error of sd 0.22 is fatal, because noise averages out and a fixed
wrong answer does not.

**So: a tally works when the output is a category. It fails when the output is
a continuous quantity that must vary within a category.** MNIST scored 0.8672
on a tally because a digit *is* a category. Thrust is not. 33 hypercolumns
with a formula inside beat a billion with an answer.

Full writeup: [`drone_rl/README.md`](drone_rl/README.md).
Full writeup: [`drone_tracks/README.md`](drone_tracks/README.md).
Full writeup: [`drone_imitate/README.md`](drone_imitate/README.md).
Full writeup: [`drone_stack/README.md`](drone_stack/README.md).

---

## What is settled, and what is not

**Settled.** Competition creates identity. Partition beats capacity, with an
optimum. The gate must match what the experts are. Specialisation resists
forgetting. Tallies are for categorical outputs.

**Not settled.** Everything above is MNIST, which saturates at ~15 experts and
whose classes are ten labelled digits — almost too easy for the claim. The
composition question (does layer two find structure in layer one's
co-occurrence?) has not been asked. And the architecture has never met a task
where a linear baseline is not already close.

**Next**: two convolutional rungs on Fashion-MNIST, where logistic gets ~0.84
against ~0.93 for composition — a real gap instead of MNIST's three points.
Then CelebA 48x48 (in `data/`, 39,910 faces, 40 binary attributes)
where multi-label genuinely stresses a tally, since one winner has to carry
several co-occurring facts at once.

---

## `fashion/` — Fashion-MNIST, and what correction costs

The competitive40 architecture moved unchanged from digits to clothes, and it
failed legibly: 0.5952 against 0.8926 for an RBF SVM, because the fit gate
("who rebuilds this image best") no longer answers "which class is this" — the
right expert ranked 1st for 59.5% of test images but was in the top 3 for
81.6%. Correction fixed the read: reluctance plus repulsion reached 0.8186, and
grading on drawing and naming together (λ=4) reached **0.8278**, the best number
of the day — but that grading only pays given repulsion. Teacher-free (the label
as a second stream, sleep, habitual calibration) reached 0.6868. Also here: the
coefficient covariance as the missing half of the generative model, only 14 of
40 hypercolumns ever winning the gate, both normalisation steps earning their
place, and confidence being input-blind.

Full writeup: [`fashion/README.md`](fashion/README.md).

---

## `continual_newrule/` — punishment breaks the stream

Split-MNIST again with the new grading rule. Without correction it retains
0.9620 of the old digits (forgot +0.0180); with repulsion it keeps only 0.3171
(forgot +0.6657); sleep sits in between (0.9049). Allocation stayed perfect in
every arm — the damage came through correction, not learning, because
punishment is aimed at whoever currently wins, which on a stream is the old
specialists.

Full writeup: [`continual_newrule/README.md`](continual_newrule/README.md).

---

## `conv/` — the convolutional rung

H hypercolumns of K minicolumns on 5x5 patches, slid over the image with shared
weights. Identity + dense code on a 2x2 grid is the first message in the project
to beat raw pixels (0.9445 from 192 numbers against 0.8925 from 784). The full
teacher-free stack on Fashion reached 0.7950; spatial resolution was worth ten
points, capacity nothing, and about thirteen layer-two experts did the work in
every configuration.

Full writeup: [`conv/README.md`](conv/README.md).

---

## Where this leaves the project

**Settled today.** The gate must match what the experts are, and on Fashion it
does not — silhouette is not class. Correction buys discrimination and costs
generative fidelity and, on a stream, everything. The label belongs in the
*selection*, never in the *correction*. Grading on drawing and naming together
has no tradeoff between the two. The coefficient distribution is the missing
half of the generative model. Normalisation earns its place.

**Open.** L2 has not been built. The binding problem is the known wall one rung
further on: our message is a superposition, and sums are commutative, so
`john + loves + alice` and `alice + loves + john` are the same vector. Position-
concatenation within a window handles the first rung; time (`[present ; lagged
keys]`) handles sequences; multiplication (tensor products, HRR) is the only
option that gives role structure without slots, and our unit has no
multiplication anywhere.

---

## The conversation that changed the design

The user asked a question about their own experience that turned out to settle
an architectural argument.

> "In my mental state I see both the image and the category at the same time.
> So I logically conclude the image data should go up as well. And I can imagine
> things vividly. But in the architecture you describe, V2 encodes combinations
> of V1 features and so on until I recognise an object — that is fine, object
> recognition is okay. But then how do I not only have the concept of the object
> in my mind but also at the same time the vivid image of the scene?"

The premise to drop is that the hierarchy is a **pipeline that empties**. It
isn't. V1 does not hand its contents upward and go quiet — it keeps firing for
as long as you are looking, and so does everything above it. The state of the
system is the union of all the active maps, not the contents of the top one. The
image never had to travel anywhere, because it never left.

What makes the low level look like a coherent object rather than a field of
edges is the feedback — roughly ten times more connections run down than up, and
the high-level interpretation continuously reshapes the low-level maps. Imagery
with the eyes closed is plausibly the same pathway driven without sensory input:
V1 does activate during visual imagery, more strongly in people reporting more
vivid imagery. (Caveats: imagery activity is weaker and coarser than perception,
and people with aphantasia recognise objects fine with no imagery at all.)

The user then sharpened it further, and was right to:

> "I have a concept of an edge, but that is a totally different edge from what V1
> detects. When I say edge I refer to the edge that is a referrable object in my
> mind. Anything I can refer to is also an object — which is still a higher
> region thing."

Correct, and it corrects the paragraph above. The *appearance* draws on the whole
active hierarchy, but what can be **referred to** is only what some region holds a
token for. V1 responds to things that are never consciously seen (masked stimuli,
the suppressed eye in rivalry), and nobody can introspect the four hundred
oriented filters covering one degree of visual field — you experience a boundary
between two surfaces. Appearance and reportability are different questions.

## And so: identity only, again, and this time all the way down

The conclusion the user drew from this is the design:

> "Through feedback we should be able to get a vivid, input-like image
> reconstructed from the higher layers and their projections to lower layers,
> because I can imagine vividly. So we never needed the dense code that the
> hypercolumn carried in today's experiments."

Which is right, and it is what the measurements had already been saying without
our understanding why. The dense code does not need to travel; it needs to
**exist**, down where the content is. What goes up is the address. What comes
down selects among content the lower layer already holds. Vividness is
manufactured locally, on demand, driven from above.

That makes the identity-only message not a compromise that happened to score
better, but the correct shape — and today's dense-only failure (0.7240, and
unable to reconstruct its own input at 0.9794 error) was the architecture saying
so.

**The redesign that follows.** If only identity travels, the whole apparatus of a
hypercolumn holding a dense subspace is unnecessary:

    one hypercolumn per patch position, weights shared across positions
    K minicolumns inside it, competing -- the best-matching one wins
    only the winner learns, rotating toward the patch
    the message up is which minicolumn won, and how strongly
    the projection down is that minicolumn's template, which IS a picture

The terminology snaps back to the original: the minicolumn is the thing that
competes, the hypercolumn is the competing group.

Two consequences worth predicting before we build it. Generation becomes exact
rather than statistical — no coefficient means, no covariances, no sampled
faces, just paste the named templates where they were named. And the templates
themselves should become interpretable: today they looked like noise because a
dense subspace has no preferred basis, while the hypercolumn *faces* were crisp
edges. Winner-take-all learning breaks that symmetry, so the thing we had to
reconstruct indirectly becomes the primitive.

What carries forward unchanged: competition plus conscience, teacher-free; the
label only ever as a co-occurring second stream, never as a verdict; centre and
L2-normalise each patch and drop the flat ones; judge a message format by whether
a reconstructive layer can use it, not by a linear probe; and spend on spatial
resolution before capacity.

---

## `kmeans/` — identity only, all the way down

The redesign above, built with plain online spherical k-means: one hypercolumn
per patch position, competing minicolumns, only the winner learns, the message
is the winner's index. No writeup was kept for this folder; the numbers below
come from its result files. One layer on whole images joined with the label:
0.9086 (`results/joint_mnist.json`). Two rungs unpooled: 0.7504
(`results/stack.json`); with the L1 identity map max-pooled first: **0.9492** at
pool 4 with 103,400 parameters, against 0.921 for one layer on pixels and 0.9074
for logistic (`results/stack_pool.json`). Three rungs: 0.9302
(`results/three.json`). Then the same stack on 48x48 CelebA faces with the 40
attributes as the second stream, and composition experiments (a woman with a
mustache) on it.

Full writeup: [`kmeans/README.md`](kmeans/README.md).

---

## `antihebb/` — anti-Hebbian lateral inhibition

Several templates may be active at once; units that fire together learn to
suppress each other, and each unit's threshold drives it to a target activity.
Run on MNIST patches over several template counts and inhibition strengths. No
writeup was kept; the per-run numbers are in `results/ah_*.json`.

Full writeup: [`antihebb/README.md`](antihebb/README.md).
