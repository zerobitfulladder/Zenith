# The convolutional stack, from scratch

A complete account of `experiments/2026_08_29/conv_stack/run_conv_stack.py`:
what it is built from, how each piece works, how they compose, and what
it scores. Concepts first, with the small amount of maths that is
actually load-bearing.

---

## 0. The task

A curve

    f(x) = 1.5·sin(1.3x) + 0.35x + 0.6·cos(0.7x)

over x ∈ [−12, 12]. 1600 training pairs (x, y), y carrying noise of
σ = 0.08. **A slice of x from 1.2 to 2.8 is deleted from training** — no
pair inside that band is ever shown. Inside it the curve crests and then
falls by 2.29.

The question: can a memory built from competing templates reconstruct
that missing stretch, having never seen a single example of it?

It has to do this without being told anything about the function — no
period, no trend, no formula. Only the pairs.

---

## Part I — the pieces

### 1. A number is written as a patch of cells, not stored in one

A channel is a strip of `nb` cells with centres `μ[0..nb-1]` spread
across the range of values that channel can take. A value `v` lights a
run of them, brightest at `v` and fading with distance:

    w[i] = ½·(1 + cos(π·d[i]/r))     if d[i] < r,  else 0
    d[i] = | μ[i] − v |
    r    = the fall-off radius, in value units

Two things follow, and everything else in the design leans on them:

- **similar values are written similarly.** 1.90 and 1.95 light almost
  the same cells, so anything that recognises one partly recognises the
  other.
- **distant values are orthogonal.** They share no cells, so knowing
  about one says nothing about the other.

The radius is measured in *value* units rather than cell indices, so the
pattern slides continuously: a value halfway between two cells lights
both, and a small change in `v` is a small change in the code. A
one-cell code would have neither property and nothing downstream could
recover them.

### 2. Many numbers live in one array

Each channel owns its own scattered set of `nb` positions inside a
single array of `D` slots. The sets are **disjoint** — carved out of one
permutation of `0..D−1`, never drawn independently — and each channel
writes its lit cells into its own positions. Where several channels
write, the array takes the element-wise maximum.

Because the sets are disjoint this is, mathematically, just a
concatenation of the channels with the slots shuffled; a dot product
does not care about slot order. The scatter-and-max machinery only earns
its keep if positions are allowed to collide, and they are not (§13).

The result is one sparse vector per input, a few hundred slots lit out
of thousands.

### 3. The unit: a hypercolumn of competing templates

A hypercolumn holds `K` templates in `W ∈ R^{K×D}`. Every row is kept

    Σⱼ Wᵢⱼ = 0        (mean-centred)
    ‖Wᵢ‖ = 1          (unit length)

**Matching.** With `m = mean(x)` and `n = ‖x − m·1‖`, template `i`'s
agreement with the input is the Pearson correlation

    cᵢ = ⟨Wᵢ, x − m·1⟩ / n  =  ⟨Wᵢ, x⟩ / n

The mean subtraction is free because `Wᵢ` is already centred, so scoring
touches only the lit slots. `n` is the same for every template, so the
winner is just `argmax ⟨Wᵢ, x⟩`.

Unit length is not housekeeping. Without it the argument is won by
whichever template is loudest rather than whichever fits best, and the
hypercolumn collapses onto a handful of greedy templates.

**Learning — exactly one template moves.** An empty hypercolumn adopts
its first inputs as templates. After that, the winner rotates a small
angle *toward* the input and nothing else changes. With `c` the winner's
correlation and `x̂ = (x − m·1)/n`:

    θ = η·c                    (η = 0.05)
    t = √(1 − c²)
    a = cos θ − c·sin θ/t
    b = sin θ/t
    W ← a·W + b·x̂

This is the exact rotation by `θ` in the plane spanned by `W` and `x̂`
(substituting `u = (x̂ − c·W)/t`, the orthogonal component, into
`W' = cos θ·W + sin θ·u` gives the line above). Both invariants survive
exactly: `W'` stays unit length and stays centred. On a sparse input it
costs a scale, a scalar shift and a short add.

The step being proportional to `c` means a template that nearly
recognised the input moves a lot and one that barely recognised it
barely moves, so templates settle onto whatever recurs.

**Reading with part of the input missing.** This is how the whole system
answers anything. Write the part you know, leave the rest of the array
empty, compete, and read the winner's own values where the emptiness
was. The score must be taken over the written slots `S` only, and
divided by how much of each template lives there:

    sᵢ = ⟨Wᵢ|ₛ , x|ₛ⟩ / max( ‖Wᵢ|ₛ‖ , floor · maxₖ‖Wₖ|ₛ‖ )

A bare dot product would assume every template holds the same amount of
itself inside `S`, which is false: a template that has gone *vague about
the missing part* keeps its whole norm in the written part and outbids
templates that genuinely match. The `floor` (0.25) stops the opposite
failure — a template holding almost nothing in `S` dividing a tiny
overlap by a tiny norm and bidding absurdly high.

---

## Part II — the architecture

Three levels. Only the middle two are learned.

### 4. Level 0 — the raw signal

A grid over x at spacing 0.1 (241 points). At each grid point, average
the y of every training pair within ±0.1 of it:

- **226 points have data.** Local density is ~70 pairs per unit of x, so
  each average is over ~14 pairs and the noise falls to about 0.02.
- **15 points have none** — x from 1.3 to 2.7, the deleted slice.

Nothing here is learned. It is just "what the pairs say here", and
inside the slice the honest answer is *nothing*.

### 5. Layer one — one shared hypercolumn over a small patch

This is the convolution, and it is the whole difference between this
design and a lookup table.

**A patch** is 5 grid samples spaced 2 grid steps (0.2) apart, so it
spans **0.8 of x**. That is deliberately far too small to reach across
the 1.6-wide hole — layer one cannot solve the problem by itself.

**The patch is written relative to its own mean.** Take the 5 values,
subtract their mean, and encode each residual as a bump in its own
channel (5 channels × 64 cells over [−2, 2], disjoint positions in an
array of 1024). The patch's *height* therefore never enters the code —
only its shape does.

This subtraction is the one piece of arithmetic outside the templates,
and it cannot be avoided: value is carried by *position* in these codes,
while the unit's mean-centring acts on *magnitudes*, so normalisation
can never reach the height. But it is local, uniform, computed by each
patch from itself, and identical at every position.

**One hypercolumn, applied at every x.** 64 templates, trained on 20 000
patches drawn from all 210 positions where every one of the 5 samples
has data. This is weight sharing: the same templates see the curve
everywhere, so what they learn is *what shapes occur*, not *what is at
x = 1.4*.

The vocabulary that results (`l1_vocabulary.png`) is exactly a
dictionary of local shapes — sorted by slope it runs from steep descents
through flats, crests and troughs to steep rises. Template #53 is
`[0.76 0.39 0.03 −0.43 −0.74]`, a steep fall. It has no location.

### 6. What layer one emits, and why it is the point

At each position layer one emits its **graded code**: the 8 best-matching
templates with their match strengths intact. "Graded" means the match
strengths survive; "top-8" is how much of the profile is kept. They are
separate choices, and only the first one matters much here:

| what is sent upward | cells lit | gap RMSE |
|---|---|---|
| top-1, magnitude dropped (a one-hot) | 1 | 0.168 |
| graded, top-8 | 8 | **0.086** |
| graded, top-16 | 16 | 0.109 |
| graded, every positive match | 64 | 0.086 |

A one-hot costs **2x**: cell #37 and cell #38 share nothing even when
templates 37 and 38 mean nearly the same thing, so the layer above has
no similarity to work with. Everything from top-8 upward is the same
within noise — with a vocabulary of only 64 shapes there is no penalty
for sending all of it. (A large dictionary is a different story: sending
every positive match out of 1024 templates lights so much at every
position that the differences become a small part of a large common
signal, and error rose 10x in that setting. At 64 that does not happen.)

The property this buys is the one the design exists for. Take two
21-sample windows over the curve — one over the deleted slice at x = 2.0,
one two wave-periods away at x = −7.6, where the curve does the same
thing 3.2 lower down — and correlate their codes:

| layer one built as | same shape, far apart | different shape | margin |
|---|---|---|---|
| one template per location | −0.004 | −0.004 | **0.000** |
| **shared vocabulary (64)** | **0.414** | **0.193** | **+0.221** |

A place-bound layer one has margin **exactly zero**: "the same shape
somewhere else" and "something unrelated" are literally the same number
to it, because its templates for the two places are strangers that share
no cells. The shared vocabulary separates them.

`l1_code_field.png` shows this directly. Because the templates are
sorted by slope, layer one's code field draws the derivative of the
curve — and the same rows light up at every crest and every trough
across the whole axis, even though the curve trends upward throughout.
A place-bound layer would draw a diagonal that never repeats.

### 7. Layer two — a window of those codes

Layer two is the same unit again, one level up. Its input is a **window
of 51 consecutive layer-one codes**, one grid step apart, spanning 5.0
of x. Each tap of the window is a channel holding that position's graded
code over the 64 shared shapes, all written into one array of 8192
slots with disjoint positions.

So a layer-two vector is a *sequence of shape-words* — what the curve
does, step by step, over 5.0 of x. Its 96 templates become the grammar:
which shape-words follow which.

It is trained on 12 000 windows drawn from the 110 anchor positions
where all 51 taps have a layer-one code behind them. It never sees the
deleted slice.

### 8. Completing what layer one cannot say

Layer one is silent wherever a patch would need missing data. Its patch
is 0.8 wide, so silence extends 0.4 beyond the hole on each side:
**23 positions, x from 0.9 to 3.1** (plus 8 at the two ends of the axis,
where a patch runs off).

Place one layer-two window centred on the hole. It spans x from −0.5 to
4.5: **28 taps carry a code, 23 are blank.** Write the 28. Leave the 23
empty. Compete using the partial-cue rule of §3. The winning template's
own values at the 23 empty taps say which shape-word belongs at each.

This is the identical read as everywhere else — write what you know,
blank what you do not, take the winner, read it where the blanks were.
The only difference is that the missing thing is a stretch of *shape
words* rather than a number.

### 9. Decoding shape-words back into values

A shape-word says how the curve bends over 0.8 of x. It says nothing
about where it sits, because layer one threw the height away.

Neighbouring patches overlap in 4 of their 5 samples, so their shapes
have to agree, and that constraint fixes everything. Collect one
equation per adjacent pair of samples in every completed patch: if patch
`p` covers grid points `q₀…q₄` and stands for shape `s`, then

    u[q_{a+1}] − u[q_a] = s[a+1] − s[a]        for a = 0..3

plus a pinning equation `W·u[q] = W·v[q]` (with `W = 10`) for every grid
point whose value is already known. Solve the whole system in one
least-squares step.

Solving jointly rather than marching left to right is what keeps this
honest: a marching decode feeds each error into the next step and drifts
(§12).

---

## Part III — how it runs, end to end

**A — build level 0.** Local-average the training pairs onto the grid.
Mark which points have data.

**B — train layer one.** Sample 20 000 patches from positions where all
5 samples exist. Each is written relative to its own mean and learned by
the geodesic rule, one winner at a time.

**C — emit codes.** At every codeable position, run the patch through
layer one and keep the 8 best matches with their strengths.

**D — train layer two.** Sample 12 000 windows of 51 consecutive codes
from positions where all 51 exist. Learn them the same way.

**E — complete.** One window over the hole: 28 taps written, 23 blank,
compete, read the blanks.

**F — decode.** Turn the 23 completed shape-words into values by the
least-squares stitch, pinned to the known data at both rims.

Total runtime: **16 seconds**.

---

## Part IV — results

### The reconstruction

RMSE inside the deleted slice, all measured at the same 15 points:

| method | RMSE |
|---|---|
| one template per (x, y) location, no second layer | 0.817 |
| hold the value at the nearer rim of the hole | 0.743 |
| the shared vocabulary alone, walking step by step | 0.428 |
| MLP (64-64, tanh) fitted on the same pairs | 0.144 |
| **shared vocabulary + layer two over its codes** | **0.086** |

Point by point:

| x | predicted | truth | | x | predicted | truth |
|---|---|---|---|---|---|---|
| 1.3 | 2.291 | 2.313 | | 2.1 | 1.264 | 1.395 |
| 1.4 | 2.203 | 2.278 | | 2.2 | 1.106 | 1.205 |
| 1.5 | 2.153 | 2.217 | | 2.3 | 0.894 | 1.008 |
| 1.6 | 2.064 | 2.131 | | 2.4 | 0.727 | 0.807 |
| 1.7 | 1.981 | 2.022 | | 2.5 | 0.488 | 0.606 |
| 1.8 | 1.867 | 1.891 | | 2.6 | 0.310 | 0.408 |
| 1.9 | 1.634 | 1.741 | | 2.7 | 0.140 | 0.217 |
| 2.0 | 1.487 | 1.575 | | | | |

### What layer two actually wrote

Not one shape — a sentence:

| x | shape-word | the shape it means | slope |
|---|---|---|---|
| 0.9 | #15 | `[−0.43 −0.06 0.12 0.17 0.21]` | +0.64 |
| 1.0 | #29 | `[−0.32 −0.07 0.10 0.16 0.13]` | +0.45 |
| 1.1 | #51 | `[−0.26 0.03 0.13 0.13 −0.03]` | +0.23 |
| 1.2 | #24 | `[−0.07 0.05 0.10 0.03 −0.10]` | −0.03 |
| 1.3 | #40 | `[0.01 0.12 0.11 −0.05 −0.19]` | −0.20 |
| 1.4 | #14 | `[0.10 0.14 0.08 −0.07 −0.24]` | −0.34 |
| 1.5 | #28 | `[0.18 0.20 0.09 −0.08 −0.39]` | −0.58 |
| 1.6 | #8 | `[0.30 0.20 0.09 −0.09 −0.50]` | −0.79 |
| 1.7–1.8 | #55 | `[0.42 0.31 0.13 −0.24 −0.62]` | −1.04 |
| 1.9–2.0 | #54 | `[0.52 0.35 0.04 −0.32 −0.59]` | −1.11 |
| 2.1–2.6 | #53 | `[0.76 0.39 0.03 −0.43 −0.74]` | −1.51 |
| 2.7–2.8 | #9 | `[0.73 0.26 −0.04 −0.34 −0.62]` | −1.35 |
| 2.9–3.1 | #21 | `[0.57 0.26 −0.05 −0.31 −0.47]` | −1.04 |

Still rising into the crest, flattening at x ≈ 1.2 (the true crest is at
1.25), accelerating to steepest at 2.1–2.6, then easing as the trough
approaches. 13 distinct words across 23 positions, and the repeats fall
exactly where the slope is genuinely constant.

### The vocabulary has to be smaller than the number of places

Sweeping the shared vocabulary size, with the same order-preserving
measurement as §6:

| K₁ | positions per template | reuse span in x | same shape | diff shape | margin |
|---|---|---|---|---|---|
| 8 | 21 | 19.8 | 0.548 | 0.596 | −0.048 |
| 16 | 12 | 19.4 | 0.518 | 0.422 | +0.096 |
| 32 | 6 | 19.2 | 0.537 | 0.353 | +0.184 |
| 64 | 3 | 14.3 | 0.414 | 0.193 | **+0.221** |
| 128 | 1 | 0.0 | 0.355 | 0.145 | +0.209 |

Too few shapes and everything looks alike — at 8 the margin goes
negative. Note that at 128 the margin is still good even though each
template *wins* at only one place: the sharing that matters lives in the
**graded** code, not in who wins.

---

## Part V — the three things that had to be right

Each of these cost real accuracy before it was found, and none of them
showed up in aggregate metrics — retrieval looked healthy while the
emitted answer was wrong.

### 10. Channels must not share array positions

Independent per-channel scatter lets two channels land on the same slot.
One shared position puts a phantom peak into the *answer* region of
every template whose *input* patch covers it, and the read emits the
phantom confidently across a whole band. Carving all positions from one
permutation instead: error 0.157 → 0.107 in a two-channel test, and
0.675 → 0.160 in a smaller array with 3 collisions.

### 11. A partial read must be scored on the part it asks about

The masked rule of §3, rather than a bare dot product. With the bare
rule, adding templates makes a hypercolumn **worse** — error 0.102 →
0.546 going from 128 to 1024 templates, while it collapses onto 13 % of
itself. With the correction it holds at ~0.09 and uses 46 %.

### 12. The reference must come from the same samples in training and reading

Layer one's patches are written relative to their own mean. If that mean
is taken over all 5 samples while learning but over the 4 surviving
samples while reading, every query is silently shifted against the
stored templates. For a *falling* patch the mean of the first four sits
above the mean of all five, so the query rides high and matches
templates that turn upward — the layer predicted an upturn from four
clearly descending samples. Fixing the reference subset turned a walk
across the hole from **1.022** into **0.428**.

The general rule: whenever part of the input is missing, whatever you
subtract or divide by must be computed from the same subset in both
phases.

---

## Part VI — what this shows, and what it does not

**Shows.** A hierarchy of competing templates, with one winner learning
at each level and nothing but pattern completion for inference, can
reconstruct a stretch of signal it was never shown — better than a
fitted network on the same data — provided each layer sends upward a
code that overlaps by *content*. Weight sharing is what produces such a
code: the same templates applied everywhere learn what shapes exist
rather than what sits at each location.

**Shows, second.** The two failures are separable and both were
measured. A place-bound layer one has content-margin 0.000 and nothing
above it can transfer. And the shared vocabulary *alone*, run
autoregressively, manages only 0.428 — a local vocabulary can carry a
curve for roughly its own receptive field and then drifts, because 0.8
of context cannot tell it where in the larger wave it sits. Content-
overlap below and long-range structure above are both necessary; neither
suffices.

**Does not show.** Layer two did not invent anything from nothing. It
recognised that the missing stretch is an instance of shape-sequences it
has seen elsewhere on the curve. If the signal contained no recurring
structure — a one-off feature in the hole and nowhere else — there would
be nothing to transfer and it would fail. The honest description is
*reuse of structure seen elsewhere*, not extrapolation.

**A remaining hand-supplied piece.** "Each patch is written relative to
its own mean" is an inductive bias, not something discovered. It is a
cheap and generic one — it works on any signal and requires knowing
nothing about this one — but it is supplied, and it is the only reason
the height drops out. The alternative, encoding a patch so that its
height *is* the DC component and letting the unit's own mean-centring
remove it, would trade the blur away for it.

---

## Files

    conv_stack.py       level-0 sampling, PatchLayer (layer one),
                        CodeWindowLayer (layer two), the stitch decode
    conv_viz.py         the pictures
    run_conv_stack.py   the run

    .venv/bin/python experiments/2026_08_29/conv_stack/run_conv_stack.py

Writes `results/`: `l1_vocabulary.png` (all 64 shared
shapes) · `l1_code_field.png` (what layer one says everywhere — the
repeat is visible) · `l2_templates.png` · `completed_codes.png` (the
shape-words layer two invented) · `regeneration.png` · `metrics.json`.

Knobs: `CS_K1` (vocabulary size, 64), `CS_K2` (layer-two templates, 96),
`CS_TAPS` (window taps, 51), `CS_WINDOWS` (training windows, 12000).
