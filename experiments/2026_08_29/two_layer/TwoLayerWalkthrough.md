# Two layers on a curve — what was actually built, and why it works

A plain walkthrough of `experiments/2026_08_29/`. No maths
beyond what is needed; the formulas are in [`HypercolumnReference.md`](../regress/HypercolumnReference.md).

**First, a correction to something worth getting right.** The second
layer does *not* fill the gap without having seen the shape. It saw the
shape — two wave-periods earlier and 3.2 units lower down. The whole
trick is that it was written down in a way that makes "two periods
earlier and 3.2 lower" count as **the same shape**. Section 7 traces
this exactly. Everything else follows from it.

---

## 1. The task

A curve, `f(x) = 1.5·sin(1.3x) + 0.35x + 0.6·cos(0.7x)`, over
x from −12 to +12. 1600 training pairs with a little noise on y.

**A slice of x from 1.2 to 2.8 is deleted from training.** No pair in
that band is ever shown. Everything is measured on two questions:

- can it answer for x values inside the range it trained on but never
  saw exactly? (ordinary generalisation)
- can it answer for x values inside the deleted slice? (the real test)

In the deleted slice the curve falls from about 2.3 down to 0. The model
has to produce that fall without a single example of it.

---

## 2. The encoding — the one thing everything rests on

**A number is never stored in one place.** It is spread across a strip
of cells: its own cell brightest, neighbours fading with distance.

For x: 192 cells spread over −12 to +12. A value lights **40 of them**,
brightest at the value and fading to nothing about 2.5 either side. So
x = 2.0 lights cells covering roughly −0.4 to +4.5, brightest at 2.0.

For y: the same, 192 cells over −6 to +6, fading over about 1.25.

Two consequences, and they are the reason anything works:

- x = 2.0 and x = 2.3 share almost all their lit cells. They are nearly
  the same pattern, so anything that recognises one partly recognises
  the other.
- x = 2.0 and x = −5.0 share nothing at all. They are unrelated
  patterns, and nothing that knows about one has any opinion about the
  other.

**Then both numbers go into the same array.** x owns 192 scattered
positions in an array of 16384 slots; y owns a different 192; the two
sets never overlap. Both strips are written into that one array. A
single vector — 79 lit slots out of 16384, half a percent — now says
one whole thing: *at this x, the answer was this y.*

There is no input side and output side. There is one vector with an x
region and a y region.

---

## 3. Layer one — what it stores and how it answers

Layer one is 1024 **templates** competing over the same input. Each
template is a vector the same shape as the code.

**Learning.** Show it a training pair as one combined vector. Every
template says how well it matches. The single best match wins, and
**only the winner adjusts**, turning slightly toward what it just saw.
Repeat. Each template drifts onto one recurring thing — here, one small
neighbourhood of the curve. After training, each template is one x-blob
bound to one y-blob: *"around x ≈ 1.4, the answer is around y ≈ 2.3."*

**Answering.** Write only the x part. Leave every y slot empty. Let the
templates compete on what is written. Take the winner, look at its own
y region, and read it out.

That read comes out **blurry** — a hill over the y cells rather than a
number — which is honest, because that is what the template contains.
Take the brightest cell and you have your answer. Look at the hill
itself and you can see how sure it is.

Nothing else exists. No output weights, no decoder, no second pass. The
answer was already sitting inside the winning template.

Layer one alone gets **0.067** error on unseen x inside the training
range. For comparison, a small neural network trained on the same pairs
gets 0.115.

---

## 4. Why layer one cannot bridge the gap

Ask it for x = 2.0, inside the deleted slice. The x-strip for 2.0
overlaps templates sitting just left of the slice, so one of those
wins, and it answers with *its* y — the value at the edge of the slice.

That is all a single winner can ever do. **One winner emits one stored
answer.** It cannot produce a value that is not written in some
template already, so it cannot invent the descent.

Measured in the slice: layer one gets **0.817**. A rule that just holds
whatever the curve was at the nearest edge gets **0.819**. They are the
same thing.

This is structural, not a tuning problem, and it is exactly the point
you made: **a single (x, y) pair contains nothing about which way the
curve is going.** "x = 2.0, y = 1.6" does not know it is on a descent.
That information only exists across several pairs. So no amount of
capacity or tuning at layer one can help.

---

## 5. Layer two — templates that are shapes, not points

Layer two is the same machinery, fed something different.

**A window** is 21 samples along x, spaced 0.2 apart, so it covers 4.0
of x. Call each sample a **tap**. Each tap holds layer one's answer at
that x, written as a blurry strip in exactly the way section 2
describes. Each tap owns its own 96 positions in the array, all
disjoint, and all 21 strips are written into that one array.

So one layer-two vector is **a whole piece of curve** — 21 heights in a
row. And a layer-two template, after training, is a *shape*: a crest, a
trough, a rising ramp, a falling ramp. (`trend_shapes.png` shows them;
they look like exactly what they are.)

Layer two is trained only on windows where every tap has real data
behind it — so it never sees the deleted slice.

**Filling the gap.** Put one window over the slice. Of its 21 taps, 14
are outside the slice and 7 are inside. Write the 14 you know. Leave
the 7 blank. Compete. The winning template's own values at those 7 taps
are the answer.

It is the identical read as layer one, one level up: **write what you
know, leave the rest empty, read the winner where the emptiness was.**
The only difference is that the missing thing is a stretch of curve
rather than a single number.

### 5.1 What layer two's input vector actually is

Not 21 numbers. **One sparse vector, exactly the same kind of object as
layer one's input** — the only difference is how many channels it has.

| | layer one | layer two |
|---|---|---|
| channels | 2 (x, y) | 21 (tap 0 … tap 20) |
| cells per channel | 192 | 96 |
| array | 16384 slots | 4096 slots |
| one code says | "at this x, the answer was this y" | "across these 21 steps of x, the curve did this" |

**x never appears as a number in layer two's input.** Each tap holds one
value — layer one's answered y at that point — and position along x is
carried by *which channel you write into*. Tap 7 IS "0.6 to the right of
the window's start", the way a pixel's position in an image patch is
carried by which pixel it is.

The real window that fills the gap, anchored at x = 2.0, reference =
first known tap's y = 0.534:

| tap | x | layer 1 says y | minus ref | writes |
|---|---|---|---|---|
| 0 | 0.0 | 0.534 | 0.000 | 16 cells, #40–55 |
| 1 | 0.2 | 1.099 | 0.565 | 16 cells, #45–60 |
| 2 | 0.4 | 1.539 | 1.005 | 16 cells, #49–64 |
| 3 | 0.6 | 1.853 | 1.319 | 15 cells, #52–66 |
| 4 | 0.8 | 2.105 | 1.571 | 15 cells, #54–68 |
| 5 | 1.0 | 2.168 | 1.634 | 16 cells, #54–69 |
| 6 | 1.2 | 2.168 | 1.634 | 16 cells, #54–69 |
| 7–13 | 1.4 … 2.6 | — | — | **nothing written** |
| 14 | 2.8 | −0.346 | −0.880 | 15 cells, #33–47 |
| 15 | 3.0 | −0.346 | −0.880 | 15 cells, #33–47 |
| 16 | 3.2 | −0.408 | −0.942 | 16 cells, #32–47 |
| 17 | 3.4 | −0.597 | −1.131 | 16 cells, #30–45 |
| 18 | 3.6 | −0.660 | −1.194 | 16 cells, #30–45 |
| 19 | 3.8 | −0.660 | −1.194 | 16 cells, #30–45 |
| 20 | 4.0 | −0.471 | −1.005 | 16 cells, #31–46 |

Each tap's lit cells scatter to that tap's own slots in the 4096 array —
tap 5's to slots 84, 145, 689…, tap 6's to 14, 67, 368… — never
colliding. The values written are the hill itself: 0.006, 0.073, 0.205,
0.382, 0.577, 0.760, 0.904, 0.986, … peaking at the tap's value.

Three numbers worth holding on to:

- **328 slots lit** — a full training window, every tap written. What
  layer two learned on.
- **220 slots lit** — the query. 14 taps written, 7 left completely
  empty. 5.4 % of the array.
- **7 × 96 = 672 slots** — the region the answer is read from.
  Competition runs over the 220 written slots only; the winner's own
  values in those 672 empty slots, sharpened tap by tap, are the seven
  filled-in y values.

One detail visible in the table: taps 5 and 6 both say 2.168, and taps
14 and 15 both say −0.346. That is layer one's staircase — it returns
the same stored answer for nearby x. Layer two is working from
genuinely imperfect input, not from the true curve.

---

## 6. The four interfaces — what a tap is allowed to say

This is the part you asked about, and it turns out to be the whole
question. All four options below use the *same* everything —
same layer one, same windows, same competition, same learning. Only
what one tap of the window contains differs.

Say layer one's answer at some tap is **y = 1.9**, and that answer came
from its template #412.

**value-absolute** — the tap holds a blurry strip lit at 1.9. Simple
and direct. Two windows overlap in this code when their curves pass
through the same heights.

**value-relative** — the tap holds a blurry strip lit at **1.9 minus
the first tap's y**. If the window's first tap was 0.5, this tap holds
1.4. The whole window is described as *rise and fall from where it
started*, with the absolute height thrown away. The first tap is always
one you know, so nothing is smuggled in. Two windows overlap in this
code when they have the same **shape**, wherever they sit.

**winner (one-hot)** — the tap holds a single lit cell, number 412, out
of 1024 cells (one per layer-one template). This is literally what a
top-1 layer emits. Cell 412 and cell 413 have no relationship
whatsoever even if templates 412 and 413 mean nearly the same thing.

**graded top-16** — the tap holds the 16 templates that matched best,
each with its match strength kept. So maybe cell 412 at 0.9, cell 411
at 0.7, cell 87 at 0.4, and so on. Nearby x values light overlapping
sets of cells with similar strengths, so overlap is back.

The picture `trend_window_codes.png` makes this vivid: in the two value
codes you can *see* the curve running across the window. In the one-hot
and graded codes the vertical axis is template number, which is
arbitrary, so the same curve is scattered into unreadable dots.

---

## 7. Where the answer actually came from

I traced it. The template that filled the gap was built from **exactly
one window position: x = −7.6.**

- x = −7.6 is **1.99 sine-periods** away from the gap at x = 2.0.
- At those two places the curve does the same thing — but the `0.35x`
  trend means x = −7.6 sits **3.23 lower**.

Now measure how close the gap's true shape is to anything in the
training data, two ways:

| measured as | closest training window | how different |
|---|---|---|
| a **shape** (relative to the window's start) | x = −7.6 | **0.079** |
| **absolute** y values | x = +6.8 | **1.840** |

That is the entire result. As a shape, the missing piece is almost
exactly repeated elsewhere in the data. As absolute numbers, nothing in
the data comes close. So:

- the **relative** code finds it and reproduces the descent: **0.033**
- the **absolute** code cannot find it and fails: **3.221**

Same data, same machinery, same overlap in both codes. The only
difference is whether the code can express *"this shape, at whatever
height."*

**The control that proves it is not a broken mechanism.** I reran
everything on a strictly periodic curve, where the same absolute values
really do recur. There, absolute becomes the *best* interface (0.063).
So absolute was never broken — it just had no word for the thing that
needed saying on the trended curve.

---

## 8. How it runs, end to end

### The four phases

**A — train layer one.** 1600 (x, y) pairs, each as one combined
vector. Then frozen; layer one never changes again.

**B — build a table of layer one's answers.** Sweep x across the range
in steps of 0.1 and ask layer one for each (x in, y out). Write the
answers down. Also mark which entries have real training data behind
them — the entries inside the deleted slice do not.

**C — train layer two.** Slide a 21-wide window along that table. Keep
only windows where all 21 entries have data behind them. For each:
subtract the first entry's y from all 21, encode each result as a
strip, drop the strips into the 21 tap slots. That vector is one
training input. 12000 of them, at random positions. Layer two's
templates become shapes.

**D — answer.** Place a window over the slice. 14 taps have data behind
them: ask layer one for those 14 x values and encode its answers. 7 do
not: **write nothing at all** — layer one is never even asked there,
because we know it would just repeat the value at the edge. Compete on
the 14. Read the winner at the 7.

Two things people get wrong reading this:

- Layer two never sees a single prediction. It sees fourteen at once,
  spread along x, because one prediction contains no shape and shape is
  the only thing layer two knows about.
- The x values are not passed to layer two in any form. Only y values
  are. Position along x is carried by which tap slot a value lands in.

### What "read the winner at the 7 taps" means

The winning template is a full 21-channel vector. It has values in
**all** of its slots — including the 7 channels that were left empty in
the query. Those values are what that shape says belongs there.

Reading tap 7 (the x = 1.4 slot) out of the winning template #371:

    gather the 96 numbers at tap 7's slots:
       cell 61   +0.0779   = relative y +1.563
       cell 62   +0.0866   = relative y +1.679
       cell 63   +0.0880   = relative y +1.795   <- brightest
       cell 64   +0.0817   = relative y +1.911
       cell 65   +0.0716   = relative y +2.026
    brightest cell 63  ->  relative y = +1.795
    + reference 0.534   ->  y = 2.329       (truth at x = 1.4 is 2.278)

Three steps: gather that tap's 96 numbers, take the brightest cell,
convert the cell number back into a y value and undo the subtraction.
Exactly how layer one reads its y channel — the same operation, done
seven times.

All seven, from the actual run:

| tap | x | brightest cell | relative y | + ref = y | truth | error |
|---|---|---|---|---|---|---|
| 7 | 1.4 | #63 | 1.795 | 2.329 | 2.278 | +0.051 |
| 8 | 1.6 | #61 | 1.563 | 2.097 | 2.131 | −0.034 |
| 9 | 1.8 | #59 | 1.332 | 1.866 | 1.891 | −0.026 |
| 10 | 2.0 | #57 | 1.100 | 1.634 | 1.575 | +0.059 |
| 11 | 2.2 | #54 | 0.753 | 1.287 | 1.205 | +0.081 |
| 12 | 2.4 | #50 | 0.289 | 0.824 | 0.807 | +0.017 |
| 13 | 2.6 | #47 | −0.058 | 0.476 | 0.408 | +0.068 |

That descent from 2.33 to 0.48 is the answer, and not one training pair
in that band was ever shown.

Information flows **up as values and down as values**. Layer two never
learns which template layer one used, only what it answered. That is a
deliberate choice and the reason this rig sidesteps the one-hot problem
rather than solving it (section 10).

---

## 9. Every number, side by side

Same task, same 1600 training pairs, error is RMSE (lower better).

| method | unseen x in range | inside the deleted slice |
|---|---|---|
| k-nearest neighbours (k=5) | **0.041** | 0.665 |
| random forest (300 trees) | 0.053 | 0.649 |
| small neural network (64-64) | 0.115 | 0.149 |
| just hold the value at the nearest edge | — | 0.819 |
| **layer 1 alone** | 0.067 | 0.817 |
| **layer 1 + layer 2** | 0.067 | **0.033** |

Layer one is competitive on ordinary questions and useless in the gap.
Layer two makes the gap **25× better than layer one** and **4.5× better
than the neural network**, out of the same competing-template machinery.

As a sanity check, layer two filling a blank of the same size *inside*
the data, where it certainly knows the shape: 0.059. So filling the
real gap (0.033) is no harder for it than filling a blank anywhere else.
That is the sign that it genuinely recognised the shape rather than
getting lucky.

### The interfaces

| what a tap says | blank inside the data | the real gap | the real gap, periodic curve |
|---|---|---|---|
| value, relative to the window's start | 0.059 | **0.033** | 0.102 |
| value, absolute | 0.055 | 3.221 | **0.063** |
| the winning template (one-hot) | **0.638** | 4.317 | 0.427 |
| graded, best 16 with strengths | **0.143** | 2.783 | 1.594 |

Read the first column for the overlap question and the second for the
invariance question. They are separate problems.

---

## 10. Your three calls, and what the numbers said

**"Semantic overlap is the only thing that lets this generalise."**
Right, and it is measurable. With a thin training set (60 pairs), only
the blur width changed: a one-cell code gives 0.576, a code blurred
over 4 cells gives **0.163**. 3.5× purely from making similar numbers
look similar. (With 1600 dense pairs this effect hides completely — a
one-cell code memorises fine. If you test the code's generalisation on
dense data, the measurement lies to you.)

**"The wave is a second-order property no single X-Y pair can see, so
we need convolution, or a second layer."** Exactly right, and it is the
difference between 0.817 and 0.033. The window *is* the convolution —
a fixed-width receptive field slid along x — and layer two's templates
are the filters it learned. Nothing else moved the gap at all: more
capacity, better selection, better reads all left it at ~0.7–0.8.

**"When it is expressed as a single winner it is like a one-hot, so
there is no other on-bit for the next layer to correlate — that is the
core problem."** Confirmed, and it is worth putting a number on because
it is the largest single effect in the interface table. Layer two
reconstructing a blank it definitely knows the shape of:

- winner only (one-hot): **0.638**
- best 16 with strengths: **0.143**

4.5× worse, on both test curves. Your reasoning was exactly right: the
overlap that makes layer one work is destroyed at layer one's own
output, so layer two has nothing to correlate over. The same U-curve
shows up inside a single layer too — reading one layer's answer from
the winner alone gives 0.094, from the best 16 gives 0.046, and from
*every* positive match gives 0.488. Graded and sparse, not one, not
all.

**Where I would extend your diagnosis.** Fixing the overlap is
necessary but does not finish the job. Compare the two value codes:
they overlap identically and score 0.033 and 3.221. A message built out
of **template identities** can only say which templates fired, and
since each layer-one template ties an x to a y at one fixed place, no
such message has a word for "the same shape, higher up". That
invariance is not degraded in transmission — it is unsayable in that
vocabulary. Graded speech cannot rescue it, which is why graded top-16
still scores 2.783 in the gap despite being the best activation code by
a distance.

So there are two questions to ask of any interface, not one:

1. Does it overlap? (one-hot fails; graded passes)
2. Can it express the invariance the task needs? (both identity codes
   fail; only a code measured relative to a reference the receiver also
   holds passes)

---

## 10.1 Why the relative trick cannot be pushed down to layer one

A natural follow-up: if absolute values are the problem, lift the offset
out of x and y *before* layer one, let it learn on the relative data,
and add the offset back to its output by hand — no arithmetic inside
the network.

That procedure is already exactly what layer two does (`ref` subtracted
before encoding, added back after reading). The question is whether it
can live one level lower. It cannot, for a simple reason: **at layer one
there is nothing to subtract.** You know x and you know nothing about y
— that is the whole query. The offset is a property of y, and y is what
you are asking for. An offset only becomes subtractable once you hold a
*stretch* with at least one known value in it, which is what a window
is.

**Something in the stack has to be absolute.** Layer one is the *where*
— absolute, indexes the world, supplies the anchor. Layer two is the
*what* — relative, reusable, transfers anywhere. The offset is the
handoff between them.

Measured anyway, at the same 7 gap positions, each row handing layer one
progressively more of the answer:

| lifted out by hand | gap error |
|---|---|
| (a) nothing — layer one untouched | 0.817 |
| (b) remove the `0.35x` trend, add it back after | 1.271 |
| (c) …and fold x by the sine period 4.833 | 0.317 |
| (d) …and remove the `0.6cos(0.7x)` term too | **0.020** |
| layer two, told nothing about the function | **0.033** |

(b) gets *worse*, and it is the informative row. Removing the trend from
y grants layer one no new ability — it still holds the value at the rim
— and the detrended curve happens to fall more steeply across the gap
(2.85 instead of 2.29), so holding the rim is a bigger mistake. Lifting
an offset out of y was never what was blocking it.

(c) is where it starts working, and note that what changed was **x**,
not y: folding makes x = 2.0 and x = 6.83 the same input, so the gap's
phase suddenly has training data from other periods. It stalls at 0.317
because `0.6cos(0.7x)` is not periodic at 4.833, so the folded data
contradicts itself.

(d) is essentially perfect and worthless: reaching it required supplying
the trend, the period and the second harmonic — the whole function bar
the sine.

### Generic versus specific invariance

Layer two's relative encoding is a hand-supplied inductive bias as well;
it did not discover the idea from nothing. The difference is in kind:

- *"measure each window from its own first tap"* — **generic**. Works on
  any signal, requires knowing nothing about the data.
- *"fold x by 4.833, subtract 0.35x and 0.6cos(0.7x)"* — **specific**.
  You had to already know the function.

Layer two matches the fully hand-solved version (0.033 against 0.020)
using only the generic rule. That is the whole value of putting the
invariance in the architecture rather than in the preprocessing: layer
two found *where* the offset was and *how much* it was, per window, by
itself — settling on a reference of 0.534 for the window over the gap
and matching a shape from x = −7.6, with nobody supplying a period or a
trend.

---

## 11. What this does and does not show

**Shows.** Competing templates with a single winner and a blurry code
can do regression well; two of them stacked can reconstruct a stretch
of missing input far better than a fitted network, using nothing but
pattern-completion. The interface between layers is a first-class
design decision that decides what the layer above can learn at all.

**Does not show.** Layer two did not invent anything. It recognised a
repeat. If the curve had no recurring structure — a one-off bump in the
gap and nowhere else — it would have nothing to transfer and would fail.
The honest description is *"reuse of a shape seen elsewhere,"* not
extrapolation.

**A caveat about my own rig.** Layer two hears layer one's *answer*
re-encoded as a fresh blurry strip. It does not hear layer one's
activity. So this experiment does not solve the problem you raised — it
avoids it, by returning to value space at the boundary. That is
defensible when the variable being passed has a natural scale (a
number, a position, an angle) — the map itself carries the overlap.
It will not carry to abstract layers where there is no scale to
re-encode into, and there the two questions in section 10 are still
open.
