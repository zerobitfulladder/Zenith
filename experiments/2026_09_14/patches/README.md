# `patches/` — one layer of patch nodes under a single parent whose value is the label

The settled picture: the lower layer is what the higher layer caused. One parent `C` with ten
values, 576 patch nodes as its effects. Ambiguity is not a choice between two parents, it is a
spread over the values of the one parent, so what has to come out two-humped is `P(C | patches)`.

## The build

* a node looks at a **5x5 window of the image, stride 1, no padding**: 24 x 24 = **576 nodes**.
* a node holds **ten tables, one per value of the parent, each shaped like the window itself**
  (5 x 5 = 25 cells). Training: pick the table by the label, add the patch's pixel values into it
  cell by cell. Then normalise each table to sum to 1, so it is a distribution over *where ink
  lands inside that window* given that value of the parent.
* **nothing is shared between nodes.** Each node's ten tables are counted only from its own window.
* 576 x 10 x 25 = **144,000 numbers**, one counting pass over 50,000 images, 2.5 s.
* how the tables are **read** is deliberately left open — that is the next conversation.

`python run.py tables && python board.py tables` -> [`results/tables.png`](results/tables.png),
weights in [`results/tables.npz`](results/tables.npz) (`mass_raw` = the raw ink sums, `tables` =
normalised).

## What the tables look like

Drawn in place, the 576 tables of one label reassemble that digit out of local pieces: in the
middle of the image each table is a little oriented ramp — ink on one side of the window, none on
the other — and the ramps turn as the stroke sweeps through. Nodes near the border collect almost
no ink (36 of the 5,760 tables saw under one unit of ink in 50,000 images) and their normalised
tables are noise: a table with nothing behind it looks just as confident as one with 83,000 units
of ink behind it, because normalising throws the mass away.

Of the 314 nodes that see real ink for every label, the label moves node 388's table most (total
variation 0.29 from its own label-average) and node 469's least (0.13 — its ten tables are very
nearly one table, so that node says almost nothing about the parent).

## Measured: the 144,000 numbers are 7,840 numbers

Neighbouring nodes' tables differ by **0.000** on the 20 cells they share. That is not an
approximation, it is exact: node (r,c) and node (r,c+1) add the very same pixels of the very same
images into their overlapping cells, so their raw sums there are identical. The raw accumulator is
just `sum of all images of label c`, read through 576 windows — **10 images of 784 pixels = 7,840
numbers**, plus 5,760 normalising divisors that are themselves determined by those images. Stride 1
buys 576 views of one picture, not 576 pictures. See panel D.

## The read collapses (for later)

Not run, but worth writing down before we choose a read. If a patch is scored the natural way —
each pixel contributes `pixel value x log(the table cell it fell on)`, summed over cells and then
over nodes — the sums can be regrouped by pixel instead of by node, and every pixel ends up
multiplied by one fixed number per label. The 576 nodes vanish and what is left is a single weight
image per label matched against the input: **a linear classifier**. This is true of any table that
keeps pixels separately, because independent per-pixel evidence is a sum, and a sum of sums is a
sum. A node only becomes more than the pixels under it when something in the read depends on the
window as a whole — looking up the whole pattern, dividing by the window's own ink total,
or letting nodes suppress each other.

## The brute-force read (`read.py`, `board_read.py`)

Every patch scores every value of the parent, the scores are summed, the largest wins. Three ways
of scoring one patch against one table, over the 25 cells of a window:

| | per-patch score | summing it is | held out |
|---|---|---|---|
| ink | `Σ x · log T` — a bag of ink units, each landing at a cell drawn from the table | Bayes | **0.8397** |
| shape | `Σ (x / patch ink) · log T` — every node weighted equally however much ink it holds | a vote | 0.8172 |
| corr | correlation of the patch with the table | a vote | 0.7964 |

A blank patch scores 0 under all three. These tables say where ink lands, so where there is no ink
there is nothing to say — this read cannot use the absence of ink as evidence, and for MNIST that
is a lot to give up.

`python read.py tables && python board_read.py tables` -> [`results/tables_read.png`](results/tables_read.png).

**The collapse, measured.** Summing the 576 per-patch log-likelihoods was rebuilt as one 28x28
weight image per label matched against the raw pixels. Over 10,000 held-out images the two scores
differ by at most **0.025 out of scores averaging 8,400** (float rounding) and agree on **100%** of
predictions. Read 1 is a linear classifier; the 576 nodes are not doing anything the weight image
could not do alone. Panel A draws the ten weight images — and their loud border frames are the
mass problem again: a table with no ink behind it still votes at full volume.

**Summing works, and costs.** It lands below the 3x3 whole-pattern tables (0.9251) even though it
has five times the nodes, because a table over where-ink-lands is weaker than a table over which
shape appeared. Worse, the posterior is destroyed: median gap between the best two values of C is
**236 nats** (21.3 at 3x3 stride 3), mean top-1 posterior 0.9982, 99% of images over 0.99, because
each pixel sits in up to 25 windows and is counted 25 times over. And the gap stops meaning
anything: accuracy is flat at 0.31-0.46 for every gap below 40 nats and 0.885 above, where at 3x3
it climbed monotonically from 0.44 to 1.00. The two-humped posterior we want is gone.

Panels B-E draw the per-patch scores as maps — each patch's score for one value of C minus its own
average over the ten values, so red means this patch prefers this label. The evidence is spatially
sensible (a 0 is argued for by its ring and against by its empty middle), and on a torn image two
labels light up in different places, which is the thing a resolution step would have to arbitrate.

## Fix 2: a grid should speak only as loudly as the evidence behind it

429 of the 5,760 grids were built from under 100 units of ink, and scaling every grid to add up to
1 made them look exactly as certain as a grid built from 83,000 units. The fix is to add flat
imaginary ink to every grid before scaling: a well-fed grid does not notice, a starved one goes
flat, and a flat grid gives every value of C the same score.

The user proposed the same thing from the other end — **Eigengrau**, a floor under every pixel so
black is never truly black, applied in training *and* in reading. In training the two are the same
operation: a floor of E collects `n_c * E` units of flat ink in every grid, so E = 0.002 is 10
units per cell, and the two sweeps agree to four decimals. `fix2.py` is the knob as imaginary ink,
`eigengrau.py` as a pixel floor and it also measures the reading half separately.
[`results/tables_eigengrau.png`](results/tables_eigengrau.png).

| floor E | ink/cell | acc, floor in training only | acc, floor in both | mean top-1 | edge ÷ centre weight |
|---|---|---|---|---|---|
| 0 | 0 | 0.8397 | 0.8397 | 0.9982 | 2.05 |
| 0.0005 | 2.5 | 0.8385 | 0.8383 | 0.9980 | 0.93 |
| **0.002** | **10** | **0.8369** | **0.8370** | 0.9979 | **0.76** |
| 0.01 | 50 | 0.8318 | 0.8319 | 0.9981 | 0.57 |
| 0.05 | 250 | 0.8215 | 0.8214 | 0.9968 | 0.39 |

Kept: E = 0.002, by the rule "the largest floor that costs at most 0.005 accuracy".

**Three results, one of them the point.**

*The fix does what it was aimed at.* The frame of noise around the edge of every scorecard is gone
(edge-to-centre weight 2.05 -> 0.76, where 1.0 is even) and the digit shapes underneath are
untouched. Panels A and B.

*It buys no accuracy.* 0.8397 -> 0.8370, and it keeps falling as the floor rises. The border noise
was never costing us anything; the read was already ignoring it. So this was a correctness fix, not
a performance one, and it should be kept for that reason alone.

*The reading half of the floor does nothing at all.* Floor-in-training and floor-in-both differ by
at most 0.0006 anywhere in the sweep. This is the prediction confirmed: at read time a uniform floor
adds the same number to a digit's score for every picture, so it is a standing per-digit offset, not
evidence. **Problem 3 is untouched — a blank patch still cannot speak.** Only a channel that reads
the blankness itself (an inverted copy) or a stored "how much ink is here" would do that.

*And the over-confidence is untouched*: mean top-1 flat at 0.998 across the whole sweep. It does not
come from this problem. It comes from problem 1, each pixel being counted by 25 overlapping windows.

## Does the floor let a blank patch speak?  Under one read yes — and it makes things worse

The claim that Eigengrau cannot fix problem 3 was only true of the **ink** read, where the floor's
contribution factors out as `E * sum(log T)`, one constant per digit added to every picture alike.
Under the **shape** read it does not factor out: each patch is divided by its own ink total, so a
floored blank patch becomes a flat patch with a full-weight vote. Measured
([`shape_floor.py`](shape_floor.py), [`results/tables_shapefloor.png`](results/tables_shapefloor.png)):

| floor E | shape read, blanks voting | blanks muted | cost of letting blanks vote | images guessed "1" (true 11%) |
|---|---|---|---|---|
| 0 | 0.8172 | 0.8172 | 0 (a blank patch has nothing to divide by) | 14% |
| 0.0005 | 0.5841 | 0.8234 | **-0.2393** | 28% |
| 0.002 | 0.6508 | 0.8304 | -0.1796 | 24% |
| 0.01 | 0.7389 | 0.8362 | -0.0973 | 19% |
| 0.05 | 0.7938 | 0.8341 | -0.0403 | 16% |

Blank patches genuinely speak here — muting them moves the answer by up to 24 points, so this is
not a constant. But what they say is wrong. Floored, **every blank patch is the same flat patch**,
so each votes for whichever digit's grid is flattest at its window, and the digit that leaves most
of the picture blank is flattest nearly everywhere: the machine guesses "1" on 28% of images when
the true share is 11%. And 42% of all patches are blank, each carrying the same voting weight as a
patch that actually saw a stroke, so they outvote the evidence.

The lesson is about **weight, not voice**. A blank patch should not count for as much as a patch
that saw something, and it should say something more specific than "I am flat". An inverted channel
does both: a patch's blankness vote is weighted by how much blankness it holds, and it is about
*where* the blankness sits. That is still the thing to try for problem 3 — this run says why the
cheap version of it fails.

(The muted column also shows fix 2 working on its own: 0.8172 -> 0.8362 as the floor rises, purely
from starved grids going flat.)
