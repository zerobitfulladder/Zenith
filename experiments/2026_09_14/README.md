# 2026-09-14

Theory-first reset. The day began with a discussion instead of a run: every build is four separate
choices (how an image is composed from templates, how the code is found, what a template learns
from, how the code is read), the last three days moved several at once, and what a cell learns is
decided by the composition rule (pick one -> wholes; signed sum -> wholes plus corrections;
nonnegative shared sum -> strokes, which is Lee & Seung's nonnegative factorisation in online form).
Settled: outputs nonnegative (the private drive is the signed log-odds, the output is its
rectified form), parts as causes, settling as the picker. Open: what a template learns from, and
how a layer above can be more than a copy of the layer below.

---

## `slots/` — L2 as a counted belief network with named parent slots

Pixels and nodes at four levels, no pixel table, L1 = frozen share-then-scale strokes, L2 nodes are
parents of L1 nodes through four named slots per child and one full 256-row count table per child,
root priors as counts, mean-field settling, learning by counting the settled state. Held out: L2 on
0.73 per image, 28% dead, 78% of L2 nodes raise at most one child, full model -27.2 nats per image
against -25.9 for independent children. Two failures: every node that fires is carried by one child whose row
alone clears the prior (6 to 8 nats against 3.5), and 28 nodes die because naming is rich-get-richer
(the busiest node is named by 50 children, the dead ones by 0.4). Lever not yet run: a table floor
that caps one child's evidence at the prior's price.

Full writeup: [`slots/README.md`](slots/README.md).

---

## `patches/` — ambiguity as a value of one parent, not a choice between parents

Second line of the day, after the correction: an apparent circle that could be a tilted ellipse is
not two parents to choose between, it is one parent whose posterior is two-humped, so the machine
to build is a parent whose VALUE is inferred. One parent C with ten values (the label), 576 patch
nodes as its effects: 5x5 windows at stride 1, no padding, nothing shared between nodes, and ten
tables per node shaped like the window itself, filled by adding pixel values and normalised to sum
to 1. 144,000 numbers, one counting pass, 2.5 s. Drawn in place the tables reassemble each digit
out of little oriented ramps. Two things measured: normalising throws the mass away, so a border
table with no ink behind it looks as confident as a centre one with 83,000 units; and neighbouring
tables agree EXACTLY (0.000) on the 20 cells they share, so the 144,000 numbers are 10 images of
784 pixels seen through 576 windows. The read is left open, but the natural one regroups by pixel
and collapses to a linear classifier.

Brute-force read of those tables (`patches/read.py`): each patch scores each value of C, the scores
are summed. Ink (sum x.logT, the real likelihood) 0.8397, shape (patch normalised first) 0.8172,
correlation 0.7964 -- all below the 3x3 whole-pattern tables at 0.9251. The collapse was then
confirmed by measurement, not argument: the 576-node sum was rebuilt as one 28x28 weight image per
label against raw pixels and the two agree to 0.025 out of 8,400, same prediction on 100% of 10,000
images. Stride-1 overlap counts each pixel up to 25 times, so the median gap between the best two
values of C is 236 nats (21.3 at 3x3), 99% of images sit above 0.99 posterior, and accuracy by gap
goes flat -- the two-humped posterior the whole line is aimed at does not survive this read.

Fix 2 (`patches/fix2.py`, `patches/eigengrau.py`): 429 of 5,760 grids were built from under 100
units of ink and still scaled to look as certain as grids built from 83,000. Adding flat imaginary
ink before scaling -- identical, measured to four decimals, to the user's Eigengrau floor under
every pixel at E=0.002 -- removes the frame of noise round every scorecard (edge/centre 2.05 ->
0.76) and costs 0.8397 -> 0.8370. Three findings: the fix is a correctness fix, not a performance
one; the reading half of the floor does nothing anywhere in the sweep (<=0.0006), confirming that a
uniform floor at read time is a per-digit offset and not evidence, so problem 3 stands; and the
over-confidence does not move (0.998 flat), so it comes from the 25x window overlap, not from this.

Follow-up (`patches/shape_floor.py`): the claim that Eigengrau cannot reach problem 3 held only for
the ink read, where the floor factors out as one constant per digit. Under the shape read it does
not factor out and blank patches genuinely vote -- muting them moves the answer by up to 24 points.
What they say is wrong: floored, every blank patch is the same flat patch, so each votes for
whichever digit's grid is flattest there, the machine guesses "1" on 28% of images against a true
11%, and at 42% of all patches they outvote the patches that saw a stroke. Letting blanks vote costs
0.8304 -> 0.6508 at the kept floor. The lesson is weight, not voice: blankness must be weighted by
how much of it there is, which is what an inverted channel does and a uniform floor does not.

Full writeup: [`patches/README.md`](patches/README.md).

---

## `tiles/` — non-overlapping windows, so each piece of evidence is counted once

Problem 1, fixed the cheap way: stride = window size. 4x4 windows at stride 4 tile 28x28 exactly,
49 windows, every pixel with exactly one parent, 49 x 16 = 784 with nothing shared. 7,840 numbers,
one counting pass, 0.33 s, floor 0.002 in training only. Read of C 0.8046 against 0.8397 at stride
1 -- 3.5 points paid for a machine that knows when it does not know. Mean top guess 0.929 not 0.998,
median gap to the second answer 6.5 nats not 236, and 39% of pictures now sit within 5 nats of a
second answer instead of 1.3%. The calibration curve is monotone again (0.39 / 0.51 / 0.71 / 0.90 /
0.98 / 1.00 by gap, against a flat 0.31-0.46 at stride 1), and the most torn pictures carry real
two-humped beliefs over the parent's value. Still unfixed: the read still collapses to one scorecard
per digit, blank patches still say nothing, and a 4x4 grid is a coarse description of a patch.

Follow-ups in `tiles/` on "is this the theoretical maximum?". No: the same functional form fitted
rather than counted reads 0.9195. `redundancy.py` -- neighbouring pixels still correlate 0.665 once
the digit is known, and thinning them apart shrinks the counted-vs-fitted gap from 9.7 to 6.0
points, so neighbour redundancy is about a third of it. `pixels.py` -- one count per pixel per
digit, read with ink AND blank, gives 0.8319, and absence of ink alone is worth 14.3 points (0.6887
ink-only), much the largest effect of the day; counted and fitted there have identical form, so
their 8.1-point gap is purely counted-versus-fitted. `importance.py` -- the user's proposal of a
learned weight per (window, label) on top of untouched tables: not backprop, it does learn, and it
buys +1 point (0.8046 -> 0.8145) because 24 of the 32 adjacent pixel pairs around a 4x4 window are
INSIDE it, where one number per window cannot reach them; it also costs the probabilistic reading
(mean top guess 0.47). Finer granularity would reach the redundancy but one weight per cell IS the
fitted scorecard. The redundancy that matters is within a window, which is what a whole-pattern
table does not assume -- and why the 3x3 whole-pattern build still reads 0.9251.

Full writeup: [`tiles/README.md`](tiles/README.md).

---

## `arrangements/` — counting whole arrangements, and how finely to cut a pixel

Raised by the user: the morning's 3x3 differed from the afternoon in TWO ways, not one -- whole
arrangements vs one number per position, and binarised vs the float input. Measured by holding the
window and varying the levels. The float does carry something: at 2x2, four levels instead of two is
worth +2.6 points (0.8981 -> 0.9242). But levels and window size buy the same thing and compete for
one budget -- past a point more levels only thin the counts (3x3: 0.9338 at 2 levels, 0.9307 at 3;
2x2: 0.9242 at 4 levels, 0.9139 at 8), and every arm above 0.92 sits between 256 and 65,536
arrangements. My prediction that 4x4 binary (65,536) would be too many to count was refuted: it is
the best arm, 0.9343, with 0.8% of pictures on a once-seen arrangement. That is the best number of
the day, above the fitted scorecard ceiling of 0.9195 and the whole cell-by-cell line near 0.83 --
a counted generative model beating the discriminatively fitted ceiling of the per-pixel form.
Still over-confident (mean top guess 0.989) but monotone in the gap again.

Full writeup: [`arrangements/README.md`](arrangements/README.md).
