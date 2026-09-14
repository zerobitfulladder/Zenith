# `tiles/` — non-overlapping windows, so each piece of evidence is counted once

Problem 1 from the `patches/` line: at stride 1 every pixel sat under 25 windows and was counted 25
times over. The machine was then always certain (mean top guess 0.998, 99% of pictures over 0.99)
and being unsure stopped meaning anything — accuracy was flat at 0.31-0.46 for every gap under 40
nats. The two-humped belief the whole line is aimed at could not exist.

The cheap fix, chosen over the joint-parent-table version: **stride = window size**. 4x4 windows at
stride 4 tile 28x28 exactly, no padding, 7x7 = **49 windows**, and every pixel has exactly one
parent. 49 x 16 = 784: the windows are the picture, nothing shared and nothing left over.

Everything else is carried over: 10 grids per window shaped like the window, filled by adding pixel
values into the grid the label picks, with a floor of 0.002 under every pixel in training only (the
reading half of the floor does nothing under the ink read and hurts under the shape read). 49 x 10 x
16 = **7,840 numbers**, one counting pass, 0.33 s.

`python run.py tiles && python board.py tiles` -> [`results/tiles.png`](results/tiles.png).

## Held out, 10,000 pictures

| | 4x4 tiles, no overlap | 5x5 stride 1 |
|---|---|---|
| read of C (ink) | 0.8046 | 0.8397 |
| read of C (shape / corr) | 0.7854 / 0.7492 | 0.8172 / 0.7964 |
| mean top guess | **0.929** | 0.998 |
| pictures claiming over 0.99 | **62%** | 99% |
| median gap to the second answer | **6.5 nats** | 236 nats |
| pictures within 5 nats of a second answer | **39%** | 1.3% |

Accuracy against how torn it is — the thing that was broken:

| gap (nats) | 0-1 | 1-2 | 2-5 | 5-10 | 10-20 | 20-40 |
|---|---|---|---|---|---|---|
| **tiles** | 0.39 | 0.51 | 0.71 | 0.90 | 0.98 | **1.00** |
| stride 1 | 0.43 | 0.44 | 0.31 | 0.41 | 0.41 | 0.46 |
| pictures (tiles) | 907 | 874 | 2166 | 3013 | 2580 | 459 |

**3.5 points of accuracy bought a machine that knows when it does not know.** The curve is
monotone again: when it is torn it is right about 4 times in 10, when it is sure it is right every
time, and 39% of pictures now sit within 5 nats of a second answer instead of 1%. Panel D shows the
twelve most torn pictures with genuinely two-humped beliefs over the parent's value — a 9 against a
4, a 2 against a 7. That is the material a resolution step needs, and it did not exist before.

## What this build does not fix

*It still collapses.* With no overlap the read is even more obviously one scorecard picture per
digit: the 49 grids stitched together are the scorecard, cell for cell. Nothing here escapes that —
only a read that asks a window about its patch as a whole would.

*Blank patches still say nothing* (problem 3). 4x4 windows are large enough that fewer of them are
blank than before, but the ones that are contribute zero to every digit.

*The windows are coarse.* A 4x4 grid is a weak description of a patch, which is where the 3.5 points
went. The whole-pattern tables of the 3x3 build still read better (0.9251) than anything here.

## Follow-ups: where the ceiling is, and what a weight on top can reach

Three questions came out of "is this the theoretical maximum?".

**[`redundancy.py`](redundancy.py) — is the gap caused by pixels repeating each other?** Each pixel
is read exactly once, so nothing is added twice; the claim is about the assumption. Thinning the
picture out spreads the survivors apart so they repeat each other less:

| keep | pixels | neighbour corr *within a digit* | counted | fitted | gap |
|---|---|---|---|---|---|
| every | 784 | 0.665 | 0.8225 | 0.9195 | 9.7 |
| every 2nd | 196 | 0.313 | 0.8096 | 0.9032 | 9.4 |
| every 3rd | 100 | 0.103 | 0.7725 | 0.8393 | 6.7 |
| every 4th | 49 | 0.019 | 0.6751 | 0.7348 | 6.0 |

The assumption is plainly false (neighbours still correlate 0.665 once the digit is known) and the
gap does shrink as pixels spread apart — but only from 9.7 to 6.0. Neighbour redundancy is worth
about a third of it; the rest is longer-range structure thinning does not remove.

**[`pixels.py`](pixels.py) — one count per pixel per digit.** Note first that 1x1 windows under our
rules are degenerate: a one-cell grid always sums to 1, so it says nothing. The meaningful version
is `P(pixel inked | digit)` read with both halves.

| | reads | mean top guess | median gap | within 5 nats |
|---|---|---|---|---|
| 4x4 tiles (where ink lands) | 0.8046 | 0.929 | 6.5 | 39% |
| per pixel, ink only | 0.6887 | 0.969 | 15.0 | 21% |
| per pixel, ink + blank | **0.8319** | 0.986 | 29.8 | 10% |
| same form, fitted | 0.9132 | 0.890 | 3.9 | 68% |

**Absence of ink is worth 14.3 points** — much the largest effect measured all day, and problem 3
falls out of this form for free. And the counted and fitted models here have *identical* functional
form (`sum x * [log p - log(1-p)]` plus one number per digit, the same count of free numbers), so
the 8.1-point gap between them is purely counted-versus-fitted with no structural difference at all.

**[`importance.py`](importance.py) — a weight on top, tables untouched.** The user's proposal: keep
the counted tables, learn one number per (window, label) saying how much that window's testimony is
worth, raised for the true label and lowered for the label that wrongly won. Not backpropagation —
the weights sit directly on the output, nothing is differentiated through anything. Votes are
centred per window first, so "more important" means "counts for more", and w = 1 everywhere
reproduces the tiles model exactly.

| arm | reads | mean top guess | median gap |
|---|---|---|---|
| flat (w = 1) | 0.8046 | 0.929 | 6.54 |
| counted (w from how far the label moves the table) | 0.7651 | 0.948 | 9.35 |
| learned (the raise/lower rule, 4 passes, step 0.001) | **0.8145** | 0.471 | 0.77 |

It learns, and it gains **+1 point**, against the 8 points on the table. The reason is structural:
**24 of the 32 adjacent pixel pairs around a 4x4 window are inside it**, so three quarters of the
redundancy that matters lives where a single number per window cannot reach it. Finer granularity
would reach it — but one weight per *cell* is 7,840 free numbers multiplying 7,840 table entries,
which is exactly the fitted scorecard again. A weight on top has a low ceiling by construction.

It also costs the probabilistic reading: with the weights shrunk (mean 0.55) the score is no longer
a log-probability, mean top guess falls to 0.47 and the gap to 0.77 nats, so "how sure am I" stops
meaning anything. Setting importance by counting was worse than not weighting at all.

**Where that points.** The redundancy the weight cannot reach is *within* a window — cells of one
patch repeating each other. That is exactly what a table over the whole pattern does not assume, and
it is why the 3x3 whole-pattern build reads 0.9251 while every cell-by-cell model here sits near
0.83. The fix for within-window redundancy is a table that never assumed the cells were independent,
not a weight applied afterwards.
