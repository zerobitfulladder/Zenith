# `arrangements/` — counting whole arrangements, and how finely to cut a pixel

A window's state is the **whole arrangement** of its pixels, not one number per position. The table
therefore never assumes the pixels inside a window are independent — it stores the combinations it
actually saw. "Everything off" is one arrangement among the rest, so a blank window speaks like any
other, and problem 3 does not arise.

The price is that with L levels per pixel and p pixels there are L^p arrangements to fill by
counting. The input is a float, so L is a free choice, and it competes with window size for the
same budget. `python run.py levels` -> [`results/levels.json`](results/levels.json).

| window | levels | alphabet | windows | numbers | reads | mean top guess | median gap |
|---|---|---|---|---|---|---|---|
| **4x4** | **2** | 65,536 | 49 | 32.1M | **0.9343** | 0.989 | 26.2 |
| 3x3 | 2 | 512 | 100 | 512K | 0.9338 | 0.989 | 24.3 |
| 3x3 | 3 | 19,683 | 100 | 19.7M | 0.9307 | 0.991 | 32.1 |
| 2x2 | 4 | 256 | 196 | 502K | 0.9242 | 0.992 | 39.3 |
| 2x2 | 8 | 4,096 | 196 | 8.0M | 0.9139 | 0.995 | 53.0 |
| 2x2 | 2 | 16 | 196 | 31K | 0.8981 | 0.986 | 24.0 |

Imaginary mass is spread flat over the whole alphabet (A/S a column) so that it means the same
thing whatever the alphabet size; A was picked from {1, 10, 100} per row.

**The float value does carry something.** At 2x2, cutting a pixel into 4 levels instead of 2 is
worth **+2.6 points** (0.8981 -> 0.9242). Binarising is not free when the window is small.

**But levels and window size buy the same thing and compete for it.** Once the alphabet is large
enough, more levels only thin the counts: 3x3 goes 0.9338 -> 0.9307 from 2 to 3 levels, and 2x2 goes
0.9242 -> 0.9139 from 4 to 8. What matters is the alphabet, not either factor alone, and every arm
above 0.92 sits between 256 and 65,536 arrangements.

**A prediction of mine that the measurement refuted.** I said 4x4 binary (65,536 arrangements) would
be too many to count against 50,000 pictures. It is the best arm here: 0.9343, with only 0.8% of
training pictures landing on an arrangement seen exactly once. The real edge is somewhere between
4x4 (0.8% singletons, works) and 7x7 (28.5% singletons, memorises) — untested, and it would need
sparse counting since 5x5 binary is 33M arrangements.

**Best number of the day: 0.9343**, above the best-fit scorecard's 0.9195 and the whole of the
cell-by-cell line near 0.83. This is a counted, generative model — it can still generate and still
answer about a missing patch — beating the discriminatively fitted ceiling of the per-pixel form.
Counting the right object beats fitting the wrong one.

**Still open: it is over-confident.** Mean top guess 0.989, median gap 26 nats. 49 or 100 windows
each contribute a full log-probability as though they were independent given the digit, and they are
not — the same disease as stride-1, milder. Calibration is at least monotone again (0.48 / 0.53 /
0.59 / 0.77 / 0.94 / 1.00 by gap for the 4x4 arm), unlike the overlapping build where it was flat.
