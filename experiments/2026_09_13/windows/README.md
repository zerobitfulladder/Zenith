# Layer 1 as a grid of windows

2026-09-13. Layer 1 in terms of neurons, with limited receptive fields:

    cells      a grid of windows on the image (12x12, stride 4 -> 25 windows),
               12 cells per window. A cell sees only its window. No weight
               sharing. Random start.
    matching   Pearson: patch and template are mean-centred and unit length
               inside the window, so a cell's drive is a correlation in [-1, 1].
    settling   u <- (1-dt) u + dt ( c - (G - I) a ),  a = u where u > theta.
               G is the overlap between cells' templates in image space: zero
               for cells whose windows do not overlap.
    learning   every cell that is on after settling rotates toward the patch it
               saw, by the angle eta * (its correlation). Nothing is picked.
    threshold  each cell's theta drifts to hold its firing rate near a target;
               silent cells come in by their threshold falling.

No writeup was kept for this run; the `neurons/` line continued with whole-image
cells instead (see [`../neurons/README.md`](../neurons/README.md) and the day
README). The held-out numbers in the result files, 300 cells:

| target rate | cells on / image | inked windows left silent | windows with 2+ on | patch variance unexplained | dead |
|---|---|---|---|---|---|
| 0.04 | 10.3 | 61% | 3% | 0.78 | 3% |
| 0.08 | 16.8 | 48% | 13% | 0.73 | 1% |
| 0.15 | 21.1 | 45% | 19% | 0.72 | 0% |

## Files

    layer1.py    python layer1.py      one pass over 8k images; held out: unexplained,
                                       cells on, dead; writes results/layer1_<tag>.json/.npz
    board.py     the figure, results/board_<tag>.png
    results/log_r*.txt   training logs

`results/layer1.json` / `board.png` are the same run as the `_r0.04` files.
