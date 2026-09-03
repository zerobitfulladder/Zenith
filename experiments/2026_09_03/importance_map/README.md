# An exact importance map from the tally read

2026-09-03. The read is a sum over positions of table rows, so each position's
share of the decision margin (winning class minus runner-up) is its exact
contribution. Painted over the 9x9 window each patch covers: red argued for
the decision, blue against. No gradient, no approximation. 9x9 strokes, 6x6
cells, MNIST 12k/3k, no pressure, seed 7, table accuracy 0.9697.

`results/importance.png`: ten correctly read digits and four misreads. Row 2
is the margin map, row 3 the evidence for the read class alone.

- The decisive regions are where the class is: the top junction of the 4, the
  top bar of the 7, the loop of the 9, the spine of the 3.
- Blue inside a correctly read digit is where its own strokes resemble the
  runner-up (the 3's left side, the 0's inner edges).
- The misreads show the argument: the 6 read as 1 is carried by its long
  vertical stroke against its own loop; the 3 read as 9 by its top curve
  against its lower one.
- Two limits: 6x6 cells and 9x9 windows make the map blocky; and near-blank
  patches at stroke edges carry real evidence ("no ink here" is a fact about
  the class), which shows as broad red in background regions.

    uv run python importance.py    # ~20 s
