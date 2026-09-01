# The pipeline on Fashion-MNIST, and brightness channels beside L1

2026-09-01 / 09-02.

`run.py` moves the whole MNIST pipeline to Fashion-MNIST unchanged (same 5x5
patches, 30 experts of 8 templates, ink-only competition, the counted table by
4x4 cell). No writeup of that run was kept; its result file
`results/fashion.json` gives 0.7793 for the cell-conditioned table.
`templates.py` draws what the Fashion experts learned to look at
(`results/templates.png`).

The 09-02 addendum below is moved verbatim from the day writeup.

## Brightness channels beside L1 (`channels.py`) — all refuted

The hypothesis: centring, normalising and gating discard brightness three ways,
and the pixel-counting control priced brightness at +9.5 points on Fashion. So
four counted channels were laid beside the champion, weights chosen on a
2,000-image carve, tables rebuilt on the full 12k before the test was touched:
`flat` (brightness of the dropped patches), `int` (raw mean of every patch),
`magbin` (the magnitude-bin evidence the binned table conditions on but never
counts), `silh7/4` (the whole image average-pooled coarse).

```
                        fashion (3 seeds)       mnist (control)
binned4 anchor          0.8440 +- 0.0015        0.9697
combo of channels       0.8437 +- 0.0007        0.9697
best single, ANY weight +0.0007  (flat)         nothing
int at weight 1.0       -0.0537                 the shouting law again
```

Nothing, at any weight, on either dataset. The pixel control's brightness gain
is **subsumed** by the champion, not additive to it: which positions are live,
in which cell, at which contrast bin already pins the brightness structure —
and Fashion is textured nearly everywhere (77.5% of positions live, against
57.2% on MNIST), so "flat and bright fabric fill" barely exists as an event.
The gap to logistic-on-pixels (0.8512) stays open; the untried Fashion moves
are the ones that paid on MNIST and were never run here: the per-cell linear
readout (+0.70 there) and width x data (800 x 40k).
