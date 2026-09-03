# OR-ing the winners: a position-free code, measured

2026-09-03. The champion describes an image as one winner per position and
counts them in a table indexed by (template, 6x6 cell, class). The question
raised: collapse the winners into ONE 400-bit vector with no position at all,
and count on that. Same trained templates, three readouts, test set shifted by
0-3 pixels. MNIST 12k/3k, 3 epochs, seed 7.

```
size  readout   shift 0   shift 1   shift 2   shift 3   active bits
 5    cells     0.9683    0.9263    0.7397    0.3673
 5    count     0.7247    0.7213    0.6967    0.6737
 5    OR        0.7547    0.7453    0.7237    0.7023      163/400
 9    cells     0.9687    0.9227    0.6897    0.3113
 9    count     0.8837    0.8770    0.8573    0.8230
 9    OR        0.8777    0.8687    0.8370    0.7930      153/400
13    cells     0.9530    0.8973    0.6283    0.2417
13    count     0.9043    0.8927    0.8637    0.8210
13    OR        0.8920    0.8850    0.8610    0.8270       92/400
```

`cells` = the champion's readout. `count` = position summed out, a template
present k times votes k times. `OR` = presence only, each template votes once.

- **Position is worth 21 points at 5x5 and 8 at 9x9 / 13x13.** A bag of 5x5
  strokes is nearly the same bag for every digit; a bag of 9x9 or 13x13 pieces
  carries much more of the digit in each piece, so the OR code gets to 0.88-0.89.
- **The OR code is what shift-tolerance looks like.** At 3 pixels the champion
  is at chance-plus (0.24-0.37) while the OR code loses only 5-7 points. The
  per-cell table is not a description of the image, it is a description of the
  image at this alignment.
- **OR vs count is a wash** (within a point either way), so presence loses
  nothing over multiplicity. About 90-160 of the 400 bits are on per image, so
  the code is sparse but not very: a 13x13 OR code is 92 bits.
- The 2026-08 pooling result sits between these rows: pooling to 4x4 cells was
  the best of both (0.85 at 1px shift where unpooled gave 0.22). OR is pooling
  to 1x1.

## Files

| | |
|---|---|
| `or_readout.py` | trains the rf_sweep rig at each size, evaluates three readouts under shift |
| `results/or_readout.json` | the table |

    uv run python or_readout.py    # ~2 min

## Position as a dial: `grid_sweep.py`

Pool the winners into g x g cells before counting; g = 1 is the OR code, g = 6
is the champion. Trained per grid, seed 7.

```
size  grid  cells   shift 0   shift 1   shift 2   shift 3
 5     1      1     0.7317    0.7310    0.7083    0.6883
 5     2      4     0.9293    0.9147    0.8597    0.7597
 5     3      9     0.9517    0.9370    0.8497    0.6797
 5     4     16     0.9653    0.9393    0.8267    0.5703
 5     6     36     0.9677    0.9250    0.7367    0.3683
 9     1      1     0.8960    0.8923    0.8743    0.8357
 9     2      4     0.9573    0.9510    0.9167    0.8017
 9     3      9     0.9640    0.9547    0.8687    0.6737
 9     4     16     0.9673    0.9450    0.8067    0.5177
 9     6     36     0.9680    0.9217    0.6903    0.3090
```

Going from 1 to 4 cells recovers 20 of the 21 lost points at 5x5 and 6 of the
7 at 9x9, and keeps most of the shift tolerance. **9x9 templates in 2x2 cells
is the practical position-light code**: 0.957 unshifted, 0.80 at 3 px, a
1600-bit vector with ~150 bits on. Every extra cell after that buys under a
point unshifted and costs 10-15 points at 3 px.

Why a CNN does not face this trade: it pools a little, re-describes, and pools a
little more, so arrangement survives across several small pools instead of one
global one. This rig pools once. The two-rung result of 2026-08 (0.9492 with
4x4 pooling between rungs) is the one step of that ladder the project has
taken; the identity-only message between rungs is what limits the next.
