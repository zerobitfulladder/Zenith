# Results

Full MNIST: 50,000 to count and learn, 10,000 to choose the cost and the trigger, the 10,000
test images for every number here. One seed. Two standard errors on a test accuracy of
0.97 over 10,000 images is 0.003, so differences under a third of a point are noise.

## Arm 1, the counted table

| readout | test | tokens per image |
|---|---|---|
| 400 templates, stride 2, table over 6x6 cells, nothing fitted | 0.9725 | 81.7 |
| the same at stride 1 (the 2026_09_01 shape) | 0.9771 | 329.9 |
| CNN, 118,346 parameters, 8 epochs | 0.9908 | |
| linear map on pixels | 0.9242 | |

The table has 144,000 entries; the CNN has fewer parameters than that.

## Arm 2, store: name neighbouring pairs, count again

| names chosen by | how the name is used | round | test | tokens per image |
|---|---|---|---|---|
| count | replaces the pair | 1 | 0.9738 | 68.5 |
| count | replaces the pair | 2 | 0.9729 | 60.6 |
| count | replaces the pair | 3 | 0.9707 | 55.9 |
| how class-telling | replaces the pair | 1 | 0.9728 | 71.5 |
| how class-telling | replaces the pair | 3 | 0.9718 | 60.4 |
| count, 1000 names | replaces the pair | 1 | 0.9734 | 59.2 |
| count | added beside the parts | 1 | 0.9722 | 94.8 |
| how class-telling | added beside the parts | 1 | 0.9726 | 91.9 |
| count, 1000 names | added beside the parts | 1 | 0.9708 | 104.2 |

## Arm 3, substitute: each goal takes the moves that pay for it

| version | test | flipped right | flipped wrong | moves taken on |
|---|---|---|---|---|
| baseline, no moves | 0.9725 | | | |
| nine single moves, gated by the goal-by-move table, chosen cost 0 | 0.9737 | 26 | 14 | 25% of images |
| singles and pairs of moves, gated, chosen cost 0.05 | 0.9733 | 14 | 6 | 16% |
| every goal every single move, free, no table | 0.9774 | | | |
| every goal every single and paired move, free | 0.9722 | | | |
| oracle: the single move that most helps the true digit | 0.9959 | | | |
| oracle over singles and pairs | 0.9987 | | | |

The static merge, the same move applied to every image and the table counted on moved images:

| always | test |
|---|---|
| deskew | 0.9766 |
| thin | 0.9658 |
| thicken | 0.9679 |
| shift up / down | 0.9717 / 0.9717 |
| shift left / right | 0.9733 / 0.9734 |
| scale up / down | 0.9730 / 0.9736 |

The goal-by-move table, deskew column (lift of the goal's margin per token, true digits minus
the rest): 0: -0.06, 1: +0.19, 2: +0.07, 3: 0.00, 4: -0.01, 5: -0.16, 6: +0.07, 7: +0.02,
8: -0.06, 9: -0.01. Every other move is negative for every goal. Figure: `goal_by_move.png`.
