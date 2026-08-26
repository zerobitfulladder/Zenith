# Encoder/decoder bridge (unification layer)

E/D stacks: 8x8s2/64 -> 3x3s2/128; unification KU=400 over [enc ; dec], 15000 images x 2 epochs.
Identity preserved (pixel-LR judge): 7/10 on unseen test digits.

| class | judge | corr(input) | corr(class mean) |
|---|---|---|---|
| 0 | 0 | 0.851 | 0.801 |
| 1 | 9 | 0.746 | 0.643 |
| 2 | 2 | 0.637 | 0.744 |
| 3 | 3 | 0.720 | 0.790 |
| 4 | 4 | 0.692 | 0.430 |
| 5 | 5 | 0.677 | 0.316 |
| 6 | 6 | 0.651 | 0.506 |
| 7 | 7 | 0.811 | 0.732 |
| 8 | 5 | 0.528 | 0.507 |
| 9 | 7 | 0.711 | 0.613 |

Figure: reconstructions.png
