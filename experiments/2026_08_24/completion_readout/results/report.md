# Completion readout — 8x8, K1=512 (retrained arms)

| L1 out | top-1 | top-10 raw vote | top-10 margin vote | softmax vote (t=.02) | all-200 vote |
|---|---|---|---|---|---|
| dense | 0.9080 | 0.8468 | 0.8984 | 0.9080 | 0.6894 |
| top3 | 0.8560 | 0.8094 | 0.8598 | 0.8584 | 0.7294 |
| top1 | 0.8594 | 0.7610 | 0.8220 | 0.8594 | 0.7682 |

Voting: relu-correlation weights, each memory pushes one-hot(owner); class argmax.
