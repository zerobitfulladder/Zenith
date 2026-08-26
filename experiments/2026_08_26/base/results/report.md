# Base rig 2026-08-26 (3-layer, L1 8x8/1024, L2 1024, dense, top-1)

Train wall-time: 74.4s (cupy, B=128).
wide3level K=512 dense reference: probe .9660, hard .8972.

| probe L2 | hard | consistent | train_s |
|---|---|---|---|
| 0.9570 | 0.9112 | 10/10 | 74 |

Generation: generation.png. Templates: templates_L{1,2,3}.png. Weights: weights.npz.
