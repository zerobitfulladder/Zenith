# Big-window L2 (8x8 s2, footprint 15x15, K2=2048)

Train wall-time: 379.2s (cupy, B=64).
Base (3x3 L2, footprint 10x10, K2=1024) reference: probe .9570, hard .9112, consistent 10/10.

| probe L2 | hard | consistent | train_s |
|---|---|---|---|
| 0.9572 | 0.9132 | 10/10 | 379 |

Generation: generation.png. Templates: templates_L{1,2,3}.png. Weights: weights.npz.
