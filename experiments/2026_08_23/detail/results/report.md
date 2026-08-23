# Detail build — stride-1 dense code (MNIST)

Config: 4x4 windows at stride 1 (625 positions, code dim 22500), learning at the stride-2 subgrid only, K1=36, K2=200, lam=0.5, EPOCHS=2, SEED=42.
Batch-2 baselines: hard 0.743, top10 0.782, probe 0.9146 (seed 42).

| hard | top10 | probe | label-only consistent |
|---|---|---|---|
| 0.8008 | 0.8084 | 0.9516 | 10/10 |

Figures: generation_labelonly.png, recon_test.png. Weights: weights.npz
