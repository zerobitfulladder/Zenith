# Batch 2 — shared dictionary, stride-2 overlap, feathered rendering (MNIST)

Config: 4x4 windows, stride 2 (169 positions), shared K1=36, K2=200, lam=0.5, ETA1=0.02, ETA2=0.04, EPOCHS=2, TRAIN_N=20000, SEED=42.
Baseline (batch 1, independent nodes, no overlap, K2=200): hard 0.648, soft 0.705, soft_norm 0.725, soft_top10 0.737, probe 0.916.

| hard | soft | soft_norm | soft_top10 | probe | label-only consistent | dic entropy | dead | templates per class |
|---|---|---|---|---|---|---|---|---|
| 0.7430 | 0.7456 | 0.7160 | 0.7822 | 0.9466 | 10/10 | 0.994 | 0 | 15 14 24 23 24 21 21 24 18 16 |

Figures: shared_dictionary.png, generation_labelonly.png, recon_test.png
