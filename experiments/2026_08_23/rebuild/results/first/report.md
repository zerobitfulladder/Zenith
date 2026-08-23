# Rebuild — local receptive fields + concat top layer (MNIST)

Config: 7x7 grid of 4x4 patches, K1=25/position, K2=100, ETA1=0.05, ETA2=0.04, EPOCHS=2, TRAIN_N=20000, SEED=42.
Whole-image baselines from earlier runs: selector acc 0.699, concat acc 0.727, L1 probe 0.871.

| lam | label-only consistent | acc without label | code probe | mean per-position entropy | templates per class |
|---|---|---|---|---|---|
| 0.5 | 100.0% | 0.6548 | 0.9170 | 0.944 | 7 8 15 14 10 11 4 12 11 8 |
| 1.0 | 100.0% | 0.4892 | 0.9170 | 0.944 | 7 8 15 14 10 11 4 12 11 8 |

Figures: generation_labelonly_lam*.png, l1_templates_center.png, recon_test.png
