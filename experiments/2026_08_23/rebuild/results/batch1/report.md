# Rebuild — local receptive fields + concat top layer (MNIST)

Config: 7x7 grid of 4x4 patches, K1=25/position, K2=100, ETA1=0.05, ETA2=0.04, EPOCHS=2, TRAIN_N=20000, SEED=42.
Whole-image baselines from earlier runs: selector acc 0.699, concat acc 0.727, L1 probe 0.871.

| lam | label-only consistent | acc hard | acc soft | acc soft norm | acc soft top10 | code probe | mean per-position entropy | templates per class |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 100.0% | 0.6210 | 0.6646 | 0.6722 | 0.7124 | 0.9146 | 0.955 | 7 8 16 14 9 11 4 12 11 8 |
| 1.0 | 100.0% | 0.4596 | 0.5592 | 0.6174 | 0.5826 | 0.9146 | 0.955 | 7 8 16 14 9 11 4 12 11 8 |

Figures: generation_labelonly_lam*.png, l1_templates_center.png, recon_test.png
