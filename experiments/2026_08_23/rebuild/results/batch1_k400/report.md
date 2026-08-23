# Rebuild — local receptive fields + concat top layer (MNIST)

Config: 7x7 grid of 4x4 patches, K1=25/position, K2=400, ETA1=0.05, ETA2=0.04, EPOCHS=2, TRAIN_N=20000, SEED=42.
Whole-image baselines from earlier runs: selector acc 0.699, concat acc 0.727, L1 probe 0.871.

| lam | label-only consistent | acc hard | acc soft | acc soft norm | acc soft top10 | code probe | mean per-position entropy | templates per class |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 100.0% | 0.6558 | 0.6786 | 0.7632 | 0.7352 | 0.9152 | 0.955 | 31 42 38 50 31 38 41 54 43 32 |

Figures: generation_labelonly_lam*.png, l1_templates_center.png, recon_test.png
