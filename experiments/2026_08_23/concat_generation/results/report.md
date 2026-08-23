# Concat generation test — label-only queries at the top layer

Config: TRAIN_N=20000, K1=100, K2=50, ETA1=0.04, ETA2=0.04, EPOCHS=2, SEED=42. L2 is a free-competition Zenith over [L1 code ; lam * onehot]; the selector baseline shares the same L1.

| lam | label-only retrieval consistent | gen corr (concat) | gen corr (selector) | acc without label | templates per class |
|---|---|---|---|---|---|
| 0.25 | 100.0% | 0.187 | 0.698 | 0.5666 | 4 4 5 4 10 4 5 3 5 6 |
| 0.5 | 100.0% | 0.532 | 0.698 | 0.6854 | 4 4 5 4 10 4 5 5 4 5 |
| 1.0 | 100.0% | 0.265 | 0.698 | 0.7272 | 4 4 5 4 10 4 5 5 4 5 |

'Templates per class' lists how many of the K2 templates store each
digit 0-9 as their dominant label. Figures: generation_labelonly_lam*.png
