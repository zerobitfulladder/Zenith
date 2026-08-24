# Residual learning at L1 (8x8, K1=512; L2/L3/L4 standard)

| speech | probe L2 | probe L3 | hard | consistent | pairs cos>0.8 | >0.9 | margin | train_s |
|---|---|---|---|---|---|---|---|---|
| dense | 0.9656 | 0.9680 | 0.8984 | 10/10 | 0.0001 | 0.0000 | 0.052 | 22 |
| resid | 0.9674 | 0.9686 | 0.8030 | 10/10 | 0.0001 | 0.0000 | 0.052 | 21 |

Std-learning references: dense .9094 (margin 0.038) / top3 .8642 / resid speech .8502. Std palette: adjacent-pair median cos 0.816, max 0.968.
