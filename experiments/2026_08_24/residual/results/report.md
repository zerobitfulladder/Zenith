# Residual competition speech (8x8, K1=512, K2=64, K3=100, cap 3)

| arm | probe L2 | probe L3 | hard | consistent | speakers L1/L2/L3 | train_s |
|---|---|---|---|---|---|---|
| L1resid | 0.9658 | 0.9682 | 0.8502 | 10/10 | 2.97/0.00/0.00 | 26 |
| allresid | 0.9624 | 0.9618 | 0.7898 | 10/10 | 2.97/3.00/3.00 | 39 |

References (same geometry): L1-only top3 .8642 / top1 .8590 / dense .9094; all-layer top3 .7596 / top1 .7092.
