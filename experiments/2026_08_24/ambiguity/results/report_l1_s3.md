# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nan | nan | 0.9146 | 10/10 | 0.213 | 0.590 | 0.255 | 0.00 | 0.037 | 0.988 | 0 | 19 |
| dir_cmax | nan | nan | 0.9180 | 10/10 | 0.207 | 0.585 | 0.256 | 0.00 | 0.037 | 0.990 | 0 | 19 |
| dir_margin | nan | nan | 0.9170 | 10/10 | 0.248 | 0.572 | 0.234 | 0.00 | 0.046 | 0.969 | 0 | 20 |
| dir_marginx | nan | nan | 0.9202 | 10/10 | 0.244 | 0.577 | 0.235 | 0.00 | 0.046 | 0.972 | 0 | 20 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
