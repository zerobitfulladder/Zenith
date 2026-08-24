# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nan | nan | 0.8888 | 10/10 | 0.211 | 0.576 | 0.257 | 0.00 | 0.039 | 0.986 | 0 | 19 |
| dir_cmax | nan | nan | 0.8970 | 10/10 | 0.211 | 0.577 | 0.259 | 0.00 | 0.039 | 0.988 | 0 | 19 |
| dir_margin | nan | nan | 0.9030 | 10/10 | 0.252 | 0.569 | 0.236 | 0.00 | 0.048 | 0.967 | 0 | 21 |
| dir_marginx | nan | nan | 0.8932 | 10/10 | 0.251 | 0.581 | 0.238 | 0.01 | 0.047 | 0.970 | 0 | 21 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
