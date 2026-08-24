# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nan | nan | 0.9010 | 10/10 | 0.230 | 0.589 | 0.246 | 0.00 | 0.040 | 0.988 | 0 | 19 |
| dir_cmax | nan | nan | 0.8952 | 10/10 | 0.223 | 0.589 | 0.248 | 0.00 | 0.038 | 0.991 | 0 | 19 |
| dir_margin | nan | nan | 0.9010 | 10/10 | 0.256 | 0.583 | 0.229 | 0.01 | 0.047 | 0.970 | 0 | 20 |
| dir_marginx | nan | nan | 0.8986 | 10/10 | 0.263 | 0.579 | 0.229 | 0.00 | 0.046 | 0.972 | 0 | 20 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
