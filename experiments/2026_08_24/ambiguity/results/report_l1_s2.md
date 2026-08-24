# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nan | nan | 0.8884 | 10/10 | 0.224 | 0.562 | 0.246 | 0.00 | 0.040 | 0.986 | 0 | 19 |
| dir_cmax | nan | nan | 0.8844 | 10/10 | 0.220 | 0.584 | 0.246 | 0.00 | 0.038 | 0.989 | 0 | 19 |
| dir_margin | nan | nan | 0.8904 | 10/10 | 0.260 | 0.582 | 0.229 | 0.00 | 0.047 | 0.968 | 0 | 20 |
| dir_marginx | nan | nan | 0.8858 | 10/10 | 0.246 | 0.561 | 0.228 | 0.01 | 0.047 | 0.971 | 0 | 20 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
