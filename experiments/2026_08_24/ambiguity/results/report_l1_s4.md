# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nan | nan | 0.9024 | 10/10 | 0.215 | 0.570 | 0.249 | 0.00 | 0.040 | 0.985 | 0 | 20 |
| dir_cmax | nan | nan | 0.8970 | 10/10 | 0.205 | 0.572 | 0.251 | 0.00 | 0.039 | 0.988 | 0 | 19 |
| dir_margin | nan | nan | 0.9042 | 10/10 | 0.246 | 0.581 | 0.231 | 0.01 | 0.047 | 0.969 | 0 | 20 |
| dir_marginx | nan | nan | 0.9036 | 10/10 | 0.255 | 0.591 | 0.234 | 0.00 | 0.046 | 0.971 | 0 | 20 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
