# F8: ambiguity in plasticity (8x8, K1=512, scope=l1, 2 epochs, seed 42)

| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 | L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.9664 | 0.9660 | 0.9020 | 10/10 | 0.228 | 0.592 | 0.257 | 0.00 | 0.037 | 0.989 | 0 | 19 |
| dir_cmax | 0.9668 | 0.9664 | 0.9058 | 10/10 | 0.226 | 0.590 | 0.258 | 0.01 | 0.036 | 0.991 | 0 | 19 |
| dir_margin | 0.9662 | 0.9660 | 0.9056 | 10/10 | 0.271 | 0.589 | 0.235 | 0.01 | 0.045 | 0.971 | 0 | 37 |
| dir_marginx | 0.9666 | 0.9688 | 0.9142 | 10/10 | 0.275 | 0.587 | 0.236 | 0.00 | 0.045 | 0.974 | 0 | 38 |

Step sizes rate-matched across arms (mean step = mean c_max).
Reference (dense@512, same rig): hard .9094, margin 0.038.
clones% = share of off-diagonal L1 template pairs above 0.95.
