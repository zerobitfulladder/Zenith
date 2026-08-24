# 3-level wide rig (L1 8x8/512 -> wide L2 -> memory)

| K2 | out | probe L2 | hard | consistent | train_s |
|---|---|---|---|---|---|
| 256 | dense | 0.9660 | 0.8728 | 10/10 | 19 |
| 512 | dense | 0.9618 | 0.8972 | 10/10 | 25 |
| 256 | top3 | 0.9644 | 0.7616 | 10/10 | 21 |

4-level references: dense .9094 / probe .9668; all-top3 .7596; L1-only top3 .8642.
