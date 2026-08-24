# Rich palette / sparse speech (stride-1 flagship geometry)

Merged table (first four rows from the initial run, last two from the
resumed run after the argpartition-OOM fix):

| K1 | L1 out | probe L2 | probe L3 | hard | consistent | L1 crowd | top1-2 margin | train_s |
|---|---|---|---|---|---|---|---|---|
| 36 | dense | 0.9624 | 0.9646 | 0.8972 | 10/10 | 0.349 | 0.111 | 13 |
| 36 | top3 | 0.9666 | 0.9644 | 0.8742 | 10/10 | 0.348 | 0.111 | 16 |
| 36 | top1 | 0.9584 | 0.9618 | 0.8242 | 10/10 | 0.349 | 0.111 | 16 |
| 512 | dense | 0.9646 | 0.9666 | 0.9066 | 10/10 | 0.347 | 0.017 | 24 |
| 512 | top3 | 0.9514 | 0.9580 | 0.8172 | 10/10 | 0.348 | 0.017 | 26 |
| 512 | top1 | 0.9324 | 0.9492 | 0.7350 | 10/10 | 0.348 | 0.018 | 24 |

Verdict and reading: see the day README, "Rich palette / sparse speech".
Figures: generation_K{36,512}_{dense,top3,top1}.png
