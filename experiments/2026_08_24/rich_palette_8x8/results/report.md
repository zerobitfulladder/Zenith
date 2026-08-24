# Rich palette at 8x8 L1 windows (63 dims after centering)

| K1 | L1 out | probe L2 | probe L3 | hard | consistent | L1 crowd | top1-2 margin | train_s |
|---|---|---|---|---|---|---|---|---|
| 64 | dense | 0.9666 | 0.9658 | 0.8990 | 10/10 | 0.272 | 0.092 | 10 |
| 64 | top3 | 0.9698 | 0.9652 | 0.8864 | 10/10 | 0.272 | 0.092 | 10 |
| 64 | top1 | 0.9660 | 0.9658 | 0.8434 | 10/10 | 0.272 | 0.092 | 10 |
| 512 | dense | 0.9666 | 0.9668 | 0.9094 | 10/10 | 0.256 | 0.038 | 19 |
| 512 | top3 | 0.9694 | 0.9676 | 0.8642 | 10/10 | 0.256 | 0.038 | 21 |
| 512 | top1 | 0.9654 | 0.9698 | 0.8590 | 10/10 | 0.257 | 0.038 | 21 |

4x4 reference: dense@512 hard .9066 margin 0.017; top1@512 hard .7350; crowd 0.348 everywhere.
Figures: generation_K{64,512}_{dense,top3,top1}.png
