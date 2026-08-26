# Trained-with-quantization (hidden accumulator), base rig

f32 base reference: hard .9112, consistent 10/10.
Post-hoc comparison arms: precision/results/post_hoc/report.md.

| bits | hard | consistent | train_s |
|---|---|---|---|
| 6 | 0.9266 | 10/10 | 80 |
| 4 | 0.9258 | 10/10 | 81 |
| 3 | 0.9136 | 10/10 | 79 |
| 2 | 0.8934 | 10/10 | 78 |

Figure: generation_qat.png (f32 reference row on top).
