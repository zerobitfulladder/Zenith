# Weight precision sweep (base rig, post-training quantization)

Per-template symmetric uniform rounding + one f32 scale per row
(renormalization). f32 base reference: hard .9112, consistent 10/10.

| bits | hard | consistent |
|---|---|---|
| f32 | 0.9112 | 10/10 |
| f16 | 0.9110 | 10/10 |
| 8 | 0.9114 | 10/10 |
| 6 | 0.9114 | 10/10 |
| 4 | 0.9110 | 10/10 |
| 3 | 0.9012 | 10/10 |
| 2 | 0.8906 | 10/10 |

Figure: generation_vs_bits.png (one generation row per arm).
