# GPU mini-batch trainer (cupy, B=128)

Train wall-time: 8.3s (CPU online baseline: ~1200s).
CPU sparselearn baselines: probes .9528/.9592/.9590, hard .8748, peak .718/.534.

| probe L1 | probe L2 | probe L3 | hard | consistent | peak2 | peak3 |
|---|---|---|---|---|---|---|
| 0.9550 | 0.9600 | 0.9588 | 0.8842 | 10/10 | 0.622 | 0.424 |

Figure: l3_parts.png. Weights: weights.npz
