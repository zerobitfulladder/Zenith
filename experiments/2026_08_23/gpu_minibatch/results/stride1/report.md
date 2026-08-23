# GPU mini-batch trainer (cupy, B=128)

Train wall-time: 27.2s (CPU online baseline: ~1200s).
CPU sparselearn baselines: probes .9528/.9592/.9590, hard .8748, peak .718/.534.

| probe L1 | probe L2 | probe L3 | hard | consistent | peak2 | peak3 |
|---|---|---|---|---|---|---|
| 0.9542 | 0.9626 | 0.9658 | 0.8492 | 10/10 | 0.690 | 0.562 |

Figure: l3_parts.png. Weights: weights.npz
