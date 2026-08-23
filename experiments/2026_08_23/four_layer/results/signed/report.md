# 4-layer hierarchy, reporting mode: signed (MNIST)

k_out per level (topk mode): L1=3, L2=2, L3=1; learning top-1 everywhere.
Naive-stack baselines: probes 0.943/0.919/0.855, hard 0.463, top10 0.553.

Config: L1 4x4s2 K=36 (grid 13); L2 3x3s2 K=64 (grid 6); L3 3x3s1 K=100 (grid 4); top K=200, lam=0.5, EPOCHS=2, TRAIN_N=20000, SEED=42.
Batch-2 baselines: probe(L1 code) 0.9146-0.947, hard 0.743, top10 0.782.

| probe L1 | probe L2 | probe L3 | hard | top10 | label-only consistent |
|---|---|---|---|---|---|
| 0.8984 | 0.9194 | 0.9224 | 0.8688 | 0.7594 | 10/10 |

Figures: generation_labelonly.png, l2_motifs.png, l3_parts.png, recon_roundtrip.png. Weights: weights.npz
