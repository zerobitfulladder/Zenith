# Softmax-sampled, unnormalized L1 (MNIST)

Validated normalized 2-layer baselines: probe 0.9146, hard 0.743 (batch 2).

| tau | probe | hard | usage entropy | dead | label-only consistent |
|---|---|---|---|---|---|
| 0.5 | 0.9134 | 0.1388 | 0.995 | 0 | 10/10 |
| 0.2 | 0.9150 | 0.1086 | 0.988 | 0 | 10/10 |
| 0.1 | 0.9020 | 0.1068 | 0.986 | 0 | 10/10 |

Figures: generation_tau*.png, dictionary_tau*.png
