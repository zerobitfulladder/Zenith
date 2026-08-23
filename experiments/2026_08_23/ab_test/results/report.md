# A/B test — Variant B contrast feedback on Fashion-MNIST

Config: TRAIN_N=20000, TEST_N=5000, K1=100, ETA1=0.04, ETA2=0.1, EPOCHS=2, gamma_on=2.0, seeds=[0, 1, 2, 3, 4, 5, 6, 7].
Evaluation is feedforward with neutral gain (no label at test).

## Condition means (± sd over seeds)

| k_active | feedback | acc | probe_l1 | probe_l2 | pair_margin | top1_corr | entropy |
|---|---|---|---|---|---|---|---|
| 1 | off | 0.5756 ± 0.0095 | 0.7942 ± 0.0063 | 0.6695 ± 0.0061 | 0.0963 ± 0.0073 | 0.8549 ± 0.0014 | 0.8973 ± 0.0089 |
| 1 | on | 0.5735 ± 0.0109 | 0.7930 ± 0.0055 | 0.6715 ± 0.0051 | 0.0987 ± 0.0060 | 0.8466 ± 0.0017 | 0.9542 ± 0.0040 |
| 5 | off | 0.5679 ± 0.0128 | 0.7763 ± 0.0060 | 0.6589 ± 0.0079 | 0.0990 ± 0.0047 | 0.8366 ± 0.0020 | 0.9678 ± 0.0068 |
| 5 | on | 0.5645 ± 0.0152 | 0.7746 ± 0.0065 | 0.6541 ± 0.0127 | 0.0996 ± 0.0058 | 0.8235 ± 0.0020 | 0.9841 ± 0.0028 |

## Paired per-seed deltas (feedback on − off)

### k_active = 1

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | -0.0020 | 2/8 | 0.2890 |
| probe_l1 | -0.0012 | 2/8 | 0.0871 |
| probe_l2 | +0.0020 | 5/8 | 0.3766 |
| pair_margin | +0.0024 | 5/8 | 0.0851 |
| top1_corr | -0.0083 | 0/8 | 0.0000 |
| entropy | +0.0569 | 8/8 | 0.0000 |

### k_active = 5

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | -0.0034 | 3/8 | 0.2461 |
| probe_l1 | -0.0016 | 1/8 | 0.1763 |
| probe_l2 | -0.0048 | 2/8 | 0.1920 |
| pair_margin | +0.0005 | 4/8 | 0.6544 |
| top1_corr | -0.0130 | 0/8 | 0.0000 |
| entropy | +0.0162 | 8/8 | 0.0002 |

## Graded-learning main effect (k=5 − k=1, feedback off)

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | -0.0076 | 0/8 | 0.0104 |
| probe_l1 | -0.0179 | 0/8 | 0.0000 |
| probe_l2 | -0.0106 | 1/8 | 0.0126 |
| pair_margin | +0.0027 | 7/8 | 0.0954 |
| top1_corr | -0.0183 | 0/8 | 0.0000 |
| entropy | +0.0705 | 8/8 | 0.0000 |
