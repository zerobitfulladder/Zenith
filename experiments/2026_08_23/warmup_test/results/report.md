# Warm-up test — unsupervised settling, then Variant B contrast feedback

Config: Fashion-MNIST, TRAIN_N=20000, TEST_N=5000, K1=100, ETA1=0.04, ETA2=0.1, gamma=2.0, seeds=[0, 1, 2, 3, 4, 5, 6, 7], top-1 learning.
Conditions (gamma per epoch): off=(0,0), always=(2,2), warmup=(0,2).

## Condition means (± sd over seeds)

| condition | acc | probe_l1 | probe_l2 | pair_margin | top1_corr |
|---|---|---|---|---|---|
| off | 0.5756 ± 0.0095 | 0.7942 ± 0.0063 | 0.6695 ± 0.0061 | 0.0963 ± 0.0073 | 0.8549 ± 0.0014 |
| always | 0.5735 ± 0.0109 | 0.7930 ± 0.0055 | 0.6715 ± 0.0051 | 0.0987 ± 0.0060 | 0.8466 ± 0.0017 |
| warmup | 0.5736 ± 0.0119 | 0.7935 ± 0.0059 | 0.6732 ± 0.0042 | 0.0989 ± 0.0073 | 0.8463 ± 0.0011 |

## Paired per-seed deltas (warmup − off)

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | -0.0019 | 4/8 | 0.3876 |
| probe_l1 | -0.0007 | 4/8 | 0.3170 |
| probe_l2 | +0.0037 | 5/8 | 0.1899 |
| pair_margin | +0.0026 | 6/8 | 0.0582 |
| top1_corr | -0.0086 | 0/8 | 0.0000 |

## Paired per-seed deltas (always − off)

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | -0.0020 | 2/8 | 0.2890 |
| probe_l1 | -0.0012 | 2/8 | 0.0871 |
| probe_l2 | +0.0020 | 5/8 | 0.3766 |
| pair_margin | +0.0024 | 5/8 | 0.0851 |
| top1_corr | -0.0083 | 0/8 | 0.0000 |

## Paired per-seed deltas (warmup − always)

| metric | mean delta | seeds improved | paired t p-value |
|---|---|---|---|
| acc | +0.0001 | 4/8 | 0.9277 |
| probe_l1 | +0.0005 | 5/8 | 0.2250 |
| probe_l2 | +0.0017 | 5/8 | 0.0665 |
| pair_margin | +0.0002 | 5/8 | 0.7924 |
| top1_corr | -0.0003 | 2/8 | 0.4086 |
