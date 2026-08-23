# Variant B Gain Feedback — gamma sweep report (gain mode: prototype)

Config: TRAIN_N=20000, TEST_N=5000, K1=100, ETA1=0.04, ETA2=0.1, EPOCHS=2, SEED=42. Evaluation is always feedforward with neutral gain (no label at test time).

## Headline table

| gamma | pipeline acc | L1 probe acc | mean top-1 corr | L1 usage entropy | dead templates |
|---|---|---|---|---|---|
| 0.0 | 0.6994 | 0.8708 | 0.784 | 0.949 | 0 |
| 0.25 | 0.7014 | 0.8710 | 0.782 | 0.950 | 0 |
| 0.5 | 0.6994 | 0.8730 | 0.781 | 0.956 | 0 |
| 1.0 | 0.6974 | 0.8720 | 0.778 | 0.963 | 0 |
| 2.0 | 0.7054 | 0.8718 | 0.767 | 0.965 | 0 |

## Confusable-pair margins (higher = better separated)

| gamma | 1v7 | 3v5 | 4v9 | 5v8 | 7v9 |
|---|---|---|---|---|---|
| 0.0 | 0.7168 | 0.1864 | 0.0854 | 0.1532 | 0.0844 |
| 0.25 | 0.7216 | 0.1915 | 0.0841 | 0.1637 | 0.0828 |
| 0.5 | 0.7272 | 0.1841 | 0.0851 | 0.1491 | 0.0870 |
| 1.0 | 0.7185 | 0.1814 | 0.0834 | 0.1471 | 0.0824 |
| 2.0 | 0.7245 | 0.1774 | 0.0812 | 0.1384 | 0.0806 |

## Confusion matrix (rows true, cols predicted)

### gamma = 0.0

```
[[455   1   4   9   2  27   7   0   4   1]
 [  0 525   7   3   0   3   0   0   3   1]
 [ 14  54 343  33  21   1  32   6  17   1]
 [  3  12  26 376   5  14   4   5  38  21]
 [  2   8   1   0 317   0   6   3   9 130]
 [ 15  47   5 132  13 178   7   9  11  19]
 [ 14  23   7   6  19  15 422   0   5   0]
 [  6  30   4   0  67   0   0 371  18  29]
 [  6  41  22  78   3  21   5   0 234  30]
 [  3  23   0  10 165   0   1  35  21 276]]
```

### gamma = 0.25

```
[[457   1   4   8   2  26   7   0   4   1]
 [  0 525   7   3   0   2   1   0   2   2]
 [ 13  55 345  31  18   1  34   6  18   1]
 [  2  12  26 379   5  12   4   6  37  21]
 [  1  11   1   0 321   0   7   3   7 125]
 [ 16  44   5 127  15 183  10   9   9  18]
 [ 14  24   7   5  19  14 424   0   4   0]
 [  6  29   4   1  71   0   0 366  19  29]
 [  6  44  20  78   4  20   6   0 234  28]
 [  3  23   0  10 171   0   1  33  20 273]]
```

### gamma = 0.5

```
[[454   1   4   9   1  29   6   0   5   1]
 [  0 525   7   3   0   3   0   0   3   1]
 [ 14  53 342  35  20   1  34   6  16   1]
 [  3  12  27 376   5  14   4   6  37  20]
 [  2  11   1   0 319   0   6   3   7 127]
 [ 16  43   7 131  13 185   7   9   9  16]
 [ 14  23   5   7  19  15 422   0   6   0]
 [  5  27   4   0  68   0   0 373  18  30]
 [  7  40  25  84   4  23   5   0 223  29]
 [  3  22   0  10 163   0   1  37  20 278]]
```

### gamma = 1.0

```
[[459   1   4   8   2  26   5   0   2   3]
 [  0 525   7   3   0   3   0   0   2   2]
 [ 17  61 334  30  19   1  32   8  18   2]
 [  4  13  27 376   5  14   4   6  34  21]
 [  3   9   1   0 319   0   5   4   7 128]
 [ 17  42   6 127  13 185  10  10  10  16]
 [ 13  23   6   4  20  17 425   0   3   0]
 [  6  27   5   0  68   0   0 371  20  28]
 [  7  37  25  78   5  26   5   0 229  28]
 [  3  21   0  10 166   1   1  45  23 264]]
```

### gamma = 2.0

```
[[453   1   4   8   2  30   6   0   3   3]
 [  0 525   8   3   0   2   0   0   3   1]
 [ 13  51 347  31  17   1  31   8  20   3]
 [  2  12  28 376   4   9   4   7  38  24]
 [  2  11   2   0 324   0   4   3   7 123]
 [ 15  46   7 133  13 172  12  12  11  15]
 [ 14  18   6   7  18  11 429   0   7   1]
 [  6  24   5   0  56   0   0 385  20  29]
 [  5  38  21  78   4  19   6   0 236  33]
 [  3  20   1  10 160   0   1  37  22 280]]
```

## Figures

- `summary_accuracy.png` — both accuracies vs gamma
- `summary_margins.png` — pair margins vs gamma
- `coherence.png` — class-gain distinctness over training (self-bootstrap check)
- `generation_gamma*.png` — the 10 downward projections W2 @ W1 as images
- `gain_fields_gamma*.png` — final per-class gain fields
- `pair_diff_gamma*.png` — where downward projections disagree for confusable pairs

See README.md for the a-priori predictions and pass/fail criteria.
