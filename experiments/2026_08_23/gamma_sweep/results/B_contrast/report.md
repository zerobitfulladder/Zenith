# Variant B Gain Feedback — gamma sweep report (gain mode: contrast)

Config: TRAIN_N=20000, TEST_N=5000, K1=100, ETA1=0.04, ETA2=0.1, EPOCHS=2, SEED=42. Evaluation is always feedforward with neutral gain (no label at test time).

## Headline table

| gamma | pipeline acc | L1 probe acc | mean top-1 corr | L1 usage entropy | dead templates |
|---|---|---|---|---|---|
| 0.0 | 0.6994 | 0.8708 | 0.784 | 0.949 | 0 |
| 0.25 | 0.6998 | 0.8716 | 0.783 | 0.958 | 0 |
| 0.5 | 0.7012 | 0.8726 | 0.781 | 0.964 | 0 |
| 1.0 | 0.6994 | 0.8726 | 0.777 | 0.973 | 0 |
| 2.0 | 0.7008 | 0.8774 | 0.769 | 0.982 | 0 |

## Confusable-pair margins (higher = better separated)

| gamma | 1v7 | 3v5 | 4v9 | 5v8 | 7v9 |
|---|---|---|---|---|---|
| 0.0 | 0.7168 | 0.1864 | 0.0854 | 0.1532 | 0.0844 |
| 0.25 | 0.7463 | 0.1954 | 0.0890 | 0.1635 | 0.0898 |
| 0.5 | 0.7493 | 0.1856 | 0.0884 | 0.1536 | 0.0949 |
| 1.0 | 0.7548 | 0.1989 | 0.0938 | 0.1704 | 0.0955 |
| 2.0 | 0.7596 | 0.1967 | 0.0875 | 0.1629 | 0.0895 |

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
[[459   1   4   9   1  22   8   0   4   2]
 [  0 525   7   3   0   2   1   0   3   1]
 [ 14  53 346  32  20   1  34   7  14   1]
 [  4  13  27 376   5  14   4   6  32  23]
 [  2  11   1   0 325   0   6   3   7 121]
 [ 16  44   5 132  14 185   9   8   9  14]
 [ 12  24   6   6  22  12 425   0   4   0]
 [  6  31   5   0  75   0   0 363  17  28]
 [  7  45  23  79   7  24   4   0 223  28]
 [  3  23   1  10 170   1   1  32  21 272]]
```

### gamma = 0.5

```
[[457   1   4  10   3  23   7   0   4   1]
 [  0 526   7   3   0   2   1   0   2   1]
 [ 13  53 348  33  19   1  32   6  16   1]
 [  4  14  27 380   6  12   4   6  30  21]
 [  1   9   1   0 322   0   7   3   8 125]
 [ 15  47   7 131  15 179  10   9   9  14]
 [ 14  24   6   5  19  14 425   0   4   0]
 [  6  28   6   0  70   0   0 371  18  26]
 [  6  41  27  83   7  21   4   0 224  27]
 [  3  23   1  10 166   1   1  34  21 274]]
```

### gamma = 1.0

```
[[460   1   4  10   3  21   7   0   3   1]
 [  0 526   7   3   0   2   0   0   3   1]
 [ 13  60 338  32  18   1  32   9  18   1]
 [  2  16  26 379   5  11   4   7  32  22]
 [  3  13   3   0 320   1   3   3   8 122]
 [ 16  47   3 126  15 190  10   9   8  12]
 [ 15  24   7   4  20  12 425   0   4   0]
 [  6  29   6   0  69   0   0 373  16  26]
 [  6  47  21  79   7  21   5   0 228  26]
 [  3  25   0  10 171   1   1  45  20 258]]
```

### gamma = 2.0

```
[[456   1   4   9   3  24   9   0   4   0]
 [  0 523  11   3   0   2   0   0   2   1]
 [ 12  47 348  34  17   0  35   7  20   2]
 [  5  11  27 376   6  10   4   8  38  19]
 [  2   8   2   0 326   0   5   4  10 119]
 [ 17  47   7 125  15 179  15   9   8  14]
 [ 14  22   3   3  17  10 439   0   3   0]
 [  6  23   6   0  67   0   0 380  21  22]
 [  6  38  27  83   7  22   5   0 225  27]
 [  3  21   0  10 176   0   1  50  21 252]]
```

## Figures

- `summary_accuracy.png` — both accuracies vs gamma
- `summary_margins.png` — pair margins vs gamma
- `coherence.png` — class-gain distinctness over training (self-bootstrap check)
- `generation_gamma*.png` — the 10 downward projections W2 @ W1 as images
- `gain_fields_gamma*.png` — final per-class gain fields
- `pair_diff_gamma*.png` — where downward projections disagree for confusable pairs

See README.md for the a-priori predictions and pass/fail criteria.
