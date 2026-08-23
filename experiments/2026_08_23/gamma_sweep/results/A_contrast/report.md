# Variant A Gain Feedback — gamma sweep report (gain mode: contrast)

Config: TRAIN_N=20000, TEST_N=5000, K1=100, ETA1=0.04, ETA2=0.1, EPOCHS=2, SEED=42. Evaluation is always feedforward with neutral gain (no label at test time).

## Headline table

| gamma | pipeline acc | L1 probe acc | mean top-1 corr | L1 usage entropy | dead templates |
|---|---|---|---|---|---|
| 0.0 | 0.6994 | 0.8708 | 0.784 | 0.949 | 0 |
| 0.25 | 0.6944 | 0.8724 | 0.733 | 0.889 | 0 |
| 0.5 | 0.6704 | 0.8684 | 0.659 | 0.806 | 0 |
| 1.0 | 0.5622 | 0.8646 | 0.583 | 0.804 | 0 |
| 2.0 | 0.5502 | 0.8536 | 0.498 | 0.804 | 0 |

## Confusable-pair margins (higher = better separated)

| gamma | 1v7 | 3v5 | 4v9 | 5v8 | 7v9 |
|---|---|---|---|---|---|
| 0.0 | 0.7168 | 0.1864 | 0.0854 | 0.1532 | 0.0844 |
| 0.25 | 1.0456 | 0.1296 | 0.0533 | 0.1150 | 0.0472 |
| 0.5 | 1.0041 | 0.0688 | 0.0248 | 0.0516 | 0.0382 |
| 1.0 | 0.4098 | 0.0686 | 0.0078 | 0.3507 | 0.2422 |
| 2.0 | 0.7304 | 0.2363 | 0.0065 | 0.0259 | 0.2614 |

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
[[452   1   2   3   8  16  21   0   2   5]
 [  0 522   7   5   0   0   0   1   6   1]
 [ 29  34 235  20  32   0 132  11  22   7]
 [ 13  16  11 327   0  61   4  18  32  22]
 [  1  12   0   0 295   0  10   2   0 156]
 [ 16  17  11  93  14 207  25  18  14  21]
 [ 17   6   9   2  22  16 425   0  13   1]
 [  4  30   2   0  20   0   1 431   5  32]
 [  6  23  14  57   4  42   8   6 243  37]
 [  4  24   0   6 116   3   2  39   5 335]]
```

### gamma = 0.5

```
[[452   0   3   1   5  13  28   2   4   2]
 [  1 508  18  11   0   0   0   2   2   0]
 [ 31  15 329  30  29   0  38  10  36   4]
 [ 19  25  12 293   1  41   2  15  84  12]
 [  0  10   5   0 248   0  19   7   0 187]
 [ 23   8  24  92   5 192  24  25  37   6]
 [ 21   2  40   9  11  19 390   0  19   0]
 [  6  18  11   0   7   0   3 453   8  19]
 [ 17  26  41  45   6  23  11  16 228  27]
 [  3  29   2   8 105   5   3 113   7 259]]
```

### gamma = 1.0

```
[[447   0   8   4   3  12  19  16   0   1]
 [  1 461  63  11   0   3   0   1   1   1]
 [ 22  12 380  55  15   2  15   2  13   6]
 [ 21  21  22 374   2  34   1   6  14   9]
 [  0   7   8   0 206   0   8   1  52 194]
 [ 67   9  43 134   7 109  10  48   1   8]
 [ 10   3 165  24  16   9 282   1   0   1]
 [  7   8  38   3  40   0   0 182  41 206]
 [ 22  17 106 120   4  35   3   5 118  10]
 [ 11  31   5  18 144   8   4  32  29 252]]
```

### gamma = 2.0

```
[[319   0   4   1   1  83  12  26  63   1]
 [  1 443  83   9   0   3   0   2   0   1]
 [ 11   7 342  32  30   2  45   4  41   8]
 [  8  31  16 321   3  45   4  19  45  12]
 [  0  12  18   0 294   1   9   5  12 125]
 [ 72  10  38  92   6 108  12  36  49  13]
 [ 10   4 148  21  16   4 300   1   0   7]
 [ 15   2  32   6  17   4   1 369  49  30]
 [ 38  24  67  69  10  22   8  19 167  16]
 [  6  20  14   8 184  10   3 122  79  88]]
```

## Figures

- `summary_accuracy.png` — both accuracies vs gamma
- `summary_margins.png` — pair margins vs gamma
- `coherence.png` — class-gain distinctness over training (self-bootstrap check)
- `generation_gamma*.png` — the 10 downward projections W2 @ W1 as images
- `gain_fields_gamma*.png` — final per-class gain fields
- `pair_diff_gamma*.png` — where downward projections disagree for confusable pairs

See README.md for the a-priori predictions and pass/fail criteria.
