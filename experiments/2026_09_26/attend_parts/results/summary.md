# Results

Judge 0.9884, forward 0.9855 on real test digits.  mse = pixel error of the composite to the original;
judge = fraction of composites named correctly, as rendered / brightness-normalised.

| window set | windows | window size | mse | judge |
|---|---|---|---|---|
| whole | 1 | 32 | 0.0159 | 0.952 / 0.953 |
| tiles16 | 4 | 16 | 0.0128 | 0.973 / 0.973 |
| overlap16 | 9 | 16 | 0.0106 | 0.983 / 0.983 |
| tiles8 | 16 | 8 | 0.0040 | 0.985 / 0.985 |
