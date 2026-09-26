# Results

300 test faces as goals; library of 1000 faces x 49 positions; 8 candidates per window.

agreement = cosine between the goal and the read-back of the finished canvas.  fine = mean cosine between each chosen part and the read-back of its window.

| arm | agreement | pixel error | sharpness | fine agreement |
|---|---|---|---|---|
| blur | 0.585 | 0.0194 | 0.0126 | 0.000 |
| first | 0.592 | 0.0265 | 0.0165 | 0.866 |
| random | 0.561 | 0.0281 | 0.0161 | 0.857 |
| coarse1 | 0.746 | 0.0225 | 0.0177 | 0.853 |
| coarse2 | 0.768 | 0.0232 | 0.0183 | 0.837 |
| both2 | 0.562 | 0.0259 | 0.0126 | 0.923 |
| from_real | 0.800 | 0.0061 | 0.0191 | 0.000 |

Imagined goals (8 pca64 samples):

| arm | agreement | sharpness |
|---|---|---|
| coarse | 0.502 | 0.0091 |
| first | 0.576 | 0.0149 |
| coarse2 | 0.763 | 0.0181 |
| both2 | 0.534 | 0.0090 |
