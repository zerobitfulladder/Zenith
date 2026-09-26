# Results

Judge on real test digits: 0.9880.  One seed, 8 epochs.

readout = linear classifier trained on the frozen top map afterwards (the only place labels enter, as a measurement).
recon = judge on the render from the top map alone, as rendered / brightness-normalised.  concept = judge on the ten class-mean renders.
code = cosine between the top map and the top map recomputed from its render (floor: shuffled).  alive = fraction of top channels that vary.

| forward trained by | readout | recon | recon mse | concept | code (floor) | alive | activity |
|---|---|---|---|---|---|---|---|
| labels | 0.9888 | 0.950 / 0.950 | 0.0153 | 1.0 / 1.0 | 0.96 (0.65) | 0.92 | 1.111 |
| random | 0.8622 | 0.854 / 0.855 | 0.0272 | 1.0 / 1.0 | 0.97 (0.95) | 0.94 | 0.042 |
| agree | 0.1135 | 0.114 / 0.114 | 0.0756 | 0.1 / 0.1 | 1.00 (1.00) | 0.00 | 0.001 |
| reconstruct | 0.9384 | 0.947 / 0.949 | 0.0200 | 1.0 / 1.0 | 0.98 (0.92) | 0.81 | 0.211 |
| both | 0.6441 | 0.888 / 0.888 | 0.0161 | 0.9 / 0.9 | 1.00 (1.00) | 0.66 | 0.014 |
