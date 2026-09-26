# Results

Judge on real test digits: 0.9876.  Forward net (trained arm) on real test digits: 0.9865.

Each cell is the fraction of renders the judge names correctly, as rendered / after dividing each render by its own maximum.
10,000 test digits for recon and imag; 10 class-mean renders for concept.  mse = pixel error of the reconstruction to the original.
The corner mode (value in one fixed position of the pool) is in metrics.json; it was below 0.25 everywhere.

| arm | recon | recon mse | imag spread | imag spread4 | concept spread | concept spread4 | test mismatch L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|
| trained_tied | 0.853 / 0.873 | 0.0518 | 0.817 / 0.817 | 0.103 / 0.788 | 0.9 / 0.9 | 0.1 / 0.9 | 0.0111 / 0.0655 / 0.2885 |
| trained_untied | 0.984 / 0.984 | 0.0060 | 0.077 / 0.077 | 0.408 / 0.452 | 0.1 / 0.1 | 0.4 / 0.4 | 0.0030 / 0.0038 / 0.0307 |
| trained_mixed | 0.985 / 0.985 | 0.0059 | 0.104 / 0.104 | 0.613 / 0.609 | 0.1 / 0.1 | 0.6 / 0.6 | 0.0045 / 0.0061 / 0.0506 |
| random_tied | 0.609 / 0.727 | 0.0739 | 0.102 / 0.141 | 0.103 / 0.141 | 0.1 / 0.1 | 0.1 / 0.1 | 0.0347 / 0.0165 / 0.0055 |
| random_untied | 0.972 / 0.972 | 0.0098 | 0.114 / 0.114 | 0.279 / 0.341 | 0.1 / 0.1 | 0.2 / 0.3 | 0.0055 / 0.0011 / 0.0002 |
