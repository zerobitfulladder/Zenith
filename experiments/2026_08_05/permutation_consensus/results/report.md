# Zenith — Consensus Decoding (marginalize over ranked candidates)

Near-negative model, pools: T=0.5, N=1000.

Truth energy percentile in pool: mean 3.1% (median 0.7%) — rank-1 rate was 0%.

| decode | Jaccard |
|---|---|
| one_shot | 0.3995 |
| argmax | 0.2810 |
| vote@10 | 0.3191 |
| vote@25 | 0.3366 |
| vote@50 | 0.3404 |
| vote@100 | 0.3518 |
| soft | 0.3088 |
| pool_mean | 0.3972 |
