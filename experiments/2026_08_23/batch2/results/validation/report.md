# Batch 2 — 8-seed validation

Config identical to run_batch2.py; split/order/init vary per seed. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

| seed | hard | top10 | probe | label-only consistent |
|---|---|---|---|---|
| 0 | 0.7480 | 0.7806 | 0.9474 | 10/10 |
| 1 | 0.7618 | 0.7620 | 0.9496 | 10/10 |
| 2 | 0.7408 | 0.7462 | 0.9414 | 10/10 |
| 3 | 0.7550 | 0.7678 | 0.9474 | 10/10 |
| 4 | 0.7592 | 0.7314 | 0.9392 | 10/10 |
| 5 | 0.7374 | 0.7372 | 0.9466 | 10/10 |
| 6 | 0.7444 | 0.7662 | 0.9480 | 10/10 |
| 7 | 0.7428 | 0.7372 | 0.9480 | 10/10 |

- **acc_hard**: 0.7487 ± 0.0084  (min 0.7374, max 0.7618)
- **acc_topk**: 0.7536 ± 0.0168  (min 0.7314, max 0.7806)
- **probe**: 0.9460 ± 0.0034  (min 0.9392, max 0.9496)
- **label-only retrieval**: 80/80 across all seeds

Figure: style_grid.png (first seed)
