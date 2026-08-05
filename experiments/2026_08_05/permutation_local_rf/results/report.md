# Zenith — Local Receptive Fields (factorize by limiting what nodes see)

Plain training (no negatives), 600 epochs, k=10, eta=0.03. Equal expected wire coverage (~5 nodes/wire) across configs.
Global anchors (plain): one_shot 0.336 | far-field 97.7% | truth pctl med 0.7% | truth_wins 0%.

| F | M | wire% | wire% null | one_shot J | far-field | truth pctl (med) | truth_wins | argmax J | pool_best J |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 1280 | 4.4% | 0.8% | 0.6611 | 29.0% | 29.1% | 0% | 0.4310 | 0.7778 |
| 32 | 320 | 17.0% | 0.8% | 0.5879 | 62.0% | 7.1% | 0% | 0.4157 | 0.7445 |
| 64 | 80 | 60.8% | 0.4% | 0.6411 | 100.0% | 21.6% | 0% | 0.5002 | 0.7616 |
