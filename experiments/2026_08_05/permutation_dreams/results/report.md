# Zenith — Dream-Phase Training (self-generated negatives)

Rounds=6, epochs/round=100, neg_scale=0.5, dream sources/round=100.

Train-time hack rate per round: 100% -> 100% -> 100% -> 100% -> 100% -> 100%

| ranking acc | one_shot J | iterative J | energy J | energy P | hacked% | search-true gap |
|---|---|---|---|---|---|---|
| 98.70% | 0.3639 | 0.1963 | 0.1809 | 0.3008 | 100% | +0.9345 |

Compare: one-phase control 0.336/0.196/0.182, hacked 100%, gap +0.98;
mismatch-negatives 0.388/0.224/0.209, hacked 100%, gap +0.80.
