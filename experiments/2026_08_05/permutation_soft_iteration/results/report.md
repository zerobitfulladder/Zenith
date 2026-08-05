# Zenith — Iterative sampling (population refinement, no verifier)

Near-negative model. N=100 particles/round, 6 rounds.
Anchors: one_shot 0.3995 | hard iteration 0.196 | pool_mean 0.3972.

## Decode Jaccard (top-k of round's mean distribution)
| config | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| T0.5 | 0.2725 | 0.2248 | 0.2027 | 0.1951 | 0.1908 | 0.1840 |
| T1.0 | 0.2830 | 0.2443 | 0.2274 | 0.2211 | 0.2133 | 0.2144 |
| anneal | 0.2830 | 0.2348 | 0.2137 | 0.1988 | 0.1853 | 0.1836 |

## Oracle best sample in pool (coverage of truth)
| config | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| T0.5 | 0.4999 | 0.3773 | 0.3372 | 0.3096 | 0.3010 | 0.2975 |
| T1.0 | 0.3985 | 0.3505 | 0.3280 | 0.3181 | 0.3074 | 0.3133 |
| anneal | 0.3985 | 0.3733 | 0.3431 | 0.3215 | 0.2906 | 0.2664 |

## Collapse (1 = pool unanimous, 0 = max disagreement)
| config | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| T0.5 | 0.8763 | 0.8905 | 0.8969 | 0.9020 | 0.9050 | 0.9065 |
| T1.0 | 0.8194 | 0.8264 | 0.8304 | 0.8335 | 0.8352 | 0.8361 |
| anneal | 0.8194 | 0.8474 | 0.8778 | 0.9079 | 0.9345 | 0.9553 |
