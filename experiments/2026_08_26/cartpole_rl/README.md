# `cartpole_rl/` — cart-pole without a teacher: expectation-gated learning

Script: [`run_cartpole_rl.py`](run_cartpole_rl.py). Results: [`results/`](results/) — [`learning_curve.png`](results/learning_curve.png), `report.md`.

## Cart-pole WITHOUT imitation (run_cartpole_rl.py) — the morning's
## dopamine mechanism, in pure layers

User spec: no tally sidecars — units store [state ; action ;
EXPECTED outcome] (value population-coded as a third pathway in the
same row); act by comparing the two actions' retrieved expectations;
learn by rotating the owning unit toward experienced truth with THETA
SCALED BY PREDICTION ERROR (surprise moves weights; fulfilled
expectation moves nothing). 600 self-played episodes, no demonstrator.

### Outcome — learns 4x over random, plateaus at ~100 steps

Greedy eval climbs 23 (random floor) -> ~90-167, final mean 97.
P1 CONFIRMED (self-taught learning works through pure
expectation-gated rotation). P2 REFUTED: surprise stays flat (~0.10,
never calibrates) — and that flatness is P4's watch-item CONFIRMED as
the binding constraint: value nonstationarity. As the policy
improves, true returns from every state keep rising, so stored
expectations chase a moving target and the surprise gate never
closes; the museum-jumble disease, RL edition. P3 REFUTED: stress
108/400, below both imitation pupils. Suspected co-factor: bootstrap
allocation — all 256 units born in random-flail states (the
curriculum-order lesson again). Levers queued: bootstrapped TD
targets (stationary-er), staged expectation freezing, allocation
refresh, longer training. Mechanism demonstrated; engineering open.
