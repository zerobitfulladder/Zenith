# `cartpole_model/` — cart-pole with a babbled world model and the goal as a query

Script: [`run_cartpole_model.py`](run_cartpole_model.py). Results: [`results/`](results/) — [`calibration.png`](results/calibration.png), `report.md`.

## Two-phase: babbled world model + goal-as-query (run_cartpole_model.py)

User design: phase 1 = random babbling, units learn [state ; action ;
NEXT state] (stationary dynamics, no goal); phase 2 = the goal given
as a STATE (upright/centered/still, sensor-encoded), control = per
action, retrieve the predicted future and act toward the one most
resembling the wish. Goal lives at inference; only surprise-gated
dynamics writes anywhere.

### Outcome — calibration CONFIRMED, greedy control REFUTED

P1 lands beautifully: prediction error 0.148 -> 0.028 during
babbling — the surprise gate CLOSES on the stationary target, the
exact curve the RL run could never produce. The nonstationarity
diagnosis is thereby confirmed twice over: same gate, moving target
(values) = flat surprise; still target (physics) = calibration.

P2/P3 refuted: control mean 97 (= the RL plateau), stress 20/400
(worst of all pupils). Named cause, P4's watch-item at full strength:
GREEDY MYOPIA. Cart-pole recovery is fundamentally non-greedy — to
save a leaning pole you must drive the cart UNDER it, which makes the
next moment look LESS like the goal (more velocity, still leaning)
before it looks more. One-step wish-similarity picks exactly the
wrong action at a 7-deg lean (dies at 20). A calibrated model is not
a policy; between model and wish there must be LOOKAHEAD (multi-step
rollout through the model / MPC) or a learned value bridging the
distance (model + expectation — Dyna-flavored: the model generating
imagined experience for the value learner). Both queued.
