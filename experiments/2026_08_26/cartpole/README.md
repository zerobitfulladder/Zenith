# `cartpole/` — cart-pole by imitation: first motor control

Script: [`run_cartpole.py`](run_cartpole.py) (imported by `../cartpole_rl/` and `../cartpole_model/`). Results: [`results/`](results/) — [`trajectory.png`](results/trajectory.png), [`balance.gif`](results/balance.gif), [`balance_recovery.gif`](results/balance_recovery.gif), `report.md`.

## Cart-pole by imitation (run_cartpole.py) — first motor control

Sensors -> population code (16 overlapping Gaussian bumps per state
variable, 64 dims) -> association bank cn([sensors ; 0.5*action]),
standard rule, KU=200 -> closed-loop control by retrieval. Expert =
hand-tuned bang-bang PD (baseline 500/500); random floor 23.

### Outcome — balances at expert level; TWO prediction reversals

pure top1: mean 491 / median 500 (expert-level). pure graded: mean 49
(collapse!). shoved top1: 445. shoved graded: 459.

Reversal 1 — the span prediction FAILED benignly: pure-expert demos
sufficed. Cart-pole's reachable tube from gentle inits is narrow, and
overlapping population codes generalize locally, so the pupil's small
drifts stay within the demonstrated span. (Span law not violated —
the span was simply sufficient for this task's operating region.)

Reversal 2 — READ-MODE LAW REFINED, the day's counterweight to
"graded reads generalize": graded DESTROYED the pure-demo policy (49
steps) while top-1 was near-perfect. A bang-bang policy is a sharp
decision boundary; population-vote smoothing blurs exactly that
boundary -> lag near the switch -> oscillation death. Top-1 = nearest
prototype = piecewise-constant, boundary PRESERVED. Content
interpolation (drawing) wants graded reads; discontinuous DECISIONS
want hard reads. Read mode must match the geometry of the target
function. (Shoved+graded 459: the vote's noise-averaging recovers the
15% label noise; note the shove arm recorded the EXECUTED random
action, behavior-cloning a noisy demonstrator — DAgger-style expert
relabeling was NOT used, so top1's dip to 445 is label noise.)

First full sensor->motor loop of the project: perception, association,
action — one local rule end to end.

### Stress-test coda (user noticed the gif "didn't move")

The original gif was undramatic because the pupil is TOO good: 4cm of
cart travel and +-0.8deg over 300 steps, actions chattering 197/299 —
sub-pixel motion. New demo (balance_recovery.gif): 7-deg start + two
angular kicks, shoved-top1 pupil — catches the lean, survives kick 1,
dies fighting kick 2 (273/400). The stress test RESURRECTS the span
prediction: calm eval had hidden it (pure 491 vs shoved 445), under
kicks the order flips decisively (pure dies at 165, right after kick
1; shoved 273). The span law wasn't wrong — calm cart-pole never
leaves the demonstrated tube. Robustness = span made visible by
stress. Graded reads remain bad for control in every condition.
