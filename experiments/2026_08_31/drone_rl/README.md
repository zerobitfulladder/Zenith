# Drone by reward alone — FAILED. The credit assignment is wrong.

2026-08-31. One layer of competitive experts, each a subspace over
`[situation ; motor]`, so completing the blank motor half from a written
situation is the policy. Situation is dx, dy, vx, vy, tilt, gyro; motors are
left/right thrust; both place-coded, 12 cells per channel, 48 experts x 8
templates. Value only at the selector: fit picks who is competent here, a
scalar V per expert picks among those.

## It does not fly

| | reward | success |
|---|---|---|
| teacher (PD autopilot) | −49.8 | **0.942** |
| random thrusts | −41.3 | 0.000 |
| **this, after 8000 episodes** | −20.0 | **0.000** |

(The reward column is misleading — this arm "wins" it by dying fast, which
stops the per-step penalty accruing. Success is the honest column.)

## Run one: two outright bugs

* **The value function diverged to −4e64.** Accumulating eligibility traces
  (`e[h] += 1`) with 48 experts over 300 steps pile up on the same expert and
  the update runs away. Fixed with replacing traces (`e[h] = 1`), alpha
  0.05 → 0.02, and a clip.
* **The starting policy tipped the drone in 0.30 s.** Random templates give
  arbitrary motor completions — measured mean differential thrust 2.96
  against a hover of 4.90, about 30 rad/s^2 of torque. Every episode was a
  tumble, so all experience was crash. Fixed by seeding each expert in its
  own pocket of state space with near-hover thrust.

## Run two: the real problem

The seeding worked — episodes started at **49.7 steps** instead of 20. Then
training made it **worse**, monotonically:

    ep  500   49.7 steps
    ep 1500   28.5
    ep 4500   14.7
    ep 8000   15.6

**Learning degraded the policy below its own initialisation.** That is not a
weak learner, it is a harmful one, and it isolates the fault precisely.

The rule was: consolidate `[situation ; the action just taken]` whenever
`delta > 0`. But at the single-step level `delta > 0` mostly means *the value
estimate was pessimistic*, not *the action was good* — it fires on roughly
half of all steps regardless of merit. So each expert absorbs whatever noisy
thrust it happened to apply, random-walking its motor completion away from
the hover seed and into tumbling.

**The architecture is not implicated.** The gate works, all 48 experts get
used, the encoder and the viewer path are fine. What failed is the one piece
invented here with no precedent behind it: credit assignment.

## Run three and four: episode return, and freezing the territory

`drone_cem.py` replaced the credit rule: roll out a batch, compute
return-to-go, and let each expert consolidate only its top 20% of experience
ranked by **advantage** (`G - V[h]`, so the ranking is not just the expert's
easy states). V became Monte-Carlo, removing the bootstrapping that diverged.

`drone_cem2.py` added two more fixes: the correct multi-sample tangent
instead of rotating toward the *mean* of the kept experiences (averaging
place codes of different situations gives a target that is not a valid code
for anything), and **freezing the situation half** of every template so
consolidating a policy cannot drag the expert's fit region out from under it.

| learning rule | at the seed | after training |
|---|---|---|
| TD(lambda), consolidate on delta>0 | 50 steps | **15.6** |
| episode return, top-20% by advantage | 54.6 | **27.8** |
| + correct tangent, frozen territory | **71.2** | **33.5** |

Success **0.000** in all three. Freezing helped — best start and best end,
which supports the coupling diagnosis — but every rule still ends below its
own initialisation. **Hovering beats everything learning produces.**

## Why, and it is not the consolidation rule

Three different credit schemes failing identically puts the fault upstream of
all of them. The confound:

**Return-to-go here mostly measures when the drone died.** The −8 terminal
penalty dominates, so an action 5 steps before a crash scores about
`−8·gamma^5` and one 50 steps before scores `−8·gamma^50`. Early actions
systematically look better — and early actions are taken from the *initial*
state, which is random. So the top-20% slice largely selects "what I did at
the start of episodes that happened to begin somewhere easy", not "what I did
well". Subtracting `V[h]` does not fix it, because an expert's region still
spans a wide range of starting conditions.

That needs real variance reduction — a state-dependent baseline, or
normalising advantage within comparable starts — not another tweak to how
experiences are consolidated.

There is also a task-shape problem worth naming. The drone is **unstable**: a
policy that is 90% right still crashes, so the reward landscape is sharp and
narrow. Local random search over 48 regions with roughly 125 episodes each
has very little to go on, whatever the credit rule.

## What is and is not established

**Not implicated: the architecture.** The same experts, gate and encoder
classify MNIST at 0.9392 (`../competitive40`) and resist catastrophic
forgetting (`../continual_fixed`). Here the gate works, all 48 experts get
used, and the partial-cue motor read functions — the policy it reads is
simply bad.

**Implicated: this family of RL.** Local random search that consolidates
whichever perturbations happened to work is not adequate for unstable
continuous control. Parked here rather than tuned further.

The next thing to try, if it is picked up again: give each expert a **local
linear policy** fitted by least squares on its own experience, weighted by
advantage — a real regression rather than a subspace rotation hoping to drift
toward competence. That keeps the experts and the gate exactly as they are
and changes only what lives inside an expert.

## Watching it

    VIEW_CKPT=experiments/2026_08_31/drone_rl/results/bank.npz \
        .venv/bin/python viewer.py

`viewer.py` is not forked — `drone_rl.load_policy` plus the `bank.json`
sidecar is all it takes, per the convention in the viewer's own docstring.
Select "pupil". It tumbles.

## Files

| | |
|---|---|
| `drone_rl.py` | encoder, expert bank, TD(lambda) loop, `load_policy` |
| `drone_cem.py` | episode-return credit, top-q by advantage |
| `drone_cem2.py` | + correct tangent, frozen situation half |
| `results/bank.npz`, `bank.json` | the checkpoint and its viewer config |
| `results/01_learning.png` | the curves, such as they are |

    python drone_rl.py     # ~80 s
