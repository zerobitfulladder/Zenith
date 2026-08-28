# `single_layer_drone/` — single-layer sparse-OR drone

2026-08-28, late. Moved here from `experiments/experiments_single_layer_drone/`.

## Where things are

- `sl_drone.py` — the rig module (world, oracle, encoder, hypercolumn). Other folders import it:
  `viewer.py` at the repo root, the 08-30 and 08-31 drone work, and the 09-03 vision drone.
- `sl_viz.py` — template pictures used by the run scripts.
- `run_sl_drone.py`, `run_sl_rl.py`, `sl_viewer.py` — see Files below.
- `results/` — the main continual run (`weights_best`, `checkpoint`, metrics.json, samples,
  template pictures, `run.log`).
- `results/consol/` — a run with `SL_TAG=_consol` (its config records K=32768; best 0.89 at
  tick 45000), log `results/run_consol.log`.
- `results/rl_rehearse/` — the `run_sl_rl.py` run with `RL_TAG=_rehearse` (`weights_final`).


The user's architecture, built fresh: **no tracks, no cascade, no
tally, no second layer.** One hypercolumn, one giant sparse vector per
moment carrying both the sensed state and the oracle's motor commands.

## Encoding

Ten channels, all encoded identically — a bump of 5 adjacent cells with
a sharp fall-off `[.3 .7 1 .7 .3]` over NB=128 cells, so near values
share most of their active bits and far values share none:

| channel | source | cells |
|---|---|---|
| dx, dy | target offset | foveated (fine near zero) |
| vx, vy | velocity (integrated) | uniform |
| ax, ay | acceleration (IMU) | uniform |
| tilt | angle (integrated) | foveated ring |
| gyro | angular rate (IMU) | uniform |
| mL, mR | **the oracle's thrust commands** | foveated thrust ladder |

Every channel owns its own random scatter of NB positions inside ONE
array of SIZE=8192, and all channels are **OR'ed** (element-wise max)
into it. 1280 of 8192 positions are claimed, ~50 bits active per moment
(0.5% density), so collisions exist but are rare — and tolerated by
design.

The motor channels go into the SAME array. One sparse vector therefore
says: *in this state, these thrusts were commanded.*

## Layer

ONE hypercolumn, K=4096 minicolumns of dimension 8192. Minicolumns
compete on each moment; only the winner learns (geodesic rotation).
Templates stay centred, so scoring exploits sparsity exactly:
`<template, centred query> == <template, raw query> / ||centred||`,
and the raw query touches only ~50 of 8192 entries.

**Inference:** write the sensory bits, leave the motor bits empty, let
the minicolumns compete, and read the winning minicolumn's own motor
bits back out (gather its values at each motor channel's positions,
take the peak cell, decode to a thrust level). That projection drives
the thrusters. Nothing else.

## Learning

Continual and per-tick: the pupil flies, the oracle is asked at every
tick what it would have done, and the hypercolumn learns that moment
immediately. WARM ticks let the oracle fly first so the adopted
templates come from sane states. No rounds, no rebuilds, no growth —
episodes reset, weights never do.

Teacher: cascaded PD autopilot, recovery rate ~1.00.

## Files

- `sl_drone.py` — world, oracle, encoder, hypercolumn (self-contained)
- `run_sl_drone.py` — the continual run; writes `checkpoint` (now),
  `weights_best` (peak, never overwritten by worse), `weights_final`
- `sl_viewer.py` — human / oracle / pupil, click-to-set-target

    .venv/bin/python experiments/2026_08_28/single_layer_drone/run_sl_drone.py
    .venv/bin/python experiments/2026_08_28/single_layer_drone/sl_viewer.py
    SL_CKPT=best .venv/bin/python .../sl_viewer.py

## Results

Smoke (25k ticks, 10k warm): flies at **0.80** rolling success within
5k ticks of the handover, then declines (0.64, 0.52). All 4096
minicolumns adopted within the first 4096 ticks; code occupancy 0.45%.
Full run in progress — see `results/metrics.json` (`best` records the peak and
its tick).

Comparison points on the same task: cascade + sliced read 83/100,
cascade + tally 61/100, single 6-D hypercolumn with tracks 0/100.

## The continual-drift result (2026-08-28, late)

The single-layer rig peaks well and then always decays to zero. The
mechanism is now measured, and it is not what it looked like.

**It is not forgetting.** Retrieval quality (`bestmatch`) stays flat at
~0.54 the whole way down. The memory keeps matching states fine.

**It is action drift under DAgger data imbalance.** Commanded thrust
walks steadily away from the oracle's:

| phase | collective | \|L-R\| | success |
|---|---|---|---|
| oracle flying | 6.5 (hover) | 0.54 | 1.00 |
| pupil, decayed | 8.0 (full power) | 1.7 | 0.00 |

The pupil spends most of its time in bad states, so most learned
moments are emergency moments where the oracle's correct answer really
is extreme. The memory fills with emergency responses; nearest-neighbour
retrieval then issues them in calm states too; that manufactures more
emergencies. A positive feedback loop with correct labels at every step.

**Ruled out, each with a run:** capacity (K = 4096 / 8192 / 16384 /
32768 — a higher peak, 0.89, same decay), consolidation (win-count
thresholds 300 and 5 — note 300 never engaged, since with tens of
thousands of minicolumns each wins only a handful of times), oracle
rehearsal episodes (30% — slows the drift, does not stop it).

**The structural reading.** Real DAgger keeps every demonstration in a
dataset forever and retrains; its correctness depends on aggregation.
A continual competitive memory has no dataset — it tracks the recent
stream — so it cannot implement DAgger, and drifts with the pupil's own
state distribution. The batched cascade (rebuilt each round on the
accumulated set) never showed this: 61-83/100, stable.

**The principled fix, untested: replay.** Keep a reservoir of past
moments and learn from a mixture of the live tick and randomly sampled
history, so the calm distribution is re-presented forever rather than
merely re-visited occasionally. That is what the batched rebuild was
doing implicitly, and it is what hippocampal replay does literally.

**RL is blocked behind this.** The user's three-factor rule (rotate the
winner toward the moment when closing on the target, away when
receding, with an eligibility trace) is implemented and mechanically
healthy — advantage centred, ~60-70% of ticks rewarded, credit assigned
to the minicolumn that caused the action. But it inherits a policy that
collapses for reasons unrelated to RL, so it cannot be evaluated yet.
Fix the base first.

Bugs found and fixed along the way, worth not repeating: a running
baseline over a potential that charged for speed and tilt made
*loitering* the optimum (rewarded 0.59 -> 0.67 while arrivals fell 0.74
-> 0.06); a living cost in absolute units was ~70x typical progress and
punished every tick; the same constant was subtracted twice; credit
initially went to the minicolumn matching state+action rather than the
one that actually produced the action.
