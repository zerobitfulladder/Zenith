# The chase on the GPU — the split rig carried to the position task

*No writeup of this run was kept; this page is written from the script's
docstring and its `results/metrics.json`. Shared modules and the viewer hook
(`chase_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md).*

## What it tests

Everything the balance rung proved ([`../balance/`](../balance/README.md),
[`../split/`](../split/README.md)), carried to the target-chasing task and run
as 64 drones at once on the GPU:

- six sensed channels (dx, dy, vx, vy, tilt, gyro), each with its own fovea;
- per-signal tracks (layer one over 240 ms, layer two over 720 ms), as in
  `td_tracks`;
- a sensory-only top layer (8192 templates): present values at half the cue,
  track codes at the other half;
- a cetele (count table) of the teacher's (left, right) thrust pairs per
  template, read as the duty-cycle average;
- a teacher phase, then tight-envelope DAgger (the pupil flies, the teacher
  labels).

## What came out

Over 4 million ticks (43 min) the frozen strict EVAL stayed at **0.0** at every
report. Closest approach fell from 2.39 m to about 1.2 m, and on-path agreement
with the teacher fell from 0.76 to about 0.5. The next step in the record is
[`../cascade/`](../cascade/README.md), which split the learned outer loop from a
fixed attitude reflex and reached 30/30.

## Run

    .venv/bin/python experiments/2026_08_29/chase/gpu_chase.py

Env: `GC_B GC_TICKS GC_PHASE1 GC_STAGE1 GC_STAGE2 GC_K3 GC_REPORT GC_EVERY
GC_TAG GC_CPU`. Writes `results/` (checkpoint viewer-ready through
`chase_view`); console output `results/gpu_chase.log`; smoke test
`results/smoke/`. `gpu_chase` is also imported as a module by
`../cascade/gpu_cascade.py` and the scripts that build on it.
