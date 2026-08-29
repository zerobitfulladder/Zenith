# THE CASCADE CHASER: 30/30 (2026-08-30)

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

`gpu_cascade.py` + `../temporal_drone/cascade_view.py`, `results/`. The user's
calls, in sequence: cascade, motor vocabularies, GPU, small and fast —
then, after the per-report decomposition isolated the stages, the
final one: "low level precise motor control is a dumb processor
either way" — the inner loop need not be a memory.

    OUTER (learned, 10 Hz): 256 sensory templates over the foveated
      position error; per-template cetele over 32 velocity-command
      prototypes (hard-count notching, duty-mean read). Trained on
      the GPU in ~3 minutes. Measured alone: 0.98-1.00.
    INNER (reflex, 50 Hz): the teacher's own nested-P attitude law —
      the brainstem. Not learned, per the user's decision; consistent
      with the value-only-at-the-selector law (nobody's cortex learns
      spinal reflexes from dopamine).

End-to-end strict EVAL through the viewer path: **30/30, closest mean
0.067 m.** First perfect score in the folder.

The learned-inner question stays open as science: five variants all
left the inner memory at ~0.0 (mode/mean reads, hard counts, present
share, on-policy isolation, and a real mixed-reference bug fixed on
the way — the reference law's fourth appearance); a 4x duration run
and, if flat, a faithful transplant of the balance rig's inner design
are the queued experiments. But the TASK is solved, and the cognitive
organ that solved it is the memory.

View:  VIEW_CKPT=experiments/2026_08_29/cascade/results/checkpoint.npz \\
         .venv/bin/python viewer.py    (pick "pupil", click targets)

## Inner co-tuning: REFUTED, with a law worth keeping

Random-search of the reflex gains UNDER the trained outer memory
(120 candidates, batched) claimed a speed win in its own harness
(median 272 -> 250) — but head-to-head through the real viewer path
on 100 common episodes: old reflex 100/100, closest 0.063, median
266; co-tuned 100/100, 0.097, 268. A wash on speed, a loss on
precision. The harness delta was noise.

THE CO-ADAPTATION LAW: the memory was trained on demonstrations whose
inner law WAS this reflex — the commander learned to speak to exactly
this servo. Tuning the servo away from the training-time one cannot
help and mildly hurts; conversely, swapping in a stiffer servo
post-hoc collapsed the same memory 0.98 -> 0.10 (the PID-oracle run).
Commander and reflex are one system: tune them together or not at
all. (Biology agrees: cortex and brainstem co-evolved.)

Final standing config: fast-oracle-trained outer (256 templates,
cetele over 32 velocity prototypes, duty-mean, 10 Hz) + original soft
reflex + ALPHA=0.15 brainstem smoothing: **100/100, closest 0.063 m,
median 266 ticks**, calm flight. Viewer: results/checkpoint
as pupil; tuned full-PID on the dropdown beside it for comparison.

## Where the results are

`results/` holds the champion (`checkpoint.npz`, viewer-ready). Other runs of
`gpu_cascade.py`: `results/long/` (console `results/cascade_long.log`),
`results/pid/` (a run tagged `pid`; its best strict EVAL in `metrics.json` is 0.0), `results/smoke/`.
The reward-refinement and grid-ablation files that used to sit beside the
champion moved to [`../reward_rl/`](../reward_rl/README.md) and
[`../grid_ablation/`](../grid_ablation/README.md).
