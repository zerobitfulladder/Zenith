# Two sparse tracks and an L2 that learns their pairing

2026-08-31. A sensory track and a motor track, each emitting a top-k graded
sparse code, and a layer-two of winner-take-all templates over
`[sensory code ; motor code]`, trained on stored PID flight data. Flying is the
partial-cue read: write the sensory block, leave the motor block blank, find the
best-matching L2 template and read its motor block back out. The codes are
graded rather than one-hot so that L2 does not degenerate into the tally that
already scored 0.000.

No writeup was kept and `results/` is empty — no run output survived. The day
README ([`../README.md`](../README.md), section on the drone) lists it with the
drone experiments.

Run: `.venv/bin/python experiments/2026_08_31/drone_stack/stack.py` (writes
`results/stack.npz`, `stack.json`, `metrics.json`).
