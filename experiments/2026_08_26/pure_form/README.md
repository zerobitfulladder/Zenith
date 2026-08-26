# `pure_form/` — the "pure form" rigs: no slots, no arrows (square with fatigue, digit carousel)

Scripts: [`run_square_pure.py`](run_square_pure.py) and [`run_digits_pure.py`](run_digits_pure.py).
Results: [`results/square/`](results/square/) ([`generated_pure.gif`](results/square/generated_pure.gif),
`report.md`) and [`results/digits/`](results/digits/) ([`cycles.png`](results/digits/cycles.png),
`input.gif`, `generated.gif`, `report.md`, `weights.npz`).

Run:
`.venv/bin/python experiments/2026_08_26/pure_form/run_square_pure.py`,
`.venv/bin/python experiments/2026_08_26/pure_form/run_digits_pure.py`

No writeup was kept for these two runs; this README is written from the scripts'
docstrings and their `report.md` files. By file times they were the last runs of the day.

**Square, pure form + fatigue** (`run_square_pure.py`). L1 stores
[leaky-integrated input ; gain x down-projection]; L2 is a slower leaky integration of L1's
activity. No next-slots, no arrows, nothing split: everything queries with everything. In
free-run one unit property is added — fatigue: winners carry a decaying penalty. The idea
under test: a decaying "been here" trace is the arrow of time, because the backward
neighbour was visited two steps ago and is still warm while the forward one is fresh.
Result recorded in the report: `mean 9.10, max 19` — no better than the one-trail (8.12) and
two-trail (8.24) ablations in [`../square_nonext/`](../square_nonext/), against 0.00 for the
forms that store an arrow.

**Digit carousel, pure form** (`run_digits_pure.py`). L1 stores [leaky input ; gain x down];
L2 stores [era activity ; gain x label], label only at the top Layer. The intended arrow is
the lag between Layers: the down channel during a digit's first frames still describes the
previous era. Result recorded in the report: the judge reads `9` on every one of the first 120
frames — the playback sat on one digit.
