# Exp 7 (night): run_square_bounce_updown.py — user's up-down bounce vision

Layers learn ONLY [cargo ; own lagged trail] (banks reused from the
stacked rig). Inference: climb (perceive, trails advance), ONE
successor-read at the chosen TOP, descend by pure reconstruction
(each row's cargo names the next unit below; L1's cargo = the next
frame), emit, re-see own emission, bounce again. No down segments,
no gates, no joint learning; disambiguation is positional (decide
where the trail is slowest).

Results: TOP=L1 0.00 (= flat baseline). TOP=L2 6.69 — but the trace
is the story: after a 4-tick stumble it locks on and tracks
PHASE-EXACTLY for a long stretch (15..1 perfect), then derails
somewhere mid-run (max 20). The decision-above/reconstruct-below
chain genuinely carries phase-exact generation through L2 — the
vision works in kind at one level up. TOP=L3 6.84: locks onto a
frozen frame within 3 ticks — same-rate gamma=.975 trails change
too little per tick for a one-tick successor read (L3 units own
7-17 phases; era-grain choice, phase-grain question). Open levers,
untested: instrument WHERE TOP=2 breaks (suspect the wall turns);
refinement descent (each layer's own read picks within the
candidate set named from above — gate-then-match, but the gate is
the cargo chain); coarser upward strides for the top's read.
Next-session queue, joining the three-movies test.

## Dense upward speech variant (`run_square_bounce_dense.py`)

No writeup was kept for this run; this note is from the script's docstring and
its `results/dense/report.md`. Same learning rule as the stacked rig, but the
message sent upward is the whole graded match profile instead of the winner's
one-hot. The descent is tried two ways: committed (argmax, one unit per layer)
and graded (weighted reconstruction, committed only at the pixel step).

| top | argmax | graded |
|---|---|---|
| L1 | 0.00 (max 0) | 0.00 (max 0) |
| L2 | 8.62 (max 20) | 3.72 (max 8) |
| L3 | 4.65 (max 15) | 3.41 (max 11) |

(mean |x err|; one-hot references TOP1 0.00 / TOP2 6.69 / TOP3 6.84)

---

Files: `run_square_bounce_updown.py` -> `results/onehot/` (loads its layers
from `../lagcargo_stack/results/stack/weights.npz`), `run_square_bounce_dense.py`
-> `results/dense/` (trains its own stack). Both import the rig from
`../completion/`. Each results folder holds `generated_top{1,2,3}*.gif` and
`report.md`.

Run:

    .venv/bin/python experiments/2026_08_27/bounce_updown/run_square_bounce_updown.py
    .venv/bin/python experiments/2026_08_27/bounce_updown/run_square_bounce_dense.py
