# Exp 6 (added after Exp 5, same evening): stacked + wired variants

run_square_lagcargo_stack.py: the delayed-write recipe at all three
same-rate layers (decays .5/.9/.975, top-1 speech up, no feedback
wiring). Free-run (L1-driven) holds 0.00. Galleries
(templates_L{1,2,3}.png): every layer is the same [keys;cargo] arrow
with progressively longer comet-keys (3 -> 10 -> 30 frames) and
coarsening ownership (one phase -> one era; L3 units own 7-17
phases). Used units 48/30/10. Capacity reconciliation confirmed
visually: surplus units harmless once the arrow is written (40-unit
perfect phase tiling at L1 in the flat rig, one owner per moment).

run_square_lagcargo_wired.py: FULL-STACK GENERATION — rows =
[GC*cargo ; lagged own trail ; GD*lagged down-projection], learned
jointly; playback runs the whole loop every tick (L1 successor-read
-> emit cargo -> echo winner up -> L2/L3 perceive -> contexts down).
Ordering matters: playback absorbs-then-reads (keys were written
pre-arrival); upward echo uses full-row perception matches mirroring
training. GATE PASSED: 0.00 max 0 — wiring the feedback does not
hurt the unambiguous case. This rig is the complete standard unit;
next session: three-movies task, where the context is load-bearing
(shared frames) and the no-era chassis should cure the rotation
stickiness (ref: 0.39 vs 1.00 scans).

---

Files: `run_square_lagcargo_stack.py` -> `results/stack/` (`generated.gif`,
`homes.json`, `report.md`, `weights.npz`, `templates_L{1,2,3}.png`),
`show_stack_templates.py` (draws those galleries), `run_square_lagcargo_wired.py`
-> `results/wired/` (`generated.gif`, `report.md`). All import the rig from
`../completion/`. The stacked weights in `results/stack/weights.npz` are reused
by `../bounce_updown/`.

Run:

    .venv/bin/python experiments/2026_08_27/lagcargo_stack/run_square_lagcargo_stack.py
    .venv/bin/python experiments/2026_08_27/lagcargo_stack/show_stack_templates.py
    .venv/bin/python experiments/2026_08_27/lagcargo_stack/run_square_lagcargo_wired.py
