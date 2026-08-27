# Exp 2: run_square_refractory.py — Arm R, forbid the self-match

User's isolation test (their pick over starving): full capacity, rig
untouched, refractory at L1 (winner can't repeat) + oracle
runner-up instrument: teacher-forced, when the self-twin wins,
exclude ALL self rows — does the best survivor sit ahead or behind?
User's mechanism predicts forward-dominant; symmetry predicts 50/50.

Outcome: EXACTLY 50/50 forward/backward (n=100, median offset +0.0)
— the descending future-tint does NOT break the fwd/back symmetry at
this grain; the claim "L3's context through L2's projection advances
the sequence" is refuted at full capacity. Bonus discovery: the best
non-self candidate is not a temporal neighbor at all — mean |offset|
8.78 frames, i.e. the MIRROR PASS (same position, opposite travel
direction): position information in a trail vastly outweighs
direction information, so forbidding self lands you on your mirror
twin, not your successor. Free-run with refractory: ping-pongs
between same-phase twins (x locked at 17), err 7.49 — better than
no-refractory (10.02) but still no motion, nowhere near arrow forms
(0.00). Era-grained context can pick the era, not the phase —
the additive-band lesson again, and consistent with the hypercolumn
day (gate-then-match worked; additive tint has no working band).

Standing verdict after Exps 1+2: at full capacity the completion
arrow has no mechanism to fire — self-match wins, and with self
removed the choice is a coin flip. Remaining route for a PURE
completion arrow: coarse grain (Arm S, fewer units than moments).

---

Files: `run_square_refractory.py` (imports the rig from `../completion/`).
Results in `results/` (`generated.gif`, `report.md`).

Run:

    .venv/bin/python experiments/2026_08_27/refractory/run_square_refractory.py
