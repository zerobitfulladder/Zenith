# Exp 10: two labeled sequences over the same three digits (run_two_movies.py) — the user's two-scale test

User's spec: 3 layers, 3 digits, a 2-bit one-hot at the top. Bit 0 ->
1,2,3,1,2,3...; bit 1 -> 3,2,1,3,2,1... Generation gets ONLY the label.

Why it is the fair two-scale test: both streams have IDENTICAL long-run
statistics (each digit equally often), so direction cannot live in a
slow average — only in short-range order or in the label. Fresh exemplar
every lap, so pixels never repeat. Streams INTERLEAVED tick by tick,
each carrying its own trails (08-26 allocation lesson). Staged training
(L1, then fresh L2 on frozen L1, then fresh L3 on both) per the Exp 8
fix. L1 = the Exp 9 winner (4x4 K36, window-dominant). L2 = the arrow
over the code. L3 = [0.5*L2 speech ; lagged speech trail ; 0.5*label].
Judged by a probe trained on renders (Exp 9b's law), val 0.97.

Three read modes on ONE trained network, cold start from the label
alone (all trails zero, no priming, no frames shown):

| read | up (label 0) | down (label 1) |
|---|---|---|
| top — L3 names an L2 unit, pure descent | 0.237 | 0.288 |
| l2 — L2's own successor read, L3 ignored | 0.627 | 0.220 |
| **refine — L3 gates a pool of 5, L2 picks within it** | **0.831** | **0.475** |

THE LABEL WORKS, AND GATE-THEN-MATCH IS WHY. panel_refine.png: label 0
emits 1,2,3,1,1,2,3,1,2,3... and label 1 emits 3,2,1,3,2,1,1,3,2,1... in
real handwriting, from nothing but two bits. The control is decisive —
panel_l2.png shows the two labels producing THE IDENTICAL IMAGES,
because that read ignores L3: the trail alone cannot select direction,
it just falls into whichever attractor it finds (both labels emit the
ascending sequence). And pure top-down descent is weak (0.24/0.29):
L3 naming an L2 unit outright is too coarse a decision. The label must
GATE and the layer below must MATCH inside the gate — the hypercolumn
day's law (gate-then-match works, additive tint has no band), now
confirmed as the mechanism by which a top-level label steers a sequence.

MONOPOLY GUARD TESTED AND THE HYPOTHESIS REFUTED. New mechanism
(classic "conscience" from competitive learning, never tried here):
during LEARNING only, bias the winner choice against rows winning more
than their fair share; reads and stored content untouched.

| conscience | L2 top-row share | refine up | refine down | TF |
|---|---|---|---|---|
| 0 | 0.987 | 0.831 | 0.475 | 0.38 |
| 0.5 | 0.181 | 0.780 | 0.407 | 0.37 |
| 2.0 | 0.060 | 0.525 | 0.475 | 0.39 |

It fixes the monopoly outright (0.987 -> 0.060) and generation does NOT
improve — it degrades, and the teacher-forced read does not move at all
(0.38/0.37/0.39). THE MONOPOLY WAS A SYMPTOM, NOT THE BLOCKER. The
"next lever is L2 allocation" call, made twice earlier today (Exp 9 and
9b), is WRONG and is withdrawn. Evenly-spread rows read no better than
monopolized ones; whatever limits this rig is not how the rows are
handed out.

FAILURE MODE, characterized — and it re-points the queue. Classify every
consecutive pair of emissions:

| run | advance | REPEAT | backward |
|---|---|---|---|
| no guard, up | 0.76 | 0.17 | 0.07 |
| no guard, down | 0.59 | 0.38 | 0.03 |
| conscience 0.5, up | 0.69 | 0.28 | 0.03 |
| conscience 0.5, down | 0.66 | 0.31 | 0.03 |

Errors are overwhelmingly REPEATS, not wrong turns (backward 0.03-0.07).
The sequence STALLS — it re-states the present instead of advancing — it
does not turn around or wander. So DIRECTION is solved (that is the
label's job, and the label does it); ADVANCE is what still fails, and it
fails as the self-match freeze that every arrow experiment today (Exps
1-4) diagnosed: the retrieved row's best match is the present, not the
successor.

NEXT LEVER, re-pointed by this: the ANTI-FREEZE machinery already banked
today, not allocation. Exp 3 explicitly filed subtractive
(explaining-away) feedback as "an ANTI-FREEZE, not an arrow" with a
measured effect (square free-run 10.02 -> 5.56) and named "anti-freeze
in playback loops" as its natural fit. This is that fit. Apply the
fired-history subtraction to the L2 candidate pool inside the L3 gate
and re-measure the repeat rate.

OPEN: descending is worse than ascending in every arm and every guard
setting (0.475-0.66 vs 0.69-0.83). Not explained. Candidates: interleave
order (stream 0 leads every tick pair, and 08-26 showed allocation order
surfaces at ties), or MNIST class geometry making 3->2 a harder step
than 1->2. One-line test: alternate which stream leads per epoch.

Artifacts: results/{base,c0.5,c2.0}/ — panel_{top,l2,refine}.png
(both labels side by side), gen_<mode>_label<n>.gif, templates_L1.png,
metrics.json, weights.npz. TM_CONSCIENCE / TM_TAG / TM_TICKS env-set.

---

Files: `run_two_movies.py` (imports the rig from `../completion/`). The
conscience runs were `TM_TAG=_c0.5` / `TM_TAG=_c2.0`; the tag (without the
leading underscore) names the folder under `results/`, and no tag writes to
`results/base/`.

Run:

    .venv/bin/python experiments/2026_08_27/two_movies/run_two_movies.py
