# `square_nonext/` — time without a next-slot, and the trail-only ablation

Scripts: [`run_square_nonext.py`](run_square_nonext.py) and [`run_square_trailonly.py`](run_square_trailonly.py). Results: [`results/nonext/`](results/nonext/) ([`generated_nonext.gif`](results/nonext/generated_nonext.gif)) and [`results/trailonly/`](results/trailonly/) ([`generated_trailonly.gif`](results/trailonly/generated_trailonly.gif)).

## No-next-slot mechanism VALIDATED (run_square_nonext.py)

User's question: does the temporal architecture need the explicit
next-half at all, or can time come from the leaky integration itself?
Every validated rig had used a next-slot; the lag-advance conception
(store [present ; trails-of-the-PAST]; query with the present empty;
the stored present, one step ahead of the matched trails, IS the
prediction) had never been fairly tested.

Bouncing-square head-to-head (capacity provably a non-issue):
next-slot = 0.00 error (reference). No-next with the present at the
LARGEST gain = 7.28 mean error — because the empty-at-query channel
carried most of each row's mass, collapsing retrieval margins.
No-next REBALANCED (present gain 0.4, trails dominant): **0.00 —
exact parity.**

Verdict: the two designs are equivalent re-slottings of the same
associations; the user's form is leaner by one pathway (the present
channel doubles as emission cargo). NEW LAW, general: THE KEY MUST
DOMINATE THE ROW — whatever channel is empty at query time is cargo,
and cargo must be a small-gain minority of the stored vector, or
matching runs on scraps. (Also retroactively voids the arm no-next
result: it had present-dominant gains AND trails-include-present
ordering AND the partition wall — three confounds, zero verdicts.)
Two implementation requirements for lag-advance: trails must LAG the
stored present (else self-match freeze), and emission gain small.

### Trail-only ablation (run_square_trailonly.py) — ARROWS, NOT POINTS

User's maximal simplification: state = leaky integration(s) only, no
present channel, no next slot. Mechanism attempted: refractory
retrieval (self excluded) + residual emission (retrieved trail minus
its projection onto mine = the new content; prediction-error made
literal). Ladder on the square: next-slot 0.00; present+lagged-trails
0.00; ONE trail 8.12; TWO trails (fast+slow) 8.24.

LAW — MEMORIES MUST STORE ARROWS, NOT POINTS: every working temporal
form holds, inside each stored vector, content from a strictly LATER
time than its matching channels (next-slot, or present over lagged
trails — the same arrow drawn differently). Pure trails store points:
retrieval finds WHERE (position needs two rates — the two-hands law
stands) but a point has no direction; forward and backward neighbors
of a trail are cosine-symmetric, so similarity+refractory walks a
coin flip. The within-vector time-offset is irreducible — relocatable
(the user's leaner 0.00 form) but not deletable.
