# `drone/` — planar drone, unified and cascaded (Exps 18-19)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_drone.py` — Exp 18, the unified 6-D top (`results/unified/`).
- `run_drone_cascade.py` — Exp 19, the cascade. `DC_ASSOC` picks the association form,
  `DC_TAG` the results subfolder: `results/cascade/` (first run), `cascade_grow/` (growing
  memory, 61/100), `cascade_bound/`, `cascade_bound_t1/`, `cascade_sliced/`.
- `drone_viewer.py` — fly it yourself or watch the pupil; picks among the checkpoints above
  (and `../drone_continual/results/`).
- `results/` — one subfolder per run, logs at the top.

The sliced arm (`cascade_sliced/`, the "third triptych arm" above) has no writeup here; its
log records 76, 74, 83, 61 /100 over rounds 0-3 (the single-layer drone writeup quotes it as
"cascade + sliced read 83/100").

## Exps 18-19 — planar drone (run_drone.py / run_drone_cascade.py)

Exp 18 UNIFIED (three tracks -> one 6-D top): 0/100 all arms. The
three-way control split MEASURED the curse: teacher-through-quantizer
0.95 (actuator fine), MLP 0.78 (encoding fine), 1-NN 0.53 (similarity
retrieval starving in 6-D), bank 0.18-0.23 (quantization on top).
MLP-vs-1NN gap = the part of the task nearest-neighbor cannot do at
this experience density. Also: thrust fovea added (hover sat exactly
between uniform levels 4.0/5.0 -> permanent bang-bang; now the middle
level IS hover). Per-round viewer-loadable checkpoints added.

Exp 19 CASCADE (user-approved serial factorization; uniform
conv-memory arm deferred): cascaded PD teacher rate 1.00 first
audition. THE FACTORIZATION HALVED THE PROBLEM: inner attitude loop
HEALTHY (MLP 0.89, 1-NN 0.77, TF 0.75) — pole-sized world, dense as
designed. Outer position loop = the new bottleneck (MLP 0.67, 1-NN
0.46) with a diagnosed cause: FOVEA-LAG INTERACTION — approach
velocity is read as the difference of two lagged foveated position
codes, and far from target the foveated bumps are ~1 m wide while the
8-tick velocity baseline moves ~0.24 m: velocity is invisible except
near the fovea (third aliasing of the day; the pole escaped it only
because its fovea and its velocity-critical region coincided).
Closed loop 0/100 (a 0.75-per-tick attitude loop also compounds at
50 Hz). TF declines over DAgger rounds again (0.75->0.63; fixed-K
rows thin as data grows). VELOCITY-CHANNEL FIX (same night): approach velocity computed by
finite difference of the offset history and given ITS OWN uniform
bump population (a speedometer, not sharper photos — position keeps
its fovea for WHERE, velocity gets flat resolution for HOW FAST; law:
derived quantities need their own encoding, never a difference of
foveated codes). RESULT: **THE DRONE FLIES** — closed loop 0 -> 14 ->
20 -> 14, first DAgger ignition on the drone, best 20/100 saved
(weights_best, viewer-ready). Outer 1-NN 0.46 -> 0.59. Round-2
regression = the known fixed-K thinning (tf_inner 0.75 -> 0.62 as
DAgger grows data). HEADROOM LEVERS, next session: K grows with
dataset; more rounds; inner-loop closed-loop compounding; the user's
uniform conv-memory comparison arm. The pole's trajectory was
3 -> 15 -> 58 -> 87 across its iterations; the drone now stands at
its "15-20" stage with the same toolbox waiting.

GROWING-MEMORY RUN (user: "why did we cut training?"): 6 rounds, K
grows with the dataset (moments-per-row held ~constant; the measured
fixed-K dilution removed). **25 -> 19 -> 39 -> 48 -> 57 -> 61, still
climbing at budget's end** — DAgger compounds once memory grows with
experience. Best 61/100 (results/cascade_grow/weights_best,
viewer-live). tf_inner falls 0.77 -> 0.60 while flying improves:
mimicry != competence, drone edition. Unsaturated; more rounds =
the obvious next lever.

BOUND-vs-TALLY A/B (user's original vision, DC_ASSOC=bound): rows =
[raw sensor channels ; 0.4*motor code] learned jointly top-1, query
motor-empty, DENSE projection of all rows' motor halves, motor track
argmax commits. Same rig otherwise. RESULT: **0/100 all six rounds**,
inner tf 0.51 -> 0.37 (tally: 0.75, 61/100). Three banked laws
convicted it in concert: action-in-the-metric (rows organize by what
queries can't supply), and the dense projection re-imported mean-mush
at the population level — near boundaries, rows with OPPOSING motor
halves co-fire and their summed projection peaks between them; the
downstream motor argmax commits to the blur (the cart-pole
graded-read collapse, association edition). The bound form's real
virtue (bidirectional queries — action->situations, simulation)
remains untested and is its future home. Tally stands as champion.
Third triptych arm (bound storage + SLICED compositional read) still
open for next session.

BOUND_T1 (the user's de-confounding pushback, honored): sensory
hypercolumns RESTORED + strict top-1 read (winner minicolumn's own
stored command half projects; no pre-choice blending). The confound
was real — round-0 inner agreement recovered 0.51 -> 0.68 — but the
closed loop stayed dead: best **3/100** (2,3,1,0,0,1), agreement
decaying 0.68 -> 0.39 as data grew. FINAL VERDICT, two controlled
runs deep: on this rig the association hypercolumn must compete on
SENSORY CONTENT ALONE; gluing the command into the template loses
61-to-3 even with top-1 reads. Surviving mechanism: boundary
aliasing — where the encoding cannot separate two true states, the
glued command fragments minicolumns by action variant, and
sensory-only queries cannot target the fragments; fragmentation
worsens as DAgger adds boundary-dense data (the decay). The user's
refinement stands in the record: with a deterministic teacher and a
PERFECT encoding the two forms would coincide; the gap measures the
encoding's blind spots. Still open from the user's vision: sliced
compositional read; bidirectional (action->situation) queries.
