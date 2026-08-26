# `temporal_compose/` — temporal v5: generating label combinations never seen

Script: [`run_temporal_compose.py`](run_temporal_compose.py). Results: [`results/`](results/) — [`filmstrip.png`](results/filmstrip.png), `gen_<arm>_<combo>.gif` for arms G, L, Gq, H, A, J on the held-out combos 1110 and 1111, [`debug_armA.png`](results/debug_armA.png), `report.md`.

## Temporal v5: compositional generalization (run_temporal_compose.py)

User's test: 4 rotating lines (one per quadrant), 4-bit label = each
line's direction. Train on all <=2-bit combos (11 of 16), hold out
1110 and 1111 entirely; cold-start generate from the label. User's
prediction: the current architecture fails because generation is
top-1. Refinement under test: the blocker is top-1 at GLOBAL scope —
arm G (one whole-frame temporal bank) vs arm L (four LOCAL banks, one
per quadrant: own code slice, own bit, own trails; frame = assembly).

### Outcome — both predictions confirmed, exact signatures

G on 1110: decoded 1100 (3/4) — locked onto the hamming-nearest
TRAINED combo, as predicted. G on 1111: decoded 0011 (2/4) — another
trained neighbor. A global winner can only recite a stored
whole-frame; unseen conjunctions have none. L: **4/4 on both held-out
combos** — every quadrant bank had seen its own bit locally, so the
globally-novel movie is locally familiar everywhere. Same rule, same
top-1, only the scope changed.

LAW (day's capstone): COMPOSITION COMES FROM LOCALITY OF DECISION,
NOT SOFTNESS OF DECISION. Many local top-1s compose (the spatial
render's overlap-add always did this); one global top-1 recites.
16 behaviors from 8 learned local ones. Caveat: locality was
hard-wired (quadrants), as it is in the spatial rigs' windows;
DISCOVERING the factorization is the open problem.

### Arm Gq — the sharpening (user's "top holds the entire seen thing")

Same trained GLOBAL bank as arm G, but per-quadrant LOCAL READS: each
quadrant matches [own trails ; own bit] against the corresponding
SLICE of every stored whole-conjunction memory and emits only the
winner's slice. Result: **4/4 on both held-out combos** — identical
to the local-banks arm, zero retraining.

FINAL FORM OF THE LAW: storing whole conjunctions is fine — every
novel combination already exists piecewise across the stored wholes.
COMPOSITION LIVES IN THE READ, NOT THE STORAGE. One global argmax
recites one memory; many local argmaxes over the same memories quote
chapters from different books. (Brain-flavored: memories as bound
wholes, recall as partial reinstatement.) The open problem narrows
usefully: not "how to store factored memories" but "how to learn the
read partition" — where the slices are.

### Arms H and A — the user's hypercolumn design, refuted then fixed

H (label IN each column's match metric): FAILS (1/4, 1/4), worse than
global. Diagnostic: label-dim variance flat across units — no
relevance discovered; the label fragments each column's units by
whole-label context (conjunctive coding relocated one level down).
F1's law in temporal clothing: parts are label-neutral; identity
cannot be pushed into the metric of parts.

A (label BESIDE the metric): units compete on trail alone; a Hebbian
tally records each unit's win-average label; the label acts at
retrieval. Relevance discovery MEASURED: own-bit variance 0.23 vs
0.14 (clean diagonal, all four columns) — marginal statistics find
which label dim belongs to which patch, no oracle. Read design took
three iterations, each failure diagnostic: additive bias @0.3 =
direction oscillates; @1.0 = constant bias tramples phase choice
(third damping-band lesson of the day); GATE-then-match (bias selects
candidate pool, trail picks phase) = direction solved, self-loop
stalls remain; + REFRACTORY rule (winner can't repeat; in this rig
phase always advances) = **4/4 on both held-out combos**, matching
the oracle arms.

Final recipe, all local, all biologically flavored: content-only
competition + Hebbian outcome association + top-down gating +
refractoriness. Still hard-wired: the spatial column partition
itself (label relevance is learned; slice boundaries are not).

### Arm J — the user's uniform-architecture version (association IS a
### layer; influence via graded reprojection)

Top layer jointly learns [label ; columns' activity] with the
standard rule; generation reads it as a POPULATION (all units,
weighted sharply by label match) and reprojects each unit's
column-slice as that column's gate. Three engineering lessons on the
way (each measured): instantaneous activity snapshots make the top's
advice phase-specific — the top must watch through a SLOWER trace
(timescale hierarchy earning its keep: slower clock = phase-invariant
content); median-thresholded gates no-op when half the bank sits on
the centering floor (gate by top-third instead); 5-memory votes are
too thin (whole-population votes).

RESULT: J composes 1110 at 4/4 — the population vote over memories
0110+1100+1010 reconstructs a combination none contains — but ties at
1/4 on 1111, and the failure is STRUCTURAL, not parametric: 1111 is
equidistant from all six 2-bit trained combos, and each column is CW
in exactly 3 of those 6 — the label-neighborhood vote is 50/50 by
symmetry on every column. INSIGHT: global-similarity population reads
compose only where the label neighborhood is asymmetric; per-column
MARGINAL association (arm A's tally — each column's own-bit
statistics, unconditioned on the rest) decides even at symmetric
corners. The marginal is not an implementation shortcut; at symmetric
corners it is informationally necessary. (Caveat: real curricula are
rarely perfectly symmetric; J-style reads would usually work.)

Day's final standings: no-oracle composition solved by arm A (4/4
both); uniform-layer arm J 4/4 asymmetric / structurally tied at the
symmetric corner. Open: a layer-native read that computes per-column
marginals (learning arm A's tally as synapses of the top layer).
