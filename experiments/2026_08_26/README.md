# 2026-08-26 — precision, time, and first motor control

Continuation of the log in `../2026_08_24/README.md`.

The day ran in three stretches: the static MNIST rig (base, bigger L2 window, bits per
weight, bounce generation), then time (five temporal rigs from a bouncing square up to
held-out label combinations), then control (cart-pole, pole from angle only, reaching arm),
and it ended back on time (no next-slot, hierarchical time). The day log was split into the
experiment folders below; every paragraph of it is in one of their READMEs.

---

## `base/` — the 3-layer base rig, widened to 1024 templates

The best 3-layer network rebuilt standalone, with 1024 templates in L1 and L2 instead of
512. Hard readout **.9112** (ties the project record at 2 epochs, +1.4 points over the
512-wide version), linear probe on L2 .9570, label-only generation recognizable for all ten
digits. The galleries show L2 is barely more than L1: its pixel footprint is only 10x10, so
the whole jump in scale happens at L3. These weights are reused by `precision/` and `bounce/`.

Full writeup: [`base/README.md`](base/README.md).

---

## `bigwin/` — a bigger L2 window (8x8 over L1) and the generation-mush diagnostic

L2's footprint grown from 10x10 to 15x15 pixels (with more templates at L2 and L3). L2
templates became larger strokes — hooks, arcs, S-turns — but the probe stayed flat
(.9570 -> .9572), hard held (.9132), and generation got visibly worse. Re-rendering from
fewer positions did not sharpen it; the cause is that each big template is 6x flatter
(winner's share 0.215 -> 0.035), and generation paints that flatness directly.

Full writeup: [`bigwin/README.md`](bigwin/README.md).

---

## `precision/` — how many bits a weight needs

Lavender's hypothesis (synapses hold ~4-5 bits): f32 is overkill. After fixing a
quantization artifact in round 1, rounding the trained base weights costs nothing down to
4 bits (hard .9110-.9114 vs .9112) and slopes gently below (.9012 at 3 bits, .8906 at 2).
Training with 4-bit expressed weights scored **.9258**, above the f32 base, but its drawings
at 3 and 2 bits are messier than the rounded ones.

Full writeup: [`precision/README.md`](precision/README.md).

---

## `bounce/` — generation by bouncing between image and code

Generation as settling: label -> top -> pixels -> re-encode -> repeat, keeping fewer
templates per position each round. Without the label clamped, the loop drifts into stripe
and blob patterns; with it clamped, the final images are cleaner than one-shot generation
for several digits. Side finding: the network recognizes only 1/10 of its own one-shot
drawings (vs 45/50 real test images) — its renders are off-distribution to its own encoder.

Full writeup: [`bounce/README.md`](bounce/README.md).

---

## `temporal_square/` — temporal v1: a bouncing square replayed from two trails

Stored vector = [fast trail ; slow trail ; next frame], no labels. Free-run replay was
perfect on the first run (0.00 mean error over 100 frames, every wall reversal), because
each emission is a stored memory that snaps the state back onto the learned loop.

Full writeup: [`temporal_square/README.md`](temporal_square/README.md).

---

## `temporal_line/` — temporal v2: a rotating line, spatial L1 + temporal L2

A frozen spatial L1 under the same two-trail temporal L2. Free-run angle error 3.3 degrees
(under one frame step), and L1 grew orientation-tuned templates from nothing but the
rotating line.

Full writeup: [`temporal_line/README.md`](temporal_line/README.md).

---

## `temporal_movies/` — temporal v3: three labeled movies in one network

Three movies sharing frames, played back from the label alone with no priming. After
switching to interleaved training (sequential episodes let the first movie take all the
templates): purity 1.00 on all three, advance 1.00 on the two scans, 0.72 on the rotation.
Given two labels at once, it plays one movie 100/0 — no blends.

Full writeup: [`temporal_movies/README.md`](temporal_movies/README.md).

---

## `temporal_digits/` — temporal v4: the MNIST digit carousel

MNIST 0..9 looping with fresh exemplars every cycle. With a slower fast-trail it loops
through all ten digits indefinitely (irregular rhythm); with a faster one it froze on 4 —
held input stops an input-driven clock. The emissions are neither copies nor class means
but crisp canonical drawings, the same drawing every loop; copies where a class is tight
(0, 1), new drawings where it is varied.

Full writeup: [`temporal_digits/README.md`](temporal_digits/README.md).

---

## `temporal_compose/` — temporal v5: generating label combinations never seen

Four rotating lines with a 4-bit label, two combinations held out. One global Hypercolumn
recites the nearest trained combination; four local ones get **4/4** on both. The same
global memories read locally per quadrant (arm Gq) also get 4/4 — composition lives in the
read, not the storage. Arms H, A, J test Lavender's hypercolumn and uniform-layer versions:
A reaches 4/4 on both; J 4/4 on 1110 but structurally tied at 1/4 on 1111.

Full writeup: [`temporal_compose/README.md`](temporal_compose/README.md).

---

## `bridge/` — encoder/decoder bridge: see a digit, redraw it with another stack

Two independent stacks joined by a Layer that learns [encoder ; decoder]. On unseen digits
it keeps identity 7/10, redrawn in the decoder's own handwriting. Trained only on random
images: noise gives nothing; straight scribbles transfer a digit's dominant strokes
(graded read corr .351) but cannot draw closed loops.

Full writeup: [`bridge/README.md`](bridge/README.md).

---

## `cartpole/` — cart-pole by imitation: first motor control

Sensors as bumps, action stored beside them, control by retrieval. Top-1 read balances at
expert level (mean 491, median 500); the graded read collapses (49) because it blurs a
bang-bang decision boundary. Under kicks, the pupil trained on noisy demos survives longer.

Full writeup: [`cartpole/README.md`](cartpole/README.md).

---

## `cartpole_rl/` — cart-pole without a teacher

Templates store [state ; action ; expected outcome] and move only as much as they are
surprised. It learns 4x over random (final mean 97) but plateaus, because the values it
chases keep moving as the policy improves.

Full writeup: [`cartpole_rl/README.md`](cartpole_rl/README.md).

---

## `cartpole_model/` — babbled world model, goal as a query

Learn [state ; action ; next state] from random babbling, then act toward the predicted
future most like the goal. The model calibrates (prediction error 0.148 -> 0.028), but
one-step greedy control fails (mean 97, stress 20/400): saving a pole first makes things
look worse, so it needs lookahead or a learned value.

Full writeup: [`cartpole_model/README.md`](cartpole_model/README.md).

---

## `pole_angle/` — recovering the pole from its angle alone

Only the pole angle is sensed; the trails must carry velocity. Two fixes were needed: a
bipolar force code ("nothing" must never be a valid symbol), then DAgger for the pupil's
own drift. Recovery rose 4 -> 43 -> 46 -> 68 /100 over four rounds. Live viewer included.

Full writeup: [`pole_angle/README.md`](pole_angle/README.md).

---

## `pole_swingup/` — swing-up from any state

Every state counts as recoverable. Two factors were both needed: a lag-free current-angle
channel beside the trails, and DAgger. The strict metric said 47/100; watching it, Lavender
disputed that, and without the arena walls (which the pupil cannot sense) it succeeds
**100/100**. Also here: a single leaky-sum trace instead of separate present and past
channels drops to 13-19/100 vs 58 — present and past belong in separate channels.

Full writeup: [`pole_swingup/README.md`](pole_swingup/README.md).

---

## `pole_tracks/` — the two-track pole architecture (open, parked)

Lavender's full design (perception track, motor track, unification over both) scores 0/100
in every round vs 58/100 for the flat version on the same demos. The motor track
round-trips perfectly; something in the perception recoding is lost. A bisect plan is noted.

Full writeup: [`pole_tracks/README.md`](pole_tracks/README.md).

---

## `arm/` — two-joint reaching arm, parked at the dimensionality wall

The emission must be a command, not a state. Three emission forms fail in the same band;
4x the templates buys only 2-3x the success (3/8/6 -> 10/11/17). Flat memory pays
exponentially per state dimension; the escape noted is to factor the policy.

Full writeup: [`arm/README.md`](arm/README.md).

---

## `square_nonext/` — time without a next-slot, and the trail-only ablation

Storing [present ; lagged trails] and querying with the present empty matches the
next-slot exactly (0.00) once the present gets a small gain — the key must dominate the
row. Trails alone fail (8.12 / 8.24): a stored vector must hold something later than what
it is matched on — arrows, not points.

Full writeup: [`square_nonext/README.md`](square_nonext/README.md).

---

## `hier_time/` — hierarchical time: the slow hand comes from the layer above

Lavender's original design: L1's query includes the down-projection of a slower L2 with a
5-tick stride. Square replay within one pixel (0.95, max 1) once L2 also stores its own
arrow. Movies keep purity 1.00 but the rotation advances at 0.39 vs 0.72 flat. Label-free
carousel: best run plays 1-2-3-4-5-7 in order with correct dwells, then stalls.

Full writeup: [`hier_time/README.md`](hier_time/README.md).

---

## `digits_bridge/` — digit carousel that sees with one vocabulary and draws with another

The carousel with separate eye and hand vocabularies, re-seeing its own drawings through
pixels. One full clean 0-9 lap, then frozen on 4. No separate writeup was kept.

Full writeup: [`digits_bridge/README.md`](digits_bridge/README.md).

---

## `pure_form/` — no slots, no arrows: square with fatigue, digit carousel

Two rigs where everything queries with everything and the arrow of time is meant to come
from fatigue or from the lag between Layers. Square: mean error 9.10 (max 19); carousel:
stuck on 9. No writeup was kept.

Full writeup: [`pure_form/README.md`](pure_form/README.md).
