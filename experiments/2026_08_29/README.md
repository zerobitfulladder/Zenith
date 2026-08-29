# 2026-08-29 — blurry numbers on a curve, then the drone over time

Two lines of work. The day started on the simplest task where "overlapping
inputs let this generalise" can be checked with a ruler — (x, y) pairs from a
known curve with a slice cut out — and built one, two, and a convolutional
stack of layers on it. In the evening the same machinery went back to the
drone, now over time; that line ran through the night into 08-30 and 08-31
and is kept here as one piece. Its shared modules are in
[`temporal_drone/`](temporal_drone/README.md), with the full account in
[`temporal_drone/ARCHITECTURE.md`](temporal_drone/ARCHITECTURE.md).

---

## `regress/` — does one hypercolumn become a regressor?

One hypercolumn of 256 minicolumns learns (x, y) pairs written as bumps in one
array; testing writes only x and reads the winner's y. Inside the range it is
close (RMSE 0.107); inside the deleted slice it scores 0.729, about the same
as k-nearest neighbours (0.739) and a random forest (0.765) — its answer there
is split between the two rims. Two machinery
bugs came out that also apply to the drone: channels sharing array positions
make phantom answers (0.675 → 0.160 once positions are disjoint), and a partial
cue must be scored by the template's norm over the queried cells — without it,
adding minicolumns makes things worse (0.546 at 1024), with it they help (0.094).

The unit's formulas are in [`regress/HypercolumnReference.md`](regress/HypercolumnReference.md).

Full writeup: [`regress/README.md`](regress/README.md).

---

## `two_layer/` — a second layer that carries the wave

Layer two is handed a window of 21 taps along x, so a template is a piece of
curve. Bridging the hole from its rims: **0.033**, against 0.817 for layer one
alone and 0.149 for an MLP. What a tap is allowed to say decides everything:
a value relative to the first known tap bridges (0.033), absolute values fail
on the trended wave (3.221), and messages made of layer-one minicolumn
identities (one-hot 4.317, graded 2.783) cannot say "the same shape, higher up"
at all.

A plain walkthrough: [`two_layer/TwoLayerWalkthrough.md`](two_layer/TwoLayerWalkthrough.md).

Full writeup: [`two_layer/README.md`](two_layer/README.md).

---

## `conv_stack/` — a convolutional layer one, shared shapes below, structure above

The old layer one overlapped by place, not by content (margin between "same
shape elsewhere" and "unrelated" exactly 0.000). One shared hypercolumn over a
small patch at every x, each patch relative to its own mean, gives 64 local
shapes (margin +0.221). Layer one alone walks into the hole to 0.428; layer two
over its codes, decoded by one least-squares solve, reaches **0.086** — better
than the MLP (0.144), though not part two's hand-relative 0.033. A third
instance of the same bug: whatever is subtracted must come from the same
subset of the input in training and at read time (1.022 → 0.428).

From-scratch account: [`conv_stack/ConvStack.md`](conv_stack/ConvStack.md).

Full writeup: [`conv_stack/README.md`](conv_stack/README.md).

---

## `temporal_stack/` — two (and three) layers over time: the drone does not fly

The drone rig rebuilt as a convolution over time (layer one over 240 ms, layer
two over 540 ms of codes), with replay. Frozen evaluation is 0.00 for every
version. The training-time success (0.79) was a one-tick-delayed copy of the
teacher. Measured wall: the task has about 5 independent quantities, so a
nearest-example memory's error falls only 1.15x per doubling of templates.

Full writeup: [`temporal_stack/README.md`](temporal_stack/README.md).

---

## `tracks/` — one hierarchy per signal; where it actually loses

Every signal gets its own two-layer track and layer three binds them. Also
EVAL 0.000 over 300k ticks. Probes then showed the stack was not at the
dimensionality wall: decoding through a chain of winners threw away graded
structure (steering 0.34 → 0.56 when decoded graded), and the present command
was about 3% of the evidence.

Full writeup: [`tracks/README.md`](tracks/README.md).

---

## `bound/` — the bound form on sensor tracks

One top-layer vector per moment: three sensor tracks' codes plus the two
commands on dedicated cells. Reads improved sharply (collective 0.898, steering
0.586, agreement 0.672 vs 0.63 / 0.34 / 0.28) but EVAL stayed 0.00. A
handover probe showed why: the oracle wearing the pupil's full error still
succeeds 1.00; what was missing was the first second (blind windows) and any
data of holding at the goal.

Full writeup: [`bound/README.md`](bound/README.md).

---

## `balance/` — the validation rung: stop and level

Teach it only to balance. The hold first looked solved (0.95) but was a
short-window artefact; the cause was traced step by step — the success band
smaller than one cell, then the present being 3% of the cue (fixed by giving
the present its own cells: hold 1.3 s → 4-6 s), then a glide trap from missing
data. Tight-envelope DAgger solved the hold (balanced-frac 0.76-0.79) and the
600k run reached strict EVAL **0.72** from random tumbles.

Full writeup: [`balance/README.md`](balance/README.md).

---

## `critic/` — value only at the selector

The settled RL law (value is a per-template scalar on the top layer only;
content is never reward-edited), then a critic on the 0.72 policy. Its gates
passed (honesty corr 0.712), but biasing selection by value hurt monotonically
(beta 0.1 → EVAL 0.22): a template's value measures how nice its situation
was, not how good its action is.

Full writeup: [`critic/README.md`](critic/README.md).

---

## `tally/` — the cetele: proposer and chooser split

A count table of executed actions beside each template chooses the action. The
mode read failed for regulation (0.02 vs 0.72): the teacher regulates by
flickering between neighbouring levels, so the average, not the most common
action, carries the law.

Full writeup: [`tally/README.md`](tally/README.md).

---

## `split/` — the joint encoding retired

A sensory-only top layer with the cetele doing all generation. The smoke run
reached agreement 0.908; the target was to match the joint rig's 0.72.

Full writeup: [`split/README.md`](split/README.md).

---

## `chase/` — the split rig on the position task, on the GPU

64 drones at once, six foveated sensors, per-signal tracks, sensory-only top
layer, cetele, DAgger. No writeup was kept; its metrics show strict EVAL 0.0
over 4 million ticks, closest approach about 1.2 m.

Full writeup: [`chase/README.md`](chase/README.md).

---

## `cascade/` — the cascade chaser: 30/30

A learned outer loop (256 templates over the foveated position error, a cetele
over 32 velocity words, 10 Hz) over a fixed attitude reflex (50 Hz). First
perfect score: 30/30, then **100/100, closest 0.063 m, median 266 ticks**.
Re-tuning the reflex under the trained memory did not help: commander and
reflex are one system.

Full writeup: [`cascade/README.md`](cascade/README.md).

---

## `deep_outer/` — layers restored to the outer loop, and the tax measured

Giving the champion's outer loop temporal tracks cost half its score (EVAL
0.52), adding the angle channel almost all of it (0.05). A layer's cue must
carry exactly what its decision depends on.

Full writeup: [`deep_outer/README.md`](deep_outer/README.md).

---

## `attend/` — attention, rung 1

Choosing which cue blocks matter works: conditional information picks present
dx and dy and refuses the rest, rediscovering the champion's cue. Using it
failed the hold — traced to a velocity vocabulary with no word for "stay".
With the vocabulary fixed, the co-adaptive loop reached **EVAL 1.00, closest
0.06 m, median 265**.

Full writeup: [`attend/README.md`](attend/README.md).

---

## `reward_rl/` — refining the champion by reward alone

Teacher off, value tilt only where the habit says "move", directed
exploration. At 200k steps, beta 0.1 gives success 0.98 at median 246 ticks
against the base's 0.98 / 270 — the first improvement from reward that costs
nothing. Without the reflex (raw motors from scratch) reward learning was
indistinguishable from random.

Full writeup: [`reward_rl/README.md`](reward_rl/README.md).

---

## `grid_ablation/` — does the learned partition matter?

A fixed 16x16 grid in place of the learned templates did better (0.99 / 0.033 m
/ 229 vs 0.95 / 0.081 / 291). With few cells the learned partition wins (K=64:
0.99 vs an 8x8 grid's 0.64): adaptive partitions buy about 4x allocation
efficiency, not better asymptotic quality.

Full writeup: [`grid_ablation/README.md`](grid_ablation/README.md).
