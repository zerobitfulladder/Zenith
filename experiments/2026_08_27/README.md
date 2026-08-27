# 2026-08-27 — the completion arrow

Theme: the COMPLETION ARROW — the user's proposed alternative to
lag-advance channels. No next slots, no cargo/key channel splits, no
empty query channels. The arrow of time is claimed to come from the
hierarchy itself: a mid-era trail is a half pattern to the layer
above; the layer above's stored era snapshot was written later and
contains the era's future; retrieval completes half -> whole; the
downward projection tints the lower layer's query with that future,
recursively; L1's retrieved row, deblurred, is the emission.

Uniform unit everywhere (user spec): leaky integrate every arrival
(T = x + 0.5*T, never reset), speak every 3rd arrival, store snapshots
[own trail ; 0.5 * lagged top-down projection], projection = layer
above's retrieved-row input-trail segment, refreshed at every arrival
(event-driven reads), consumed with one-tick lag and learned jointly
with the input (user's amendment).

---

## `completion/` — Exp 1: does the arrow of time come from completion up the stack?

A bouncing 4x4 square, three layers, no next-slots. The question was whether
a half-finished trail, completed by the layer above, would retrieve the future.
It froze: free-run error 10.02 (worse than no arrow at all, 8.12; the arrow forms
score 0.00). With 64 templates for 40 moments every query finds a template stored at
exactly its own moment, so L1 retrieves itself 1.00 of the time and completion
never gets a chance to fire.

This folder also holds `run_square_completion.py`, the rig module that every
other experiment of the day imports.

Full writeup: [`completion/README.md`](completion/README.md).

---

## `refractory/` — Exp 2: forbid the self-match

Same rig, but the winner may not repeat. With the self-match removed the next
best template sits ahead or behind exactly 50/50, and it is usually the mirror pass
(same position, opposite direction), not the successor. Free-run error 7.49:
better than 10.02, still no motion.

Full writeup: [`refractory/README.md`](refractory/README.md).

---

## `residual/` — Exps 3 and 3b: same-rate layers + subtract what is already explained

Every layer ticks every input tick; timescale comes from decay alone. A starved
L2 (8 templates) organises itself into clean contiguous arcs of the cycle. Subtracting
the already-explained part of an arc gave the first forward retrievals of the day
(forward 0.41) but backward stayed 0.56. Subtraction works as an anti-freeze
(free-run 10.02 -> 7.49 -> 5.56, and 4.98 with dense upward speech in 3b), not as
an arrow. The day's law: order has to be written into the stored vector at
storage time; no read rule can recover it.

Full writeup: [`residual/README.md`](residual/README.md).

---

## `residcargo/` — Exp 4: store the residual as the "next" part of the row

Stores [trail ; residual] so the network's own completion plays the next-frame
role. The stored guess inherits the residual's backward leak (forward 0.38 /
backward 0.56) and free-run oscillates, error 6.65. The cargo is only as good as
whoever writes it.

Full writeup: [`residcargo/README.md`](residcargo/README.md).

---

## `lagcargo/` — Exp 5: write the next frame when it is actually seen

Same row layout as Exp 4, but the cargo is the frame observed one tick later
(delayed write, the lag-advance form). Free-run error **0.00, max 0** over 100
frames. Inferred cargo 6.65 vs observed cargo 0.00 on the same chassis.

Full writeup: [`lagcargo/README.md`](lagcargo/README.md).

---

## `lagcargo_stack/` — Exp 6: the delayed-write recipe at all three layers, then wired

Stacked: every layer learns the same [keys ; cargo] arrow with longer keys
higher up (3 -> 10 -> 30 frames) and coarser ownership; free-run holds 0.00.
Wired: the full loop with top-down contexts learned jointly also holds 0.00, so
wiring the feedback does not hurt the unambiguous case.

Full writeup: [`lagcargo_stack/README.md`](lagcargo_stack/README.md).

---

## `bounce_updown/` — Exp 7: decide at the top, rebuild downward

Layers learn only [cargo ; own lagged trail]; one successor read at the chosen
top, then each layer's cargo names the template below. TOP=L1 0.00; TOP=L2 6.69 but it
tracks phase-exactly for a long stretch before derailing; TOP=L3 6.84 freezes.
A follow-up with dense upward speech (no writeup kept) scored TOP=L2 3.72 and
TOP=L3 3.41 with the graded descent.

Full writeup: [`bounce_updown/README.md`](bounce_updown/README.md).

---

## `carousel/` — Exp 8: MNIST carousel, and convolution as the layer

MNIST 0..9 loop with a fresh exemplar each lap. With the whole-frame stack,
TOP=L1 is perfect (advance 1.00, timeline 1.00); higher tops freeze on a mushy 8,
because this task has structure at one timescale only. The convolutional version
learns a proper 8x8 stroke sheet at L1, but its L2 over the code failed.

Full writeup: [`carousel/README.md`](carousel/README.md).

---

## `conv_codec/` — Exps 9 and 9b: the conv codec was the real failure, then the judge was

Re-opened Exp 8 with measurements. The L1 row was 74% trail, so decoded digits
were one scribble (decoder ceiling 0.26-0.27 against a judge at 0.87). Giving
the current window the majority of the row lifts the ceiling to 0.81-0.87. Then a
judge trained on the network's own renders re-ranked the arms: **w4k36** counts
all ten digits phase-locked to the carousel (advance 0.879, timeline 0.920),
gc4 counts as well but one tick behind. Law: never judge generated output with a
gate calibrated on real input.

Full writeup: [`conv_codec/README.md`](conv_codec/README.md).

---

## `two_movies/` — Exp 10: two labelled sequences over the same three digits

Label 0 plays 1,2,3,... and label 1 plays 3,2,1,...; generation gets only the
two-bit label. Letting the top gate a pool of 5 and L2 pick inside it scores
**0.831 / 0.475**, vs 0.237 / 0.288 for pure top-down descent and identical
images for both labels when the top is ignored. The monopoly guard ("conscience")
spreads L2's templates (top share 0.987 -> 0.060) but does not help generation, so
the monopoly was not the blocker. Errors are mostly repeats (stalls), not wrong
turns.

Full writeup: [`two_movies/README.md`](two_movies/README.md).

---

## `static/` — the pre-temporal rig, rebuilt clean (no time)

Three convolutional layers plus a label memory, no trails. 8x8 L1 windows beat
the 4x4 control on every measure: probes 0.9594 / 0.9606 / 0.9548, hard readout
0.8762, round trip 0.651 vs 0.487. Every template used at every level; label alone
generates ten readable digits. (Moved in from the old `experiments_static/`
folder.)

Full writeup: [`static/README.md`](static/README.md).
