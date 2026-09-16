# HelloInductor — vocabulary

The shared terminology for this project. One meaning per word.

The substrate is named after music, because the correspondence is structural rather than
decorative: the units really are monophonic, the space really wraps like pitch, and
superposition really does behave like several chords struck at once.

Nothing here is a result. Results live in `experiments/2026_09_16` through `2026_09_21`.

---

## The substrate, smallest to largest

| term | what it is | size |
|---|---|---|
| **note** | the position a voice is sounding — the one that is on | — |
| **gamut** | the positions a voice *could* sound | 41 notes |
| **voice** | one ring. **Monophonic**: exactly one note at a time | — |
| **choir** | the voices that all look at one cell | `B = 16 … 128` voices |
| **cell** | a location in the image — the patch a choir looks at | `5×5` patch, stride 2 |

`gamut` is a collective noun, like *range*: "a voice has a gamut of 41 notes", not "41 gamut".

`cell` always means *a place in the image*. It never means a neuron.

A choir is what the cortex calls a hypercolumn: several competing populations, all looking at
the same patch. One voice ≈ one feature map, one note ≈ the column that won.

**Why 41.** It is prime, so a K-fold transposition stays invertible. That 41 is also a real
microtonal tuning — one of the good equal divisions of the octave, 12, 19, 31, 41, 53 — is a
coincidence, but a convenient one.

---

## Chord and polychord

This is the distinction the vocabulary exists to keep straight.

| term | meaning | geometry | voices |
|---|---|---|---|
| **chord** | names **exactly one** thing | a point on the torus `T^B` — the *surface* | all at full strength |
| **polychord** | several chords sounding at once | a point in the *interior* | weakened |

- A **chord** is pure. Every voice sounds one definite note. Pattern A has a chord; so does
  pattern B.
- A **polychord** is what you get when A and B are both in the input. Each voice is pulled
  between the notes its contributors want, and **the strength of a voice is the agreement of
  its contributors**. Voices pulled in opposite directions cancel to nearly nothing, which
  means *no opinion*.

So **chords are on the surface, polychords are inside**. The closer to the surface, the more
certain; dead centre is silence.

The name is the point: a listener hears a polychord as *two chords*, not as one strange chord.
Recovering the constituents is the whole job.

- **mud** — a polychord past capacity. Too many chords stacked, no constituent recoverable any
  more. This ceiling is the reason attention and chunking are forced rather than chosen.

---

## Cluster

A **cluster** is a voice with weight spread over *adjacent* notes instead of one clean note —
a tone cluster in the musical sense, where adjacency is the defining property.

Clusters are how graded codes are written. Note what the name admits: **a voice playing a
cluster is no longer monophonic.** That is a real departure from the one-note rule, not a
sloppy edge of the metaphor, and it should stay visible.

Two readout modes, one dial between them:

- **clean** — one note per voice. The reading is forced back onto the surface.
- **cluster** — weight over neighbouring notes. The reading is allowed to stay inside.

---

## Transposition

**Transposition** moves a chord to a place. Intervals add, it is invertible, and transpositions
compose — which is the algebra we want and the reason the musical word fits.

**It is not musical `T_n`.** A musical transposition moves every voice by the *same* interval.
Ours moves **each voice by its own interval**.

That difference is load-bearing. A uniform transposition would preserve every relative
interval, so the same chord at two different positions would stay perfectly correlated and the
whole scheme would collapse. **Per-voice independence is the decorrelation.**

Consequence worth stating plainly: content gives a point on the torus, position gives a vector
of intervals, and a thing-at-a-place is their sum. There is no per-cell hardware — the same
choir serves every cell, with the cell folded in as a transposition. Cortex does the opposite:
position is *which* hypercolumn, so its hardware scales with area.

---

## Not named yet

Deliberately open, to be settled when the work needs them:

- resolving a polychord back into its chords
- one entry of the vocabulary
- all the cells of one image, taken together
