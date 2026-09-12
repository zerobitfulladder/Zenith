# Templates as the only memory — no notebook

2026-09-12. Same 100-light world as [`../sparse_patterns`](../sparse_patterns/README.md).
One seed, two runs (2,000 and 5,000 vectors), nothing here is a measurement.

The question this tests: **do we need the pair-count notebook at all?** The
argument for dropping it was that a template's weights already *are* a stored
co-occurrence — "these bits appear together when I'm chosen" — so a separate
table of what goes with what is redundant. Templates learn, the search picks
which, the residual says from what. Three things, one memory.

## The design

The vector is a 4,000-slot sparse space. Every light and every template has a
fixed random **signature** of 12 slots. Light 5 on = its 12 slots on. Template
40 fired = its 12 slots on. Same space, so a template can learn "light 5 with
template 40" as easily as any pair of lights, and substitution never collides
with anything.

```
collision arithmetic   two signatures share 144/4000 = 0.036 slots on average
                       a light that is NOT on has ~1.7 of its 12 slots lit by accident
                       decode threshold 7 of 12 never confuses them
```

Reading is sequential explaining — the pursuit rule — with a beam of 3 over
the order: pick the template that best accounts for the residual, switch off
what it claimed, pick the next from what is left, stop when nothing pays.
Substitution switches on the chosen templates' signatures and the same encoder
reads the result. Learning follows the sequence: each chosen template moves
toward the residual it was shown, by counting. Templates are hired by copying
a residual that is still large after the search, and refined from there.

150 templates. No pair counts, no base rates, no notebook of any kind.

## It recovers the atoms

```
atoms represented alone: 7/11    {A} {B} {C} {F1} {F2} {F3} {F4}
5,000 vectors in 8.8 seconds
```

Every atom that appears on its own gets its own template. The four that don't
are G, H and their contexts, which only ever appear as a pair — and the engine
has them as pairs (`{F3,G,CTX1}`, `{F3,H,CTX2}`), which is honest.

So for the first level, the answer is **no, you don't need the notebook.**
Templates picked by the search and refined by the residual find the parts.

## It cannot form the groups

```
             speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2   ambiguity
encoder            0.00            0.00             0.00             0.00        0.79
recursive          1.00            0.00             0.00             0.00        1.00
```

Zero speculation — the mute regime from this morning. Given A alone, nothing
supplies B or C. The constraints hold only because nothing is ever proposed.

The reason is in the upper templates:

```
upper templates by size in signatures:   {1: 63,  2: 5,  3: 4}
```

**Sixty-three of the seventy-two upper templates expect exactly one signature.**
They are wrappers — a template that means "template 12 fired" and nothing more.
The five that expect two are junk (`{F1,F4}`, a template and its own wrapper).
Not one of them is `{A,B}`, `{A,C}` or `{B,C}`, which is what the two-floor
version minted unprompted in five seeds of five and which is what legal
speculation requires.

## Why: erosion has no reason to stop at a pair

The first run had a real bug — hiring accepted a single signature as something
to name, which manufactured wrappers directly. Fixed by requiring two
signatures' worth. The wrappers came back anyway, by a different route.

A template hired from a three-signature residual, say `{T_A, T_B, T_F1}`, is
picked whenever 40% of what it expects is present — any two of the three. So it
gets picked on A-B scenes, on A-C-F1 scenes, on A-B-F3 scenes. Across all of
those, **T_A is the common thread**; T_B and T_F1 are each present about half
the time. Counting drives them below threshold, and the template ends up
expecting T_A alone.

That's the whole mechanism. A template's only signal is *what is consistently
present when I'm picked*, and a single thing is always more consistent than a
pair. Left to that signal alone, groups erode to their most reliable member,
and a most-reliable member is a wrapper.

**The notebook was what stopped this.** In the two-floor version the upper
carve used pair counts over template signatures to decide which ones belong
together — "T_A and T_B keep company more than chance." That is exactly the
information a template cannot get from its own record, because a template only
knows what accompanies *it*, never what accompanies *each other*.

## Verdict

```
recovering parts from lights      templates alone are enough
forming groups of parts           they are not
```

Dropping the notebook was right for the first level and wrong for the second.
What a template can learn on its own is *what goes with me*. What a group needs
is *what goes with each other*, and that is a fact about two things neither of
which is the template. Something has to hold pairwise evidence at the level
where groups form — a notebook over template signatures, or a rule with the
same content — or the search will keep finding the most reliable single thing
and calling it a group.

---

## Putting the notebook back — over things, for one job

`EncoderTally` adds one table: pair counts over **things** — the 100 lights and
150 templates, 250 in all — not over the 4,000 slots. It is consulted for
exactly one decision: when a residual is still large after the search, which of
the things in it belong together? The carve from `../sparse_patterns` runs on
that, and a template is hired for the lump rather than for the whole residual.

With the carve doing the grouping, templates no longer have to erode from blobs
to parts, so the evidence bar for being picked in the search goes up from 40%
to 75% of what a template expects. That is what stops pairs eroding into
wrappers.

Two runs: hiring from input 1, and hiring after a 1,000-vector warm-up.

```
                            speculates   adds BOTH B,C   cue A+B adds C   F1 adds F2   ambiguity
no notebook                       0.00            0.00             0.00         0.00        0.79
notebook, no warm-up              0.61            0.00             0.00         0.00        0.50
notebook, warm-up                 0.61            0.00             0.00         0.00        0.71
../recursive (beam 4)             1.00            0.00             0.00         0.00        1.00
```

```
upper templates by size    no notebook          {1: 63, 2: 5,  3: 4}
                           notebook, no warm-up {1: 9,  2: 59, 3: 15, ...}
                           notebook, warm-up    {1: 31, 2: 39, 3: 20, ...}
```

**Speculation came back, and the mechanism is the one predicted.** With the
warm-up the vocabulary contains `{A,B}T` — a template-level pair of T_A and
T_B — and the worked example runs exactly as it should:

```
cue = A's lights
  round 0:   used {A}L
  proposed:  {A,B}T     -> atoms out: A, B
```

No stored rule, no company term: T_A fires, `{A,B}T` half-matches it, and
expanding `{A,B}T` back down supplies B. The notebook's only contribution was
to make `{A,B}T` exist.

**But the vocabulary is messy, and the probes are middling.** Light-level blobs
`{A,B}L`, `{B,C}L`, `{A,C}L` sit alongside the atoms — each used almost exactly
as often as its scene occurs (414, 437, 360 against ~417 expected), so they are
stable and correct, just redundant with `{A,B}T`. One of A/B/C is always missing
as a standalone atom. And some upper templates expand to all three of A, B, C —
pairs of light-level blobs glued at the upper level — which never get proposed
(the violation rate is still 0.00) but should not exist.

Warm-up bought the clean `{A,B}T` and better ambiguity resolution (0.50 → 0.71)
at the cost of more wrappers (9 → 31). Neither run reaches the two-floor
version's 1.00.

## What the gap is

The two-floor version had four things this does not: it **prunes** templates
that stop being used, it **refuses to hire** what already exists, it uses a
**different seed floor** on each level, and floor 0 is **frozen** before floor 1
learns. This single encoder has none of them, and the mess above is what their
absence looks like: redundant blobs nobody removes, near-duplicates nobody
refuses, and upper groups formed while the lower vocabulary was still moving.

So the notebook was the missing *information*. The housekeeping is the missing
*discipline*, and it is a separate thing.

## Files

```
engine.py   Encoder (no notebook) and EncoderTally (thing-level notebook, one job)
run.py      `python run.py plain` or `python run.py tally`
```
