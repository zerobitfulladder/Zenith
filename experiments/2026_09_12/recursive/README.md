# One vector, one notebook, recursion in time

2026-09-12, same day as [`../sparse_patterns`](../sparse_patterns/README.md).
One seed, no sweeps — this is a "does the shape work" test, not a measurement.

The two-floor version worked but needed two of everything: two vocabularies, two
sets of thresholds, and a hand-decided number of floors. This replaces it with
**one vector, one vocabulary, one notebook**, and puts the recursion in time.

## The idea

The vector is 100 light slots plus 120 blank ones. A blank slot becomes the
*name* of a pattern. Reading is:

```
round 1   recognise patterns among the active slots
          switch OFF the slots they accounted for
          switch ON the slots that name them
round 2   look again -- now the active slots are pattern names
...       until it stops getting smaller
```

A pattern made of lights and a pattern made of other patterns are the same kind
of object in the same vocabulary. And because a pattern's name *is* a slot, the
question "do these two patterns go together" is just the slot-pair count between
their names — **there is no second table.** The rule "{A,B} and {B,C} never
occur together" lives in the same notebook as "these two lights go together."

## Three rules that turned out to be load-bearing

**Compress.** A rewrite is only accepted if it leaves *fewer* slots active.
Without this the thing churns forever — the first version swapped six slots for
six slots and ran until the round cap, with the active count *growing* (6 → 6 →
9 → 10).

**Evidence going up.** A pattern may only be used if most of its members are
actually active. Without it, reading admitted any pattern that touched a single
active slot, and with 120 patterns around that is all of them.

**Propose once, at the top.** Guessing is a separate, deliberate step: after
settling, allow exactly one partly-matched pattern, then expand it back down
into lights. That is the synthesis half, and it is the only place a guess can
enter.

## It does the thing

```
cue = A's six lights

  round 1:  6 slots active  ->  used {A}
  proposed:                     {A,B}
  settled:  {A}, {A,B}      ->  expands down to atoms: A, B
```

It climbed from lights to a pattern, proposed a group containing that pattern,
and expanded the group back down into lights — producing B, which was never in
the cue. Analysis up, synthesis down, one engine.

## Beam search is worth it

`greedy` = commit to the best pattern at each step. `beam 4` = carry the four
best partial answers and let them compete.

```
             price   speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2   ambiguity
greedy         3.0         0.62            0.00             0.00             0.00        1.00
greedy         2.0         1.00            0.00           * 0.28             0.00        1.00
beam 4         3.0         0.23            0.00             0.00             0.00        1.00
beam 4         2.0         1.00            0.00             0.00             0.00        1.00
beam 4         1.0         1.00            0.00             0.00             0.00        1.00
```

**Beam 4 at price 2 or below is clean on every probe**: it fills in a legal
partner every time, never turns on both B and C, never produces C when given A
and B, never puts F2 with F1, and always resolves the G/H ambiguity correctly.

Greedy has a real 28% failure rate at the same price. So beam search isn't
tidiness — it removes an error that greedy cannot avoid, because greedy commits
to a first pattern that looks best locally and then has to live with it.

## What is still wrong

**The vocabulary is bloated.** It hits the 120 cap and holds obvious duplicates
— `{F1}` appears three times, `{F4,H,CTX2}` twice. The right patterns are all in
there and the behaviour is right, but the "do I already have a name for this"
check is too weak once patterns can be made of patterns. That is the next thing
to fix, and it is the same complaint as the carving problem from this morning
wearing different clothes.

**One seed, one world.** Nothing here is a measurement.

## Files

```
engine.py   the whole thing -- notebook, recognise, rewrite, mint, expand
run.py      one seed, greedy vs beam, the worked example and the four probes
```

Runs in about 25 seconds.
