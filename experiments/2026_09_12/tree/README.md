# Top-down: one category that splits, instead of parts that glue

2026-09-12. Same 100-light world as [`../sparse_patterns`](../sparse_patterns/README.md).
One seed, nothing here is a measurement except the scaling table, which is
exact.

Every engine before this one built **upward**: find bits that co-occur, glue
them into a group, glue groups into bigger groups. All of them had the same
disease — the number of candidate groups is combinatorial from the first step,
so every version needed caps, thresholds, duplicate checks, pruning and decay to
hold it back, and the last one saturated anyway at 17,462 entries.

This goes the other way. **Everything starts as one category** covering all the
data, predicting each light at its overall rate — vague, but never wrong. When a
category fails to account for what lands in it, it splits, and the split is
chosen from the same co-occurrence counts we have had all along.

## The thing that makes it different

```
leaves containing A, B, C all three:  0
leaves containing F1 and F2:          0
```

The illegal combinations are not refused by a stored rule. **They are never
created**, because no data ever lands there.

That matters more than it sounds. Bottom-up, we spent the whole day getting the
units right so that "the A-B group never occurs with C" would be *expressible*,
and then relied on that fact being consulted and outweighing everything else at
read time. It kept slipping, because a rule is a number and numbers can be
outvoted. An absence cannot.

And the knobs that plagued every previous version simply have nothing to do
here: no leftover to mint from, no duplicate check, no cap, no pruning, no
warm-up, no decay. Every input lands in some category from the very first one.

## The probes

```
             speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2   ambiguity
tree               0.65            0.00             0.00             0.00        1.00
../recursive       1.00            0.00             0.00             0.00        1.00
```

Every constraint held. It is worse at speculating — given one atom it supplies a
legal partner 65% of the time against the bottom-up engine's 100% — because
picking a single best leaf commits to one reading of an ambiguous cue, and with
33 leaves the cue often matches several about equally.

Training takes **under a tenth of a second**, against 25 seconds for the
recursive engine.

33 leaves against 45 legal scenes in this world (3 triangle options × 5 filler
options × 3 ambiguity options). Some scenes are merged, which is the split price
doing its job.

## The cost: it pays the product

A world with `f` independent blocks, each with 4 options and nothing constraining
one block given another, has 4^f legal scenes. The question is whether the tree
needs a leaf for each.

```
 blocks  legal   clean   drop .05   drop+noise      <- leaves the tree builds
      1      4       4         28           60
      2     16      16         37           67
      3     64      64         83           87
      4    256     251        252          253
```

The `clean` column is exact. **Leaves = 4^f.** (251 rather than 256 at f=4 only
because 8,000 samples over 256 scenes leaves 31 per scene, near the minimum node
size.)

So the tree pays the **product** of every independent choice in the data, where
a pairwise notebook pays the **sum** — for f=4 that is 256 leaves against 4,560
pair counts that would cover it and everything else besides. Independence is
exactly what a tree cannot exploit and what pairwise counting represents for
free.

Which means the two approaches are not rivals, they are suited to opposite
structure:

```
tree       where combinations are CONSTRAINED -- few legal options out of many
           possible. Illegal ones vanish by never being populated.

notebook   where combinations are FREE -- independent things in any mixture.
           One pair count each instead of one leaf each.
```

## A new failure, and it is not small

The other two columns of that table are the tree splitting on **noise**.

Four legal scenes, and with 5% per-bit dropout it builds 28 leaves; add two
stray lights per vector and it builds 60. Nothing is being discovered — it is
splitting on individual noisy lights, because once a node is big enough, a split
that pins down even one light pays for itself.

The arithmetic is against us. The gain from splitting on light *j* grows with
the node size *n*. The price charged for a split grows with **log** *n*. So for
any light that is not perfectly deterministic, there is a node size above which
splitting on it is worth doing, and large datasets guarantee you get there.

At f=4 all three columns converge (251 / 252 / 253) because the real structure
finally dominates. But at small f the leaf count is mostly noise, and in a world
with any real per-item variability that is the regime you are in.

This is the classic decision-tree problem and it has known answers — hold out
data and prune against it, or charge a price that scales with *n* rather than
its logarithm. Neither is exotic. But it is a live hole, and any claim about
this approach being cheap has to survive it first.

## Files

```
tree.py      the category tree: split chosen from co-occurrence counts
run.py       the four probes, what the leaves mean, and the scaling table
scaling.py   the product claim, isolated from the noise-splitting
```
