# The notebook as the learning target

2026-09-12. Same 100-light world as [`../sparse_patterns`](../sparse_patterns/README.md).
One seed, two runs (one bug fix between them), nothing here is a measurement.

`../encoder` ended on: templates alone can find parts but cannot form groups,
and putting the notebook back fixed that but left a mess that looked exactly
like missing housekeeping — pruning, duplicate refusal, per-level floors, a
frozen lower level. The proposal here was to make the notebook the **target**
instead of a side consultation, so that the housekeeping becomes a consequence
of the objective rather than four separate rules.

## The design

A template is a clique carved out of the notebook — a set of things whose
mutual friendship is high — and nothing else. **It never learns from an
input.** Reading uses templates; counting updates the notebook; carving makes
templates from the counts. Things are the 100 lights and up to 150 templates;
the vector is one slot per thing. The signature layer from `../encoder` is
gone, because once templates stop learning over the vector there is nothing
left for a codec to do.

```
read      sequential explaining with a beam over the order (the pursuit rule)
count     the vector AFTER substitution, once per input -- as understood
carve     from what the search left unexplained: seed on the friendliest
          unmasked pair, grow while candidates stay within 70% of the seed
mask      the pairs inside a carved clique can never be seeded again
```

Three things were predicted to fall out of this without a rule for them.

## Two of the three did

```
                                 ../encoder+tally      carve
templates at the cap                       150            82
upper templates by size          {1:31, 2:39, ...}    {2: 43}
atoms represented alone                     6/11         8/11
ambiguity                                   0.71         1.00
```

**No wrappers.** Every one of the 43 upper templates has exactly two members.
Not one single-member template exists, because the carve starts from a pair
and a one-member thing explains zero pairs. The wrapper problem — which every
version today had to fight with a threshold — has no mechanism to occur.

**No duplicates.** The vocabulary stopped at 82 and stayed there, for the first
time today. Masking the pairs inside a carved clique means the same clique
cannot be seeded twice, and that turned out to be the whole of the duplicate
problem.

Both consequences held exactly as argued. And the vocabulary is the cleanest
yet — 8 of 11 atoms alone, best of the day — with perfect ambiguity resolution.

## The third did not, and it points at the counting decision

```
             speculates   adds BOTH B,C   cue A+B adds C   F1 adds F2   ambiguity
carve              0.38            0.00             0.00         0.00        1.00
encoder+tally      0.61            0.00             0.00         0.00        0.71
recursive          1.00            0.00             0.00         0.00        1.00
```

Speculation is the worst of the three, and the reason is visible in what gets
proposed:

```
cue A    proposed {A,CTX2}T
cue B    proposed {B,F2}T
cue C    proposed {B,C,F2}T
```

The legal pairs `{A,B}T`, `{B,C}T`, `{A,C}T` all exist. But so do pairs of
**independent** things — A with CTX2, B with F2 — and they win the propose
step as often as the real ones. "No blobs" was supposed to prevent this: A and
CTX2 are at chance, so the carve should refuse. It didn't, because the notebook
was telling it they weren't at chance.

### Why: counting as-understood inflates stale pairs

The first run had a real bug — a template born 20 inputs ago has a 20-input
window, and one coincidence in a window that short clears any floor. A per-thing
minimum age (300 inputs) fixed that, and cut the vocabulary from 150 to 82.

The junk pairs survived the fix, through a mechanism that is the counting rule
itself. Once `{A,B}T` and `{A,C}T` exist, T_A is absorbed upward on nearly
every scene it appears in, so it almost never reaches the final vector — **its
`seen` count freezes while its window keeps growing**, and its apparent rate
falls toward zero. Same for T_CTX2. But `co[T_A, T_CTX2]` was accumulated
before either was absorbed, and it is frozen too. The friendship ratio is
`co / (rate_A x rate_CTX2)`: numerator frozen, denominator shrinking
*quadratically*. **Stale co-occurrences don't fade. They inflate.**

So "count only what survives to the top" is wrong as stated. A thing absorbed
into a group is not absent — it is present at a higher level — and treating
absorption as absence corrupts every rate below it. The `?3L` junk at the
light level (twenty-odd three-light lumps matching no atom) is the same
mechanism one floor down: cross-atom light pairs whose within-atom neighbours
were masked and whose owners were substituted away.

The alternative I named and didn't take — count the vector at every round,
before each substitution — avoids this and reintroduces the redundancy
(`{A,B}L` alongside `{A,B}T`). Neither is right. What is needed is a rate that
asks *"of the inputs where this thing could have appeared, how often did it?"*
— which means the window for a template is not "since it was born" but "since
it was born, minus the inputs where it was absorbed." That is a bookkeeping
change, not a design change, and it is the next thing to try.

## Verdict

```
no wrappers        held, exactly
no duplicates      held, exactly
no blobs           broken by the counting rule, not by the carve
```

The idea is right: making the notebook the target does turn most of the
housekeeping into consequences. What it exposed is that **"count as
understood" needs a definition of understood that includes what was absorbed**,
and without it the notebook slowly lies about anything that has been named.

## Files

```
engine.py   the whole thing, ~200 lines, no per-template learning path
run.py      one seed, the vocabulary, three worked cues, the four probes
```
