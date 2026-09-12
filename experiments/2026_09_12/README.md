# 2026-09-12

## Where things stand (end of day) and what's next

Seven folders, one line of thought: **a sparse vector that names the groups it
finds, then names the groups of groups, using nothing but counting and a
search.** What survived and what broke, in order:

```
sparse_patterns   pair counts over bits miss third-order structure. Raising the unit
                  helps ONLY if the units are the legal combinations. A second floor
                  of the same engine mints those unprompted (5/5 seeds). Floor 0 must
                  never speculate.
recursive         one vector, one notebook, recursion in time. Works (beam 4: perfect
                  on every probe). Three load-bearing rules: rewrites must compress,
                  going up needs evidence, guessing happens once at the top.
hashed            remove the naming decision entirely. Refuted: removing it also
                  removed the brake that kept the vocabulary small.
tree              start coarse, split on exposure. Illegal combinations become
                  ABSENCES rather than stored rules -- cleaner than any rule. But a
                  tree pays the product of independent choices, and splits on noise.
encoder           templates as the only memory, no notebook. Finds parts (7/11);
                  cannot form groups -- erodes them to singletons, because a template
                  can learn what goes with IT but never what goes with EACH OTHER.
carve             notebook as the learning target; templates are cliques, never learn
                  from inputs. No wrappers and no duplicates hold EXACTLY. Counting
                  as-understood inflates stale pairs (absorbed != absent).
mnist_carve       on real digits, 3,000 templates: a five-level tower, depth-4/5
                  templates are whole digits. The label never climbs above depth 2.
```

**Next, in order of how much each is likely to move things:**

1. **Gate the write by explanation quality.** Weight `observe` (carve) or the
   learning update (encoder) by the search's normalised total gain. This is
   the "search releases the dopamine" idea: the sequence says WHO, the gain
   says HOW MUCH. It also grounds the self-reference loop for free -- bad
   readings stop being counted as evidence.
2. **Re-score MNIST label reading transitively.** The 0.137 read labels from
   direct members only; the fix is in, the engine save is in. Smaller run
   (3,000 train, ~1,500 templates) to stay under five minutes.
3. **Fix the as-understood window.** A template's rate must be taken over the
   inputs where it could have appeared -- exclude the ones where it was
   absorbed upward. Bookkeeping, not design.
4. **Put the label at the top.** It is a global property; reading it against
   the deepest fired template rather than hoping it joins a local clique.
5. **Prune by contribution.** The housekeeping rule `carve` argued for and
   never built; the cap should stop being the binding constraint.
6. **Layers instead of recursion, as a control.** Dedicated per-level template
   sets cannot produce cross-level junk (`T#371`); worth knowing what that
   costs against the shared version.

---

## `sparse_patterns/` — counting pairs of patterns instead of pairs of bits

No images, no convolution, no layers, no templates. A 100-bit sparse vector with
eleven hidden atoms and three pieces of planted structure, and two engines that
learn to autocomplete it by counting: one over pairs of **bits** (Hopfield with
counts), one over a learned vocabulary of **patterns** and pairs of patterns.

**It finds the vocabulary.** Seven of the eleven atoms come out exactly — F1 =
1.00, alone, from scratch — and 10/11 at F1 ≥ 0.8, in ~13 patterns, in four
seconds. The only misses are two atoms that genuinely share four bits with each
other.

**Pair counts over bits handle second-order illegality and are helpless against
third-order.** Two things that never co-occur are refused outright by the
baseline (0.000). But given A and B, where every pair is positively correlated
and only the triple is absent, the baseline produces the illegal C **every single
time**, and there is no price at which it both completes and refuses — refusal
arrives only once recall is already zero.

**The pattern engine refuses it — for the wrong reason.** It also never adds
anything on the strength of company alone, so its refusal is silence, not a
learned constraint. Sweeping the read price gives two regimes, mute or illegal,
with nothing in between.

**The carving is what decides.** Handing the engine a vocabulary and changing
only where it cuts:

```
                   speculates?   produces the illegal triple?
atoms  (A,B,C)         mute or   always
unions (A+B,B+C,A+C)   always    never, at every price -- but only with the
                                 pattern-pair statistics switched on
```

Raising the unit of description does lower the order of statistics you need —
but only if the vocabulary carves at the **legal combinations** rather than at
the parts. And the description-length price, which is what makes the vocabulary
economical and makes it find the atoms, pushes in exactly the wrong direction:
nine 12-bit unions is the cheaper description by the measure being minimised, and
it is the one that can hold the constraint.

So the statistics were never the problem, and the open question moved: it is now
about what decides where a thing begins and ends, not about what to count.

**And then a second floor answered it.** Running the *identical engine* on floor
0's own explanation — which is already a sparse on/off vector, over patterns
instead of bulbs — mints `{A,B}`, `{A,C}`, `{B,C}` unprompted, in **5 seeds of
5**. That is exactly the vocabulary that had to be hand-built. Reading goes up
then down: floor 1 says which floor-0 patterns it expects, and floor 0 is re-read
with that as a bonus, with its own price held fixed so it never speculates on its
own.

```
                 speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2
one floor              0.00            0.00             0.00             0.00
two floors             1.00            0.00             0.00             0.00
```

At a price where a single floor is **mute**, the stack supplies a legal partner
every time and never once produces the illegal triple. The parts stay downstairs
where they are economical; the combinations live upstairs where the rule about
them is a fact between two things. No new machinery — the same engine pointed at
its own output.

Full writeup, the three implementation bugs that each turned out to be a design
error, and the board: [`sparse_patterns/README.md`](sparse_patterns/README.md).

---

## `recursive/` — the same thing with one floor, recursing in time

Built after the above, on the same world. Instead of two floors with two
vocabularies and two sets of thresholds, **one vector and one notebook**: 100
light slots plus 120 blank ones, where a blank slot becomes the *name* of a
pattern. Recognise, switch off what you recognised, switch on its name, look
again. Because a pattern's name is a slot, "do these two patterns go together"
is just the slot-pair count between their names — there is no second table, and
the rule `{A,B}` never with `{B,C}` sits in the same notebook as "these two
lights go together".

```
cue = A's six lights
  round 1:  6 slots active -> used {A}
  proposed:                   {A,B}
  settled:  {A}, {A,B}     -> expands down to atoms: A, B
```

**Beam search earns its keep.** Carrying the four best partial answers instead
of committing to the best one removes a failure greedy cannot avoid:

```
           price   speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2
greedy       2.0         1.00            0.00          * 0.28             0.00
beam 4       2.0         1.00            0.00            0.00             0.00
```

Three rules turned out to be load-bearing, each found by breaking it: a rewrite
must **compress** or the thing churns forever; going up, a pattern needs most of
its members actually **present** or everything matches everything; and **guessing
happens once, deliberately, at the top**, then expands back down into lights.

Still bloated — it hits its pattern cap with visible duplicates, because the
"do I already have a name for this" check is too weak once patterns can be made
of patterns. One seed, no sweeps.

Full writeup: [`recursive/README.md`](recursive/README.md).

---

## `hashed/` — removing the naming decision altogether

Every version above has the same weak point: naming is a **discrete,
irreversible act**. That decision needed a warm-up (so the counts weren't noise),
produced duplicates (two naming events, two slots, one meaning), and was
permanent (a bad name minted on input 30 stays forever).

This removes it. A friendship between slot *a* and slot *b* lives at
`hash(a,b)` — nobody creates it, its address is a function of what it is, so
**duplicates are impossible by construction**. Groups are built by chaining
pairs (`{4,5,6}` = `hash(hash(4,5), 6)`) so a missing member costs the top of
the chain rather than everything. Slots hold strengths rather than bits, and
counts decay, so there is no warm-up and nothing is permanent.

**All four mechanisms work as designed. The thing as a whole does not.**

```
                        speculates   adds BOTH B,C   cue A+B adds C   F1 adds F2
hashed, 2000 vectors          0.77            0.00             0.00         0.00
hashed, 1500 vectors          0.77            0.39             0.36         0.00
recursive (beam 4)            1.00            0.00             0.00         0.00
```

Same engine, same settings, 500 fewer training vectors — and the violation rate
moves from 0.00 to 0.39. That is noise, not a system that needs tuning.

It addresses **17,462 of its 20,000 hash slots**, so almost nothing recurs often
enough to stabilise. Loosening the truncation makes it categorically worse, not
better: at K=60 the space is completely full and every constraint is violated,
including the second-order one plain bit-counting gets right.

So the truncation was not the disease — it was the only thing holding the disease
back. **Removing the naming decision also removed the thing that kept the
vocabulary small.** A mint-with-duplicate-check was doing two jobs, and the
non-obvious one was refusing to create anything new most of the time: 11 atoms
at 13 patterns, against 17,462 here.

Full writeup: [`hashed/README.md`](hashed/README.md).

---

## `tree/` — going top-down instead of bottom-up

Everything above builds **upward**: find bits that co-occur, glue them, glue the
groups. All of it shares one disease — the candidate groups are combinatorial
from the first step, so every version needed caps, thresholds, duplicate checks,
pruning and decay to hold it back, and the last one saturated anyway.

This starts with **one category covering everything**, predicting each light at
its overall rate, and splits only when a category fails to account for what lands
in it. The split is chosen from the same co-occurrence counts.

```
leaves containing A, B, C all three:  0
leaves containing F1 and F2:          0
```

**The illegal combinations are not refused. They are never created**, because no
data lands there. Bottom-up we had to make that rule *expressible* and then rely
on it being consulted and outweighing everything else; a rule is a number and
numbers can be outvoted, whereas an absence cannot. Every arbitrary knob also
disappears: no leftover, no duplicate check, no cap, no pruning, no warm-up.

```
             speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2   ambiguity
tree               0.65            0.00             0.00             0.00        1.00
recursive          1.00            0.00             0.00             0.00        1.00
```

Every constraint holds, in under a tenth of a second against 25 seconds. It is
worse at speculating, because committing to a single best leaf forces one reading
of an ambiguous cue.

**And it pays the product.** A world with `f` independent blocks of 4 options has
4^f legal scenes, and on clean data the leaf count is exactly that: 4, 16, 64,
251. A pairwise notebook pays the *sum* — 4,560 pair counts cover the f=4 case
and everything else besides. So the two are suited to opposite structure:
**constrained** combinations to the tree, **free** ones to the notebook.

**New failure found:** with 5% per-bit dropout the tree builds 28 leaves for 4
legal scenes, and 60 with stray bits added. It splits on noise, because the gain
from splitting grows with node size *n* while the price charged grows with log
*n* — so above some size, splitting on any non-deterministic light pays for
itself. Classic decision-tree overfitting, with known answers, but a live hole.

Full writeup: [`tree/README.md`](tree/README.md).

---

## `encoder/` — templates as the only memory, no notebook

Tests whether the pair-count notebook is needed at all. Every light and every
template gets a fixed random 12-slot signature in a 4,000-slot space, so
substitution never collides. Templates are chosen by sequential explaining
with a beam over the order, learn from the residual they were shown, and are
hired by copying a residual that's still large. No pair counts anywhere.

```
atoms represented alone: 7/11    {A} {B} {C} {F1} {F2} {F3} {F4}     8.8 s
```

**For the first level, the notebook is not needed** — templates picked by the
search and refined by the residual find the parts on their own.

```
             speculates   adds BOTH B,C   cue A+B adds C   cue F1 adds F2
encoder            0.00            0.00             0.00             0.00
recursive          1.00            0.00             0.00             0.00
```

**For the second level, it is.** Zero speculation, and the upper templates
explain why: 63 of 72 expect exactly one signature. A template hired from
`{T_A, T_B, T_F1}` is picked on every scene containing two of the three, and
across those T_A is the common thread while T_B and T_F1 are each present half
the time — so counting erodes it to T_A alone. A template's only signal is
*what accompanies me*, and a single thing is always more consistent than a
pair. What a group needs is *what accompanies each other*, which no template
can learn from its own record. That is what the notebook was supplying.

**Putting it back, over things, for one job.** `EncoderTally` adds pair counts
over the 250 lights-and-templates — not the 4,000 slots — used only to decide
what to hire. Speculation returns:

```
                       speculates   adds BOTH B,C   cue A+B adds C   F1 adds F2   ambiguity
no notebook                  0.00            0.00             0.00         0.00        0.79
notebook + warm-up           0.61            0.00             0.00         0.00        0.71
recursive (beam 4)           1.00            0.00             0.00         0.00        1.00
```

and by the predicted route: `{A,B}T` now exists, and `cue A -> {A}L -> proposed
{A,B}T -> A, B`. Wrappers fall from 63 to 31, pairs rise from 5 to 39. What's
left is housekeeping the two-floor version had and this doesn't — pruning,
refusing duplicates, a per-level seed floor, a frozen lower level — and the mess
looks exactly like their absence: redundant light-level blobs `{A,B}L` alongside
`{A,B}T`, one of A/B/C always missing alone. The notebook was the missing
information; the discipline is a separate thing.

Full writeup: [`encoder/README.md`](encoder/README.md).

---

## `carve/` — the notebook as the learning target

Templates become cliques carved out of the notebook and nothing else: they
never learn from an input. Reading uses them, counting updates the notebook,
carving makes them from the counts. The argument was that three of the four
missing housekeeping rules become consequences of the objective.

```
                                 encoder+tally      carve
templates at the cap                       150         82
upper templates by size          {1:31, 2:39..}    {2: 43}
atoms represented alone                   6/11       8/11
ambiguity                                 0.71       1.00
speculates                                0.61       0.38
```

**No wrappers and no duplicates held exactly** — every upper template is a
pair, and the vocabulary stopped growing at 82 for the first time today. Best
atom recovery and best ambiguity of the day. **No blobs did not hold**, and the
failure is in the counting rule, not the carve: counting only the vector that
survives to the top means a thing absorbed into a group is counted as absent,
so its rate collapses while its old co-occurrences stay frozen — and stale
pairs like `{A,CTX2}` *inflate* instead of fading. Speculation suffers because
those junk pairs win the propose step as often as the real ones. The fix is
bookkeeping: a template's window must exclude the inputs where it was absorbed.

Full writeup: [`carve/README.md`](carve/README.md).

---

## `mnist_carve/` — the carve engine on real digits

k-means (k=20) on inked 4x4 patches, one light per patch, ten more lights for
the label, no supervision. 6,000 train, two and a half minutes.

```
1  read the label from what fired      0.155   (30% of images got any label)
3  linear probe on the raw codes       0.902
4  linear probe on which templates     0.727   from 4.2 binary features per image
   fired
```

At 800 templates the vocabulary is three-patch stroke fragments and nothing
above them -- the budget filled 1,000 inputs past the warm-up. At 3,000:

```
templates by depth      {1: 1020,  2: 1456,  3: 448,  4: 65,  5: 11}
labelled by depth       {1: 121,   2: 11,    3: 0,    4: 0,   5: 0}
```

**Five levels, and the depth-4 and depth-5 templates are whole digits** --
grouped from named fragments of named fragments, with no one saying there were
digits. Reconstruction improves (0.072 -> 0.062), the fired-template probe creeps
up (0.727 -> 0.741). But **the label never climbs**: it can only join a clique
locally, with the two or three patches it co-occurs with most, so the template
that recognises a whole 7 has no idea it is a 7. Reading the label stays at
~0.14 (with a test bug -- direct members only -- now fixed but not re-scored).
The run took 15 minutes; next time save the engine and probe fewer images.

Full writeup: [`mnist_carve/README.md`](mnist_carve/README.md).
