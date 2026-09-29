# 2026-09-29 — causes as trees of splats: explain, name, draw, fill

A conversation that started from one term, "fuzzy search", and ended in a design, then one
experiment. Lavender's idea: learn without forgetting by searching for the reason behind an
answer and keeping the path that worked, with the reasons as a tree of causes over a single
kind of atom, the splat. The experiment built the first version on MNIST: whole-digit causes,
each a tree of splats with a typical value and a spread for every number, learned by counting.

## The conversation, in order

**Fuzzy search.** A real term: a lookup that tolerates small differences, also called
approximate matching (typo-tolerant search, the file finder in an editor). Closer names for
what 09-27 described: locality-sensitive hashing, where similar inputs land in the same slot
on purpose; approximate nearest-neighbour search; and state aggregation, or coarse coding, in
reinforcement learning, a value table with deliberately coarse keys, where one visit teaches
nearby states and a state never visited knows nothing.

**Fuzziness inside classical search.** BFS, DFS and A\* differ only in which state they open
next, and a loose match can enter at three places around that choice. The visited check:
Hybrid A\* (Dolgov, Thrun et al. 2008) keeps the car's state exact but keys its visited list
by a coarse grid cell; Iterated Width (Lipovetzky & Geffner 2012) is BFS that drops any state
showing nothing new. The heuristic: pattern databases (Culberson & Schaeffer 1998) store the
exact distance to the goal of a coarsened puzzle and look it up by the coarse key; LRTA\*
(Korf 1990) learns such values during search. Where to grow: RRT (LaValle 1998) grows from
the nearest stored state. A loose visited check can throw away the only path; a loose
heuristic only costs speed, because the goal check stays exact. So: fuzzy for guessing, exact
for deciding. The look-back of 09-27 is the exact goal check.

**Search to stop forgetting.** Lavender's idea: do a task somehow, find the why by search,
then enforce the path that led to the result, in one shot, like changing A\*'s heuristic
values. Each state is a reason for the one below; a cause predicts what else should be
there, so the search goes back down to check it, and a failed check means another cause.
The found reasons are hypotheses: they gain plausibility, never become true, and a failure
provokes search instead of erasing. Once the tree reads a 2, its upper curve is no longer
only a curve but the 2's curve. The operations are shift, thicken, thin and rotate; the one
atom is a splat, with an orientation and a scale; the palette above it is built by the
system, scored by the fewest abstractions that explain the layer below.

The reading. A table entry changes only for the inputs that land on it, which is why the
winner-only layer of 08-31 kept old digits (0.9757 → 0.9526). Forgetting comes back through
the fuzzy match, because generalisation is sharing: a new image uses an old entry only
because the loose key lets both land on it. The cure Lavender named is to split an entry
where it works for some inputs and fails for others, so that forgetting becomes growth.
That is ART's loop, run at every level of a tree. Keeping the path of one found why is
explanation-based learning (Mitchell; DeJong 1986), and its known trap is the utility problem
(Minton 1988): every rule kept makes matching slower, so table size and search steps per
image have to be watched as well as old-task accuracy. The check going back down is what
makes learning local: the step whose prediction failed takes the blame. With one atom the
bottom level needs no search: a splat's centre, direction, length and thickness come
straight from the moments of the ink, and the operations become the splat's pose numbers.
With pose the only identity, a higher cause is a few splats at fixed relative poses, which
is Stacked Capsule Autoencoders (Kosiorek, Sabour, Teh & Hinton 2019). A new palette entry
should be judged by how much search it saves, which the flat table of 09-27 could not test.

**What a state is.** Lavender: the tolerance depends on the cause. A toothpick is still a
toothpick with thicker picks, but not with blunt tips. So each cause keeps, for each of its
parts, a typical pose and a spread per number, and which numbers move together; "is X a
valid Y under Z" becomes "is X inside Z's spread for that part". Pictorial structures
(Fischler & Elschlager 1973) drew this as parts joined by springs of different stiffness;
active shape models (Cootes et al. 1995) learn the ways a shape bends together. In the chain
input → A → B, A explains the ink up to a leftover, and B explains part of A's leftover;
what is left after the top is noise, which is predictive coding. Causes also sit side by
side: the digit, the writer's slant, the pen, each explaining its own share. Circle or tilted
ellipse: keep both, since one answer given two likely ones is false; score each by how
sharply it predicted what was seen, which favours the circle when the ink is round (the
Bayesian Occam's razor, MacKay 1992); and do not drop the weaker early, because another cause
may explain its leftovers (a slanted page turns a circle into a tilted ellipse: explaining
away, Pearl 1988). A leftover that keeps coming back in the same shape is a cause not yet
named.

**One currency.** Lavender: causes are not strictly in layers (X and Y under T, T and Z
under R); causes must be created and removed; if a lower cause is refactored, the higher
cause that used it must keep working; a bad outcome may come from higher causes; and a
part's tolerance under a chain is set by every cause above it. The way through is to price
every choice in one number, the cost of the explanation: a charge for each cause used, how
surprised each cause is by its parts' exact poses, and the ink left unexplained. Then a
tolerance under a chain falls out of scoring the whole tree (a weak curve alone pays for
appearing from nowhere; under a digit it was predicted and costs little), a cause is created
when naming a repeated leftover is cheaper than describing it each time, removed when it
saves less than it costs, and a refactor is accepted only if the total over recent images
falls, which includes the higher causes' explanations. The assumption that makes blame local
is that each part depends only on its own causes; where siblings do depend on each other,
that shows up as a repeated leftover and becomes a new cause. Precedents: Sequitur
(Nevill-Manning & Witten 1997), which names a pair that appears twice and folds back a name
used once; DreamCoder's refactoring (Ellis et al. 2021); Lake's BPL. Two warnings: compressing
is not classifying (09-27's names shortened the description a third and added nothing), and
choosing the answer by how well each option explains the image tends to learn from fewer
examples but top out lower (Ng & Jordan 2001).

**Storage, and the label.** Not a table over poses. Each cause keeps its parts in its own
frame (where it is, how big), which is found fresh in every image, so size and position come
free; only child-to-parent relations are stored, n and not n². Variation has three tools, from
small to large: a spread per number, a few moves-together directions, an OR of separate
causes. Lavender asked to start on MNIST with the whole digit rather than patches: one splat
first, more as training fails, the label part of the input and learned jointly, and at test
"it is definitely a 2 but the label is missing" rather than "no label, so not a 2". The
label is a child of the digit cause with a very tight spread; at test it is missing, not
contradicted, so it costs nothing and is predicted. 03-22 set the missing label row to zero,
which a match reads as a value, and the label weighed 28 cells of 812.

**The algorithm, and drawing.** Explaining an image: fit the root splat to all the ink,
shortlist causes, walk down each cause's tree splitting the ink where the cause splits it,
children placed where the cause expects them and then settled on the ink, which also settles
which part is which. Price the surprise of every node and the ink the leaves fail to draw;
the cheapest cause is the explanation. Learning: a right answer updates the winner by
counting; a wrong one hires a new cause (ART's hire on error); a leaf that keeps leaving
leftover grows; a cause that keeps being picked and keeps being wrong is retired, and one
that is only unused is kept, or learning 5–9 after 0–4 would forget. Given only the label,
the same rows run backwards: pick a cause by its label count, walk down its tree with
typical values or values drawn from the spreads, paint the leaves. With whole-digit causes
that is mostly retrieval, a probe of the table rather than generation; generation starts
when causes share parts, or when sampling moves along a moves-together direction. Lavender:
the tree's depth should not be set, it grows as much as it wants. Then: train it, and test
classification, drawing from the label, and filling in hidden parts.

---

## `splat_causes/` — causes as trees of splats: explain the digit, name it, draw it, fill it in

**It works end to end, and tolerance per cause does most of the naming.** Full MNIST,
50k/10k, one seed, nothing fitted by gradient.

| label hidden | test |
|---|---|
| causes as trees of splats (2,628 causes, 425k numbers) | 0.9593 |
| same table, tree surprise only / pixel term only | 0.8164 / 0.6407 |
| pixels, the same learning rule (Hart's condensed nearest neighbour, 3.4M numbers) | 0.9322 |
| pixels, all 50,000 training images | 0.9666 |
| counted table, 09-27 / CNN, 09-27 | 0.9725 / 0.9908 |
| bottom half hidden / 12x12 box hidden (1,000 images) | 0.841 / 0.862 |

Under the same hire-on-error rule, a tree of splats per cause beat the raw image by 2.7
points with an eighth of the numbers. It did not reach keeping every training image, and it
is further still from the counted table and the CNN. Almost all of the naming comes from
how far the parts moved, measured against the cause's own spreads: the pixel term alone
reads 0.64, since any fitted tree can draw the ink. The trees read part by part: a 7 splits
into bar and stem. Depth was left free and settled at a median of 7 leaves. Drawing from the
label gives a readable typical digit for all ten; drawn from the spreads, strokes come apart
at the joins, because nothing stores what moves together. With the bottom half hidden, the
fill continues the right strokes, faintly. The table keeps growing (47 causes per 1,000
images at the end). The next steps are shared parts, moves-together directions, and the
test of learning digits in sequence.

Full writeup: [`splat_causes/README.md`](splat_causes/README.md).
