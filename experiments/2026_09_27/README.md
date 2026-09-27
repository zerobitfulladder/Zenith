# 2026-09-27 — search in place of the stack: tables, store, substitute

A morning's conversation and one experiment. Lavender proposed replacing the deep stack with a
search over a fixed palette, learning by distributing values along the path instead of
changing weights. The conversation turned that into a flat test of two of the actions, and
the test went against both on MNIST.

## The conversation, in order

**The proposal.** A network is match and compose. Give it a rich palette to begin with, the
4x4 or 5x5 primitives; let the actions be compose, store and substitute; represent states
fuzzily so that similar states meet; and when the label is found, distribute heuristic values
instead of changing weights.

**What already exists.** The 2026_09_01 layer is the palette, its win counts are a fuzzy
state, and its counted table is a value table: 0.973 with nothing fitted, 0.984 with a fitted
linear map on the same counts. The deep stack's whole contribution on MNIST is that gap. The
question that decides the idea is what makes two states the same. In a network the layers are
the fuzziness, trained so that inputs needing the same answer land in the same place. A
hand-made fuzziness merges what looks alike, but a thin 1, a thick 1 and a slanted 1 must
merge while 4 and 9 must stay apart. Two ways without weights: coarse in place and exact in
identity, which is pooling and which the counts already do; or let store do the merging, so
that two digits built from different pieces share the state "loop over stem". Precedents:
DreamCoder, Lake's Bayesian Program Learning, the 2000s compositional hierarchies of patches.
A value per fuzzy state is a weight that shares nothing: local and additive, so nothing is
erased, but a never-visited state knows nothing.

**Substitution as a chosen merge.** Lavender's answer: the rule for what counts as equal is
not a definition but a choice, made under a goal, taken only when the direct lookup has
nothing, and checked. Thin 1 and thick 1 are not the same token; under the goal "is this a
1" the move "thin it" is worth taking, under another goal it is not. A CNN has one merge for
every question, baked into its pooling. This has a merge per question, and a chosen move can
be reported. Two tables then: token by goal, the fast path; goal by operation, which ways of
looking pay for this goal, the attentive rule. The problem: a free substitution proves
anything, so a move must cost or be checked by the look-back. Copycat's slippages under
pressure, with a cost per slip, are the closest precedent.

**Where the search is.** It lives where the tables have no answer, and it is the only thing
that ever writes a value. States are partial explanations; moves are compose, substitute and
stop, each with a cost; the goal test is a confident value plus a look-back; the value table
is the heuristic; learning is the backup along the path of every finished search, as in
Korf's learning real-time A*. Depth 0 is a lookup, depth 1 one move, depth k a frontier with
backtracking. Store turns a path of five moves into one; SOAR's impasse, search and chunking
is this loop in other words, and Logan's shortcut is chunking. Early on the search is blind,
which is the price of no weights; the number to watch is nodes expanded per image over
training.

**Tokens, states, keys.** A token is a part at a pose. Position matters twice: inside a
token, relative and exact, which is what compose builds; of a token in the state, absolute
and coarse. The state is exact, the key it is looked up by is fuzzy: fine shape gone to the
palette, fine position to a coarse cell, order of composition sorted away, unexplained ink
left out. Substitution is the fourth kind of fuzziness, the chosen kind. A table is a
dictionary: an unseen key returns nothing and a write touches no other row. Four of them:
palette, library of names, values as counts per key and goal, and moves per goal.

**The test.** Flat before sequential. Arm 1 the counted table; arm 2 store, naming
neighbouring pairs and counting again; arm 3 substitute, a menu of moves each goal may take,
against the control that matters most, the same move applied to every image always. The
search proper only if both arms show something. Readings written down before the run.

---

## `store_substitute/` — tables instead of weights: does naming pairs or choosing a move close the gap?

**Neither did.** Full MNIST, 60k/10k, one seed, nothing fitted by gradient.

| | test |
|---|---|
| counted table, 400 templates at stride 2 | 0.9725 |
| store: 300 names a round replacing pairs, best round | 0.9738 |
| store: names beside the parts | 0.9722 |
| substitute: each goal takes the moves that pay for it | 0.9737 |
| always deskew, no choosing | 0.9766 |
| every goal every move, free | 0.9774 |
| CNN, 118k parameters | 0.9908 |

Store compressed the description by a third at no cost and added nothing: a name beside its
parts changed nothing, so the joint count of a pair holds no more than the two counts. The
goal-by-move table did learn the predicted pattern, deskew paying most for 1 and least for 0
and 5, and it did not matter: deskewing every image with no choosing did better than
choosing, and letting every goal take every small move for free did better still, until the
menu grew to forty-five versions and the freedom gave nothing back. MNIST is delivered centred
and sized, so slant is the one variation left, and a fixed move takes it. A chosen merge can
only show on variation without a canonical form. The search was not built.

Full writeup: [`store_substitute/README.md`](store_substitute/README.md).
