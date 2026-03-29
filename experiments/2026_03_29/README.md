# 2026-03-29 — top-1 templates as a memory, and what it can and cannot do

Standalone numpy experiments on synthetic sparse binary vectors with the Zenith
unit: one hypercolumn of templates (a "node" in the scripts), top-1 (only the
winning template learns), templates rotated on the unit sphere. The step to
top-1 is explained in
[`../2026_03_28/topk_label/TopKAndLabelConcatenation.md`](../2026_03_28/topk_label/TopKAndLabelConcatenation.md);
the partial-correlation and inhibition ideas tried just before it are in
[`2026_03_26/cshl/`](../2026_03_26/cshl/). The first two experiments are written up in
[`ensemble_completion/EnsemblePatternCompletion.md`](ensemble_completion/EnsemblePatternCompletion.md);
the other four only have their generated reports.

## `zenith_node/` — the Zenith unit

The rule every experiment below uses: Pearson correlation, only the winning
template learns, rotation on the unit sphere. Kept here as the AxonForge node,
with the random sparse pattern generator that fed it.

Full writeup: [`zenith_node/README.md`](zenith_node/README.md).

---

## `pattern_completion/` — one hypercolumn as a memory

Train one hypercolumn on N sparse patterns, mask part of each, check the right
template still wins. Perfect even at 80% masked when there are few patterns
per template; about 61% at 80% masked when there are two patterns per
template. Sparser patterns hold up better.

Full writeup: [`pattern_completion/README.md`](pattern_completion/README.md).

---

## `ensemble_completion/` — many hypercolumns side by side

M hypercolumns see the same input and their joint code is the memory. 50
hypercolumns of 10 templates give 200 patterns distinct codes; retrieval is
99.61% at 40% masked but 42.22% at 80%. With 40 patterns it is 95.25% at 80%
masked. A lookup memory, not a generaliser.

Full writeup: [`ensemble_completion/README.md`](ensemble_completion/README.md).

---

## `generalization_test/` — category through nuisance bits

Category = shared signal bits plus random nuisance bits. Unmasked, raw
similarity and the joint code both get 100%. Under masking the code pulls
ahead, up to +21.88 points at 80% masked with dense nuisance.

Full writeup: [`generalization_test/README.md`](generalization_test/README.md).

---

## `intersection_test/` — learning AND

Train on [A, B, A AND B], ask for X AND Y on new pairs by looking up the nearest
stored triplet. Raw lookup 0.0718 Jaccard, code lookup 0.0333: lookup cannot
compute AND.

Full writeup: [`intersection_test/README.md`](intersection_test/README.md).

---

## `permutation_relation/` — recognising a hidden permutation

Train on correct pairs [X, P(X)] only; rank the true P(X) of a new X against 15
wrong ones. Direct template score 92.54%, raw lookup 86.38%, lookup on the
normalised code 4.17% (chance 6.25%).

Full writeup: [`permutation_relation/README.md`](permutation_relation/README.md).

---

## `permutation_projection/` — producing P(X)

Read P(Y) out of the reconstruction of [Y, 0]. Direct readout 0.3438 Jaccard
against 0.2261 for memory lookup; a two-stage version through a learned code
collapses to 0.0764. Starting point of the August arc in
[`../2026_08_05/`](../2026_08_05/README.md).

Full writeup: [`permutation_projection/README.md`](permutation_projection/README.md).
