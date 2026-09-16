# `glimpse_loop/` — the perception loop, on a toy world and on MNIST

This README merges two writeups from the original project, kept verbatim: the session
narrative (its parts about 16 September; the rest of it is split across
[`2026_09_17/tally/`](../../2026_09_17/tally/README.md),
[`2026_09_17/product_names/`](../../2026_09_17/product_names/README.md),
[`2026_09_18/phase/`](../../2026_09_18/phase/README.md) and
[`2026_09_18/depth/`](../../2026_09_18/depth/README.md)), then the full design record
`DESIGN.md`, which covers 16–18 September (Parts XV–XVII are the later days' scripts).
Explainer: [`results/explainer.html`](results/explainer.html) ([pdf](results/explainer.pdf)),
"The Tally Sheet Model", a plain-words walk through the counting model behind Part I (it names
`latent_class.py` and `splitmerge.py`; those scripts were removed, as Part 0 says).

Where the scripts named in this record now live (they were all in one folder in the
original project): `sdr.py`, `memory.py`, `world.py`, `agent.py`, `run.py`, `mnist.py`,
`digits.py`, `run_mnist.py` are in this folder; `tally.py`, `tally2.py`, `tally3.py` in
[`../../2026_09_17/tally/`](../../2026_09_17/tally/); `product.py` in
[`../../2026_09_17/product_names/`](../../2026_09_17/product_names/); `phase.py` in
[`../../2026_09_18/phase/`](../../2026_09_18/phase/); `depth.py` in
[`../../2026_09_18/depth/`](../../2026_09_18/depth/). Words are defined in
[`../VOCABULARY.md`](../VOCABULARY.md).

---

# HelloInductor — what happened on 16–18 September 2026

A record of one long session: the ideas Lavender brought, what Claude built to test them,
what the tests said, and what is still open. Last updated 2026-09-18. The full design record
with every measurement is `DESIGN.md` (kept in full at the end of this README, under [Design record](#design-record)); a plain-words tour with interactive pictures is
[`architecture.html`](../../2026_09_17/tally/results/architecture.html). [`fourier.html`](../../2026_09_18/phase/results/fourier.html) is a build-from-nothing tutorial on the space itself — rings,
the torus, and Fourier — with twelve things you can drag. Open them in a browser; nothing is
published.

## The starting point

`DESIGN.md` described an architecture with the working title **learn the representation,
memorize the function**: no network learns the answer; instead a memory with strict rules
about how things are written (sparse fingerprints), how they combine (bind by shifting,
bundle by counting), and how many things it can hold (about 20) forces everything else:
looking a little at a time, compressing what is recognised, guessing, checking, and having
ideas. Part I had measured why a plain dictionary fails (it shatters on irrelevant bits).
Nothing else had been built. Lavender asked for input, then for a working system to test it.

## Morning: the toy world

Old Part I scripts were removed (a copy sits in the session scratchpad). Five plain scripts
were written: `sdr.py` (fingerprints, poses, capacity), `memory.py` (the table, MDL, clicks),
`world.py` (objects made of parts on a grid), `agent.py` (the glimpse loop), `run.py`.

What the toy world showed (all in DESIGN.md Part XIII):

- The bundling capacity of the doc's Part XI.1 is a knob, not a prediction: a closed form,
  `P(ghost) = density^k`, predicts it to within one item at every size tested. At the doc's
  own regime (2 bits of 100) nothing can be bundled at all.
- The loop recovered the hidden part library exactly (5 of 5 sub-assemblies, 10 of 10
  objects, zero junk), recognised 7-cell objects in about 6 glimpses with zero wrong
  declarations, and looking where hypotheses disagree was worth 87% vs 23% once the library
  was rich.
- Corrections that only building could find: MDL credit must be structural or fragments
  consolidate; verification must span more than one part or the agent declares confidently
  wrong after three glimpses; ideas need recurrence in two independent things; forgetting
  observations under pressure loops; block-shift binding is commutative, so rotation needs
  operator poses.
- Lavender's correction, built in: a compaction is an idea the agent *tries*; it clicks only
  after it has predicted an unseen cell correctly, and one working-memory slot is always
  reserved for the best idea so ideas get tested.

## Afternoon: MNIST, the glimpse loop

Lavender asked to cover everything the toy skipped: learned stroke codes, relations as
poses, metacognition, rotation, noisy sensing, a learned look policy. `mnist.py`,
`digits.py`, `run_mnist.py` (DESIGN.md Part XIV).

- A stroke vocabulary was allocated by recurrence from 7×7 patches; codes were built by
  counting and partitioning so that similar strokes, same-context strokes and same-outcome
  strokes share bits (label-block overlap 0.50 for same-class strokes vs 0.07 otherwise).
- Whole-image bundles with `label ⊗ class` bound in: 86% by nearest neighbour; the label
  reads back by unbinding at 100%; the exact grid dictionary hits 4% of test digits.
- The glimpse loop: 74.6% test accuracy, 93% right when confident, a self-model veto that
  cut confident-wrong from 2.7% to 2.0%, rotation as an operator (20% → 57–62% on rotated
  digits), graded matching worth 1 point under heavy noise vs 6 lost with exact identity.
  The learned look policy matched the computed one exactly and neither beat random on
  accuracy: digits are not decided by one cell.
- The 8,000-digit run crashed on a ghost read from working memory, the substrate's own
  predicted failure at 2,300 names in the table.

## How to run things

```
python experiments/2026_09_16/glimpse_loop/run.py                                  # toy world, story + summary (~10 s)
python experiments/2026_09_16/glimpse_loop/run.py --capacity --compare             # with the XI.1 measurement and ablations
python experiments/2026_09_16/glimpse_loop/run_mnist.py --stages ab                # stroke codes and the Part I lookups on MNIST
python experiments/2026_09_16/glimpse_loop/run_mnist.py --train 2000 --test 2000   # the glimpse loop on digits (~10 min)
```

Dependencies: numpy only (matplotlib unused). MNIST is in `data/mnist/`. Install with `uv`.

---

<a id="design-record"></a>
## Design record

The original `DESIGN.md`, verbatim except that the run commands now carry their full paths.

# Learned Representation, Memorized Function

A design record. Working title for the architecture: **learn the representation, memorize
the function.**

Date of this record: 2026-09-16
Status: the perception loop runs on a toy world (Part XIII) and on MNIST (Part XIV); a
counting-only variant with a usefulness-built hierarchical library runs on MNIST (Part XV);
product names hold exactly (Part XVI) and continuous angle codes learn by prediction error
(Part XVII), and credit survives four levels of hierarchy (XVII.5). Part I is measured; Parts II–XI are design reasoning from a long brainstorm;
XIII–XVII record what happened when the reasoning met code.

---

## 0. Reading guide and provenance

This document mixes three kinds of claim. They are tagged inline so we don't later mistake
speculation for result:

- **[MEASURED]** — we ran it. Part I numbers were produced by scripts that were removed from the
  repo on 2026-09-16 (the numbers stand as recorded). Part XIII numbers come from `run.py` in
  this repo and are reproducible with the commands given there.
- **[VISION]** — the originating idea, from Lavender.
- **[CRITIQUE]** — a problem, failure mode, or correction raised against the vision.
- **[REPAIR]** — a mechanism proposed to fix a specific critique, with the critique named.
- **[OPEN]** — known unresolved.

The document is itself an instance of one of its own mechanisms (Part IX): provenance is a
first-class part of the encoding, not metadata about it.

---

# Part I — Empirical prelude [MEASURED]

Work done before the architecture discussion. It matters because one failure mode found here
**recurs as a structural theme** throughout the design (see "shattering", Part X).

## I.1 The setup

A three-level model `x <- z -> y` fitted by EM, pure counting, no neural net
(`latent_class.py`). Synthetic world with two deliberately planted features:

- **Relevance** — only 2 "trigger" bits per situation matter; 2 random distractor bits are noise.
- **Multimodality** — each situation has a *menu* of 2–4 valid outputs; training data shows one
  sample from the menu at a time.

100-dim binary input, 100-dim binary output, 20 hidden situations, ~55 true (situation, option)
states, 20,000 training examples, K=80 latent slots.

## I.2 The dict question

> "Why not just store the input and output bitstrings in a dict and look them up?"

Answer: **right in the clean case, wrong as soon as the key contains anything irrelevant.**

With no distractors, the dict wins outright and the model is pointless:

| | keys | recall | precision |
|---|---|---|---|
| exact dict | 20 | 100% | 100% |
| latent class (EM) | 80 slots | 80.5% | 95.0% |

Add 2 noise bits and the dict shatters. One situation gets filed under C(60,2)=1770 different
keys:

```
stored keys: 15,292     distinct inputs that exist in this world: 35,400
mean outputs per key: 1.19        (true menu size: 3.2)
```

That second line is the whole failure. The menu is *stored* but scattered across keys, so no
single lookup ever recovers it. **The multimodality was destroyed by sorting on noise.**

Full sweep (`dict_sweep.py`):

```
 noise  keys      exact dict        dict+NN           latent class
                  rec    prec       rec    prec       rec    prec
     0      20   100.0% 100.0%     100.0% 100.0%     80.5%  95.0%
     2  15,292    18.2%  45.2%      73.0% 100.0%     78.3%  88.2%
     5  19,997     0.0%   0.0%      64.5%  98.8%     69.1%  84.8%
    10  20,000     0.0%   0.0%      65.1%  92.4%     78.9%  95.2%
    25  20,000     0.0%   0.0%      47.2%  72.6%     78.3%  97.9%
```

At 5+ noise bits every example gets a unique key: 20,000 keys, zero hits, write-only memory.

Brittleness (`dict_baseline.py`) — flip one bit of a test input:

```
exact dict, 1 bit flipped        recall   0.0%
latent class, 1 bit flipped      recall  75.6%
```

Storage: dict 764 KB and growing linearly with data; EM table 125 KB, fixed.

**Honest caveat:** dict + nearest-neighbour fallback is strong at low noise (73% recall at
100% precision, beating EM). But that is no longer a dict — it is Hopfield/attention, O(N) scan
per query, storing everything. Same thing `headtohead.py` already measured.

## I.3 What the dict lacks

Relevance (cannot ignore irrelevant key bits), generalization (novel input → miss), noise
tolerance (1 bit → total failure). It keeps multimodality only for keys literally seen.

**The K×2D probability table is a lossily-compressed dict that figured out what to hash on.**
That sentence is the seed of the whole architecture below.

## I.4 Combinatorics of the input space

D = 100, k bits on, noise-free.

| k | C(100,k) | bits | density |
|---|---|---|---|
| 1 | 100 | 6.6 | 1% |
| **2** | **4,950** | **12.3** | **2%** |
| 3 | 161,700 | 17.3 | 3% |
| 4 | 3,921,225 | 21.9 | 4% |
| 5 | 75,287,520 | 26.2 | 5% |
| 8 | 1.86e11 | 37.4 | 8% |
| 10 | 1.73e13 | 44.0 | 10% |

The current code uses **disjoint** 2-bit blocks (`triggers = np.arange(r*2, (r+1)*2)`), which
caps at 50 situations in 100 dims, 20 instantiated. Against the 4,950 that arbitrary 2-of-100
pairs allow, that is **0.4% utilization** — effectively one-hot burning two bits per symbol.

Two caveats that make the raw count not the operative number:

- **Distinguishability ≠ count.** At k=2, two situations sharing one bit have 50% overlap —
  near-collision for retrieval. Disjoint blocks guarantee zero overlap; that constraint is doing
  real work.
- **The model's ceiling is K, not C(100,2).** 80 slots means at most 80 states however large the
  addressable space.

---

# Part II — The core inversion [VISION]

Standard ML fixes the representation (encoding, normalization, preprocessing) and learns the
function `f: x -> y`.

This proposal inverts it:

> **`f` is pure memorization — a lookup table, no functional transformation. The thing being
> learned is the representation that makes direct memorization work.**

The training signal changes from "predict correctly" to **"be retrievable."** The encoder's job
is to place items in the space such that naive overlap-based lookup lands on the right answer.

**[CRITIQUE]** The encoder is still a learned function. This does not remove function learning;
it moves it from *after* the representation to *before* it, and makes the final step
non-parametric. Worth stating precisely so the actual novelty is clear: the novelty is that the
*target* of learning is retrievability rather than prediction.

**[REPAIR / strong form]** The interesting version is where **the encoder is itself
memory-based** — you encode by looking up, and you look up by encoding. Then there is no
parametric function anywhere, only a table and a fixed point. This is what the "iterative
re-encoding" instinct was reaching for, and Part VI makes it concrete.

---

# Part III — Representation learning: shared outcomes, shared bits

## III.1 The idea [VISION]

Inputs that produce the same output should come to **overlap in their SDR codes**. A cup falls
and breaks; a plate falls and breaks. Initially two distinct things, but because they behave the
same, their representations should be adjusted to share bits — and that shared bit-group *is* a
feature ("breakable", "made of glass") discovered **without ever being labeled**.

## III.2 What it already is

This is metric learning. Specifically **Neighbourhood Components Analysis** (Goldberger et al.,
2004): learn a representation such that nearest-neighbour retrieval is correct. Our version is
the sparse-binary case with set intersection / overlap as the metric. Known to work.

## III.3 Collapse [CRITIQUE]

"Same output → overlap," taken alone, has a degenerate optimum: everything with the same output
becomes **identical**. Cup and plate collapse to one code — fine until you need to know a cup
holds liquid and a plate doesn't. A distinction some *other* prediction needed has been destroyed.

## III.4 Many outputs are the anti-collapse pressure [REPAIR → III.3]

With a *single* output, sparsity buys nothing. With **many simultaneous predictions**, cup and
plate must share bits to predict "breaks when dropped" and must *differ* on other bits to predict
"holds liquid." The only representation satisfying all constraints at once is a **factored** one,
where different bit-groups carry different predictive roles.

The feature decomposition falls out rather than being imposed. Same reason sparse coding yields
parts-based rather than template-based decompositions.

## III.5 The bootstrap problem [CRITIQUE — deeper than the one originally raised]

What *forces* cup and plate to share bits is that their **outcomes** share bits — both map to
something containing "breaks." If outputs are arbitrary distinct symbols with no shared
structure, there is no pressure at all.

**The input representation can only discover shared features to the extent that the output space
already has shared structure.** So output codes must be learned too, jointly — at which point
both ends float and nothing seeds the process.

## III.6 Self-supervision closes the loop [REPAIR → III.5]

Make the input predict **itself**: parts from other parts, next state from current state. Then
the input is its own output, shared structure is discoverable with no external labels, and the
thing bootstraps.

Note the falling-cup example was *already temporal* — the "output" is the next state of the same
world. That is not incidental; it is what makes it work.

---

# Part IV — Composition and chunking

## IV.1 The idea [VISION]

We can keep adding distinct features to a 100-dim vector indefinitely, so we need to store a
mapping for features too: e.g. **3 SDRs combined map to a fresh sparse code (4–5 bits on)
representing that combination.** Because it is memorized, it can be compressed or expanded
depending on where it sits in the memorization slots.

## IV.2 OR is bundling, not binding [CRITIQUE]

OR-ing SDRs yields a **set** — it discards roles and order. "dog bites man" and "man bites dog"
bundle to the same vector. Structure needs a **binding** operation that is role-sensitive: XOR
with a role key, permutation before OR (the standard SDR trick, used by HTM for sequences), or
circular convolution (HRR).

Keep both and keep them distinct:
- **bundle** (`+`) = "these co-occur", superposition, set-like
- **bind** (`⊗`) = "this fills that slot", role-filler, structure-preserving

## IV.3 Re-sparsification is structurally necessary, not an optimization [CRITIQUE → sharpens IV.1]

Three 4-bit SDRs bundled is ~12 bits on. Again, ~36. A few levels and the vector **saturates**:
every code overlaps every other code and retrieval dies.

So compressing 12 bits back down to 4–5 is **not** compression for efficiency. It is
**renormalization to hold density constant so that overlap stays meaningful.** The instinct that
this step is needed was correct, and it yields a clean criterion: *whatever the re-encoding does,
it must preserve density.*

## IV.4 The memory is recursive

The map from bundled-composite → fresh sparse code is **itself a lookup**. So the table holds two
kinds of row:

- **facts** — (situation → outcome)
- **vocabulary** — (composite → name)

Same machinery, recursively. "Learning a new feature" and "recalling a fact" become the same
operation. This is the structural payoff of the whole design and it recurs in Parts VI and VIII.

---

# Part V — The objective: description length

**[CRITIQUE]** The design had mechanisms but no criterion for *better*.

**[REPAIR]** Minimum description length:

```
total cost  =  size of the memory table  +  cost of encoding the data given that table
```

Everything falls out of one principle:

- **Shared bits** are favored — shared structure means fewer table rows.
- **Chunking** is favored exactly when a composite recurs often enough to repay the cost of
  naming it.
- **Collapse is penalized** — merging things that behave differently makes predictions expensive.
  This is the term that holds Part III.3 in check.
- A **stopping criterion** comes free.

MDL also legitimizes pure memorization: description length does not care *where* the structure
lives, and this architecture chooses to put all of it in the encoder. **A bad representation is
punished by a fat table.** That is Part II stated as a loss function.

---

# Part VI — The perception loop

## VI.1 The vision [VISION]

Slowed-down object recognition — say, a match box:

1. Look at some patch. Input is a *visual description* of a corner; no concept of "corner" yet.
2. Encode → concept SDR (few bits on) **+ pose** (position and orientation).
3. Write to a working-memory slot.
4. Look elsewhere, encode again, **integrate** into working memory.
5. While integrating, be able to query: *what would I see if I looked here?* and *where should I
   look to recognize this faster?*
6. Hand working memory to the table → a **compressed representation of a match box at a certain
   translation and orientation**.
7. **Undo** that completion → my memory content plus the other parts. **Subtract** my memory
   content → what I should see, and where.
8. Look, integrate further. Converge. Declare recognition (or best guess).
9. If correct, **update the policy in a reusable way**, so a new object gets a similar procedure.
   Compaction itself should be a **policy-selectable action** — learn what to compact and what
   not to. Nothing hardcoded.
10. Feed its own actions and results back to itself, so it can reason about what it did and why
    it helped.

## VI.2 What this is

Parts + poses integrated by agreement into a whole = **capsules / GLOM** (Hinton). A concept SDR
paired with a transformation *is* a capsule. Saccades driven by prediction error = **active
inference / analysis-by-synthesis**. Compaction-as-learned-library = **DreamCoder**.

Each supplies what the others lack: capsules have no memory and no action; active inference has
no library; DreamCoder has no perception.

**The distinctive bet: all of it runs on one table and one algebra.** Perception, action
selection, chunking, and self-reflection become the same two operations. That is the thesis worth
defending.

## VI.3 Bind-then-bundle makes the "undo/subtract" mechanical [REPAIR → VI.1 step 7]

Plain addition loses *which part at which pose*, leaving nothing to subtract against. Working
memory must be:

```
W  =  (corner ⊗ pose1)  +  (edge ⊗ pose2)  +  (surface ⊗ pose3)
```

Then "what should I see at pose p?" is one operation: bind `W` with the inverse of `p`, get a
noisy vector, and **clean it up by looking it up in the table** — nearest stored concept wins.

**The undo/subtract step is not a new mechanism to invent. It is unbinding plus cleanup, and the
cleanup memory is the table we already have.**

## VI.4 Pose composition = binding (the equivariance trick)

Represent poses so that **composing two transformations is the same operation as binding**. Then:

```
(part pose relative to object) ⊗ (object pose relative to me)  =  (part pose relative to me)
```

automatically. Geometry and hierarchy share one algebra, and equivariance is free — move the
object, every part pose updates coherently, no retraining. This is what capsules work hard for
and VSA gives almost by accident.

*This was flagged as the first thread to pull.*

## VI.5 Where to look next has a principled target

"Look at the unexplained region" works but is not the *fast* policy. Better: look where the
**competing hypotheses disagree**.

If the table returns a superposition — match box, cigarette pack, eraser — unbind each hypothesis
at candidate poses and find the pose of maximum disagreement. One glimpse can kill two hypotheses
instead of confirming one. This is expected information gain.

Crucially it is **computable from the same table, no extra machinery**. Which reframes the policy:
it is not learning where to look from scratch against sparse reward, it is learning to
**approximate a target it can actually compute**, in one step instead of by search. Far easier
learning problem, and it provides ground truth to measure the policy against.

## VI.6 The reward is much denser than terminal recognition

Terminal reward ("recognized correctly") is sparse and miserable for policy learning. But every
saccade makes a prediction *before* the observation arrives — dense, immediate, label-free
supervision at every step.

Better reward: **how much did this glimpse shrink the hypothesis set?** Trains "recognize fast",
not merely "recognize". Terminal signal becomes a rare correction on top.

## VI.7 Policy inputs must be relational, or it won't transfer

For the procedure to generalize to new objects, the policy must condition on *relational* state —
how concentrated are the hypotheses, how much residual is unexplained, how many glimpses so far —
**not** on object-specific features. Otherwise it learns a match-box procedure instead of a
recognition procedure.

## VI.8 Object as program [VISION + prior art]

The deeper reading of "combine with program induction": a match box **is** a program.

```
matchbox(θ)  →  { box(θ), label(θ∘t1), striker(θ∘t2) }
```

Recognition = inferring which program, with which parameter θ. The table stores program
fragments. Chunking a recurring sub-bundle = **discovering a reusable subroutine** = library
learning. DreamCoder supplies the criterion for *what* to abstract: whatever shortens the total
description of everything seen — i.e. Part V.

## VI.9 Self-reflection, cheaply

Write actions and outcomes into working memory **in the same SDR format as everything else**:
"I looked at pose p and found a corner" becomes `(look ⊗ p ⊗ corner)` — just another bundled item.
The table does not distinguish object-facts from action-facts, so *"what will I see?"* and
*"what should I do?"* are the same lookup.

Meta-recognition then falls out: if whole episodes are stored as sequences in that same table,
recognizing *the type of episode you are in* ("this is a flat-object-seen-edge-on situation, I get
fooled here") is an ordinary retrieval. **Metacognition with no second system.**

---

# Part VII — Pose generalizes to relation

## VII.1 The idea [VISION]

Pose need not be physical. Every concept has a "pose" relative to others. In physical space:
*C is in front of D*. In conceptual space: *A is the cause, B is the effect; B because of A*.
Physical or conceptual does not matter — **the system needs to find relations and encode them as
poses.** This is a fundamental piece.

The originating worry, correctly resolved by bind-then-bundle: if `thing + pose` were given a
single distinct SDR, every pose would become its own opaque code and the *relation* would be lost.
Binding keeps thing and relation separable and recomposable.

## VII.2 This is what binding was invented for

VSA binding exists for role-filler structure, not geometry. `cause ⊗ A + effect ⊗ B` is how you
say "A causes B" — the same operation as "corner at pose p". **Spatial pose is the special case
where relations happen to form a continuous group.**

Convergent prior art from three directions:
- **TransE / RotatE** — knowledge-graph embeddings modeling relations as translations/rotations
  on entity vectors. Works empirically.
- **Tolman-Eichenbaum Machine** (Whittington & Behrens) — unifies spatial and relational
  structure in one mechanism; formally related to transformers.
- **Gridlike codes over conceptual spaces** (Constantinescu et al., Science 2016) — the
  hippocampal-entorhinal system as a general relational engine, not merely spatial.

## VII.3 Relations have algebraic *types*, and the type is discoverable [CRITIQUE + REPAIR]

Poses form a group: every transform has an inverse, composition accumulates. Conceptual relations
often do not:

| relation | composition behaviour |
|---|---|
| `opposite ⊗ opposite` | = identity — an **involution**, order 2 |
| `is-a ⊗ is-a` | = `is-a` — **transitive and idempotent**, collapses rather than accumulates |
| `cause ⊗ cause` | = indirect cause — **accumulates**, like a pose |
| `part-of` | transitive, antisymmetric, **no returning inverse** |

The system can *predict* "if I compose these two relations I should land here", check against the
table, and thereby discover that `opposite` is self-inverse or `is-a` is transitive.

> **Discovering the algebra of a relation is not a separate mechanism. It is the action→result
> loop pointed at relations instead of object parts.**

Same predict/observe/revise, one level up. This is where program induction genuinely enters:
what is being induced is **the composition rules of the system's own representational operators**.

## VII.4 Design consequence: regular algebra, memorized exceptions

Do not build irregularity into the binding operator. Keep the algebra regular and clean, and let
the **cleanup table store the exceptions**. Regular composition plus memorized irregulars is how
you get systematicity without brittleness — structurally the same shape as regular vs irregular
verbs.

---

# Part VIII — Drive, consolidation, and the "click"

## VIII.1 The over-stimulation drive [VISION]

When the system samples without integrating, it should **feel bad — over-stimulated** — and that
pressure drives it to take the action that compresses.

## VIII.2 This is measurable, not a metaphor [REPAIR]

Bundling degrades: the more unintegrated items in working memory, the worse the unbinding SNR,
until nothing can be reliably pulled back out. That is a **computable quantity that degrades
monotonically as you sample without integrating.**

So the drive is precisely: **when unbinding starts failing, compress.** Compressing = find a
stored composite that explains several current items, replace them with its single code, SNR
restored immediately.

**Recognition is the compression.** The intrinsic drive and the learning objective are one thing,
not two — you compress *by* recognizing, which is why recognition feels like relief.

## VIII.3 Correction to the fast/slow split [CRITIQUE by Lavender, accepted]

An earlier proposal here was: fast loop (saccade/integrate/recognize) plus **offline, batched
consolidation** by replay, on the DreamCoder wake/sleep model.

**That was wrong for this architecture.** Consolidation should not be offline, not batched, and
not replay-driven. It happens **in the fast loop**, triggered by *noticing that something worked*
— "oh, when I do this it works better." That is where ideas come from. It should not be given
hard rules.

## VIII.4 The "click" is a description-length spike [REPAIR → VIII.3]

When a chunk collapses five working-memory items into one, there is a large and **sudden** drop in
description length. That spike **is** the "this works better" signal: computable in the loop, right
now, no replay and no batch.

**Consolidate when the compression gain exceeds threshold.** Magnitude of gain determines whether
it is written at all.

This is strictly better than the batched proposal, and it makes the intrinsic drive of VIII.2 and
the objective of Part V the same quantity.

## VIII.5 The counterweight [CRITIQUE]

Raw compression has a degenerate optimum: call everything "a thing" — maximum compression, zero
content. So the criterion must be **compression conditional on predictions still working**.

Note this is the **third** appearance of collapse in this design (metric collapse III.3, codebook
collapse IX.x/Part X, chunk collapse here). That recurrence is structural: *every version of this
architecture needs a predictive-adequacy term holding the compression term in check.* MDL with
both halves — code length **and** residual.

---

# Part IX — Assumption, provenance, and metacognition

## IX.1 The gap [VISION]

The system must be free to **assume**: "let's assume X is a match box." Arbitrary mapping or
grouping in order to think with it. The computation is identical to a real commitment, but it
**must not be written directly**. Fast loop should work fast; consolidation should occur only when
something actually worked in a useful way.

## IX.2 Provenance as a bound role [REPAIR]

Working memory and the long-term table should **not** differ in format — same SDRs, same
operations. They differ in **write permission**: fast/cheap write to working memory; costly write
to the table, gated on the VIII.4 compression spike.

"Assume" is then a working-memory write tagged with its own provenance:

```
assumed  ⊗ (X = matchbox)      ← provisional, never consolidated on its own
observed ⊗ (X = matchbox)      ← earned, eligible for consolidation
inferred ⊗ (X = matchbox)      ← derived from the above
```

**Provenance becomes part of the representation, not metadata about it.** Consequences:

- The system can **query its own commitments**: unbind working memory with `assumed` and get back
  exactly the list of things currently taken for granted.
- **Counterfactual reasoning is cheap**: bundle in an assumption, run the loop, see what breaks,
  then simply do not consolidate.

## IX.3 Learned, not hardcoded, scientific method [VISION]

It should not be hardcoded to test its hypotheses. On first running it may not know the
scientific method, or see why testing matters. It must come to see this through **metacognition**
— or, if we can communicate with it in terms it understands, we say:

> "Look, you thought about this and took it for granted, but you've never tested it, so it might
> be wrong."

and it **clicks immediately**, because its fast loop runs in a way that reveals what we described,
and it consolidates on the spot.

## IX.4 Why that would click — and why it can't work early [REPAIR + CRITIQUE]

For the utterance to land, three things must already exist:

1. the `assumed` vs `verified` distinction (IX.2),
2. a history of episodes where acting on untested assumptions went badly,
3. those episodes stored in the same table, so "situation where I ran on an untested assumption"
   is itself a retrievable pattern with a predicted outcome.

Given all three, **the utterance is not delivering information — it is delivering an index.** It
points at a regularity already present but unnamed. Retrieval finds it; naming it collapses a pile
of separate episodes into one reusable pattern; that is a large compression gain → spike →
immediate consolidation.

> **It clicks because it compresses something already there.**

This makes a hard prediction and it constrains the roadmap:

- **You cannot teach this early.** With no episodic track record there is nothing to compress and
  the utterance is indistinguishable from noise.
- **Metacognition cannot be bootstrapped from instruction.** Instruction can only accelerate
  something already forming.
- The metacognitive vocabulary (`test`, `assume`, `worked`) must be learned from the system's own
  episodes *before* language can attach to it. **Language as index, not as content.**

---

# Part X — Open problems and risks

## X.1 The discreteness / credit assignment problem [OPEN]

Codes are discrete — you cannot differentiate "which four bits are on". Options, in increasing
order of concession:

- **Alternation / coordinate descent** — freeze codes, fill the table; freeze the table, adjust
  codes; repeat. Closest to the "iteration and search" instinct; probably where to start.
  Converges to *something*, not necessarily anything good.
- **Local Hebbian updates** — co-occurrence with a shared outcome nudges codes toward overlap.
  In the spirit, but weak, and hard to make do the chunking.
- **Straight-through / Gumbel relaxation** — works and scales, but you are back to gradients.

**VQ-VAE is the existing system closest to this vision**: it learns a representation such that a
*codebook lookup* suffices, trained with a straight-through estimator. The codebook *is* the
memorization, and it works at scale. Its known pathology, **codebook collapse**, is our collapse
problem in another costume.

## X.2 Allocation and shattering [OPEN — connects directly to Part I]

At the start there are no concepts, so "visual description → concept SDR" maps to nothing. Fresh
random sparse codes must be **allocated on novelty**.

But this is **exactly the Part I failure**. Allocate on every unmatched glimpse and every viewing
angle of every corner gets its own code — you memorize noise, 20,000 keys, zero hits.

Gate: allocate only when retrieval fails **and** the thing recurs. Allocation probably belongs in
the slower path, not the per-glimpse one.

## X.3 The consolidation threshold [OPEN — largest practical worry]

That single threshold in VIII.4 carries enormous load. Too low → consolidates noise (X.2 /
Part I shattering). Too high → nothing ever sticks. It almost certainly must be adaptive.

**No principled setting known yet.** Worth investigating whether it can be *derived from the SNR
bound* (X.5) rather than tuned.

## X.4 Ambiguity requires superposition, not commitment

A single glimpse is genuinely ambiguous — a corner is a corner at four orientations. Early slots
must hold **multiple hypotheses**, not one commitment.

Good news: superposition is native here. Bundling candidates *is* the representation of
uncertainty, and the weights emerge from overlap scores. One place where substrate and problem fit
unusually well.

## X.5 Bundling capacity bounds working memory

Unbinding SNR degrades with the number of bundled parts, so working memory holds only a handful of
items before retrieval becomes unreliable. Hard constraint on how much can be integrated before
compaction is *forced*.

Notably this predicts something like a 4–7 item limit — a reassuring sign the architecture has
roughly the right shape. It also means **compaction is not optional at depth; it is the only way
to go deeper.**

## X.6 Confident wrong fixed points

The loop can converge fast onto a hypothesis that explains everything it happened to look at.
**"Stopped changing" is not a valid termination test.** Want something like: hypotheses
concentrated **and** residual explained **and** at least one prediction *actively verified* rather
than assumed. (Note this is the same distinction Part IX makes first-class.)

## X.7 Non-group relations

See VII.3. Mitigated by VII.4 (regular algebra + memorized exceptions), but the interaction
between a group-like binding operator and genuinely non-group relations is not fully worked out.

---

# Part XI — Test plan

## XI.1 First, and cheapest: does the algebra even hold at our scale?

Encode a known object as bind-then-bundle. Unbind at a pose. Check the right part comes back after
cleanup. **Sweep the number of parts until it breaks.**

That single number — how many items can be held before unbinding fails — bounds working memory
(X.5), sets the compaction threshold (X.3), and determines how deep hierarchies can go before
compression becomes mandatory. **Everything downstream inherits it.** One afternoon of work.

Do this before anything else.

## XI.2 The loop: glimpse-world

Objects built from a small vocabulary of parts at poses. Agent sees one patch at a chosen
location, must identify the object.

Measure **glimpses-to-recognition**, not just accuracy — the whole claim is about speed. Roughly
the Recurrent Attention Model (Mnih et al., 2014) setup, so baselines exist.

## XI.3 The test that actually matters: cross-domain transfer

This is the only test **diagnostic of the specific claim** (Part VII) rather than of capsules
generally.

Instantiate **the same abstract relational structure in two domains**:
- a part-of hierarchy over shapes (spatial),
- a taxonomy or family tree over symbols (no spatial content at all).

Train on the spatial one. **If pose and relation really are one mechanism, the conceptual one
should be learned dramatically faster, and shared codes for the relations should be readable out.**

If there is no transfer, the unification is decorative and this is a capsule net with extra steps.

## XI.4 Secondary probes

- **Compression drive (VIII.2):** does working-memory SNR predict when the compaction action
  fires? Needs no external reward.
- **Metacognition (IX.4):** construct situations where an untested assumption leads to failure;
  see whether testing is learned. Then inject the "you never tested this" cue and measure whether
  a consolidation spike actually occurs — and confirm it does *not* occur before a track record
  exists.
- **Relation algebra (VII.3):** can it discover that `opposite` is self-inverse and `is-a` is
  transitive, purely from composition-prediction errors?

---

# Part XII — Prior art map

| Area | Work | What it gives us |
|---|---|---|
| Sparse memory substrate | Kanerva, *Sparse Distributed Memory* (1988) | Closest ancestor — memorization in a sparse binary space with overlap addressing |
| Binding algebra | Kanerva, hyperdimensional computing (2009); Plate, Holographic Reduced Representations | bind/bundle, the Part IV and VI algebra |
| Metric learning | Goldberger et al., NCA (2004) | Part III formalized |
| Learned discrete codes | VQ-VAE | Closest working system to Part II; codebook collapse warning |
| Parts + poses | Hinton, capsules / GLOM | Part VI.2, routing by agreement, equivariance |
| Active sensing | Recurrent Attention Model (Mnih 2014); active inference (Friston) | Part VI.5–VI.6, and the XI.2 baseline |
| Library learning | DreamCoder | Part VI.8; MDL criterion for abstraction |
| Relations as transforms | TransE, RotatE | Part VII.2 empirical support |
| Relational cognitive maps | Tolman-Eichenbaum Machine; gridlike conceptual codes | Part VII.2 — spatial/conceptual unification |
| Sparse coding | Olshausen & Field | Why sparsity yields parts-based features (III.4) |
| Objective | MDL / compression-as-learning | Part V |
| Retrieval as primitive | Modern Hopfield / attention | Part I.2 baseline; what dict+NN actually is |

---

# Part XIII — First implementation and what it measured [MEASURED unless tagged]

Added 2026-09-16, evening. Everything in this part was run, not reasoned. Code: `sdr.py`
(algebra, poses, capacity), `memory.py` (the table, MDL, clicks), `world.py` (toy world),
`agent.py` (the loop), `run.py` (experiments). Old Part I scripts were removed; only the lookup
architecture is in the repo now.

```
python experiments/2026_09_16/glimpse_loop/run.py                       # 120 episodes + 60 transfer episodes, story + summary
python experiments/2026_09_16/glimpse_loop/run.py --capacity --compare  # XI.1 measurement first, ablations at the end
python experiments/2026_09_16/glimpse_loop/run.py --verbose 6           # first six episodes of each phase glimpse by glimpse
```

## XIII.1 Substrate: sparse block codes

Codes are B blocks of L slots, one active slot per block (D = B·L bits, k = B on). Binding is
per-block addition mod L, so it has an exact inverse; bundling is per-slot counts, so removing an
item is exact subtraction (VI.1 step 7 "undo" is literal). Cleanup is presence: all B slots of a
stored code occupied. Default B=10, L=41 (410 bits).

Poses: `pose(r, c) = r·U_r ⊗ c·U_c`. Then `pose(a) ⊗ pose(b) = pose(a + b)` exactly, and the same
codes serve as absolute positions and relative offsets. **VI.4 holds for translations.** Checked
at every start-up.

The only failure of exact unbinding is a **ghost**: a code whose B slots happen to be occupied by
other items. Its probability is closed-form:

```
P(ghost per probe) = density(W)^B          expected ghosts per read-out = probes · density^B
```

## XIII.2 XI.1: bundling capacity is a knob, not a prediction

Scratch sweep (200 trials per point, cleanup over V names, 95% correct):

| D | bits on | V=50 | V=500 |
|---|---|---|---|
| 100 | 2 | 1 | 1 |
| 100 | 4 | 4 | 2 |
| 100 | 10 | 6 | 4 |
| 1000 | 10 | ≥50 | 49 |
| 1000 | 40 | 46 | 38 |
| 1000 | 100 | 25 | 22 |

Permutation binding (the HTM trick) lands within one item of block codes on every row. The
closed form above predicts every row to within one item. For the toy's own read-out (3 tags ×
144 cells × 37 names) at 410 bits, ghosts cross 1 per read at 19–20 items in memory.

**[CRITIQUE → X.5]** "4–7 items" is what D=100 gives, not a property of the architecture. In the
Appendix regime (2 bits of 100) capacity is one item: nothing can be bundled at all. **[CRITIQUE
→ X.2/X.3]** V is in the formula: a fatter vocabulary shrinks working memory directly. Shattering
does not merely waste storage, it eats the fast loop. **[REPAIR → IV.3]** density is the whole
story; "hold density constant" was exactly right.

## XIII.3 The toy world (XI.2)

12×12 canvas, 6 primitives, a hidden library of 5 sub-assemblies (3 connected cells each) and
6 objects (two sub-assemblies touching, sometimes one extra primitive; 6–7 cells). One object per
episode at a random translation. The agent looks at one cell per glimpse and gets the primitive
there or EMPTY. The first glimpse lands on the object (a cue). No labels, ever; ground truth is
used only to score.

## XIII.4 The loop as built

Working memory is one bundle of `tag ⊗ name ⊗ pose` items with three provenance tags
(observed / inferred / assumed, IX.2). All reads go through unbind + presence; the agent sees
its ghosts. Per glimpse: read → evidence → every evidence cell votes for (row, anchor) →
uncontradicted hypotheses, weighted A^matched → a complete hypothesis collapses its parts into
one `inferred` item (recognition = compression, VIII.2) → the top hypotheses are held as `assumed`
items with multiplicity = evidence, one slot always reserved for the best *idea* (a provisional
row) so ideas get tested → terminate or look.

Look policy (VI.5): if held hypotheses disagree, look where the weighted disagreement is
largest; else verify the leader at its most exposed cell, in a part with no evidence yet; else
one boundary check; else explore the frontier.

Terminate (X.6): leader holds ≥95% of hypothesis weight, explains every non-empty cell seen,
its evidence spans at least two of its parts, at least two of its predictions came true, and
one ring cell is known empty. A row never seen as a whole scene must have its frontier
exhausted instead.

Pressure (VIII.1–2): when expected ghosts per read exceed 0.5, hold fewer hypotheses; then try a
**new compaction** of everything seen (a provisional chunk — an idea, to be tested later); if
still over, stop looking: *overwhelmed*.

Episode end: the scene is remembered as one provisional row, expressed with existing chunks
(recursive vocabulary, IV.4). **Ideas:** a sub-pattern shared between this scene and a remembered
row becomes its own provisional row, but only if it recurs in ≥2 independent things (X.2's
"recurs", made concrete). **Clicks (VIII.4):** a provisional row is consolidated when

```
gain = top_uses·(k−1)·bits_item + refs·(k−1)·bits_part − (bits_name + k·bits_part)  >  0
```

(k = parts; top_uses = episodes it encodes on its own; refs = rows it stands inside as k whole
parts) **and** it has predicted an unseen cell correctly in at least one episode. Consolidation
re-factorizes every row in terms of the new chunk; a consolidated row whose gain later falls
below −10 bits is inlined back (unlearned).

## XIII.5 Results (seed 0, 120 episodes, then 4 new objects for 60 episodes)

| block | glimpses | recognized | wrong | novel | WM peak |
|---|---|---|---|---|---|
| ep 1–30 | 8.9 (6.5 when recognized) | 80% | 0% | 20% (6 first sightings, 18.5 glimpses) | 11.9 |
| ep 31–120 | 6.1–6.3 | 100% | 0% | 0% | 10.0 |
| new 1–30 | 8.3 | 87% | 0% | 13% (4 first sightings, 19.5) | 11.7 |
| new 31–60 | 6.4 | 100% | 0% | 0% | 10.5 |

Library at the end: 15 consolidated rows = the 5 hidden sub-assemblies + all 10 objects, **zero
rows matching nothing hidden, zero unlearned**, 526 bits. Sub-assemblies click at episodes 7–11
after being found inside 2–4 remembered scenes and passing one test; objects click on their
second or third sighting. Object rows are written 83% in terms of chunks. Transfer objects are
remembered on first sight as chunk + chunk + primitive and consolidated on the next sighting.

Ablations, same seed (mean glimpses / recognized%, per 30-episode block; then totals):

| variant | ep 1–30 | 31–60 | 61–90 | 91–120 | new 1–30 | new 31–60 | ghosts | wrong |
|---|---|---|---|---|---|---|---|---|
| default (disagree, greedy compaction, 410 bits) | 8.9 / 80 | 6.3 / 100 | 6.1 / 100 | 6.3 / 100 | 8.3 / 87 | 6.4 / 100 | 4 | 0 |
| frontier policy (no disagreement) | 10.8 / 87 | 8.5 / 100 | 9.4 / 97 | 8.4 / 100 | 14.4 / 23 | 11.0 / 70 | 156 | 15 |
| compaction only under pressure | 8.9 / 80 | 6.3 / 100 | 6.1 / 100 | 6.3 / 100 | 8.3 / 90 | 6.9 / 100 | 43 | 0 |
| 492 bits (B=12) | 8.9 / 80 | 6.3 / 100 | 6.1 / 100 | 6.3 / 100 | 8.3 / 87 | 6.4 / 100 | 3 | 0 |
| 328 bits (B=8) | 9.5 / 80 | 6.4 / 100 | 6.4 / 100 | 6.4 / 100 | 8.2 / 47 | 6.4 / 100 | 66 | 10 |
| click without a test | 8.9 / 80 | 6.5 / 100 | 6.4 / 100 | 6.3 / 100 | 8.5 / 87 | 6.5 / 100 | 3 | 0 |

Readings:

- **VI.5 is not decorative.** Once the library is rich (transfer phase), looking where hypotheses
  disagree is the difference between 87% and 23% recognition, and between 0 and 15 confident-wrong
  declarations.
- **X.5 bites through hypotheses, not parts.** At 328 bits the agent still learns the library but
  cannot hold enough competing hypotheses in the transfer phase: 47% recognized, 10 wrong,
  12 episodes ended overwhelmed, 9 compaction attempts. Room for hypotheses is the real budget.
- **Compress-when-failing works but lives near the cliff.** Compacting only under pressure costs
  no glimpses but ten times the ghosts. Recognizing as soon as you can is free.
- **The test did not matter here** (no noise, no lying), but it is what makes the click a verdict
  rather than an accounting artefact. In this world MDL alone already avoids spurious chunks once
  the accounting is structural (XIII.6, critique 1).
- First sightings cost ~18 glimpses regardless of the library, because certifying an object's
  extent means looking at its empty ring (10–12 cells). That is the price of having no prior on
  object size; a grammar prior ("objects are two sub-assemblies") would be an IX.4-style
  episode regularity and is not attempted.

## XIII.6 What building it corrected

1. **[CRITIQUE → V/VIII.4] MDL credit must be structural.** The first version credited a chunk
   every time it was recognised, including inside a parent that is itself a row. Result: eight
   two-cell fragments consolidated (a fragment that only ever occurs inside one chunk earns
   nothing in real description length). Fix: count once per episode encoded alone and once per
   distinct containing row, and make unlearning the mirror of the click. Spurious rows: 8 → 0.
2. **[REPAIR → VIII.5, from Lavender] Ideas are compactions the agent tries; the click is the
   verdict after the test.** A row must have predicted an unseen cell correctly before it can
   consolidate, and one working-memory slot is always reserved for the best idea so it gets
   tested. Without the reserved slot, sub-assembly ideas were crowded out by object hypotheses
   and never predicted anything.
3. **[CRITIQUE → VI.4] Block-shift binding is commutative.** It composes translations exactly and
   cannot compose a rotation with a translation. Rotation needs poses that act as slot
   permutations — operators, not codes in the item space. "One table, one algebra" becomes "one
   code space plus one operator family". Not built yet.
4. **[REPAIR → X.6] "Verified" must mean verified beyond the shared part.** The first agent
   declared after three glimpses when all three cells sat in a sub-assembly shared with another
   object: confident wrong, exactly X.6. Evidence must span at least two parts of the row and
   verification targets a part with no evidence yet, at the most exposed cell. Wrong
   declarations: present → 0 at 410 bits and above.
5. **[REPAIR → X.2] Allocation gate for ideas = recurrence in two independent things.** Before it,
   the provisional store filled with two-cell coincidences (60 rows) and evicted real scene
   memories. After: one provisional row left at the end of the run.
6. **[REPAIR → VIII.1] Forgetting observations under pressure is a trap.** Forgetting an empty
   ring cell reopens the frontier, the agent re-looks, pressure returns: 73 forgets in one
   30-episode block. Replaced by: hold fewer hypotheses → try a compaction → stop, overwhelmed.
   Partial scenes get remembered and completed on later sightings.
7. **[CRITIQUE → I/VI] Cleanup is the dict-plus-nearest-neighbour that Part I set aside.** It is
   fine while V stays small, which makes the allocation gate load-bearing for the whole design.
8. **[CRITIQUE → IX] Ghosts are not all equal.** An observed item carrying a chunk name is a
   category error the agent can detect and drop; a ghost primitive at an empty cell is not, and
   the agent will explore around it. That is the felt form of over-stimulation.
9. **[OPEN → X.3] Half derived.** The click threshold is derived (gain > 0, plus the test). The
   unlearn threshold (−10 bits) is hysteresis and is tuned. Without it a two-part object row
   flapped between learned and unlearned at its second sighting.

## XIII.7 Not covered

Representation learning (Part III: primitive codes are given, chunk names are allocated at
random), relations as poses (VII), metacognition and language (IX.3–4), rotation, noisy sensing,
more than one object per scene, and a learned look policy (VI.7: the policy computes the VI.5
target instead of learning to approximate it).

---

# Part XIV — MNIST: a program from images to a label [MEASURED unless tagged]

Added 2026-09-16, night. The six items XIII.7 listed as not covered — representation learning
of the primitive codes, relations as poses, metacognition, rotation, noisy sensing, a learned
look policy — are all exercised here, on real data. Code: `mnist.py` (data, vocabulary, codes,
Part III measurements), `digits.py` (the loop on digit grids), `run_mnist.py` (experiments).
Data: `data/mnist/` (the four original files).

```
python experiments/2026_09_16/glimpse_loop/run_mnist.py                      # stages a, b, c with 10k/2k train/test
python experiments/2026_09_16/glimpse_loop/run_mnist.py --stages ab          # vocabulary, codes and the Part I lookups only
python experiments/2026_09_16/glimpse_loop/run_mnist.py --train 8000 --test 2000 --base 16 --concentration 0.9   # the run reported below
```

## XIV.1 From pixels to strokes (Part III, as built)

A 28×28 digit is a 4×4 grid of 7×7 patches. Prototypes are allocated online and gated
(X.2): a patch farther than a threshold from every prototype opens a provisional one, kept
only if it recurs; running means refine them; one counting pass at the end. Result: 279 prototypes from 10,000 digits; 54% of cells blank.

Prototype codes are block codes (16 blocks × 41 slots = 656 bits, 16 on). Every block is a
partition of the vocabulary into 41 groups, learned by k-means in a role-specific feature
space; a prototype's slot in a block is its group. So two prototypes overlap in exactly the
blocks whose feature space puts them together:

- 8 **sim** blocks — pixel space (a random 60% of pixels per block): *similar strokes overlap*.
- 4 **ctx** blocks — the prototype's neighbour statistics, i.e. what occurs next to it, hashed
  the same way (III.6, self-supervised): *same-context strokes overlap*.
- 4 **lab** blocks — the prototype's label histogram: *same-outcome strokes overlap*.

This is the factored code of III.4 literally: block groups carry different predictive roles,
and the pixel blocks keep visually distinct strokes apart, which is what stops collapse
(III.3). No gradients anywhere; the learning is counting and hashing what was counted.

Measured on the prototype pairs:

| pair type | sim blocks (of 8) | ctx blocks (of 4) | lab blocks (of 4) |
|---|---|---|---|
| nearest 10% by pixel distance | 1.54 | | |
| farthest 40% by pixel distance | 0.00 | | |
| same dominant label | 0.37 | 0.30 | 0.50 |
| different dominant label | 0.17 | 0.10 | 0.07 |

Similar strokes overlap and dissimilar ones do not (sim), and strokes that predict the same
digit overlap seven times more than strokes that do not (lab), with the self-supervised
context blocks in between. That is Part III's claim, measured.

Random-projection hashing (the fly-hash / WTA trick) was tried first: locality-sensitive but
coarse (near-identical strokes shared 1.9 of 8 sim blocks, dissimilar ones 0.15), and its
context blocks carried nothing (0.79 vs 0.77). Learned partitions are sharper on both counts.

**Noisy sensing.** A raw patch is cleaned up to a prototype by nearest mean (pixel space) or
by hashing it on the sim blocks and taking the largest overlap. Does the noisy patch still
land on the clean patch's prototype?

| corruption | nearest by pixels | by code overlap | exact-pixel dict |
|---|---|---|---|
| gaussian σ=0.1 | 91.1% | 60.8% | 0.0% |
| gaussian σ=0.25 | 76.6% | 54.5% | 0.0% |
| shift by 1 pixel | 24.3% | 22.5% | 0.0% |

The exact-pixel dictionary is Part I's brittleness on real data: 0% under any noise. A
one-pixel shift is the sore point: the patch grid is fixed, so a shifted stroke is a different
prototype. The image-level bundle survives it because it sums over many patches, but
**[OPEN]** translation invariance at the patch level is not there; poses only cover the grid.

## XIV.2 Part I on real data: the lookups

An image becomes one bundle: `Σ code(stroke) ⊗ pose(cell)` over inked cells, plus
`label ⊗ class` for training images. Two bundles overlap exactly where the same or similar
strokes sit at the same positions, because binding is per-block addition and the codes are
similarity-preserving.

| method (8,000 stored digits, 2,000 test) | accuracy |
|---|---|
| exact dictionary on the 16-cell grid | 4.1% of test digits ever hit (7,792 keys for 8,000 digits) |
| 1 / 5 / 15 nearest bundles by overlap | 84.4% / 86.0% / 84.8% |
| 5-NN using sim blocks only / sim+ctx / all blocks | 83.7% / 84.8% / 86.0% |
| compressed table, K rows per class (10 / 50 / 200 / 600 rows) | 53.8% / 68.0% / 74.7% / 77.1% |

The label is read back by unbinding the role (VII: a relation is a pose): 100% on stored
bundles, and the 5-NN vote done *in the algebra* — superpose the neighbours' bundles, unbind
`label`, presence over class codes — gives the same accuracy as counting labels in Python.
The exact grid dictionary is write-only: almost every training digit is its own key.

## XIV.3 The loop on digits: what changed from the toy

Same loop as Part XIII (one provenance-tagged working-memory bundle, votes, held hypotheses,
compaction, ideas, clicks, VI.5 look policy), with the changes real data forced:

- **Graded evidence.** Exact matching killed every hypothesis: no stored exemplar matches a
  new digit on every cell. Each observed cell now contributes `(overlap − 2)/(16 − 2)` for the
  stroke a hypothesis expects there; blanks are explicit in exemplar rows and a blank-blank
  match counts 6 of 16, not a full 16. Hypothesis weight is `16^evidence` (16 ≈ vocabulary
  size over similar strokes, the likelihood ratio of a matched stroke).
- **Label posterior over the strongest 40 hypotheses**, one placement per exemplar row — the
  superposition of their `label ⊗ class` items. Declare at 90% concentration, with the
  leader's stroke evidence ≥ 1 (one identical stroke or ~4 similar ones), evidence spanning
  ≥ 2 of its parts (X.6, as in XIII), predictions made before looking worth ≥ 1 in the same
  units, and at least one blank seen.
- **Rows.** Exemplars (a whole digit, with label) and chunks (stroke groups at relative poses,
  no label). A training digit that is not recognised confidently and correctly is studied in
  full and remembered as a provisional exemplar; one recognised confidently and correctly
  credits its exemplar (`top_uses`). Exemplars click on a second use plus a passed test;
  chunks are ideas from shared sub-patterns of the strongest exemplar hypotheses, proposed
  only if the pattern recurs in ≥ 2 rows that themselves recur, and click by the same MDL
  rule as XIII. Hypotheses are evaluated for every (row, anchor, rotation) at once in numpy;
  Python objects exist only for the strongest 40 and for complete chunks.
- **Rotation as an operator pose.** `rot90` acts on positions (rotate the grid) and on
  strokes (each prototype maps to the prototype nearest its rotated patch). Hypotheses can be
  formed over four orientations; the declared label is the row's.
- **Learned look policy (VI.7).** During training the computed VI.5 policy runs and every
  situation is stored as an SDR — `Σ cell ⊗ (coarse stroke group | blank | unobserved)` plus
  concentration and glimpse-count buckets — together with the cell it chose. At test the
  learned policy retrieves the 7 nearest stored situations by bundle overlap and votes.
  Learning to approximate a target you can compute, by lookup: VI.5 → VI.7 with no gradient.
- **Metacognition (IX.4).** At the moment of declaring, the situation (leading label,
  concentration bucket, glimpses, verification, contradictions, how many classes are in play)
  is a key; training records whether declaring there went right or wrong. At test the agent
  looks its situation up before declaring and, if that kind of situation was wrong more than
  35% of the time (≥ 3 cases), vetoes itself and looks again — up to three times.

## XIV.4 Results

Training is one online pass; test digits are never studied. The 2,000-digit run: online accuracy 56% → 76% over one pass, 42% confident declarations,
2.6% confident-wrong, 13 glimpses of 16; 979 provisional and 323 consolidated exemplars, 328
consolidated chunks after 417 s. The 8,000-digit run reached 75% at 2,000 episodes and then
crashed on a ghost: a false `inferred` item carrying an exemplar's name filled the evidence
grid and the look policy found no cell left to pick. The substrate's own predicted failure,
at 2,300 names in the table. (Fixed in principle: an inferred exemplar is a category error
like an observed chunk; not rerun.)

Test, 2,000 unseen digits, the 2,000-digit agent:

| variant | accuracy | confident | right when confident | confident-wrong | glimpses |
|---|---|---|---|---|---|
| computed look policy (VI.5) | 74.6% | 41% | 93.4% | 2.7% | 13.1 |
| learned look policy (lookup of 30k situations) | 74.6% | 44% | 92.2% | 3.4% | 13.1 |
| random policy | 74.2% | 40% | 92.0% | 3.2% | 14.5 |
| raster policy | 74.2% | 45% | 94.2% | 2.6% | 14.4 |
| computed + self-model veto (IX.4) | 74.4% | 40% | 95.0% | 2.0% | 13.2 |
| self-model from the first 500 episodes only | 74.4% | 40% | 95.0% | 2.0% | 13.2 |
| pixel noise σ=0.25, graded matching | 73.6% | 38% | 93.7% | 2.4% | 13.5 |
| exact stroke identity, clean | 71.2% | 22% | 86.1% | 3.0% | 14.5 |
| exact stroke identity, noisy | 68.8% | 19% | 93.7% | 1.2% | 14.7 |

Rotation (from the 600-digit run, 80 test digits): rotated digits 20% without the operator,
57.5% with it, 67.5% upright with the operator switched on anyway; 2 of 19 sixes and nines
swapped. Applying the stroke rotation four times returns 23% of strokes exactly and the rest
to a code overlapping the original 5.6 of 16: the relation has order 4 through the cleanup
noise of a vocabulary not closed under rotation (VII.3).

Library, 600-digit run: 22 consolidated exemplars, 27 consolidated chunks, 23% of exemplar
parts are chunks, one chunk used across two classes; everything remembered is 8% shorter
written with chunks than flat.

## XIV.5 What it says about the design

- **Similarity-preserving codes are what make it robust.** Graded matching loses one point
  under heavy pixel noise; exact identity loses three points clean and six noisy, and half its
  confidence. Part I's brittleness result, on real data, with the cure the design proposed.
- **The look policy is learnable by lookup, and it does not matter here.** The learned policy
  matches the computed one exactly. Both save about 1.4 glimpses over random and gain no
  accuracy. Digits are not the toy world: no single cell decides, so where you look matters less
  than how many cells you have seen.
- **The self-model works as a veto and needs no track record.** Confident-wrong falls from 2.7%
  to 2.0% at the cost of 1% fewer declarations, and the model built from the first 500 episodes
  is as good as the full one. IX.4's "cannot work early" is not confirmed at this scale; the
  situation key is coarse enough to fill fast.
- **The library helps prediction more than classification.** Chunks shorten description by 8%
  and are reused across classes once, but recognition rides on exemplars. The 4×4 grid is the
  cap: an exemplar is 7 cells, so there is little room between a stroke and a whole digit.
- **The pile lied exactly where the formula said it would.** One ghost in 8,000 episodes, at
  the largest table, and it took the run down. Category checks make it survivable; they do not
  make it rare.

## XIV.6 Not covered, still

A learned *encoder* in the gradient sense — the codes come from counting and partitioning,
and the 4×4 grid caps what any table over it can reach. Translation invariance below the
grid. Recognising that a situation *type* is one the agent gets fooled in, beyond the
declaration-time key. The instruction-as-index experiment of IX.3. Multi-object scenes.

---

# Part XV — Counting only: a usefulness-built hierarchical library [MEASURED unless tagged]

Added 2026-09-17. Lavender's proposal: drop imagination, reasoning and the learned policy;
keep only counting and "make a new representation when it recurs", with the label always
shown, cells sampled one at a time because working memory cannot hold them all, and test
whether that alone builds a useful, reusable library. Three scripts, in the order they were
written, each correcting the last: `tally.py`, `tally2.py`, `tally3.py`.

## XV.1 Frequency is not usefulness (`tally.py`)

7×7 patches at stride 3 (64 cells), 627 strokes, random cells until 8 inked seen, a pair of
items at a relative offset becomes a chunk after 4 recurrences, chunks pair again so they grow.
10,000 digits, 102 s.

- It compresses: 445 chunks, up to 10 strokes, 8 sampled strokes written as 5.3 items; 389 of
  the 412 used chunks are shared across classes.
- The most used chunks are a cross and a bar, used by all ten classes: what recurs most is
  what every digit has, which is what says least about which digit it is.
- Label hidden, 16 strokes seen: strokes only 84.3%, with chunks 71.5%. **The library cost 13
  points**, because a recognised chunk replaced its strokes with one coarse item and filled in
  cells never looked at. The collapse the record warns about three times (III.3, VIII.5, X),
  with no counterweight.
- Part I check: with exact stroke identity the pair table shatters (2,726 chunks, almost all
  bare pairs, compression 8 → 6.5 instead of 5.3).
- Reading a class back by unbinding `label ⊗ class` from one bundle of 120 items: 6 of 10.
  The pile was five times over capacity; the formula said so.

## XV.2 Surprise as the currency (`tally2.py`)

3×3 or 5×5 patches at stride 2 (169 or 144 cells), explicit levels, and a chunk's worth
measured in bits of surprise it removes: on unseen cells (recognised from half its cells on
one half of the digit, scored on the other half) and about the label, minus its storage.
Consolidated when positive after 3 trials, dropped when negative after 12. Recognition adds a
level and never deletes stroke evidence. A beam search over the digit's items, ordered by pair
counts plus one random slot, runs when the digit is more surprising than average and proposes
label-specific combinations.

5,000 digits, label hidden at test, 24 of ~44 inked cells sampled in training:

| variant | strokes seen | class: table | with chunks | completion: table | with chunks |
|---|---|---|---|---|---|
| 3×3, hard ids | 16 | 69.7% | 74.7% | 37.0% | 43.8% |
| 3×3, hard ids | 32 | 74.0% | 80.3% | 40.0% | 47.0% |
| 3×3, graded bumps (41 centroids/block) | 32 | 70.0% | 78.3% | 28.4% | 42.6% |
| 5×5, hard ids | 16 | 84.3% | 86.0% | 44.1% | 45.6% |
| 5×5, hard ids | 32 | 87.7% | 87.3% | 43.6% | 50.5% |
| 5×5, graded bumps | 32 | 80.3% | 85.3% | 39.7% | 44.0% |

- **For the first time the library helps the task**: six points on the label and seven on
  unseen cells at 3×3, and seven on unseen cells at 5×5 where the table alone is already at
  87.7%. Seven levels deep in under a minute; 135 candidates tried and dropped.
- The library earns its keep when the view is partial. With nearly the whole digit sampled the
  plain table catches up on the label and chunks only help completion (51% vs 39%).
- **Graded codes lost twice.** Diagnosis: 41 centroids per block cut finer than the vocabulary
  itself, so a bump meant "three arbitrary neighbours" rather than "a family"; soft counts are a
  kernel estimate and 5,000 digits give ~80 observations per class-and-cell, where the histogram
  is already well estimated; central strokes collect everyone's mass (hub bias) and completion
  picks by that argmax. Lavender's proposal stands (a bump should be membership in a coarse
  family, normalised so hubs do not dominate); the construction did not test it fairly.
- **The search found nothing**, 2 consolidated chunks from 2,427 searches at beam 8, depth 5.
  Its support index was keyed by exact stroke at exact cell, so a three-item combination almost
  never had three past digits in common. The search was shattered exactly as Part I's dictionary.
- Parameters: 10,025 numbers of vocabulary, 578,880 in the class tables, 131,528 more for graded
  codes; the consolidated chunk cards themselves are 4,494 numbers. The tables are nearly all of
  it; the library is tiny.

## XV.3 Integrate as you go (`tally3.py`)

Lavender's loop: one cell at a time, nearly the whole digit; after every cell the parse may
change (a chunk with at least half its cells present and none contradicted may be placed, may
displace items it overlaps if it explains more, displaced chunks dissolve back into strokes,
repeat until stable, so 1-and-2 can become 1-and-3 when 3 arrives); at most 20 top-level items,
over the cap compact or drop the oldest stroke; a placed chunk predicts its unseen cells and is
scored when they arrive; the label is shown; at the end the class is asked, then a few held-out
inked cells, answered from the parse before the table. Graded codes rebuilt as coarse families
(8 blocks × 10 families, top-3 softmax memberships, a proper family-mixture read-out so hubs do
not dominate). Search keyed by 40 stroke families.

What building it corrected:

- **Ownership must be unique.** Two placed chunks can both cover a cell not yet seen; when it
  arrives both absorbed it, and a later drop left a chunk pointing at a cell that was gone.
- **Chunk evidence shatters across anchors.** Keying class counts by the exact anchor split one
  chunk's evidence over neighbouring positions; a 2-cell-coarse anchor fixed the class score.
- **Pressure mints junk.** Compacting under the cap by naming the most co-occurring held pair,
  with a threshold of 2, produced 5,000 provisional chunks in 300 digits and the matcher drowned
  (the first full runs timed out). Pressure-born chunks now need the same recurrence as
  pair-born ones, provisional cards are capped at 400, and untested cards are forgotten after
  600 digits. X.2's allocation gate, learned a third time.

Results, 5,000 digits, 300 test digits, 6 held-out inked cells each:

| variant (5,000 digits, WM cap as stated) | class: table, all cells | class: strokes held | class: with library | held-out cells: table | strokes held | with library | answered by a chunk |
|---|---|---|---|---|---|---|---|
| hard ids, cap 20 | 93.0% | 87.3% | 74.7% | 44.8% | 44.9% | 46.1% | 28% |
| hard ids, cap 40 | 93.0% | 91.3% | 90.0% | 44.8% | 44.6% | 46.6% | 33% |
| graded families, cap 20 | 85.7% | 70.0% | 66.3% | 56.3% | 53.4% | 60.9% | 43% |

Library at the end: hard cap 20, 31 consolidated chunks up to level 10, 4,156 tried and
dropped, 20 of them born under pressure; hard cap 40, 25 chunks, 24 from pairs; graded, 652
chunks 24 levels deep, 2,831 dropped. Training took 626 s, 556 s and 1,724 s. The search
proposed 94 label-specific combinations in 1,200 digits, every one a two-stroke pair, none of
which earned its click: deeper combinations lose support in the recent-digit index faster
than they gain label specificity.

- **Blanks are free evidence, and they carry the class.** About 77 of 144 cells are blank
  and all of them reach the class opinion in every variant; the table over all cells is at
  93%, above stage B's 86%, largely on the shape of the blanks. The cap only limits inked
  strokes, so "strokes held" at 87% is the honest capped number.
- **The library still costs the label and helps the unseen cells**, now in the memory-limited
  regime: with 20 items, class 87.3% → 74.7%, completion 44.9% → 46.1%. With 40 items the
  loss on the label shrinks to one point. The same shape as XV.1 and XV.2: a chunk item at a
  coarse anchor is weaker class evidence than the strokes it replaces.
- **Graded families change what the library is for.** Held-out cells 60.9% versus 46.1%, a
  24-level library, 43% of held-out cells answered by a chunk; and the class score at 66%.
  The soft codes make chunks match generously, which is what prediction wants and what
  classification does not.
- **Pressure-born chunks are junk unless gated.** With a co-occurrence threshold of 2 the
  system minted 5,000 provisional chunks in 300 digits; at the recurrence threshold, 20 of 31
  survivors in the hard run are pressure-born and the top one, a 5-cell corner used 234 times
  across all ten classes, is the highest earner. Under pressure the system names what it is
  holding, and what it is holding is what every digit has.

## XV.4 Where this leaves the design

1. **The pile is not doing the work in this branch.** [CRITIQUE, Lavender] The counting
   scripts keep the label, the class table, the cards and the pairs as Python data; the
   fingerprints buy only stroke similarity, which pixel distance could give. The mechanisms of
   Parts VI–IX that used the pile (votes in superposition, provenance, pressure as ghost
   probability) were replaced by a list and a cap. XIII and XIV are the pile-based branch; XV
   is the counting branch; the two have not been joined.
2. **The loop Lavender re-derived from the counting side is the XIII loop.** Search a card that
   has A as a part, expand it, subtract A, look where it predicts, integrate if it fits, go up a
   level; several hypotheses at once, jump to the top from one look, switch and come back. Every
   step exists in `agent.py` / `digits.py`. The next build is that loop with what XV added:
   cards born from recurring pairs, clicks by usefulness, the label as one more item in the pile
   completed by association.
3. **A third operation is missing: thinking of a thing as another thing.** [VISION, Lavender]
   Beyond shifting by a pose and adding to the pile, the system needs a conceptual pose: a
   transform that takes many instances (every centre of a 1, every upper loop) onto one
   canonical part, so that cards which differ only in instance become identical and merge in a
   cascade. That cascade is what a click should feel like from inside. It is Part VII's relation,
   never built. The minimal counting version, categories of strokes that fill the same card slot,
   is XV.5 below. The full version, learning shared transforms, is X.7.
4. **Forgetting is too eager.** [CRITIQUE, Lavender] A card that dies after 12 failed trials or
   600 untested digits may have been useful on a longer timescale; 4,000 cards died per run.
   Provisional storage is cheap; keep them longer, decide promotion by the bit arithmetic and
   execution by slow decay.
5. **The vocabulary is a knob in disguise.** 401 strokes with 12.6 near-duplicates each is why
   the tally needs the similarity patch at all. A proper k-means with 40–60 centroids would give
   each column real mass.
6. **Order of integration.** The raw pile is order-free; the compacted pile depends on the parse
   (which cards were placed), and two parses are two descriptions of the same evidence. Cards
   are looked up by expanded footprint, so the two routes meet at the top when both exist; the
   revision rule (displace if it explains more) is a greedy stand-in for a canonical parse.

## XV.5 Categories: interchangeable in a role [MEASURED]

Count which strokes fill each slot of each card. Strokes that co-fill the same slot (each ≥3
times) in at least two distinct roles are merged into one category (union-find, at most 12
strokes each); every card is re-expressed with categories in place of strokes; cards that
became identical merge, their earnings pooled; pair counts are remapped. Run every 500
digits from 1,000 on.

| 5,000 digits, cap 20 | class: table | strokes held | with library | held-out cells: table | strokes held | with library | by a chunk |
|---|---|---|---|---|---|---|---|
| hard ids, no categories | 93.0% | 87.3% | 74.7% | 44.8% | 44.9% | 46.1% | 28% |
| hard ids, categories | 93.0% | 87.3% | 76.3% | 51.3% | 51.1% | 51.3% | 36% |
| graded families, no categories | 85.7% | 70.0% | 66.3% | 56.3% | 53.4% | 60.9% | 43% |
| graded families, categories | 85.7% | 70.3% | 66.0% | 58.3% | 55.8% | 54.2% | 38% |

Cascade, hard: at the first pass (digit 1,000) 16 categories over 128 strokes and 106 cards
merged at once, library 660 → 616 bits; later passes merged 19, 11, 7, 5, 1, 6, 7, 0. End:
31 categories over 252 of 401 strokes, 55 consolidated chunks (31 without categories), 2 of
them used by fewer than five classes. Graded: 129 cards merged at the first pass, 394
consolidated chunks 18 levels deep, 11 of them class-specific. Note: with categories on, a
predicted stroke counts as right when it lands in the true stroke's category, so the
completion columns of the category rows are more lenient than those above them.

- **The cascade is real and happens once.** Introducing categories merged a quarter of all
  cards in one step, then almost nothing afterwards. That is the shape the click should have.
- **The categories it finds are the generic ones**, because they come from the slots of the
  chunks that exist, and those are the corners and hooks every digit has. Abstraction inherits
  the library's bias; it does not correct it. 2 of 55 chunks are class-specific after it.
- **No gain on the task.** Class +1.6 points with the library still 11 below the strokes it
  replaces; completion by chunk ties the table under the lenient score and, for graded codes,
  falls below it (60.9% → 54.2%): merging strokes into categories blurred the predictions that
  the fine-grained graded chunks were making well.
- The mechanism is right and cheap (all counting); what it needs is categories defined by
  *role in a class-specific structure*, which requires class-specific chunks to exist first.
  The two problems are the same problem.

The toy-world control (two sub-assemblies differing in one stroke should merge and nothing
else) was not run. Speed: batching the placement checks made the 5,000-digit run 3 min (hard)
and 8 min (graded) instead of 14 and 29.

## XV.6 As-if substitutions: thinking of a thing as another thing [MEASURED]

Lavender's version of abstraction, top-down and validated: a strong card placement (at
least 4 cells seen) may disagree on exactly one cell, S seen where P was expected; the
placement is allowed with the provisional reading "S as P here" (a self-transform of the
stroke's fingerprint, P − S, proposed by the higher card's expansion). At the end of the
digit, if the card survived and the label agrees with the card's majority class, the reading
is credited; three credits make it permanent for that card slot.

Hard ids, 5,000 digits: 792 substitutions tried, 34 learned, in 13 cards, all consolidated;
by class 1: 10, 3: 8, 7: 6, 4: 3, 0/2/6/8: 1–2. The centres of 1s read as one stroke, as
predicted. Task: class with library 75.3% (74.7% without), held-out cells 46.1% (same).
With categories on top: 1,000 tried, 34 learned, no change. Class-specific chunks: still 0.

The mechanism does what it should and is too rare to matter at this volume: one substitution
per placement and three agreeing confirmations keep it to seven learned readings per
thousand digits. The next step is to let a validated reading feed back into the stroke's
code (nudge S toward P so they share bits), which is the Part III representation learning
with, for the first time, a signal to learn from.


---

# Part XVI — Product names [MEASURED] and the torus [VISION, Lavender]

Added 2026-09-17, late. Lavender: why give a card a random name? Let it be its parts walked
together, `name = ⊗_i (part_i ⊗ offset_i)`, which is commutative and associative, so the
order of integration cannot matter. `product.py`, exact toy setting, 60 primitives:

- 20 integration orders of 5 parts → 1 name.
- Subtracting one part of a 2-part card and cleaning up against 960 (stroke, offset)
  candidates: 100%. Subtracting parts one by one from cards of 3 to 20 parts: 100% at every
  size. Binding loses nothing; the only limit is what the cleanup table knows.
- A whole built from two sub-card names equals the whole built from its six leaves, and
  subtracting one sub-card yields the other's name directly: 100%. **A card carries its leaf
  count**: a K-leaf card at offset t is walked by K·t (each leaf carries the offset). Moving a
  6-leaf whole by t equals walking its name by t six times: 100%. Because L=41 is prime, K·t
  is a distinct shift for K < 41 and can be divided out (the inverse of K mod 41). A card of
  more than 40 leaves cannot be moved unambiguously at this block size. [CRITIQUE]
- The price: two wholes sharing 4 of 5 parts overlap 0.58 of 16 blocks by product name
  (strangers: 0.35) versus 62 of 80 shared slots as bundles. Product names have no graded
  similarity. [REPAIR] two fingerprints per card: product for identity, order-freeness and
  subtraction; bundle for recognition by overlap.

This is the "expand, subtract, guess where to look" loop of XV.4 item 2 run inside the
algebra: hypothesise a whole from bundle overlap, subtract what was seen through the product
name, clean up the residual to get what remains and where. Not built.

Graded codes: keep the product on canonical ids and use graded similarity only in cleanup
(subtract the expected part, accept a look-alike). A fully graded product is circular
convolution (HRR); widths add per binding and unbinding is approximate; invertible exactly
when the kernel has no zero Fourier amplitude (single marks always, narrow bumps in
principle, wide bumps not).

Torus properties noted and not yet used [VISION, Lavender]: coarse-to-fine matching on a
subset of blocks ("squint"); continuous positions (fractional walks; a tally that stores the
average walked fingerprint, mean position as angle, inverse variance as strength); ordered
rings (neighbouring slots hold similar strokes, so one notch is a meaningful change and the
bump becomes natural); Fourier on the ring for capacity of graded binding. Non-commutative
composition (rotation then translation) remains outside the torus.

---

# Part XVII — Angles: codes that move by prediction error [MEASURED]

Added 2026-09-18. Lavender: the architecture has no layers and no backprop; what plays the
part of depth is **the trajectory of the search**. So make each block a continuous angle rather
than a one-hot slot, and propagate prediction success or failure back along that trajectory as
a rotation on the angles. `phase.py`, standalone: no library, no chunks, no search, only the
substrate and one prediction.

## XVII.1 Superposition when a block is an angle [VISION, Lavender + answer]

Lavender's objection, which is the right one: *if a fingerprint is angles, how do several of
them go into one pile? You cannot add angles.*

You do not. You add **arrows**. An angle is a unit vector `(cos θ, sin θ)`; binding adds angles
(rotation, length preserved, stays on the circle); **bundling adds the vectors**, and the sum
lands inside the disc. Its direction is the consensus, its **length is the agreement**. Two
opposed items cancel to a stub: no opinion. Thirty aligned items give a long arrow: certainty.
A pile block is two numbers, not one, and the second one is not an addition to the design — it
is what is left over when unit arrows disagree.

This is the old block at a different resolution. A block with counts over 41 slots is a
histogram on the ring; **the phasor sum is its first Fourier coefficient.** Keeping `m`
harmonics (`Σe^{iθ}, Σe^{2iθ}, …`) is that histogram smoothed to `m` waves: m=1 is one arrow,
m=20 recovers the 41-slot histogram exactly. **The slot substrate and the angle substrate are
one object at two settings of one knob.** [CRITIQUE] m=1 is lossy in a specific way: it cannot
represent two separate opinions, only their average and a shortened arrow. Whether that matters
is measurable, and at this task it did not (below).

## XVII.2 The backward pass is the walk run in reverse

Match is the real part of the inner product, `s = Σ_b |Z_b| cos(∠Z_b − θ_v,b)` — smooth, unlike
the presence test of XIII.1. Because binding is *addition* of angles, the derivative of `s` with
respect to **every** angle in the chain is the same scalar, `±sin(disagreement)`. One number per
block, handed to every participant, with `+` if it was bound in and `−` if it was subtracted.
No matrices, no products along the chain; B independent scalar problems.

Two consequences:

- **Depth is free.** Nothing multiplies along the chain, so the signal neither vanishes nor
  explodes. A trajectory can be as deep as the search wants, chosen at run time, per input.
- **[CRITIQUE] Credit does not discriminate.** The same fact makes every link take identical
  blame; there are no weights to separate them. The proposed substitute is the arrow length —
  well-determined angles move less — and it is **not yet tested**.

The loss is `−log softmax` over the vocabulary of the match score: the true code is pulled onto
the consensus, every other code pushed off it in proportion to how much it is currently
believed. **The contrastive term is load-bearing** — with only the single wrong winner pushed
away, all-pairs code similarity went 0.00 → 0.48 in 500 digits and the effective vocabulary fell
from 64 to 37. Collapse, for the fourth time in this record (III.3, VIII.5, XV.1), and the same
cure: predictive adequacy against every alternative, not just the leader.

## XVII.3 What was measured

Self-supervision (III.6): hold out one inked cell, let every other inked cell vote for what is
missing through the learned relation code of its relative offset (radius 4, 80 offsets),
clean up against the vocabulary. **The label is never shown at any point.** 64 strokes (the 401
allocated prototypes coalesced by weighted k-means — XV.4 item 5, taken), 5x5 patches stride 2,
5,000 train digits, 2,400 held-out cells from 300 unseen digits, one online pass, ~90 s.

The reference is a count table over (offset, source stroke, target stroke) doing the identical
job with 332,800 numbers.

| variant | learned | frozen codes | count table | near-miss | same-class sim | diff-class | all-pairs | distinct |
|---|---|---|---|---|---|---|---|---|
| 32 blocks (4,640 angles) | **49.0%** | 1.5% | 53.5% | 76.6% | +0.091 | +0.056 | +0.061 | 64/64 |
| 16 blocks (2,320 angles) | 38.2% | 1.9% | 52.2% | 67.4% | +0.163 | +0.129 | +0.134 | 64/64 |
| 8 blocks (1,160 angles) | 26.5% | 1.8% | 53.0% | 56.0% | +0.189 | +0.148 | +0.154 | 64/64 |
| 16 blocks, blanks vote | 29.8% | 1.6% | 38.7% | 60.4% | +0.188 | +0.106 | +0.118 | 63/64 |
| 16 blocks, only relations learn | 9.8% | 1.9% | — | 24.8% | +0.010 | +0.005 | +0.006 | 64/64 |
| 16 blocks, only codes learn | 12.0% | 1.9% | — | 32.6% | +0.108 | +0.060 | +0.068 | 64/64 |

Floor (always the commonest stroke): 2.8%. "Near-miss" counts a prediction right when it lands
on a look-alike stroke, the leniency XV.5 uses.

Readings:

- **The pile is a tally, and a 72x smaller one.** 4,640 angles reach 49.0% where the explicit
  count table reaches 53.5%. The claim that a bundle of phasors *is* a histogram is not a
  metaphor; it is within 4.5 points of the histogram at 1/72 the size. This is the first thing
  in the record that makes the substrate pay for itself rather than being a costume over
  Python data (XV.4 item 1).
- **Codes and relations must co-adapt.** Freeze the codes: 9.8%. Freeze the relations: 12.0%.
  Move both: 38.2%. The halves multiply rather than add. Neither "learn a representation" nor
  "learn a relation" is the operative thing.
- **Part III happened with no label.** Strokes belonging to the same digit class ended more
  alike than strokes that do not, from a purely self-supervised signal. The excess over the
  all-pairs baseline is small but consistent: +0.029 (16 blocks), +0.030 (32), +0.035 (8),
  **+0.070 with blanks voting** — blanks carry the class (XV.3) and letting them vote drives
  class structure into the codes at the cost of 8 points of prediction. Cup and plate sharing
  bits because they behave alike, measured, for the first time in this record with a live
  signal rather than a one-shot hashing of a label histogram (XIV.1).
- **The arrow length is a weak confidence signal.** Mean `|Z|` is 0.339 on correct predictions
  against 0.311 on wrong ones (0.396 / 0.358 at 8 blocks). It separates, barely. Resolved in
  XVII.3b: it is not the m=1 loss, it is the wrong quantity — the margin is calibrated and `|Z|`
  is not. [OPEN] whether `|Z|` can still serve as the per-angle learning rate for credit
  assignment (XVII.2) is untested.
- **More blocks buy prediction, not class structure.** 8 → 32 blocks nearly doubles accuracy
  and leaves the same-class excess flat at about +0.03.

## XVII.3b Harmonics and blocks: two knobs that do different jobs [MEASURED]

Added 2026-09-18, later, over several rounds in which two of Claude's explanations were tested
and failed. Two follow-ups from Lavender: *blocks are cheap, so why not use many?* and *does
keeping more Fourier coefficients per block help?*

**Blocks are paid, harmonics are free.** A code is B angles whether m is 1 or 16 — the harmonics
of a code are computed from that single angle (`e^{ikθ}`) at read time. So m describes the
*pile* only, never a stored item. Blocks cost storage linearly.

### Blocks: Lavender is right, and the codes overtake the table

| blocks (m=1) | learned | count table | near-miss | top quarter by margin | same/diff sim |
|---|---|---|---|---|---|
| 16 | 38.2% | 52.2% | 67.4% | 52.3% | +0.163 / +0.129 |
| 32 | 49.0% | 53.5% | 76.6% | 73.3% | +0.091 / +0.056 |
| 64 | 59.8% | 52.2% | 82.7% | 83.7% | +0.049 / +0.026 |
| 128 | **63.9%** | 52.7% | 86.0% | **87.3%** | +0.025 / +0.014 |

- **From 64 blocks on the learned code beats the count table it approximates** (59.8% vs 52.2%).
  Below that it is compressing a tally; above it, generalising past one, which a table of counts
  over exact (offset, stroke) pairs cannot do. 128 blocks is 8,320 angles against 332,800.
- **[REPAIR → XVII.3] Confidence was the wrong quantity, not the wrong substrate.** The margin
  (top score minus runner-up) is calibrated where `|Z|` is not: at 128 blocks the most confident
  quarter is **87.3%** correct against 39.3% for the least, overall 63.9%. `|Z|` gives 70.8% /
  54.8%. Agreement among voters is not the same thing as a decisive landing on one code.
- **[CRITIQUE] More blocks buy prediction and cost class structure.** The same-class excess over
  the all-pairs baseline falls from +0.029 (16 blocks) to +0.009 (128). Sharper codes separate
  better and therefore share less; Part III's pressure and prediction accuracy pull against each
  other along this knob.

### Harmonics: a better container that this task cannot use

Bundling capacity with **no learning anywhere** — V=2000 random codes, bundle N, precision@N:

| | m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| N=4, B=16 | 0.412 | 0.756 | 0.969 | 1.000 | 1.000 |
| N=8, B=16 | 0.228 | 0.447 | 0.731 | 0.931 | 0.997 |
| N=16, B=16 | 0.138 | 0.252 | 0.467 | 0.734 | 0.947 |
| N=32, B=16 | 0.109 | 0.176 | 0.316 | 0.502 | 0.763 |

and against blocks (N=16): B=16/m=8 → 0.718 beats B=64/m=1 → 0.575, approaching B=128/m=1 →
0.848. **m harmonics buy between m/2 and m times the blocks, with the stored code unchanged.**

But on the MNIST cell-prediction task, learning at high m fails outright:

| | m=1 | m=2 | m=3 | m=8 |
|---|---|---|---|---|
| accuracy, step size matched across m | 38.2% | 26.1% | 25.1% | 19.8% |

| train \ read | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| trained m=1 | 0.382 | **0.385** | 0.375 | 0.326 |
| trained m=4 | 0.042 | — | 0.184 | — |
| trained m=8 | 0.023 | 0.039 | — | — |

- **[CRITIQUE, two failed explanations, both Claude's]** First proposed cause was the factor of k
  in the gradient inflating the step size; normalising it changed nothing (table above). Second
  was that the sharp kernel only hurt at read time; the cross-test refutes it — codes trained at
  m=8 score **0.023 when read at m=1, which is the random-code baseline (0.019)**. Nothing was
  learned at all.
- **[REPAIR] The gradient cancels.** At m the update is a sum of terms oscillating at 1×…m× the
  rate of the disagreement. From random angles their signs are effectively independent and they
  cancel; a usable gradient appears only once the code is already close. A vanishing gradient
  arriving by oscillation rather than by multiplication along a chain — so the "depth is free"
  claim of XVII.2 survives, but only at m=1.
- **The match kernel is what m really controls.** `Σ_k w_k cos(kδ)` at a 15° disagreement is
  0.966 (m=1), 0.933 (m=2), 0.571 (m=8), 0.073 (m=16). High m is a *pickier* match, and
  generalisation is built entirely out of votes that are roughly right. This is the same trade as
  exact-vs-graded stroke identity (XIV.4) and hard-ids-vs-graded-families (XV.3), recognised for
  the first time as one dial.
- **[CRITIQUE → Lavender's proposal, MEASURED, negative] Coarse-to-fine does not rescue it.**
  Propose with m=1, rescore the shortlist sharply. The headroom is large — the true stroke is in
  the blurry top-5 for 82.0% of cells at B=16 and 94.0% at B=64, against 38.2% / 59.8% actually
  chosen — and sharpening the second pass is worse at every shortlist size and every m (B=64,
  top-5: 0.598 → 0.590 at m=2 → 0.505 at m=8).
- **[REPAIR] Capacity and recognition are different problems, and the capacity result does not
  transfer.** In the capacity test the items in the pile *are* exact stored codes: there is an
  exact match to find and a sharp kernel finds it. In cell prediction the target was never in the
  pile — it is a consensus assembled from neighbours that only roughly agree, so sharpness has
  nothing exact to reward and merely amplifies noise among look-alikes. **Harmonics buy retrieval
  of what you stored, not recognition of what you have not seen.**
- **[OPEN] Where harmonics should still pay**, by that reading: working memory holding
  already-recognised parts at known poses, to be subtracted back out (XVI). That is retrieval of
  exact stored items, the regime where they win, and it is untested. The 94% recall also says a
  better second pass would gain a great deal — but not one built from sharpness.

## XVII.5 Depth: the trajectory as layers [MEASURED]

Added 2026-09-18. The originating claim of this whole part, from Lavender: *there are no layers
here; what plays the part of depth is the trajectory of the search, and prediction error should
propagate back along it as a rotation.* XVII tested the rotation rule at a chain one bind and one
bundle deep. This tests depth. `depth.py`, synthetic so that depth is the only thing varying:
a hierarchy of cards, K children each at distinct offsets, names derived by walking (XVI), every
leaf observed but one, predict the missing one.

A pure chain of bindings cannot fail and testing one proves nothing — binding is addition, so the
derivative is identical at every link. What can fail is what sits between the links: bundling
adds crosstalk, and cleanup is a non-differentiable snap.

**Two confounds, both Claude's, both caught only after reporting a result.** The first run showed
1.000 → 0.052 across depths 1–4 and read as "depth destroys credit". But the leaf count grows as
K^d (3, 9, 27, 81), so depth was confounded with pile size; and the pose update carried a
`1/len(path)` factor, training deep paths at a systematically smaller step, which the correct
gradient does not have. With both fixed — matched leaf counts, and poses updated exactly as leaf
codes are:

| leaves | flat (depth 1) | deep (3 children per card) |
|---|---|---|
| 3 | 1.000 (moved 0.10) | depth 1: 1.000 (moved 0.10) |
| 9 | 1.000 (moved 0.22) | depth 2: 1.000 (moved 0.12) |
| 27 | 1.000 (moved 0.58) | depth 3: 1.000 (moved 0.17) |
| 81 | 0.963 (moved 1.05) | depth 4: **1.000** (moved 0.20) |

120 leaves, 32 blocks, 6,000 episodes, chance 0.008.

- **[REPAIR → XVII.2] Depth is free, confirmed.** Four levels and 81 leaves recover perfectly.
  Nothing multiplies along the chain, so there is no vanishing or exploding term; the vanishing
  gradient of XVII.3b was specific to harmonics, where terms oscillate and cancel, and does not
  arise from depth. **Lavender's originating claim holds in its strong form.**
- **Composition is not merely cheaper, it is better.** At 81 leaves the hierarchy beats the flat
  card (1.000 vs 0.963) while moving its codes a fifth as far (0.20 vs 1.05 rad). A flat card
  needs 81 distinct pose codes; a 4-deep card reaches the same 81 leaves from 3 poses per level
  combined along paths. Fewer parameters, reused combinatorially, learning faster and ending
  better. The design assumed hierarchy was forced by capacity (X.5); this says it also pays
  directly, which was never measured.
- **[CRITIQUE → XIII.2, X.5] Learned codes beat the random-code capacity bound.** XVII.3b's
  capacity table (random codes) gives ~0.11 for 32 items at 16 blocks. Here 80 items at 32 blocks
  recover 0.963–1.000. Learning arranges codes so they do not collide. The `density^B` bound of
  XIII.1 and the `sqrt(N)` law are floors for *random* allocation, not ceilings for learned
  representation — which is Part II's thesis showing up as a capacity result.
- **[OPEN] Re-sparsification is still untested.** The cleanup arm of the first run snapped each
  contributor's *walked* angle to the nearest *unwalked* leaf code — the wrong codebook, a
  category error — and destroyed depth-1 accuracy (1.000 → 0.427). That number says the
  implementation was wrong, not that IV.3 is. Depth 4 works without any cleanup at all, so
  re-sparsification is not needed at this depth and scale; whether it becomes necessary deeper,
  or with noisy leaves, is unmeasured.

## XVII.4 What this changes

1. **The tally mostly dissolves into the pile.** The class table, which was the bulk of the
   parameters in XV (578,880 numbers against 4,494 for the cards), is a bundle of phasors read
   by unbinding. What cannot dissolve is *proposing* a chunk — "these two recur, name them" is a
   discrete decision with no gradient pointing at it, and stays counted.
2. **Consolidation gains a gradient path it did not have.** A card's name under XVI is its parts
   walked together, so with continuous angles the name is a differentiable function of its
   parts: when a card predicts wrong, the correction passes through the name into the stroke
   codes. Previously a consolidated card got a fresh random name and learning stopped there.
   **Structure discrete and memorized, codes continuous and learned** — which is the title of
   this document, finally split cleanly.
3. **The as-if signal of XV.6 now has somewhere to go.** "Nudge S toward P so they share bits"
   is a rotation, and rotation is the operation this part implements.
4. [OPEN] Nothing here tests depth. The chain in `phase.py` is one bind and one bundle; the
   claim that a *search trajectory* can carry credit through many levels without vanishing is
   argued from the algebra (XVII.2) and not measured.


---

# Appendix — Glossary and notation

- **SDR** — sparse distributed representation. Binary vector, few bits on (k≈2–5 of 100 here).
- **bundle** (`+`) — superposition; "these co-occur"; set-like, loses roles.
- **bind** (`⊗`) — role-filler composition; structure-preserving; invertible for group-like relations.
- **cleanup** — snap a noisy vector to the nearest stored code via table lookup. The step that
  makes unbinding usable.
- **pose** — a transformation, spatial *or* relational (Part VII).
- **W** — working memory, a bundle of bound (item ⊗ pose) pairs.
- **shattering** — the Part I failure: keying on irrelevant bits splits one concept across
  thousands of entries, storing everything and retrieving nothing. The recurring antagonist of
  this design (X.2, X.3).
- **click** — a sudden large drop in description length when a chunk collapses several
  working-memory items into one. The consolidation trigger (VIII.4). As built (XIII.4): the row's
  description-length gain is positive *and* it has predicted an unseen cell correctly.
- **ghost** — a false item read out of working memory because its slots are all occupied by other
  items. Probability density^B per probe (XIII.1). The measurable form of over-stimulation.
- **idea** — a provisional row: a compaction the agent tries and later episodes test (XIII.4).
- **overwhelmed** — pressure above threshold with nothing left to compact; the agent stops
  sampling (XIII.4).
- **phasor** — a block written as a unit arrow `(cos θ, sin θ)`. Binding rotates it; bundling
  adds the arrows, so a pile block has a direction (consensus) and a length (agreement). The
  first Fourier coefficient of the slot histogram (XVII.1).

---

# Immediate next step

XVII gives the first learning signal in the design and the first place the pile pays for
itself. The order the evidence now points:

0. **Depth is done** (XVII.5): four levels recover perfectly, composition beats flatness, and
   learned codes beat the random-code capacity bound. Re-sparsification (IV.3) turned out not to
   be needed at this depth and remains untested rather than refuted.
1. **Read the pile sharp where the pile holds exact items** (XVII.3b). Capacity says m=8 recovers
   five times what m=1 recovers from a 16-item pile, and that advantage is real — but it did not
   transfer to cell prediction, because there the target was never in the pile. The regime where
   it should transfer is the glimpse loop's working memory: recognised parts at known poses, put
   in deliberately and taken back out by subtraction. Learning stays at m=1 (above it the
   gradient cancels); m is a read-time setting only. Cheapest test of the largest unexploited
   result in this part, and it is a genuine prediction that could fail.
2. **Join the branches** (the long-standing first item): the XIII loop with XV's card formation,
   XVI's product names for identity and subtraction, and XVII's codes that move — at 64+ blocks,
   where the codes beat the table.
3. **A better second pass.** The blurry read puts the true stroke in its top 5 for 94% of cells
   at 64 blocks while choosing it 60% of the time. Sharpening cannot close that gap (XVII.3b);
   something else might, and 34 points is the largest single margin visible anywhere in the part.
4. **Class-specific chunks.** Every library in XV is generic. XVII.3 shows blanks voting drives
   class structure into codes at the cost of prediction; whether that buys class-specific chunks
   is untested.

Older list, still open:

1. **Rotation** — poses as slot permutations (operator poses) so that SE(2) composes; this is where
   "one algebra" is actually tested (XIII.6, critique 3).
2. **Two objects per scene, then noisy sensing** — the first breaks the "residual explained" rule,
   the second brings Part I's brittleness back through cleanup-by-overlap.
3. **XI.3** — the relational world with the same abstract structure, for the transfer claim of Part VII.
4. The toy-world control for categories (XV.5) was never run.
