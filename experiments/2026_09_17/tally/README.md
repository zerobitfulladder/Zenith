# `tally/` — counting only: a library built from what recurs, then from what is useful

Three counting-only scripts, each correcting the last (design record Part XV). Verbatim from the session narrative of the original project (the rest of
it is in [`2026_09_16/glimpse_loop/README.md`](../../2026_09_16/glimpse_loop/README.md)).
`DESIGN.md` below means the design record, now kept in full at the end of that README.
Words are defined in [`2026_09_16/VOCABULARY.md`](../../2026_09_16/VOCABULARY.md).

The plain-words tour of the whole system as it stood that day is
[`results/architecture.html`](results/architecture.html). The scripts import `sdr.py` and
`mnist.py` from [`../../2026_09_16/glimpse_loop/`](../../2026_09_16/glimpse_loop/).

---

## Evening and night: understanding, then counting

Lavender went through the substrate piece by piece. The exchange that mattered:

- **Blocks are angles; the space is a torus.** A block is a ring of notches; a fingerprint is
  one point on a 16-ring torus; binding is walking each ring; two walks add. In the complex
  picture it is elementwise phase multiplication, so it is linear after all; the one-hot bits
  are just how an angle is written down. What makes a random walk act like a random rotation
  is that the space wraps.
- **Lavender's earlier template units** (winner-take-all templates rotated toward the input)
  are the same object: position as ring, identity as angle. The new layout puts identity in
  the whole pattern and position in the shift, which is what buys translation invariance
  and grouping at the price of a pile that fills.
- **The pile cannot hold a whole digit** as readable parts (about 20 items); Lavender
  re-derived from this that attention and chunking are forced, which is the design's
  central claim.
- **Graded fingerprints** (Lavender): a block should carry a small bump of memberships,
  `[0, 0.9, 0.6, 0, 0]`, not one hard bit, so the tally can use stroke similarity directly.
- **Common is not useful** (Lavender): build the library by explanatory power, not
  frequency; surprise (−log p) as the currency; search with a predetermined structure,
  ordered by the pair counts as a heuristic, wide enough that "creative" hits can happen.
- **Integrate as you go** (Lavender): show cells one at a time, let the parse revise itself
  (1-and-2 may become 1-and-3 when 3 arrives), cap the memory, form an opinion before
  predicting a patch.
- **Thinking of a thing as another thing** (Lavender): a self-transform of a fingerprint,
  proposed by a higher card's expansion and validated by the outcome; when many instances map
  to one canonical part, cards that differed only by instance merge in a cascade, which is
  what a click should feel like from inside.

Three counting-only scripts followed, each correcting the last (DESIGN.md Part XV):

| script | what it tested | result |
|---|---|---|
| `tally.py` | name what recurs (pairs → chunks), label shown, random cells | compresses (8 → 5.3 items), chunks shared across classes, but the library **cost 13 points** on the label: what recurs most is what every digit has |
| `tally2.py` | chunks scored by surprise removed on unseen cells and about the label; 3×3 and 5×5 patches; graded bumps; beam search | first library that **helps**: +6 on label, +7 on unseen cells at 3×3; 7 levels deep in a minute; graded bumps lost (too fine, hub bias); search shattered on exact ids |
| `tally3.py` | integrate-as-you-go with parse revision, 20-item cap, online prediction scoring, coarse-family graded codes, family-keyed search, categories, as-if substitutions | works; library still trades label for completion; details below |

`tally3.py` results on 5,000 digits, 300 test digits, 6 held-out inked cells each:

| variant | class: table | class: strokes held | class: with library | held-out cells: table | with library |
|---|---|---|---|---|---|
| hard ids, cap 20 | 93.0% | 87.3% | 74.7% | 44.8% | 46.1% |
| hard ids, cap 40 | 93.0% | 91.3% | 90.0% | 44.8% | 46.6% |
| graded families, cap 20 | 85.7% | 70.0% | 66.3% | 56.3% | 60.9% |
| hard ids + categories | 93.0% | 87.3% | 76.3% | 51.3%† | 51.3%† |
| graded + categories | 85.7% | 70.3% | 66.0% | 58.3%† | 54.2%† |
| hard ids + as-if substitutions | 93.0% | 87.3% | 75.3% | 44.8% | 46.1% |

† with categories on, a prediction counts as right when it lands in the true stroke's
category, so those columns are more lenient.

Readings:

- **Blanks carry the class.** Three quarters of cells are blank, all reach the class opinion
  for free in every variant, and the table over all cells reaches 93% mostly on the shape of
  the blanks. The cap only limits inked strokes.
- **The library helps prediction, not the label.** Consistently, across every variant: a
  chunk item at a coarse anchor is weaker class evidence than the strokes it replaces, and
  a chunk predicting unseen cells is better than the class table. With a 40-item memory the
  loss on the label almost vanishes.
- **Categories produce the cascade Lavender predicted**: at the first pass, 106 (hard) and
  129 (graded) cards merged in one step, then a trickle. But the categories inherit the
  library's bias: the chunks that exist are the corners and hooks every digit has, so the
  categories are generic too, 2 of 55 chunks class-specific afterwards, and no gain on the
  task; for graded codes they blurred good predictions (60.9% → 54.2%).
- **As-if substitutions work as designed and are rare.** 792 tried, 34 learned in 13
  consolidated cards; ten of them in class 1 (the centre of a 1 read as the canonical
  stroke, as Lavender guessed) and eight in class 3. Too few to move anything: strict
  validation (three confirmations, the card's majority class must agree) and one
  substitution per placement keep the volume tiny. Still zero class-specific chunks.
- **Pressure mints junk unless gated.** Naming the most co-occurring held pair whenever the
  memory overflowed produced 5,000 provisional chunks in 300 digits and drowned the matcher;
  at the recurrence threshold it is fine, and pressure-born chunks are among the top earners.
- **Search found nothing.** 94 label-specific proposals in 1,200 digits, every one a two-
  stroke pair, none earning its click: deeper combinations lose support in the recent-digit
  index faster than they gain label specificity.
- **Speed.** Batching the placement checks made a 5,000-digit run 3 minutes (hard) and 8
  (graded) instead of 14 and 29. Nothing in the algorithm is inherently slow.

## What Lavender pointed out that the code had drifted from

- The pile does no work in the counting branch: cards, pairs and the class table are Python
  data, and fingerprints only buy stroke similarity, which pixel distance gives anyway.
- The loop Lavender re-derived from the counting side ("search a card that has A as a part,
  expand it, subtract A, look where it predicts, integrate if it fits, go up a level, several
  hypotheses at once, jump to the top from one look, switch and come back") is the original
  glimpse loop of `agent.py` / `digits.py`, step for step. The two branches were never joined.
- Forgetting is too eager: thousands of provisional cards die per run before they could prove
  themselves; storage is cheap.
- 401 strokes with 12 near-duplicates each is a knob in disguise; a coarse k-means would give
  columns real mass.
- Order of integration: the raw pile is order-free; the compacted pile depends on the parse,
  and cards are looked up by expanded content so two routes meet at the top when both exist.

## Open, in the order the evidence points

*(Written 2026-09-17, before `phase.py`. Items 3 and 4 are what that run started on; the
current order is at the end of DESIGN.md.)*

1. Join the branches: the pile-based glimpse loop with card formation from counting and the
   label as one more item in the pile, completed by association.
2. Class-specific chunks. Every library so far is generic. Abstraction cannot fix that; it
   amplifies it. Either the search has to work (keyed by families, deeper support) or chunk
   scoring has to reward label information more directly.
3. Conceptual poses proper: learned transforms shared across instances, so "loop" transfers
   to loops never seen. The as-if table is the counting stand-in; the slow twin is nudging
   codes toward what they are read as, which is the Part III representation learning that
   finally has a signal.
4. Patience: slow decay for provisional cards; promotion by the bit arithmetic only.
5. The toy-world control for categories (two sub-assemblies differing by one stroke should
   merge and nothing else) was not run.

## How to run things

```
python experiments/2026_09_17/tally/tally.py                                # counting: name what recurs
python experiments/2026_09_17/tally/tally2.py --ps 5 --seen 24              # counting: usefulness-scored library
python experiments/2026_09_17/tally/tally3.py                               # integrate as you go (3 min); add --graded,
                                                                            #   --categories, --asif, --cap 40 as wanted
```
