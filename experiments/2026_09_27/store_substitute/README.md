# `store_substitute/` — tables instead of weights: does naming pairs or choosing a move close the gap?

2026-09-27. [`run.py`](run.py) builds the palette, the counted table, the store arm, the
substitute arm and the always-moved controls, about two minutes on the GPU; [`cnn.py`](cnn.py)
is the CNN and linear controls, ten seconds. Both write to [`results/`](results/)
(`summary.md`, `metrics.json`, `cnn.json`, `store_rounds.png`, `goal_by_move.png`,
`moves.png`, `run.log`).

## The idea

Lavender's proposal from the morning: replace the deep stack of a network by a search. A
network is match and compose; give it a rich palette to start from, let the actions be
compose, store and substitute, make the states fuzzy so that similar states meet, and when
the label is found distribute heuristic values along the way instead of changing weights.
The conversation that turned this into a rig is in the [day README](../README.md).

The first half already existed: the 2026_09_01 layer is the palette, its counted table is a
value table, and it reads MNIST at 0.973 with nothing fitted while a fitted linear map on the
same counts reaches 0.984. This experiment asks whether two of the actions can close that gap
with counts alone, before any search is built.

- **Store.** When two parts often sit next to each other, name the pair and write the name
  in their place. If names carry more than their parts, the table over names should read
  better with fewer tokens per image. That would be store doing what a layer does.
- **Substitute.** A menu of moves on the image: deskew, thin, thicken, shift, scale. Each
  goal digit learns which moves pay for it, takes them when it scores, pays a cost per
  move, and the answer is the goal with the best score after its own moves. The claim is
  that the merge of thin 1, thick 1 and slanted 1 should be a chosen, goal-dependent move
  rather than a fixed normalisation. The control is the fixed normalisation: the same move
  applied to every image always.

Pre-registered readings, from before the run: store works if tokens per image fall by half
and accuracy holds; substitute is a perspective if it beats always-deskew and the goal-by-move
table differs by goal, and a loophole if the moves flip about as many images wrong as right.

## The rig

Full MNIST, 50k to count and learn, 10k to choose the cost and the trigger, the 10k test set
for every number. Two standard errors at 0.97 over 10k images is 0.003.

**Palette.** 400 templates by spherical k-means over centred, unit-length 5x5 patches, flat
patches dropped, as in 2026_09_01. **Tokens.** The winning template at each patch position
at stride 2, a 12x12 grid, about 82 tokens per image. **Table.** Counts per token, 6x6 cell
and class, read as log P(token in cell | class) minus log P(token in cell), summed over the
image's tokens, largest of ten. The 2026_09_01 formula.

**Store.** Neighbouring pairs in four directions. Names chosen by count, or by count times
how class-telling the pair is, 300 per round, three rounds so names can pair with names.
Rewriting is greedy by rank, each position in at most one pair. Two uses of a name: it
replaces the pair, or it is added beside the parts. The second tells whether a name carries
anything the parts did not.

**Substitute.** Nine moves in image space (`moves.png`): deskew by the image's moments with
the centre of mass kept, thin and thicken by a 2x2 minimum or maximum, shifts of two pixels,
scale by a tenth. Every image is scored under each move and under each pair of moves with
the unmoved table. Scores are evidence per token, since thin removes tokens and scale-up
adds them, and a sum would favour whichever has more. The goal-by-move table is the mean
lift of the goal's margin under the move for true digits of that goal minus for the rest,
learned on the 50k. At read time a goal may take a move whose entry is positive, pays a
cost for it, and keeps its best; a trigger lets moves happen only when the unmoved margin is
small. Cost and trigger chosen on the 10k. Controls: every goal every move for free, with no
table; the oracle move that most helps the true digit; and each move applied to every image
always, with the table counted on the moved images.

## Results

Figures: [`store_rounds.png`](results/store_rounds.png), with the noise band drawn, and
[`goal_by_move.png`](results/goal_by_move.png). All numbers in
[`results/summary.md`](results/summary.md).

| | test |
|---|---|
| counted table, nothing fitted | 0.9725 |
| store, 300 names replacing pairs, best round | 0.9738 |
| store, 300 names beside the parts | 0.9722 |
| substitute, gated by the goal-by-move table, single moves | 0.9737 |
| every goal every single move, free | 0.9774 |
| always deskew, no choosing | 0.9766 |
| oracle move per image | 0.9959 |
| CNN, 118k parameters | 0.9908 |

**Store compresses without loss and adds nothing.** Naming 300 pairs a round took tokens per
image from 82 to 56 over three rounds, a third fewer, and accuracy moved by a tenth of a
point either way, inside the noise. Names added beside their parts changed nothing (0.9722,
0.9726); a thousand names beside the parts cost a sixth of a point, sparser rows and no new
information. The joint count of a pair says nothing the two parts' counts did not say. The
pre-registered bar, half the tokens with accuracy held, was not reached, and the deeper
point is the add rows: this table gains nothing from being told which parts sat together.

**Substitute: the prediction held and it did not matter.** The goal-by-move table did differ
by goal in the way predicted. Deskew pays most for 1 (+0.19), then 2 and 6, is nothing for 3,
4, 7 and 9, and is negative for 0 and 5. The chosen version, each goal taking deskew when
its entry is positive, gained a tenth of a point (0.9737), flipping 26 images right and 14
wrong. Deskewing every image with no choosing gained four tenths (0.9766). Letting every
goal take every single move for free, no table at all, gained five (0.9774). The static merge
did as well as the chosen one, or better. By the pre-registered reading, choosing the move
added nothing over normalising, and the claim that the merge should be a decision is dead
for this dataset.

**Why every other move was negative for every goal.** The table only knows centred,
size-normalised digits, because MNIST is delivered that way. Any move takes the image away
from what was counted, and on average lowers the true digit more than the rest. Slant is the
one variation the dataset left in, so deskew is the one move that reads as a perspective, and
a fixed deskew takes it. A chosen merge can only show on variation that has no canonical
form. MNIST offers none.

**The loophole appears as the menu grows.** With nine small moves, every goal taking every
move for free is pure tolerance, half a point up. With forty-five versions of each image the
same freedom gives nothing back (0.9722): everything reads a little more as everything. That
is the cost the conversation predicted for a free substitution, and it arrived at forty-five
moves, not nine.

**The oracle is not headroom.** A move that fixes almost every error exists in the menu
(0.9959), but picking it needs the label. Picking by confidence, which is what the free
version does, recovers a fifth of that. The rest is the classification problem itself.

**Against the stack.** A CNN with fewer parameters than the table has entries reads at
0.9908. Neither action moved the counted table towards it.

## Reading

Neither action closed the gap, and the outcomes table written before the run says what that
means: the counted table was already as fuzzy as this palette allows, and the missing points
are in the fitted map, not in naming pairs or choosing moves. The search proper, a frontier
with values on partial states, was to be built only if both arms showed something, and it
was not built.

What is not tested here, and would be needed to test the idea as Lavender stated it rather
than this flat cut of it: values on partial compositions and a per-state key, which need the
sequential version; substitution that memoises, writing a win under the token it started
from; and data whose variation cannot be undone by one fixed normalisation, which is the only
place a chosen merge could beat a static one.
