# 2026-09-26 — reasoning, attention as a window, and the backward path

Three days of conversation (the 24th to the 26th) and eight experiments at the end. The
conversation started from a question about reasoning and ended at a concrete learning rule
for the feedback path, which the experiment tests.

## The conversation, in order

**What reasoning is.** Recognition and reasoning read the same maps, in two ways. Recognition
reads them all at once through wiring built by repetition. Reasoning reads them a piece at a
time, in a stored order, holding a few results while it goes. A CNN composes in depth with
fixed wiring; reasoning composes in time with a few fixed operations over maps that do not
change (Ullman's visual routines: shift, index, mark, trace, fill, plus measure and compare).
Being taught "a square is four equal sides at right angles" writes a sequence over things
you can already see, which is why it takes one sentence and why "why is this a square" can
be answered: the sequence is stored. Repetition compiles the sequence into a shortcut from
features to label, which then wins the race (Logan's instance theory). A rule written by
words and an object learned by looking are the same data: parts at relative poses.

**The hierarchy's output is not its top.** Every level stays live and addressable. Attention
is a window, a place and a size, not a level. The whole stack runs over the window and the
description always comes out of the top, size-free, which is why attending an edge and
attending the monitor feel the same: they are the same. The top has a fixed budget, so what
is gained in the small is given up in the large. The early maps stay live so that the window
can shrink without moving the eyes. This is also the untying of depth from scale that the
2026-09-23 experiment was circling.

**Dynamic pooling.** A cell's response is its drive divided by the summed activity of a pool
of neighbours (normalisation). Multiply the drive by a gain field before the division and
three things follow: cells under the bump go up, cells outside go down, and with two stimuli
in a receptive field the attended one takes nearly all the share, so the field shrinks and
shifts onto the bump. In the column: feedforward on the basal side, feedback gain on the
apical tufts in layer 1 (apical input multiplies basal), parvalbumin cells as the divisive
pool, somatostatin cells as the surround, VIP cells as the top-down release. A column does
not send its value up a wire; it sends value times gain into a competition refereed from
above. A transformer's attention head is the same formula with the gain set by content.

**AND and OR.** The kernel is an AND across features and positions: it builds a new feature
and keeps position. Pooling is an OR across positions within one feature: it drops position
and buys tolerance. The OR must be max-like, not a mean, or a small thing is diluted into its
surroundings (and angles cancel). The description at the top needs to be comparable,
repeatable and nameable; it does not need labelled units or sparseness, though competition
produces sparseness on its own. The gradient through a max goes only to the winner, which
is competitive learning, which is the August finding that competition creates identity.
Position is dropped locally and kept coarsely, but trained CNNs are more bag-like than the
architecture forces (a patch-vote network gets most of ImageNet; shuffled tiles keep most of
the accuracy), and so is the glance (illusory conjunctions). Relative position is either
built into every level (capsules) or verified in time by a loop with poses in working
memory.

**The loop.** Glance with a flat gain; for each candidate predict the part at each pose;
attend where the candidates disagree most; compare; update the shares; settle or go again;
if nothing survived, count the new entries. One table, used three ways: forward for
prediction and rendering, backward for integrating evidence, and its disagreements for
where to look. Unlike analysis by synthesis it predicts one window at a time, compares in
description space, and has a decision in it.

**The generative path.** Feedback outnumbers feedforward (a tenth of a thalamic relay cell's
synapses come from the retina). Feedback is modulatory with input present and drives when
input is absent (imagery, dreams). It fills in, disambiguates, predicts the next window, and
teaches the forward path: the rendered prediction reaches the same cells as the input, so
the disagreement is a local teaching signal at every level. Both directions learn from the
same coincidences. The forward path discards position, so the backward path gets position
from beside, not from above: the switches (which cell won the pool, kept locally, in the
brain the winning cell's own drive multiplied by the feedback) in perception, and a pointer
(a coarse gain bump from outside) in imagination. That is why imagery is blurry.

**The rule to test.** A forward stack of conv, ReLU, max pool with switches; a backward
stack of unpool and transposed conv; at every level the mismatch between the forward map and
the backward prediction trains both, with the graph cut between levels so nothing crosses.
Tied means the backward kernels are the forward ones in transpose; untied means they are
learned on their own, which is what the brain would have to do.

---

## `feedback_render/` — a backward path trained only by mismatch: does it render?

MNIST, three levels, a judge network to name the renders. The backward half alone, from a
frozen forward stack (trained by backprop, or random), five epochs, one seed.

**Half of it held.** The mismatch rule trains a decoder that reconstructs a digit from its
top map and switches at 98%, from trained or random features alike. **The other half did
not.** That learned decoder cannot render without the switches (45% of digits, 4 of 10 class
means), because it was trained with the positions always present and learned to need them.
The forward kernels run in transpose, with nothing learned but three gains, imagine a blurry
but correct digit from the top map alone (79 to 82%) and render 9 of 10 class means, and only
from trained features. Training the learned decoder on the no-position case as well raises
it to 61% and 6 of 10, still behind the transpose. Copying the full pooled value into all
four positions saturates learned decoders to white; the mass has to be kept.

So a decoder learned by error with position given becomes an inverse that needs position,
while the forward kernels backward are position-free templates and imagine for free. The
second experiment below shows the first half of that was an artifact of training with the
positions and testing without.

Full writeup: [`feedback_render/README.md`](feedback_render/README.md).

---

## `what_where/` — content and a content-free pointer: what can the backward path render?

Lavender's proposal: extract a where code the way the top map is a what code, by summing
across channels so that it says something is here and nothing about what, and let the
backward path produce the full pre-pool map from content and where, rather than unpooling
by hand. Five where codes from a per-level energy map down to a single box and nothing; a
fixed product unpool and a learned one; two controls, where-alone (mean content, own where:
does the judge still name the digit?) and swap (own where, another class's content: which
wins?).

**With the 4x4 top map as content, content alone renders.** A learned unpool trained against
the true pre-pool map reconstructs 94% of digits and all ten class means with no where at
all, and the clean pointer (the box) adds nothing. Position inside a pool comes from the
neighbours. The previous experiment's learned decoder failed because it was trained with
positions and tested without, not because mismatch learning cannot render.

**With the content max-pooled to one 64-vector, no pointer brings the digit back.** Box and
nothing are at chance; the coarse codes that score at all are scoring on their own leak,
and the swap test says where decides there. Only the full energy map renders, and then the
content is irrelevant: it is the drawing. Nor does the network's own recogniser read those
renders (chance), and the code recomputed from them is no closer to the imagined code than
to a random digit's; with the 4x4 grid the network names its own renders at 92% and the
code comes back. A pointer says where the object is, not where its parts are. The arrangement has to live somewhere, in a grid of parts at coarse places or in
a per-part where, which is the table of parts at relative poses from the conversation.

Full writeup: [`what_where/README.md`](what_where/README.md).

---

## `attend_parts/` — attend to a part, draw the part, sweep the parts

The one-shot render is the coarse image attended as a whole, so it is blurry. The claim to
test was that attending a part draws it vividly and that sweeping rebuilds the whole. A
window is cut, zoomed to the canonical 32x32, run through the forward stack; its top map is
the what, its position and size are the where; the decoder (content alone, never shown the
position) renders it and the render is shrunk and pasted back.

**It works, and monotonically.** One window of 32 gives pixel error 0.0159 and 95.3% by
the judge; four windows of 16, 0.0128 and 97.3%; nine overlapping, 0.0106 and 98.3%;
sixteen windows of 8, 0.0040 and 98.5%, which is the judge's own accuracy on real digits.
Each window is drawn as a clean stroke at the zoomed scale. With one window attended and
the rest left to the one-look render, the picture is sharp inside the window and soft but
right outside it, and sweeping windows sharpens it region by region. But the one-look render
of a digit is already clear, so on this data the eye barely sees attention working; the
field has to be bigger than the budget for the window to matter, and that was not run. The budget is fixed, 64
features at 16 places; spend it on a quarter of the digit and that quarter comes out sharp.
The pointer only places the render, and a part at a pose is the unit that works where an
object at a pose did not.

Full writeup: [`attend_parts/README.md`](attend_parts/README.md).

---

## `both_ways/` — both directions at once, from mismatch, with no labels

Everything above used a forward stack trained on the labels. Here both networks train at
the same time with no labels anywhere, each level's loss cut off from every other level,
and the question is only whether the code the network settles on is meaningful. Two rules
that both sound like "copy your twin," plus references.

**Copy your twin collapses.** Training each forward level toward the backward network's
prediction of it reaches near-perfect agreement by switching everything off: no top channel
varies, the render is flat grey, a linear readout on the code is at chance. Silence is
perfect agreement, and nothing in the rule rewards carrying information; the image at the
bottom stays badly predicted, but that error never reaches a forward weight.

**Rebuild the level below works.** Training each forward level so that its own backward
level reconstructs the level below better, the error one level down carried up through the
feedback weights, gives a stack that renders as well as the label-trained one (95%), imagines
all ten class means, and whose top map reads out at 94% with a linear classifier it was
never trained for, above a random stack's 86% and below labels' 99%. Its level-1 kernels are
faint near-point samplers, not edges; the code is meaningful by the measures, not by the
look of its features. Adding the twin rule on top does harm. This is predictive coding's
learning rule: the comparison is made one level down, and the forward synapses answer for
it.

Full writeup: [`both_ways/README.md`](both_ways/README.md).

---

## `faces/` — a deeper stack on CelebA, a backward path from mismatch, and a mustache on a woman

Grayscale CelebA at the 160x128 crop of the original, not scaled. Five levels, 16 to 256
channels, the top map 256 features at 20 places, trained on the 40 attributes (83.8%), then
a whole-face decoder and a mouth-window decoder learned by mismatch.

**It renders faces, attributes, and sharper parts.** Pixel error 0.019 from the top map;
the mean code of smiling, bespectacled, mustached, blond, bald or lipsticked faces renders a
face with that thing in its place, and the head reads most of them. Attending the mouth
window cuts the mouth region's error from 0.0161 to 0.0101, the sharpening the digits were
too small to show.

**Woman + mustache fails as intention.** Adding the mustache difference (men with minus men
without) once does nothing visible. Three times, through the mouth window, puts a plain
mustache on the face and the head says Mustache 0.94, but also Male 0.84: the direction
carries the beard, the sideburns and the age that come with mustaches in the data. Attending
the part narrows the edit to the part, not to the attribute. There is no unbundled mustache
in the means.

**Where the mustache goes can be computed from the face, sideways.** A mouth signature
matched against every cell of a shifted face's own map finds the mouth row 95 to 97% of the
time for shifts of 20 columns, and follows the face. Vertical shifts land one cell high,
because the 32-pixel cell that holds the mouth in aligned faces holds mostly nose. The grain
is the problem, not the idea.

Full writeup: [`faces/README.md`](faces/README.md).

---

## `sample_windows/` — unique and crisp: sample a face, then sample its parts under a sliding window

Lavender's proposal: sample the top code for a face that never existed, render it coarse,
then slide a window over the whole face and sample each window's missing detail conditioned
on the coarse face, so that parts agree with the whole and the blur gets redrawn crisp.

**Unique works, along the covariance.** Sampling each top unit independently gives the mean
face every time; sampling along 64 principal directions gives distinct plausible faces at two
thirds of the real spread.

**Crisp does not, from a sampler, by the numbers.** A conditional sampler of the window's
crisp code given its blurry code, trained by squared error, adds no edge energy over the
blur and costs pixel error, as a draw around an average does. Lavender's eye reads the
samples as somewhat sharper and more often right in their details than the mean, and the
crude sharpness score cannot tell a right detail from a wrong one, so that is left open.
Picking the nearest real window instead is nearly as sharp as real codes allow and the worst
by pixel error: crisp pieces of other people that agree with nothing, and a retrieval
shortcut rather than a generation. Two fixes for the seams: feathered blending removes them
and a third of the measured sharpness with them, and 40x32 windows on a 7x7 grid show what
the window decoder can do with correct codes, a rebuild at a fifth of the coarse render's
pixel error that is recognisably the person, while the sampled codes for those small windows
come out as mottled texture. The codes are the whole problem. Sharp and consistent pull apart;
what would give both is a part chosen or redrawn under a check against its neighbours, the
routine's compare step on parts, which a fixed grid of independent draws does not do.

Full writeup: [`sample_windows/README.md`](sample_windows/README.md).

---

## `loop/` — render, read back, correct, render again

Lavender's: run the two networks as a diffusion-like loop. Ask for a code, render, read the
render back through the stack, correct the code by the difference, render again; and a
decoder trained to clean, with noise on every level's input, so the noise can be scheduled
down over the cycles.

**The correction does what it is told, and fidelity lasts one cycle.** Ten cycles raise the
agreement between request and read-back from 0.59 to 0.70 and the stack reads its final
renders better. Pixel error to the real face dips at the first cycle, the best a whole-face
render reached today, then climbs past the start: contrast pushed up, highlights blown, a
hatched texture spreading. The loop finds what the stack responds to, and the stack responds
to stripes as readily as to eyes. The denoising decoder, trained without knowing the noise
level, came out blurrier and drifted the same; blending the read into the render was harmful
everywhere. The read-back check is what today's rebuilds lacked, but it is not a prior, and
a loop that runs long needs one: a denoiser trained at matched, known noise levels, a judge
the loop is not optimising, and a stopping rule from the fidelity dip.

Full writeup: [`loop/README.md`](loop/README.md).

---

## `painter/` — draw parts from memory, look back at the whole, keep what fits

Lavender's description of imagining a face, built: a goal code, its coarse render, then 49
windows painted one at a time from a library of real parts at poses (1,000 faces), each
window's candidates looked up by the blurry region of the canvas and the winner chosen by
reading the whole canvas back and comparing with the goal. Two sweeps.

**Sharp and consistent at once, for the first time today.** Agreement with the goal goes
from 0.59 for the parts as retrieved to 0.77 chosen by the look-back, against 0.80 for the
face's own parts, with sharpness at the ceiling; random choice is worse than nearest, so the
gain is the choosing. Pixel error stays above the coarse render's because the parts are
other people's, chosen to fit, which is what imagining from memory is. Imagined faces come
out as sharp, coherent, distinct people, at 0.76 agreement against the smudge's 0.50. The
second sweep helps because the first paints against blur. The fine look-back, added with
equal weight, took over the score and chose smooth generic parts; mis-weighted, not
disproven. The rebuild machinery was never the problem; where an imagined face's part codes
come from was, and the answer is memory plus the look-back. It is the first day's loop, run
for imagination. Offered mustached parts at the lip windows, the painter picks the faintest
and the head sees nothing; with the goal also asking for the mustache, she gets a faint one
(P(Mustache) 0.11, No_Beard 0.76) and stays a woman (P(Male) 0.11), where code arithmetic
had given a mustache and a man together. Given only lines, ovals, arcs and dots on blank
paper, the painter's drawing reads back to the stack as the goal better than the coarse
render does (0.63 against 0.58) and carries the attributes, but it is not a child's drawing:
filled blobs where the hair is dark and lines along edges, not a circle with two ovals and an
arc. It draws what the stack responds to, shading and edges, and the stack has no outline
parts to draw with. The parts have to be in the representation.

Full writeup: [`painter/README.md`](painter/README.md).
