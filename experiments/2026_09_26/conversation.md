# The conversation, 2026-09-24 to 26

The thinking that led to the day's experiments, in the order it happened.

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
