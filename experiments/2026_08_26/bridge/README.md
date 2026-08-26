# `bridge/` — encoder/decoder bridge: see a digit, redraw it with another stack

Scripts: [`run_enc_dec.py`](run_enc_dec.py) (trained on MNIST) and [`run_bridge_random.py`](run_bridge_random.py) (trained only on random images). Results: [`results/enc_dec/`](results/enc_dec/) ([`reconstructions.png`](results/enc_dec/reconstructions.png)) and [`results/random/`](results/random/) ([`generalization.png`](results/random/generalization.png)).

## Encoder/decoder bridge (run_enc_dec.py) — perception->action v1

User design, aimed at robotics: two independent 2-layer stacks
(different seeds — genuinely different vocabularies), both see the
same image; a unification layer (KU=400) learns cn([enc ; dec])
jointly — cross-modal co-occurrence, the label-concat pattern
generalized to a full second modality. Inference: image -> encoder
only -> query [enc ; empty] -> winner's DECODER half reprojects down
the decoder stack -> image at the other end. (Next: decoder becomes a
temporal motor stack — see a digit, write it.)

### Outcome

UNSEEN test digits: identity preserved 7/10 (pixel-LR judge); clean
0/2/3/6/7, the odd cursive 4 still judged 4; failures on the odd 8
(->5-ish), 9 (->7-ish), and a serif on the 1 that fooled the judge.
Reconstructions are visibly the decoder's OWN handwriting, not
copies. Prediction 2 partially REVERSED, informatively: corr(input) >
corr(class mean) in 8/10 — retrieval is style-matched, so the redraw
is a cluster-canonical form closer to THIS input's style than to the
blurry class mean. "Copies your style, not your strokes." One
retrieval hop, top-1, 400 memories, ~4 min CPU total. Obvious head-
room: more units, graded retrieval, more epochs.

## Bridge generalization: random pairs -> digits (run_bridge_random.py)

User's test: train the whole bridge ONLY on random images, then show
unseen digits. Noise-trained: dead in all modes (corr .003/.047, pure
static) — the vocabulary floor: L1 never learned strokes exist.
Scribble-trained (2-5 random straight segments): top-1 corr .288
(recites nearest-scribble fragments), graded population read .351 and
visibly input-aligned — the read transfers the digit's DOMINANT
STROKES (the 7, all straight lines, reconstructs best; the 0 and 8
come out as open arcs — closed loops don't exist in a straight-
segment world). VERDICT: generalization = SPAN x SMOOTH READ. The
bridge transfers exactly those features of the novel input that its
random world contained, and only through the graded read; what the
babbling never produced, imitation cannot draw.

Framing worth keeping: this IS motor babbling. Random scribbles =
babbling; the unification layer learns see<->draw correspondence from
it; novel-shape imitation then rides on that map, limited by
babbling's repertoire — the developmental story, reproduced in an
afternoon rig. Next lever: richer babbling (curved/looped segments)
should close most of the gap on loop digits.
