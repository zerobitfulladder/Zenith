# Part three: three layers, a label on the top, read from both ends

    layer 1   4x4 patches, 7x7 = 49 positions, ONE shared hypercolumn
    layer 2   3x3 windows of layer-one codes, stride 2, 3x3 = 9 positions,
              ONE shared hypercolumn
    layer 3   all 9 layer-two codes with the one-hot label concatenated
              on the end; winner-take-all, so a template is a whole
              (image, label) pair

Trained on (image, label) pairs, 20k digits, then asked two questions
with the *same* partial-cue read — write one half, read the other off the
winner:

* **classification** — write the image half, read the label back
* **generation** — write the label half, read the image code back, and
  push it down through layer two and layer one to pixels

Layers one and two are run both ways: `dense` (all templates score and
learn, this folder's rule) and `wta` (one winner). 263 s for everything.

## Energy: the knob, not a detail

With 10 label cells against 1152 image cells the label is 0.3% of layer
three's vector and the competition ignores it outright. The label block
is therefore rescaled to **rho** times the length of the image block
before the two are joined, and rho is swept rather than guessed.

| rho | 0.1 | 0.3 | 0.6 | 1.0 |
|---|---|---|---|---|
| **dense** accuracy | 0.9096 | **0.9284** | 0.9232 | 0.9134 |
| class purity of a stored pair | 0.934 | 0.989 | 1.000 | 1.000 |
| wta accuracy | 0.4702 | 0.5010 | **0.5708** | 0.5588 |

Purity is what rho actually buys: at 0.1 a stored template is 93% one
class, by 0.6 it is 100% — the label has become strong enough to split
templates that look alike but are labelled differently. Accuracy peaks
just after that and then *falls*, because past the point where classes
are already separated, more label energy only crowds out image detail.
**rho ≈ 0.3–0.6 is the whole usable band**, and the cost of leaving it
at its natural value would have been the entire experiment.

## Classification

Image in, label read off the winner, 5000 held-out digits, never seen
with a label:

| | accuracy |
|---|---|
| **dense layers 1-2** | **0.9284** |
| *logistic regression on raw pixels (the judge)* | *0.9074* |
| wta layers 1-2 | 0.5708 |

The stack reads its own label back better than a logistic regression
reads the pixels. Full per-class report in
`results/classification_report.txt`; the confusion matrix is the
expected one, 4/9 and 3/5 taking most of the damage.

**Winner-take-all at layers one and two loses two thirds of the errors'
worth of accuracy** — 0.93 to 0.57. This is the sharpest form of the
finding from Part two: layer three is a *correlation* matcher over whole
vectors, and a top-1 code hands it one number per position where the
dense code hands it 64. The templates layer three matches on are not
things it names, they are coordinates, and it wants as many as it can
get.

## Generation

`generated_dense_rho0.6.png` — the label alone, nothing else, pushed all
the way back down to pixels. Judged by the logistic regression above:

| | digits the judge agrees with |
|---|---|
| dense, graded read (top 32 winners blended) | **10 / 10** |
| dense, top-1 read | 8 / 10 |
| wta, graded | 8 / 10 |
| wta, top-1 | 5 / 10 |

Both reads are legible. The graded row is what the class *averages* to —
a soft, thick, unmistakable digit. The top-1 row is one particular stored
digit, sharper and more idiosyncratic, and it is the one that sometimes
misses: a stored 6 that leans, an 8 that reads as a 5. That is the
codebook trade in miniature, one rung up — the average is safer, the
individual is more specific.

The grey speckle in the background is honest and worth naming: the way
down inverts two normalizations, and what comes back is a patch's *shape*
scaled per position, so flat background does not return exactly flat.
Image -> code -> image error is 0.733 for dense and 0.755 for wta, so the
stack is not a good autoencoder in either mode — layer two's 576 -> 128
bottleneck throws most of it away. It is good enough to name a digit and
to draw one, which is what was asked.

## What this adds

Part two showed a dense layer feeds an upper layer fine. This shows the
same thing holds three rungs up and in both directions, with a label
bound in at the top: a stack whose lower templates mean nothing
individually classifies at 0.93 and draws a recognizable digit from a
label alone. The identity that generation needs lives entirely in layer
three, where templates *are* nameable — every one of them is 100% one
class at rho >= 0.6 — sitting on two rungs of coordinates that are not.

Which is the rung question from Part two, answered by construction rather
than argument: identity appeared exactly where something needed to be
named, and nowhere below it.

## Files

| | |
|---|---|
| `stack3.py` | the three layers, both substrates, the up and down paths |
| `run_stack3.py` | training, both reads, the rho sweep, every figure |
| `results/label_energy_sweep.png` | accuracy, generation and purity vs rho |
| `results/generated_*.png` | digits drawn from the label alone |
| `results/confusion_*.png` | image in, label out |
| `results/classification_report.txt` | per-class precision/recall/f1 |
| `results/l1_templates_*.png`, `l2_templates_*.png` | what the rungs learned |
| `probe_stack3.py` | why generation is dithered, and why layer two's templates all look alike (no writeup kept) |
| `probe_dither.py` | is layer one overcomplete — 64 templates for a 16-dimensional patch? (no writeup kept) |
| `probe_k2.py` | with layer one fixed at k1=16, what is the right k2? (no writeup kept) |

    python experiments/2026_08_30/stack3/run_stack3.py     # ~4.5 min, both substrates, four rho values
