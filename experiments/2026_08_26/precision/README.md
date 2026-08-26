# `precision/` — how many bits a weight needs: rounding afterwards vs training quantized

Scripts: [`run_precision.py`](run_precision.py) (post-hoc rounding of `../base/results/weights.npz`) and [`run_qat.py`](run_qat.py) (trained with quantization). Results: [`results/post_hoc/`](results/post_hoc/) ([`generation_vs_bits.png`](results/post_hoc/generation_vs_bits.png)) and [`results/qat/`](results/qat/) ([`generation_qat.png`](results/qat/generation_qat.png), `weights_{2,3,4,6}bit.npz`).

## Weight precision (run_precision.py)

User hypothesis (from biology: synapses are messy; measured estimates
put synaptic precision near 4-5 bits): the f32 weights are overkill —
lower precision should hurt neither classification nor generation.

Post-hoc quantization of the trained base weights — no retraining.
Per-template symmetric uniform rounding to b bits (+ one f32 scale per
template = renormalization; template mean needs no correction since
queries are mean-centered). Arms: f32, f16, 8, 6, 4, 3, 2 bits (2 =
ternary). Per arm: hard readout, label-consistency, generation row.

Predictions (before running):
1. f16 and 8 bits indistinguishable from f32 (hard within ~.003).
2. Graceful through 4-6 bits: high-dim dot products average out
   rounding noise; hard stays >= .89 at 4 bits.
3. Breakdown somewhere at 2-3 bits, generation degrading before
   classification is destroyed (rendering shows template noise
   directly; matching is noise-averaged).

## Trained WITH quantization (run_qat.py)

User follow-up: quantized weights might need to LEARN quantized — the
network should settle into a solution that is self-consistent under
the constraint, not be rounded off a fine-grained solution afterwards.
Chosen variant: hidden accumulator ("without the trap") — fine f32
internal weights that all updates accumulate into, quantized expressed
weights used for every match; re-expressed after each update. This
avoids the sub-notch freeze (updates smaller than the rounding grid
would otherwise round away). Arms: 6, 4, 3, 2 bits on the exact base
rig; compare to the post-hoc arms at the same bits.

Predictions (before running):
1. At 6 bits, identical to f32 and to post-hoc (nothing to recover).
2. The QAT-minus-post-hoc gap grows as bits shrink; at 2 bits
   (ternary) it is large, and trained ternary stays a working network
   (hard >= .80, generation recognizable) while post-hoc ternary is
   predicted to break.
3. Locus of the gain: winner re-partitioning — competition reassigns
   windows around the representable template positions.

### Round 1 outcome — ARTIFACT, both experiments

Post-hoc: f32 .9112 / f16 .9110 / 8b .9064 / 6b .8274 / 4-3-2b all
exactly .1084 (chance), consistent 10/10 everywhere. QAT: 6b .8376,
4-3-2b the same .1084 — the accumulator recovered nothing. Three
identical chance numbers + intact label retrieval = scheme artifact,
not a precision cliff. Diagnosis (measured): every top row's max |w|
is a LABEL channel, max/std = 155; with the step scaled by the row max,
4 bits zeroes 100.0% of the top's code half (memories reduced to bare
labels — hence consistent 10/10 at chance-level hard). L2 was
half-starved too (52% zeroed at 4b; max/p99.9 = 3). L1 healthy.
Methodology lesson: max-scaled uniform quantization is outlier-scaled;
with mixed-magnitude pathways in one row it deletes the bulk signal.

### Round 2 — v2 codebook (both scripts)

Step from the 99.5th percentile of |w| per row (outliers clip), and
the top's code/label halves quantized with separate scales
(per-pathway gain). Revised predictions: f16/8b unchanged; 6b now
~f32; post-hoc cliff moves to 2-3 bits; QAT gap appears at/below the
cliff (accumulator now has something to accumulate).

### Round 2 outcome — user's hypothesis CONFIRMED, plus a surprise

Post-hoc (v2 codebook): hard .9112/.9110/.9114/.9114/.9110 for
f32/f16/8/6/4 bits — precision is FREE down to 4 bits — then a gentle
slope, not a cliff: .9012 at 3, .8906 at 2 (ternary), consistency
10/10 everywhere, generation visually intact through 4 bits and still
recognizable at ternary. The user's biological estimate (4-5 bits)
lands exactly on the knee.

Trained-with-quantization (accumulator): 6b .9266, 4b **.9258** —
BEATS the f32 base (.9112) by ~1.5pp (above the ~1pp jitter band,
single run); 3b .9136 ~ f32; 2b .8934 ~ post-hoc. Coarse expression
during learning acts as a regularizer — plausibly the skeleton
principle at the synapse level: sub-step weights express as exact
zero, so the haze never speaks at match time. PREDICTION 2 largely
REFUTED though: the big QAT-vs-post-hoc gap at low bits never
appeared (2b gap +0.3pp) — the trained network is intrinsically
robust; post-hoc ternary does NOT break. And a dissociation: QAT
generation at 3/2 bits is visibly WORSE than post-hoc at the same
bits (memories accumulated under coarse competition draw messier)
while its recognition is better — recognition/generation split again,
this time along the precision axis.

Standing conclusion: ~4 bits per weight + one gain per pathway is the
operating point; training under quantized expression is the preferred
regime for recognition (free +1.5pp) as long as generation reads
memories trained at >= 4 bits.
