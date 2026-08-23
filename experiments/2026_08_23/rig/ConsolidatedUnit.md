# The Consolidated Unit

## Settled 2026-08-23, after ~25 controlled experiments in `experiments/2026_08_23/`

The standard Zenith unit. Five commitments, each one EVIDENCED — either
derived from the measured failure of its alternatives or validated
directly. Do not change any of them casually; every alternative below
was tried and lost with numbers.

## 1. Normalized Pearson core
Mean-center and L2-normalize every input window (with a contrast floor —
blank windows stay silent). Match = dot product = Pearson correlation.
**Evidence:** the pick-two triangle. {sharp learning, diverse usage, no
normalization} — only two are ever achievable. Unnormalized + argmax ->
usage monopoly (entropy 0.32); unnormalized + sampled selection ->
diverse but blurred (sampling IS graded learning in expectation). Only
the normalized core gives sharp AND diverse. The founding axiom is
derived, not assumed.

## 2. Argmax rotation (learning)
Exactly one winner learns per window: argmax of correlation, geodesic
rotation toward the target, theta = eta * c. Bootstrap init (first k
inputs adopted as templates).
**Evidence:** the only non-blurring rule, proven twice — top-5 learning
degraded accuracy/probe in 8/8 seeds (p<0.0001), and softmax-sampled
selection blurred identically because its expectation is graded learning.

## 3. Skeleton learning (the learning VIEW, for code-level layers)
Mid/upper dictionaries learn from the per-position-top-1 skeleton of
their input window, while matching/communicating on the dense view.
**Evidence:** dense learning targets share a huge common mass -> content
collapse (flat templates, peakiness 0.23-0.26, identical rendered
parts — invisible to cosine metrics, measure PEAKINESS). Skeleton
targets restored peakiness to 0.53-0.72, produced the best parts ever,
and even improved the readout. Cost nothing anywhere.

## 4. Graded speech (communication)
The upward message is ALL POSITIVE correlations with magnitudes intact.
Never per-position-normalized (softmax), never negatives, never top-1.
**Evidence:** graded messages reversed the depth sag (probe ladder rises
.954/.963/.966) and set the readout record; bare softmax (sum-to-1)
collapsed matching to chance (0.11 vs 0.75+ scaled — magnitude IS the
matching signal); signed output lost 3-5pp at every probe (negative
half redundant via antipodal templates, and it dilutes); top-1 messages
starve the layer above (binding failure).

## 5. Mode-switched reads (the memory/downward side)
Stored codes are rich; the READER picks the nonlinearity:
- recognition: graded read (full profiles),
- recall/generation: hardened read (per-position top-1 at the memory and
  after every expansion),
- imagination: sample at the TOP only (temperature over the stored
  profile), hardened below — variety in what is drawn, determinism in
  how,
- rendering: feathered confidence-weighted overlap-add with squelch.
**Evidence:** identical weights produced porridge under the graded read
and clean digits hardened; sampling at every level broke coherence
(marginals != joint), sampling only at the top gave distinct legible
variants per ask.

## The one-sentence version
One competitive field over normalized contrast; the plasticity, the
message, and every reader apply their own threshold to it — strict to
learn, generous to speak, chosen-per-purpose to read.

## Flagship configuration (as of 2026-08-23)
4 layers (4x4 windows: stride-1 L1 -> 25x25; L2 3x3 s2 -> 12x12; L3 3x3
s1 -> 10x10; concat top K=200, lam=0.5), GPU mini-batch trainer
(`run_gpu_minibatch.py`, B=128, theta capped 0.3): 27s training, probe
ladder .954/.963/**.9658** (project best, rising with depth), best-ever
generation (first clean 8), hard readout .849 (lookup thins in 10k dims
— classification ultimately belongs to a learned readout organ; the
probe proves the information present).
