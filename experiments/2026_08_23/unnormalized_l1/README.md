# `unnormalized_l1/` — dropping mean-centring and unit length at L1 (softmax-sampled, then attention-style)

(2026-08-23, from the day log.) Scripts: `run_softmax_nonorm.py`, `run_attention_nonorm.py`.
Results: `results/softmax/`, `results/attention/`.

## Softmax-sampled, unnormalized L1 — predictions (before running, 2026-08-23)

`run_softmax_nonorm.py`: 2-layer rig where L1 drops mean-centering and
L2-normalization entirely. Logits = raw dot products / tau; LEARNING =
sample ONE winner from softmax and blend it toward the raw patch
(w += eta(x - w), the flat-space cousin of geodesic rotation);
OUTPUT = the softmax probabilities (graded); GENERATION = hardened
argmax read (dual-mode). Top layer unchanged (validated concat Zenith).
tau swept {0.5, 0.2, 0.1}. Intensity floor replaces the contrast floor
(skip near-black patches so uniform sampling can't drag templates dark).

1. The user's conjecture: stochastic selection replaces normalization as
   the anti-monopoly mechanism — usage entropy stays high, zero dead
   templates, at some tau.
2. Templates become stroke-like but carry absolute-brightness structure
   (the component centering used to remove). On MNIST's uniform black
   background this may cost little.
3. Probe/readouts below the validated normalized rig, but possibly
   surprisingly close — genuine uncertainty, that is the experiment.
4. tau is the sensitive knob: too hot -> no specialization (blurry
   near-identical templates); too cold -> monopoly returns.

### Outcome (same day) — split verdict, two surprises and two diseases

| tau | probe | hard | entropy | dead |
|---|---|---|---|---|
| 0.5 | 0.9134 | 0.1388 | 0.995 | 0 |
| 0.2 | 0.9150 | 0.1086 | 0.988 | 0 |
| 0.1 | 0.9020 | 0.1068 | 0.986 | 0 |

- **P1 CONFIRMED — the user's conjecture holds**: with zero centering
  and zero normalization, sampling alone keeps usage near-perfectly
  uniform (entropy 0.99, 0 dead). Stochastic selection IS a full
  replacement for geometry as the anti-monopoly mechanism.
- **Surprise: the representation survives** — probe 0.915, equal to the
  validated normalized rig (0.9146). The entire Pearson machinery is
  apparently OPTIONAL for linear class information at L1 (on MNIST's
  uniform-background data; caveat noted).
- Disease 1: generation is blobby because sampled learning blurs — and
  the unification is the finding: sampled-top-1 learning IS graded
  learning in expectation (each template's expected update is its
  softmax-weighted blend), i.e. the very rule the 8-seed A/B refuted;
  the blur arrived via variance instead of directly. Argmax learning
  remains the only non-blurring rule, now for a proven reason.
- Disease 2: hard readout collapses to chance (0.11-0.14) — softmax
  normalizes each position to sum 1, destroying the magnitude/energy
  signal, exactly as predicted when softmax communication was first
  debated; nearest-memory matching dies without it (the probe, which
  reweights freely, does not care). Label retrieval stays 10/10.

Net lessons, now separately measured: learning must be argmax;
anti-monopoly can be bootstrap OR sampling; output must carry magnitude
(relu-family, never per-position normalized); input normalization is
possibly droppable — untested corner: argmax learning + no
normalization (brightness-monopoly risk without sampling's protection).

## Attention-style unnormalized L1 — predictions (before running, 2026-08-23)

`run_attention_nonorm.py`: no centering/normalization, ARGMAX learning
(blend update), output = softmax(dots/tau) * relu(max dot) — attention
weights scaled by value magnitude, restoring the energy signal bare
softmax destroyed. Generation top-1. tau in {0.5, 0.2}. Top layer
unchanged.

1. The untested corner, now tested: argmax + no normalization has
   neither geometric nor stochastic monopoly protection. Entropy is the
   open question — bootstrap-in-hull + blend updates may suffice, or a
   brightness monopoly may form. No commitment; this IS the experiment.
2. Dictionary sharper than the sampled run (argmax does not blur in
   expectation) -> generation visibly less blobby.
3. Hard readout recovers well above chance from 0.11 (magnitude
   restored); if the energy theory is right, potentially into relu-all
   territory.
4. Probe holds ~0.91.

### Outcome (same day) — the corner resolves: monopoly returns; the
founding axiom is hereby DERIVED

tau=0.5: probe 0.8990, hard 0.7540, **entropy 0.318**, dead 0, 10/10.
tau=0.2: probe 0.8958, hard 0.7684, **entropy 0.318**, dead 0, 10/10.
(Entropy is bit-identical across taus — learning is argmax over raw
dots and never sees the temperature, so both arms share one learning
trajectory; the monopoly is a fact about the dynamics, not the knob.)

- P1 resolved negative: argmax + no normalization -> usage collapse
  (0.318). Bootstrap prevents literal death but not conquest; a few
  bright templates win everything. Generation renders from a tiny
  vocabulary -> fails visually.
- P3 confirmed emphatically: restoring value magnitude to the softmax
  message lifts hard readout 0.11 -> 0.754. Memory matching lives on
  the energy signal, as theorized since August.
- The completed 2x2 across three experiments:
    normalization + argmax  -> sharp AND diverse (all validated rigs)
    no-norm + sampling      -> diverse but blurred (sampled softmax run)
    no-norm + argmax        -> sharp but monopolized (this run)
  **Pick two: sharp learning, diverse usage, no normalization.** The
  only configuration with both sharpness and diversity is the
  normalized one — Zenith's founding mean-center + unit-norm decision
  is no longer an assumption; it is derived from the measured failure
  of both alternatives.
