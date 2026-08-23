# Gain Feedback (Variant A) — standalone test

## The question

When labels only exist at the top of a hierarchy, do lower layers become
task-efficient through top-down feedback — without the label ever touching
their weights? This tests Variant A from the notes
([`2026_03_20/feedback/Thoughts2.md`](../../2026_03_20/feedback/Thoughts2.md), "Hierarchical Gain-Feedback"): feedback as a
**multiplicative gain field on the lower layer's input**, produced by
projecting the top signal down through each layer's own weights.

## The rig

Two layers, MNIST, no AxonForge — plain numpy (`../rig/gain_feedback.py`).

- **L1** — a Zenith unit: 100 templates, top-1 winner-take-all, geodesic
  rotation, everything zero-mean unit-norm. Two departures from
  `experiments/2026_03_29/zenith_node/learning6.py`, both forced by raw MNIST: templates bootstrap
  from the first 100 inputs (random-Gaussian init loses the race to a
  single monopolist on cone-shaped data — seen live in AxonForge), and the
  winner is argmax(c) not argmax(|c|).
- **L2** — 10 class templates over L1's code space. The label *selects*
  which row learns (rotation toward the current L1 code, step ∝ 1 − c).
  No competition at L2; the label is the "feedback from a solved layer
  above" boundary condition from the notes.
- **Feedback loop (training only)** — for each labeled sample: take the
  true class's L2 template, project it through L1's weights into pixel
  space, z-score it, turn it into a multiplier centered at 1
  (`f = clip(1 + gamma * z, 0.05)`), and multiply the mean-centered image
  by it *before* normalization. L1 then runs completely unchanged on what
  it sees. gamma = 0 is exactly the no-feedback control.
- **Evaluation (always neutral)** — at test time there is no label, so the
  gain is all-ones. We are measuring whether the *learned structure* became
  task-efficient, not whether test-time attention helps.

`run_experiment.py` trains identical seeds/data-order at
gamma ∈ {0, 0.25, 0.5, 1.0, 2.0} and writes `results/A_prototype/`.

## What we expect (written before running)

1. **Task efficiency**: the linear probe on L1 codes and the pipeline
   accuracy improve at moderate gamma vs gamma=0, most visibly as larger
   margins on confusable pairs (1v7, 3v5, 4v9, 5v8, 7v9).
2. **Inverted U in gamma**: too weak → no effect; too strong → L1 sees its
   own expectations instead of the data and neutral-gain evaluation
   suffers. The notes flag gamma as *the* critical knob.
3. **Self-bootstrap**: class-gain distinctness starts near 0 (random L2 →
   noise gain) and grows as W2 organizes — supervision fades in by itself.
4. **Localization**: the downward difference map for 1 vs 7 concentrates
   on the horizontal-bar region — feedback pointing at the discriminative
   stroke.
5. **No collateral damage**: mean top-1 correlation (general
   representation quality) and L1 usage entropy stay roughly flat at
   moderate gamma. Generation grids (W2 @ W1) stay recognizable digits.

## How we judge

- **Works**: prediction 1 holds (probe + margins up at some gamma > 0),
  with 3 and 5 intact. Prediction 2's falling edge tells us where "too
  strong" begins.
- **Doesn't work**: metrics flat for all gamma (feedback too indirect at
  this scale) or monotonically worse with gamma (the gain only distorts).
- **Instability to watch for** (the notes' open question): usage entropy
  collapsing / dead templates rising at high gamma — the coupled
  gain-learning loop feeding on itself.

## Deliberately left out (this is the minimal test)

Iterative settling ("ping-pong" — the August experiments showed feedback
loops contract onto blends, so single down-pass only), local receptive
fields, top-k graded communication, feedback at inference time,
hypothesis-testing classification (try each label's gain, keep the one L2
likes). Each is a follow-up, not part of the core claim.

## Outcome — first run (2026-08-23)

Verdict by the criteria above: **Variant A as implemented does not work.**
Full numbers in `results/A_prototype/report.md`.

- Pipeline accuracy falls monotonically with gamma (0.70 → 0.51 at
  gamma=2); the L1 probe is flat then falls. No rising edge of the U —
  the "too strong" regime begins almost immediately.
- Mean top-1 correlation tracks gamma straight down (0.78 → 0.49):
  training on gained inputs and evaluating on neutral inputs is a
  first-order cost that swamps any task shaping.
- **The notes' stability worry is real.** The gain-learning loop
  (gain shapes W1 → W1 shapes the gain) collapses the class-specific
  content of the downward projections: at gamma=0, `W2 @ W1` renders ten
  clearly readable digits; at gamma>=0.5 all ten classes render the same
  generic blob (`results/A_prototype/generation_gamma*.png`). Label diversity did NOT prevent
  the collapse — all class gains share the average-digit structure, and
  that shared part is the attractor. Same disease as the August finding
  (feedback iteration contracts onto blends), in a new organ.
- Partial wins: the 1v7 margin *grew* under feedback (0.72 → 1.11 at
  gamma=0.5) even as everything else fell; pair-difference maps are
  stroke-shaped, not noise; L1 usage stayed healthy (0 dead templates).
- **Independent positive result:** at gamma=0 the core generative claim
  of the spec holds — the recognition stack run backwards (label → class
  template → pixel space) generates recognizable digits with no backprop
  and no decoder.
- Caveat: the class-gain distinctness metric failed as a coherence
  measure — it reads "orthogonal because informative" and "orthogonal
  because noisy" identically, so it *rose* while generation collapsed.

Ranked follow-ups:

1. **Contrast gain**: project `W2[y] - mean_j W2[j]` (the class *contrast*,
   not the class prototype) — removes the shared component that is the
   collapse attractor; this is the universal-contrast principle applied to
   the feedback path itself.
2. **Warm-up**: feedback off during epoch 1 (no bootstrap contamination,
   W2 organizes before it speaks).
3. **Hypothesis-testing evaluation**: classify by trying all 10 gain
   fields and keeping the label whose gained input L2 confirms best —
   the spec's actual inference mode; removes the train/test mismatch.
4. Small gammas (0.05–0.15) under the current protocol.
5. **Variant B** (gain on the rotation step, perception untouched) — no
   train/test mismatch by construction.

## Outcome — contrast-gain run (2026-08-23)

Prediction ("generation stays class-distinct at gamma=0.5") **failed**.
Numbers in `results/A_contrast/report.md` (gamma=0 row reproduces
the prototype run exactly, as it must).

- The collapse is *softened*, not cured: generation tiles at gamma>=0.5
  show a bit more per-class variation than prototype mode, but are still
  blobs, not digits.
- Accuracy still declines monotonically with gamma (0.70 → 0.55 at
  gamma=2; a few points better than prototype mode at 0.5 and 2.0, never
  above baseline). Top-1 correlation declines identically to prototype
  mode — the train-on-modulated / test-on-plain cost is untouched by the
  contrast fix, and it dominates.
- Why the subtraction wasn't enough: the gain is projected *through W1*,
  and every W1 template lives in the average-digit cone — so even a
  perfectly class-contrastive source re-acquires shared central-ink
  structure on its way to pixel space. The attractor is anchored in W1's
  geometry, not only in W2's shared row component.

Updated conclusion: **modifying what L1 sees during training only, with
neutral inputs at test, is net-negative regardless of gain source.** The
two designs this evidence points to instead: Variant B (gain on the
rotation step — perception untouched, so neither the mismatch cost nor
the perceptual collapse channel exists), and Variant A with feedback at
inference too (hypothesis-testing classification), which is a different
claim than the one tested here.

## Variant B — predictions (written before running, 2026-08-23)

Variant B: the gain is a relevance vector over L1's *template indices*
(source: the class's W2 row, prototype or contrast), z-scored and centered
at 1, multiplying the **winner's learning step** (theta = eta * f[winner] * c).
The winner is still chosen from raw correlations — feedback never decides
who fires, only how much the firing template learns. Run with
`GF_VARIANT=B GF_GAIN_MODE=prototype|contrast`.

1. Mean top-1 correlation stays ~flat across gamma (perception is never
   modified, so the mismatch cost cannot exist).
2. Generation grids stay class-distinct at every gamma (the collapse
   channel through pixel space cannot exist).
3. If task pressure via plasticity works at all, probe/margins/accuracy
   beat the gamma=0 baseline at some gamma.
4. Specific risk to watch: usage entropy dropping at high gamma —
   templates unfavored by the current class learn at the floor rate, a
   rich-get-richer allocation loop in template space.

Judged to work only if 3 happens with 1 and 2 intact.

### Outcome (same day)

**Variant B + contrast source is the first net-positive feedback
configuration** — weak but consistently the right sign everywhere.
Numbers in `results/B_{prototype,contrast}/report.md`.

- Prediction 1 confirmed: top-1 correlation ~flat (0.784 → 0.767 at
  gamma=2, vs 0.488 under Variant A). Perception protected.
- Prediction 2 confirmed: generation stays clearly digit-shaped at every
  gamma. No collapse.
- Prediction 3, split verdict: pipeline accuracy flat (within noise) for
  both sources. But with the **contrast** source, all five confusable-pair
  margins improve at every gamma > 0 (1v7: 0.717 → 0.760; 5v8: 0.153 →
  0.170; 7v9: 0.084 → 0.096) and the L1 probe rises near-monotonically
  with gamma (0.8708 → 0.8774). With the prototype source: pure noise.
  The shared-component lesson holds in template space too.
- Prediction 4 inverted: usage entropy *rose* with gamma (0.949 → 0.982)
  — class-gated plasticity spread the templates out rather than
  concentrating them.

Why the effect is small — structural, not tuning: with top-1 WTA, only
one template learns per tick, and the winner is usually already the
class-appropriate template. Feedback's entire lever is the step size of
that single winner. The notes' spec assumed CRG-style layers where many
templates are partially active and plasticity gating can *redistribute*
learning across the population. Next lever, if pursued: top-k or graded
learning at L1 (each active template learns ∝ f_i * c_i), giving Variant B
a population to redistribute — this is the "soft competition for
communication" thread arriving from a different direction.

## Files

- `../rig/gain_feedback.py` — the algorithm (ZenithLayer, ClassTemplates, gain field)
- `run_experiment.py` — gamma sweep, metrics, figures, report
- `results/A_prototype/report.md` — outcomes
- `results/A_prototype|A_contrast|B_prototype|B_contrast/` — one folder per variant and gain
  source (`GF_VARIANT`, `GF_GAIN_MODE`)
