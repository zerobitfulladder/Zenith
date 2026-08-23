# `ab_test/` — A/B test of feedback on the learning step (Fashion-MNIST, 8 seeds)

(2026-08-23, from the day log.) Script: `run_ab_test.py`. Shared code: `../rig/gain_feedback.py`. Results: `results/report.md`.

## A/B test — predictions (written before running, 2026-08-23)

Is the B-contrast effect real, and does graded learning give it a lever?
`run_ab_test.py`: Fashion-MNIST (harder, untouched by hyperparameter
choice — gamma=2 and contrast source were picked on MNIST), 2x2 factorial
{top-1, top-5 learning} x {feedback off, on}, 8 seeds, paired stats.
Also adds a probe on L2's code (the 10 class correlations) — note this
probe is capped by the L1 probe by construction (a 10-dim projection of
the L1 code cannot carry more information), so the question is whether
feedback closes the gap, not whether L2 exceeds L1.

1. The feedback effect replicates in sign under top-1: probe_l1 and
   pair margins up in most seeds (this is the "is it real" test).
2. Interaction: the feedback delta is larger under top-5 than top-1
   (feedback needs a population to redistribute — the lever hypothesis).
3. probe_l2 improves with feedback more than probe_l1 does (better W2
   readout subspace), under both learning modes.
4. Open, no commitment: top-5 without feedback may hurt vs top-1
   (blending risk from the August lessons) or help (smoother tiling).

"Effect is real" bar: paired delta positive in >=7/8 seeds or paired
t-test p < 0.05 on probe_l1 or pair_margin.

### Outcome (same day)

**The feedback effect did not replicate.** By the pre-registered bar
(>=7/8 seeds or p<0.05 on probe_l1 or pair_margin): not met. Full stats
in `results/report.md`.

- k=1 feedback deltas: pair_margin +0.0024 (5/8, p=0.085) — right-signed
  trend, below bar; probe_l1 slightly *negative*; acc flat. The MNIST
  single-seed positives were noise or MNIST-specific.
- Two effects ARE reliable, neither is a task benefit: usage entropy up
  (8/8 seeds, p<0.001 — feedback spreads template usage) and a small
  top-1 correlation cost (0/8 improved, p<0.001 — even plasticity-only
  feedback perturbs the templates' fit to the data distribution).
- Prediction 2 (lever hypothesis) refuted: feedback deltas under k=5 are
  *smaller*, not larger, than under k=1.
- Graded learning (k=5) is reliably harmful on its own: acc -0.008
  (0/8, p=0.01), probe_l1 -0.018 (0/8, p<0.0001). The August blending
  lesson, third sighting: multiple templates rotating toward the same
  input blurs the dictionary.

Day's ledger: Variant A harmful (mechanism identified), Variant B safe
but no reliable benefit, naive graded learning harmful, and the one
clear architectural win is the **concat top layer** — free allocation +
learned label association beat the wired selector on classification
(0.727 vs 0.699) while also providing label-only generation. Top-down
gain feedback on this rig is a dead end in all tested forms; the live
directions are the concat top, local receptive fields, and (untested)
inference-time feedback / settling.
