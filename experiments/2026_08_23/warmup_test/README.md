# `warmup_test/` — does feedback help once templates have settled first?

(2026-08-23, from the day log.) Script: `run_warmup_test.py`. Results: `results/report.md`.

## Warm-up test — predictions (before running, 2026-08-23)

`run_warmup_test.py`: does feedback help once templates have settled
unsupervised first? Fashion-MNIST, top-1, B-contrast gamma=2, 8 seeds,
three paired conditions: off/off, on/on, off/on (warm-up).

1. The hypothesis under test: warm-up beats always-on (early feedback is
   noise from a random W2 and contaminates the bootstrap; late feedback
   acts on settled templates and a meaningful W2).
2. The bar for "feedback helps at all": warm-up beats *off* on probe_l1
   or pair_margin in >=7/8 seeds or p<0.05.
3. Honest counter-hypothesis, from the A/B post-mortem: Variant B is
   inert because the top-1 winner is usually already the class-appropriate
   template — warm-up does not change that, so a null (warm-up ~= off) is
   the expected outcome under the "feedback is inference machinery, not a
   teacher" theory. This test discriminates the two stories.

### Outcome (same day)

**Warm-up does not unlock a benefit — warmup ~= always-on ~= off** on
every task metric, below the pre-registered bar (best: pair_margin +0.0026,
6/8 seeds, p=0.058). warmup − always is ~zero everywhere, so early-noise
contamination was never the binding problem. The reliable top-1
correlation cost persists even when feedback starts only after settling
(0/8 improved, p<0.0001). The `always` arm exactly reproduces the A/B's
k=1 condition (same seeds), confirming the pipeline.

Cumulative reading across all three rigorous tests: pair_margin shows the
same right-signed ~+0.0025 whisper at p=0.05-0.09 in every protocol —
possibly a real but minuscule effect (~2.5% relative), bought at a
reliable match-quality cost. Training-time feedback on this architecture
is now conclusively closed: harmful as input gain, inert as plasticity
gain, regardless of source (prototype/contrast), learning mode
(top-1/top-5), or schedule (always/warm-up). The surviving hypothesis is
feedback as inference machinery (searchlight/settling), untested by
design.
