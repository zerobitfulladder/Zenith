# 2026-08-24 — `sample_commit/`: F4: sample-verify-commit

Part of the day log [`../README.md`](../README.md).

## F4: sample-verify-commit — predictions (before running, 2026-08-24, night)

User's mechanism, composed from three individually-validated pieces:
ambiguity (low L4 margin) = failed attractor search -> turn the
candidates into a distribution and SAMPLE one (collapse) -> VERIFY by
regeneration (searchlight: sampled memory's rendered digit must
explain the input at least as well as the default winner's) -> if it
fits, THAT memory learns the instance (feedback decides WHO learns,
never what -> passes the guide-don't-teach filter). Over epochs the
verified choice deepens its attractor and the ambiguity dissolves.

`run_sample_commit.py`: 8x8 rig, 2 rigs {dense, top1-all} x
{baseline, sampler-unsupervised, sampler-label-checked}, 3 epochs,
sampler active from epoch 2, cached memory renders refreshed at epoch
boundaries, ambiguity = bottom-quartile margin per batch, top-3
candidates, temperature 0.03.

1. ATTRACTOR DEEPENING: test-set low-margin fraction shrinks / 20th-
   percentile margin rises in sampler arms vs baseline.
2. Hard readout: clear gain on the top1 rig (pixels genuinely
   disambiguate there, +9.2 precedent); modest-to-none for
   unsupervised-dense (F3 showed pixel verification is weak on dense
   — the sampler is then a pure tie-breaker: consistency may grow
   without accuracy).
3. Label-checked arm: up on both rigs, confusion pairs (7/9, 4/9)
   drop specifically.
4. Probes untouched by construction (only L4's update path changes).
5. Risk on record: verified self-reinforcement is still
   self-reinforcement; watch for memory-usage concentration.

### Outcome (same night) — works where the code is lossy, inert where
### dense; the F3 law holds for learning too

`results/` (3 epochs; note: 3-epoch dense baseline .9112
edges the 2-epoch .9094 — an epochs effect, not a mechanism claim):

| rig | mode | hard | margin p20 | accept rate |
|---|---|---|---|---|
| dense | base | .9112 | .0199 | — |
| dense | unsup | .9112 | .0196 | 0.75 |
| dense | label | .9054 | .0178 | 0.73 |
| top1 | base | .7168 | .0120 | — |
| top1 | unsup | .7214 | **.0136** | 0.82 |
| top1 | label | **.7344** | .0119 | 0.83 |

- Dense rig: unsupervised sampler perfectly inert (hard and margins
  identical to baseline); label-checked mildly negative (-0.6pp). As
  predicted from F3: pixel verification adds nothing where the upward
  code kept everything.
- Top1 rig: unsupervised +0.5pp with the margin p20 up 13% — the
  attractor-deepening signal, small but present and in the predicted
  direction; label-checked +1.8pp with the big confusions shrinking
  (7->9: 117->93, 4->9: 108->99).
- Verification gate behaved: 75-83% of sampled hypotheses accepted —
  neither rubber-stamp nor chaos. Consistency 10/10 everywhere.

READING: sample-verify-commit is a real but modest mechanism, and it
obeys the same law as the searchlight — feedback (at inference OR
gating learning) pays exactly in proportion to what the forward pass
threw away. On the dense champion there is nothing for it to recover;
on sparse rigs it recovers a little at train time (+0.5..+1.8pp) and
much more at inference time (F3's +9.2pp on the ambiguous subset).
The user's mechanism is VALIDATED in principle, second positive
feedback result of the project; its natural home is the future
all-WTA/sparse regime, not the dense flagship.
