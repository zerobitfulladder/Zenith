# 2026-08-24 — `lvq/`: F6: corrective memory updates (LVQ)

Part of the day log [`../README.md`](../README.md).

## F6: corrective memory updates (LVQ) — predictions (before running)

The missing "no, it's not a 7": the system has only ever learned by
attraction. `run_lvq.py`: when the zero-label answer is WRONG, (a)
attract-only arm: the best TRUE-class memory takes an extra step
toward the instance; (b) full corrective arm: additionally the
wrongly-winning memory steps AWAY from the instance (repulsion at
half rate, eta_neg = ETATOP/2). Dictionaries untouched (probes
identical by construction). {dense, top1} x {base, attract, lvq},
3 epochs, corrective from epoch 2.

1. Full corrective closes a LARGE chunk of the lookup->probe gap:
   dense hard .90 -> >= .93 (biggest classification jump of the day
   if it lands); attract-only gains a fraction of it.
2. The theft pairs (4->9, 7->9, 9->4) drop hardest.
3. RISK on record: repulsion pushes memories off the digit manifold —
   generation and 10/10 label retrieval are part of the verdict.
4. Top1 rig: larger relative gain (more errors to correct).

### Outcome F6 (same night) — the "no" works, both halves needed,
### museum intact; magnitude modest at 3 epochs

`results/`:

| rig | mode | hard | corrections | headline confusion |
|---|---|---|---|---|
| dense | base | .9132 | — | 4->9:37 |
| dense | attract | .9142 | 2346 | 4->9:33 |
| dense | lvq | **.9182** | 2269 | 4->9:32 |
| top1 | base | .7414 | — | 4->9:138 |
| top1 | attract | **.7138** | 4439 | (worse) |
| top1 | lvq | **.7554** | 3344 | 4->9:**62** |

- The ordering confirms the mechanism: lvq > attract > base on dense
  (+0.5pp; new best dense number .9182), lvq clearly best on top1
  (+1.4pp; the 4->9 theft cut 55%, 138 -> 62).
- SURPRISE: attract-only is HARMFUL on top1 (-2.8pp) — pulling
  true-class memories toward contested instances without repelling
  the thief destabilizes; the "no" is what makes the "look again"
  safe. The user's critique of F5 was exactly right, twice over.
- P1 magnitude missed: predicted >= .93 on dense, got .9182. Only
  ~2.3k corrections across 2 corrective epochs (~6% error rate) vs
  constant rehearsal pulling the other way; classic LVQ runs many
  epochs. The asymptote is untested — longer-run follow-up queued.
- P3 ✓: generation clean (gallery), label retrieval 10/10 everywhere.
  Repulsion at half rate did not push memories off the manifold.

FEEDBACK/SUPERVISION LADDER, final form for the day: F5 exposure
(positive-only, data-level): +0.3 dense / +2.7 top1. F6 corrective
(attract+repel at L4): +0.5 dense / +1.4 top1, composable with F5 in
principle. F3 searchlight (inference): +9.2pp ambiguous-subset on
top1, inert dense. The lookup->probe gap (.92 -> .97) remains mostly
open on dense; delta-rule readout organ and longer LVQ runs are the
remaining levers.
