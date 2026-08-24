# 2026-08-24 — `feedback_program/`: Feedback program F1-F3: label channel, gain, searchlight

Part of the day log [`../README.md`](../README.md).

## Feedback program — predictions (before running, 2026-08-24, late)

User authorization: retry top-down feedback on the SPATIAL rig — the
8-seed feedback closure was measured in the single-column era and may
not transfer. Constraints kept in mind: gain-as-teacher died there;
feedback-as-CHANNEL has a living precedent (L4's label half) and an
archaeological one (2026-03-28 CSHL note: label concatenation at a
single layer forced separate templates for confusable digits, ranking
survived zero-label queries; full-strength labels collapse to
one-template-per-class -> the channel must be SMALL). Three chained
experiments on the 8x8 rig:

F1 `run_feedback_channel.py` — label as a channel at L2/L3 windows
during training (gamma=0.35, zeros at query), arms {dense, all-top1}
x {gamma 0, 0.35}:
1. dense arm: channel costs nothing or small gain;
2. top1 arm: channel helps MORE than dense (class-splitting attacks
   the sparse regime's near-tie disease directly);
3. risk on record: label-norm wasted at query (the lam=1.0 lesson).

F2 gain retry — class-prototype multiplicative gain on the L4 message
(training only). Exploratory: if harmful/inert here too, the gain
closure generalizes to spatial rigs; if helpful, the single-column
caveat was real.

F3 inference searchlight (no training) — render all 200 memories once;
for low-margin queries, re-score top-3 candidates by pixel agreement
with the actual input:
1. low-margin subset accuracy improves;
2. net hard readout +1-2pp;
3. classic confusion pairs (1/7, 4/9, 3/5) reduce.

### Outcome (same night) — the feedback program: three clean verdicts

**F1 label channel** (`results/channel/`):

| speech | gamma | probe L3 | hard | top confusions |
|---|---|---|---|---|
| dense | 0 | .9656 | .9060 | 4->9:42 |
| dense | 0.35 | .9668 | .9060 | 4->9:64 |
| top1 | 0 | .9464 | .7268 | 7->9:106 |
| top1 | 0.35 | .9462 | **.4646** | 7->9:221 |

Free on dense; CATASTROPHIC on top-1 (-26pp). Mechanism: training-time
winners are chosen WITH the label in the window, query-time without it
— in the top-1 regime the winner's name IS the message, so stored and
query codes systematically disagree. The archaeology's "ranking
preserved at zero label" held for WHOLE DIGITS (classes differ
visually); an 8x8 fragment is class-neutral (a 1's stroke = a 7's
stroke), so split templates have no visual basis to re-derive the
split. LAW: class identity cannot be pushed into mid-level parts;
parts are class-neutral by nature.

**F2 gain retry** (`results/gain/`): class-prototype
multiplicative gain on the L4 message, beta=0.5, train only — dense
.9042 (vs .9060 baseline: inert, within jitter); top1 .7114 (vs
.7268: mildly harmful). The single-column-era gain closure GENERALIZES
to the spatial rig.

**F3 inference searchlight** (`results/searchlight/`): no training;
bottom-20% margins re-scored by pixel agreement with top-3 candidate
renders:

| rig | overall | low-margin subset (1000) |
|---|---|---|
| dense baseline | .9080 | .7940 |
| dense + searchlight | .9070-.9078 | .7890-.7930 (inert/slightly negative) |
| top1 baseline | .7092 | .4670 |
| top1 + searchlight | **.7276** | **.5590 (+9.2pp)** |

Robust across blend weights (plateau .7274-.7276). THE RESULT: going
back to the pixels recovers real information exactly when the upward
message was lossy (sparse), and nothing when it was dense — on the
dense rig L4's ambiguity is genuine (the pixels agree with both
candidates about equally).

PROGRAM SYNTHESIS: feedback in this architecture is not a teacher
(re-confirmed twice, now on the spatial rig, in both content and gain
form) — it is a CONSULTANT AT INFERENCE: re-examine the evidence when
the summary was lossy or the decision is close. Its value is exactly
proportional to how much the forward pass threw away.

### Files

- `run_feedback_channel.py` (F1) -> `results/channel/`
- `run_feedback_gain.py` (F2) -> `results/gain/`
- `run_searchlight.py` (F3, no training; loads the saved all-dense and
  all-top1 weights from `../allsparse/results/`) -> `results/searchlight/`
