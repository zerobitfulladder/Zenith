# Punishment breaks the stream

2026-08-31. Part of the Fashion-MNIST line of the day (`../fashion/`), moved
back to split-MNIST to ask what each kind of correction does on a stream.

Split-MNIST, same counts as `../continual_fixed`, today's rule:

| | 0-4 after learning 5-9 | forgot |
|---|---|---|
| MLP 256 | 0.0000 | +0.9765 |
| old rule (label-picked) | 0.9526 | +0.0231 |
| **new grading, no correction** | **0.9620** | **+0.0180** |
| new grading + sleep | 0.9049 | +0.0756 |
| new grading + repulsion | 0.3171 | +0.6657 |

Allocation stayed perfect in every arm — 0% of the new digits went to old
hypercolumns, 0 of 13 changed what they claim. The damage never came through
learning. It came through correction.

**Reinforcement is local in time; punishment is not.** A positive step touches
only the winner of a class it already owns. A negative step targets whoever
*currently* wins — and when a new class arrives, that is systematically the old
specialists, because nobody else has learned it yet.

Sleep (self-generated negatives, Forward-Forward style) is 9x safer but sits on
the same tradeoff curve: +1.4 points stationary for 5.8 points of forgetting,
against repulsion's +21.6 for 64.4. Consolidation by disuse did not break the
curve either — the win-fraction EMA froze old experts two epochs too late.

## Files

| file | arm |
|---|---|
| `split_new.py` | new grading, with and without repulsion (`results/metrics_norepel.json`, `metrics_repel.json`) |
| `split_sleep.py` | new grading + sleep in place of punishment (`results/metrics_sleep.json`) |
| `split_consol.py`, `split_batchconsol.py` | repulsion, with experts earning resistance to it (consolidation by disuse; `results/metrics_consol.json`, `metrics_batchconsol.json`) |
