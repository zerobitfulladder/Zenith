# `residual_full_gpu/` — full stack + GPU port, and the GPU batch-size sweep (Exps 5 and 10)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_residual_full_gpu.py` — the GPU trainer/evaluator (also imported as `G` by several
  other experiments of the day). `RF_B` sets the batch size, `RF_TAG` the results subfolder.
- `show_l3_roundtrip.py` — the dedicated from-L3 gallery on Exp 4's seq weights
  (writes `results/l3_roundtrip_pairs.png`).
- `results/` — Exp 5 (B=128): metrics.json, ladders, galleries, run log.
  `results/b32/`, `results/b8/`, `results/b2/` — the Exp 10 batch-size sweep; logs
  `results/run_gpu_b*.log`.

## Exp 5 — run_residual_full_gpu.py: full stack + GPU port

User request: residual learning on the whole architecture, on GPU.
CuPy mini-batch per the verified run_gpu_minibatch.py pattern (f32,
batched geodesic step toward the normalized mean of batch targets,
theta = clip(eta*sum(c), 0.3)). Residual learning generalizes as: up to
R_TRAIN rounds on the layer's OWN learning view (L1 dense window,
L2/L3 skeleton), rounds=1 == the old rule. NEW residual READS at the
deep rungs: each position's centered block is residual-coded against
its bank (few committed voices instead of dense-sum blur or top-1
loss), with block mean/norm side channels restored at expansion.
Controls: parity read (Exp 1 exact) / side-channel graded / residual.
Arms: exp4seq (no training — GPU eval of Exp 4's CPU weights),
l1only (GPU-trained, residual at L1 only), full (residual everywhere).

WALL TIME: 47s for BOTH 55k trainings plus all 5000-image evals — the
CPU equivalent this afternoon was ~50 min. GPU eval VERIFIED: exp4seq
reproduces every CPU number to the 3rd decimal (0.969/0.938/0.884/
0.847). Three findings:

1. THE DEEP RUNGS FINALLY MOVED — residual READING at L2/L3 works on
   well-trained banks (exp4seq arm): from-L2 0.884 -> 0.904, from-L3
   0.847 -> 0.857, both day records, achieved on the EXISTING Exp 4
   weights. Detail: a SINGLE committed L2 voice with side channels
   (R1 0.893) already beats the whole dense sum (0.884) — one
   committed voice beats the blurred choir; saturates at 2 voices
   (0.903/0.904/0.904). Side channels alone: wash (0.881) — the gain
   is the multi-voice coding, not the bookkeeping.

2. GPU MINI-BATCH TRAINING IS FINE FOR DENSE READS, DULL FOR RESIDUAL
   READS: l1only matches exp4seq on every parity/dense number (L2
   parity 0.882 vs 0.884) but loses under residual reads everywhere
   (L1 R6 0.960 vs 0.969; L2_res 0.843 vs 0.904; matchq rounds 2+
   lower). The mean-of-targets batch update blurs the committed
   fine structure that single-voice reads depend on. META-LESSON
   (peakiness's sibling): the residual read is a SHARPER INSTRUMENT —
   it distinguishes training regimes that dense metrics call
   identical to three decimals.

3. FULL-STACK RESIDUAL LEARNING: REFUTED IN THIS REGIME. Residual
   rounds at L2/L3 (skeleton view) made things worse even vs l1only
   (L2_res 0.819 vs 0.843; L3_res 0.752 vs 0.794; round-1 matchq
   DROPPED 0.504 -> 0.458). The L1 recipe does not transfer to the
   code levels as-is. CONFOUND, honestly held: tested only under
   mini-batch training, which finding 2 shows independently dulls
   residual structure — the online-regime full-stack run remains
   open before this verdict is final.

BEST LADDER OF THE DAY (final): L1 0.969 / L2 0.904 / L3 0.857 —
Exp 4's CPU-trained seq weights + residual reads at every rung.
Galleries: ladder_exp4seq.png (L2 residual visibly crisper than the
dense sum; the 8's double loop survives from-L3). Queue updates:
online-regime full-stack (kill the confound); mini-batch variant that
preserves committed structure (smaller B / per-round updates / cap
sweep); then the rest as previously queued.

## Exp 10 — GPU batch-size sweep: can the GPU train the recipe?

Follow-up to Exp 8's reversal: the harm was minibatch averaging, so
shrinking B should converge toward the online rule. Full-stack
residual learning, RF_B in {128, 32, 8, 2}, full eval each:

| corr | B=128 | B=32 | B=8 | CPU online |
|---|---|---|---|---|
| L1 res R6 | 0.959 | 0.966 | 0.966 | 0.969 |
| L2 parity | 0.882 | 0.890 | **0.898** | 0.892 |
| L2 res R4 | 0.819 | 0.813 | 0.881 | 0.909 |
| L3 parity | 0.851 | 0.860 | **0.868** | 0.866 |
| L3 res | 0.752 | 0.785 | 0.844 | 0.867 |
| train | 20s | 37s | 148s | ~1800s |

VERDICT: B=8 recovers the dense/parity reads FULLY (L2/L3 parity now
match or beat CPU-online) and ~2/3 of the residual-read gap, at 12x
CPU speed. The residual reads remain the sharp instrument that still
sees the batching (0.881/0.844 vs 0.909/0.867). B=32 is a bad middle
(mixed, L2_res no better than 128).

B=2 CONVERGENCE CHECK: **full parity with CPU-online** — L1 res 0.969
(=), L2 res 0.908 (vs 0.909), L3 res 0.868 (vs 0.867), and the dense
reads slightly BETTER (L1 graded 0.945 vs 0.938, L2 parity 0.901 vs
0.892) — in 602s vs ~1800s. Batch averaging was the ENTIRE
CPU-vs-GPU difference; nothing else (f32, theta-cap, frozen-batch
winners) matters at B=2. CLOSED: the GPU trains the full recipe
outright. Speed/quality dial: B=128 20s (draft) / B=8 148s
(near-parity iteration) / B=2 602s (gold). CPU-online training is
now redundant.
