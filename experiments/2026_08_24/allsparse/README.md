# 2026-08-24 — `allsparse/`: All-layer sparse speech

Part of the day log [`../README.md`](../README.md).

## All-layer sparse speech — predictions (before running, 2026-08-24)

User direction: so far only L1's output was ever sparsified; L2/L3
always spoke dense. Now the full version: sparse output at EVERY level
(L1, L2, L3), with rich palettes. Symmetry note: the generative
direction is already fully sparse (hardened reads, top-1 per position
at the memory and after each expansion) — this asks whether recognition
can mirror generation.

`run_allsparse.py`: 8x8 geometry, K1=512 fixed. Grid: palettes
{standard K2=64/K3=100, rich K2=256/K3=256} x output mode {dense,
top3, top1} applied to all three levels. L2/L3 input dims are large
(4608 / 2304-and-up), so palettes are dimension-rich by construction.
L4 learns dense from whatever arrives (masked m3). New diagnostics:
per-layer top1-2 margins + crowding; weights and readout correlations
saved this time.

1. Probes survive again: all-sparse arms within ~1.5pp of their dense
   controls at both palette sizes (the 8x8 law generalizes upward —
   sparse speech is representationally cheap when dimensions suffice).
2. Hard readout compounds: all-layer sparse costs MORE than L1-only
   sparse (naming noise at three interfaces instead of one) — top3-all
   in the low-to-mid .80s, top1-all below that. If instead it holds
   near the L1-only numbers (.86), naming noise does NOT compound —
   a strong architectural result either way.
3. Palette effect at L2/L3 is the open question we measure, not
   predict confidently: rich palettes there could help (finer parts)
   or hurt the sparse arms (finer tiling -> thinner margins -> more
   flips). The per-layer margin table adjudicates.
4. Generation stays legible in all arms — memories store sparse-ish
   codes that the hardened read would have produced anyway; possible
   that all-sparse generation is the cleanest yet (recognition-time
   code now matches generation-time statistics).
5. Consistency 10/10 everywhere (label retrieval is robust).

### Outcome (same day) — compounding CONFIRMED and worse than
### predicted; a total recognition/generation dissociation

`results/` (weights_*.npz + corrs_*.npz saved per arm):

| K2/K3 | out (all) | probe L2 | probe L3 | hard | margins L1/L2/L3 |
|---|---|---|---|---|---|
| 64/100 | dense | .9666 | .9664 | .9080 | .038/.053/.053 |
| 64/100 | top3 | .9672 | .9638 | **.7596** | .037/.188/.173 |
| 64/100 | top1 | .9562 | .9520 | **.7092** | .038/.165/.223 |
| 256/256 | dense | .9662 | .9606 | .8914 | .038/.035/.021 |
| 256/256 | top3 | .9648 | .9606 | **.7826** | .038/.138/.130 |
| 256/256 | top1 | .9524 | .9386 | **.5982** | .038/.217/.206 |

(L1-only sparse reference: top3 .8642 / top1 .8590)

- P1 ✓ mostly: probes hold within ~1-2.8pp (top3 arms ~equal dense).
- P2 ✓ and worse than the band: all-layer top3 .76 vs L1-only .86 —
  each sparsified interface costs ~5-10pp of lookup, roughly additive.
  Strict top1 with rich palettes: .5982, floor of the day.
- P3 adjudicated: rich L2/L3 palettes mildly help top3, crush top1
  (.71 -> .60), mildly cost dense — richer tiling thins dense-side
  margins (.021 at L3), more flips to commit to.
- MARGIN SUBTLETY: sparse arms show LARGE L2/L3 margins (.13-.22) —
  sparse input makes upper choices locally DECISIVE. Decisive is not
  stable: L1's coin-flip margin (.038) is unchanged, and each decisive
  upper layer commits confidently to whatever the flipped input
  implies. Compounding = confident amplification of one noisy
  interface, not indecision at three.
- P4 ✓✓ THE DISSOCIATION: generation is CLEAN in every arm — the
  .5982-hard arm (worst classifier of the day) draws one of its best
  digit sets, label retrieval 10/10 everywhere. Recognition-by-lookup
  and generation have fully decoupled: stored prototypes are coherent;
  only instance-matching of live codes dies.

### Files

- `run_allsparse.py` — the experiment above; saves `weights_*.npz` and
  `corrs_*.npz` per arm into `results/`.
- `show_templates_sorted.py` — all 512 L1 templates, sorted so near-duplicate
  families sit side by side (`results/templates_8x8_K512_sorted.png`); this is
  the sorted gallery quoted in `../residual/README.md`.
- `show_upper_templates.py` — L2 and L3 templates drawn as pixels
  (`results/parts_L2.png`, `results/parts_L3.png`).
- `show_l4_palette.py` — all 200 top-layer memories grouped by label
  (`results/palette_L4.png`).
- The saved weights here are also read by
  `../feedback_program/run_searchlight.py`.
