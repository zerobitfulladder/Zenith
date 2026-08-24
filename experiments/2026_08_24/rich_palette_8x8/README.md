# 2026-08-24 — `rich_palette_8x8/`: 8x8 windows: the repaired palette hypothesis

Part of the day log [`../README.md`](../README.md).

## 8x8 windows — the repaired palette hypothesis — predictions (before running, 2026-08-24)

The 4x4 result said sparse speech fails because a 15-dim window forces
big palettes into redundancy (margin 0.017). Repaired hypothesis:
sparse speech over a rich palette works when the palette is rich in
DIMENSIONS. 8x8 L1 windows = 63 usable dims after centering.

`run_rich_palette_8x8.py`: same 2x3 grid, new geometry — L1 8x8 s1
(grid 21), L2 3x3 s2 (grid 10), L3 3x3 s1 (grid 8), top over
[8x8x100=6400 ; 10]. K1 in {64, 512} x L1 output in {dense, top3,
top1}. K1=64 ~ complete (63 dims), K1=512 = 8x overcomplete (vs 34x
at 4x4). All else standard (L2/L3 skeleton, L4 dense). Local 8x8
feathered renderer (the shared one hardcodes 4x4).

1. Margins recover: top1-2 margin at 512 well above 4x4's 0.017
   (>= 0.05); higher still at 64. Crowding below 4x4's 0.348 at K=64.
2. THE REPAIRED BET: sparse-speech penalty at 512 shrinks vs 4x4 —
   (dense - top1) hard gap <= 8pp (was 17.2), top3 within ~2pp of
   dense on hard.
3. Scaling direction flips: the gap does NOT grow from K=64 to K=512
   (at 4x4 it more than doubled from 36 to 512).
4. Dense rows stay competitive with the 4x4 rig (probes within ~1pp);
   no commitment on whether dense@512 beats the .9066 record here.
5. Generation: top1@512 no longer wrecked (fragments gone), even if
   softer than dense.

### Outcome (same day) — dimensionality largely repairs sparse speech;
### a new hard-readout record; the last 5pp point straight at clones

`results/`:

| K1 | L1 out | probe L2 | probe L3 | hard | L1 crowd | top1-2 margin |
|---|---|---|---|---|---|---|
| 64 | dense | .9666 | .9658 | .8990 | 0.272 | 0.092 |
| 64 | top3 | .9698 | .9652 | .8864 | 0.272 | 0.092 |
| 64 | top1 | .9660 | .9658 | .8434 | 0.272 | 0.092 |
| 512 | dense | .9666 | .9668 | **.9094** | 0.256 | 0.038 |
| 512 | top3 | .9694 | .9676 | .8642 | 0.256 | 0.038 |
| 512 | top1 | .9654 | .9698 | .8590 | 0.257 | 0.038 |

- P2 ✓ on the headline: (dense - top1) hard gap at 512 collapsed
  17.2pp -> 5.0pp. Dimensionality was the dominant failure mode.
- P3 ✓: gap flat/shrinking from K=64 (5.6pp) to K=512 (5.0pp) — the
  pathological "more palette hurts sparse speech" scaling is gone.
- P1 half: margin 0.038 (up from 0.017, short of the 0.05 bar);
  crowding 0.256-0.272, below 4x4's 0.348 — and it FELL with more
  templates (room to spread beats pile-up; unpredicted).
- P4 ✓ plus a record: dense@512 hard **.9094**, new project best
  (19s train). Probes: sparse speech is FREE for representation at
  8x8 — top3 takes the best L2 probes (.9698/.9694), top1@512 takes
  the best L3 probe of the day (.9698). ALL sparse damage is L4
  nearest-memory matching, none is representational.
- P5 ✓✓: top1@512 generation is clean — arguably the best-looking
  set of the day (the 8 included). The 4x4 wreckage was pure
  near-tie noise, now confirmed by its absence.
- THE RESIDUE: top3@512 (.8642) ~ top1@512 (.8590) — two extra
  speakers buy 0.5pp. At 8x overcomplete the 3 loudest are still
  near-clones (margin 0.038 agrees). The remaining ~5pp deficit is
  the complementarity gap — exactly the residual-competition case
  (winner subtracts what it explained, survivors compete over the
  remainder), now with a precise target: close .8642 -> ~.90 at
  unchanged sparsity.

READING: sparse speech over a rich palette is viable iff the palette
is rich in dimensions (63-dim windows: probes fully intact, generation
clean, hard readout within 5pp); the palette hypothesis survives in
repaired form. What sparse speech cannot yet feed is L4's
evidence-hungry nearest-memory matching — consistent with the day's
theme (L4 runs on dense evidence). Next lever: complementary speakers
via residual competition, tested first at 4x4/512 (hostile geometry,
biggest gap) and/or 8x8/512 (close the last 5pp).

### Files

- `run_rich_palette_8x8.py` — the experiment above. It is also the shared
  8x8 rig of the day: almost every later script in this day folder (and the
  `2026_08_26` scripts) imports from it, via
  `sys.path.insert(0, ... "2026_08_24" / "rich_palette_8x8")`.
- `show_templates.py` — renders the L1 template palettes (4x4 K=36 from the
  saved `2026_08_23` flagship weights, 8x8 K=64 and K=512 by retraining L1)
  and the nearest-neighbour family panels into `results/`.
