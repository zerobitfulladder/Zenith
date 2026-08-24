# 2026-08-24 — `rich_palette/`: Rich palette / sparse speech at 4x4 windows

Part of the day log [`../README.md`](../README.md).

## Rich palette / sparse speech — predictions (before running, 2026-08-24)

User hypothesis (the sparse-coding corner): the known failure of sparse
upward messages ("top-1 output starves the layer above") was measured
at K1=36 — a poor palette. With a much richer palette the winner's
IDENTITY carries more information, so a sparse output may stop starving
L2 (biology's arrangement: overcomplete dictionary, sparse activity).
Known caveat going in: a centered 4x4 window has only 15 dims, so 512
templates tile the same small sphere more finely — more identity bits
per winner, but more near-ties, and near-ties make sparse messages
brittle (noise flips the speaker).

`run_rich_palette.py`: 2x3 grid on the stride-1 flagship. K1 in
{36, 512} x L1 OUTPUT in {dense relu-all, per-position top-3,
per-position top-1}. L1 LEARNING unchanged (dense-window top-1
winner); L2/L3 skeleton learning, L4 dense (today's verdict), all
standard. Measures: probe L2/L3 (L1 probe skipped — 320k dims at 512),
hard readout, label retrieval, generation figures; diagnostics: L1
template crowding (mean |cos|) and median top1-top2 correlation margin
(brittleness).

1. K1=36 reproduces the known ladder: dense > top3 > top1, top1
   clearly starved (probe L2 down >= 2pp vs dense, hard readout down
   more).
2. K1=512 dense: probes at least match K1=36 dense; L1 crowding rises
   strongly (mean |cos| well above 0.348 — the 15-dim ceiling), margin
   shrinks.
3. THE BET (user's): the sparse-output penalty SHRINKS with palette
   size — (dense - top1) gap at 512 less than half the gap at 36;
   top3@512 within ~1pp of dense@512 on probes.
4. Discriminator: if top1@512 still fails while top3@512 succeeds, the
   blocker is near-tie brittleness, not information capacity — then
   the palette idea survives with k=2-3 speakers, not one.
5. No committed prediction on generation; observe (render path and L4
   are standard, but everything upstream trains on the masked code).

### Outcome (same day) — the bet REVERSED: sparse speech gets MORE
### brittle as the palette grows; dense@512 sets the hard-readout record

`results/` (one OOM interlude: cupy argpartition
allocates GB-scale workspaces — replaced by k iterated argmaxes; free
the memory pool between configs):

| K1 | L1 out | probe L2 | probe L3 | hard | L1 crowd | top1-2 margin |
|---|---|---|---|---|---|---|
| 36 | dense | .9624 | .9646 | .8972 | 0.349 | 0.111 |
| 36 | top3 | .9666 | .9644 | .8742 | 0.348 | 0.111 |
| 36 | top1 | .9584 | .9618 | .8242 | 0.349 | 0.111 |
| 512 | dense | .9646 | .9666 | **.9066** | 0.347 | 0.017 |
| 512 | top3 | .9514 | .9580 | .8172 | 0.348 | 0.017 |
| 512 | top1 | .9324 | .9492 | .7350 | 0.348 | 0.018 |

- P1 half-wrong, informative: at K1=36 sparse speech barely dents the
  probes (top1 -0.4pp) — the consolidated L2/L3 absorbs it; the damage
  concentrates in the L4 hard readout (-7.3pp). The old "top-1 starves
  the layer above" was a naive-stack result; under skeleton learning
  the starvation is mostly a MEMORY-MATCHING disease, not a
  representation one. (top3@36 even beat dense on probe L2 — echoes the
  old K_OUT=3 sparsity-gradient result.)
- P2 half-right: margin collapsed 0.111 -> 0.017 as predicted, but
  crowding did NOT rise (0.348 flat): 512 templates re-tile the same
  15-dim sphere at the same mean pairwise angle.
- P3 (the bet) REVERSED: every gap WIDENED at 512 — (dense-top1) hard
  gap 7.3pp -> 17.2pp; probe L2 gap 0.4pp -> 3.2pp; top3@512 (.8172)
  is worse than even top1@36 on hard. Generation: dense@512 clean;
  top1@512 wrecked (fragments, broken digits).
- P4 discriminator: BOTH sparse modes fail at 512, and the margin
  says why — with 512 near-duplicate templates the winner among
  near-ties is noise, so a sparse message's channel identity is
  unstable input-to-input. More palette = more identity bits in
  principle, but ALSO more near-ties; at a 15-dim window the second
  effect dominates. Dense speech is immune (near-ties all speak at
  near-equal weight — smooth) and BENEFITS from the finer tiling:
  hard .9066, project record.

READING: sparse-speech-over-rich-palette (the cortical arrangement)
requires the palette to be rich in DIMENSIONS, not in count — 4x4
windows cap at 15 dims, so extra templates are forced into redundancy,
and per-position winner-picking becomes arbitrary. Two live follow-ups:
(a) bigger L1 windows (8x8 = 63 dims) x big K x sparse output — the
palette hypothesis at adequate dimensionality; (b) complementary-winner
selection (winner subtracts what it explained, survivors compete over
the residue) instead of top-k — top-3's three speakers are currently
near-clones, which is why tripling the speakers recovered only half
the top-1 damage. Also: dense@512 says palette size is a cheap win for
the standard rig; check cost (code dim 8x) before adopting.
