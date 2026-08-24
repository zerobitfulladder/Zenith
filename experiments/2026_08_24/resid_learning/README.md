# 2026-08-24 — `resid_learning/`: Residual learning (tuned inhibition in plasticity)

Part of the day log [`../README.md`](../README.md).

## Residual LEARNING (tuned inhibition in plasticity) — predictions (before running, 2026-08-24)

Message-side fixes are exhausted (dense hedges fully; top-k hedges
within families; residual speech un-hedges — worse at cloned L1,
better at clean L2/L3). The lever left is the PALETTE: L1 is the one
part-dictionary that learns dense and the one with clone families.
Tuned inhibition in learning = residual learning: speaker 1 wins and
rotates toward the window (unchanged); its contribution is subtracted;
speaker 2, won on the remainder, rotates toward the REMAINDER; cap 3,
contrast floor stops. A twin that loses round 1 can only ever learn
from remainders (family component removed) -> twins differentiate into
complements. kmax=1 reduces exactly to the current rule.

`run_resid_learning.py`, 8x8 K1=512, L2/L3/L4 fully standard: arm
(a) residual learning + DENSE L1 speech (guard: isolates the palette
change); arm (b) residual learning + RESIDUAL L1 speech (the user's
bet: rich de-cloned palette + sparse complementary speech ~ acts
dense). References: std learning dense .9094 / top3 .8642 / resid
speech .8502.

1. THE PALETTE DE-CLONES: share of template pairs with cos > 0.8
   drops sharply; the L1 winner margin rises from 0.038 (predict >
   0.08); the sorted gallery loses its clone runs.
2. GUARD: arm (a) holds .9094 +/- 1pp. If dense speech DROPS on the
   de-cloned palette, the redundancy was load-bearing (echo of the L4
   morning lesson) and de-cloning costs coverage.
3. THE BET: arm (b) beats std-top3 (.8642), target .88+; if it
   reaches ~dense, "rich diverse palette + sparse speech acts dense"
   is vindicated and the sparse-speech thread closes SOLVED.
4. Probes >= .96 in both arms; generation stays clean; round-2/3
   templates may look like difference-parts (watch the gallery).

### Outcome (same day) — de-cloning SUCCEEDED and sparse speech got
### WORSE: the clone story is refuted as the root cause

`results/`:

| speech | probe L2 | probe L3 | hard | pairs>0.8 | pairs>0.9 | margin |
|---|---|---|---|---|---|---|
| dense | .9656 | .9680 | .8984 | 0.0001 | 0.0000 | 0.052 |
| resid | .9674 | .9686 | **.8030** | 0.0001 | 0.0000 | 0.052 |

(std-learning refs: dense .9094, margin 0.038; top3 .8642; resid .8502)

- P1 half: clone pairs ANNIHILATED (>0.8 share: ~0.01%; none >0.9;
  the sorted gallery has no runs — instead many granular "residue
  part" templates). But the margin rose only 0.038 -> 0.052, far
  short of the >0.08 bar. Near-ties survive WITHOUT clones: finely
  tiling a continuous stroke manifold makes first and second place
  close even between genuinely different templates.
- P2 guard: dense speech .8984, -1.1pp — at the edge of the band.
  Residue-parts consumed some palette capacity. Probes best-tier
  (.9674/.9686): residual learning is representationally free.
- P3 THE BET LOST DECISIVELY: resid speech on the de-cloned palette
  .8030 — WORSE than on the cloned palette (.8502). De-cloning did
  not rescue sparse speech; it removed the accidental hedge the clone
  families provided and replaced clean strokes with residue parts.
- P4: generation stayed clean in both arms.

READING — the sparse-speech arc CLOSES, four refutations deep: top-k
speech (clones), voting readout (dilution), residual speech
(un-hedging), and now palette de-cloning (near-ties are intrinsic to
fine tiling, not to duplication). The lookup's need for dense speech
is not a defect to engineer away — graded evidence IS the mechanism
by which similarity matching tolerates a finely tiled dictionary.
"In generous, out strict" stands as the architectural law, now
evidenced from message, palette, and readout sides. Standard unit
unchanged: dense graded speech, argmax/skeleton learning, hardened
production reads. Residual learning: shelved (free but not helpful
here; possibly useful where palettes must be small).
