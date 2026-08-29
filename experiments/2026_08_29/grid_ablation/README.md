# The grid ablation and the scarcity sweep

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## The grid ablation (outside critic's challenge): the partition loses

16x16 fixed grid over foveated (dx,dy), same everything else:
0.99/0.033/229 vs the learned templates' 0.95/0.081/291. The learned
partition is NOT load-bearing on the outer loop — the champion is
tabular BC over a foveated grid. Load-bearing instead: the fovea, the
cetele/duty-read, the cascade, the reflex. Non-monotone (8x8 0.64,
32x32 0.60). New law #13 in [`ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md). Owed next: the same
ablation on the balance rig (d=4, 8^4 grid = 4096 cells vs its 4096
templates) and MNIST (900-dim — no grid possible), where learned
partitions must make their real case.

## The scarcity sweep: the learned partition's redemption, quantified

learned K=64: 0.99 / 0.022 (equals grid-256!)  vs grid 8x8: 0.64.
K=36: 0.94 vs 0.82. K=16: wash (both starved). Composed law #13:
adaptive partitions buy 4x allocation efficiency, not asymptotic
quality. The critic's ablation produced both the deflation AND the
crossover curve — the claim is now stated precisely instead of
believed vaguely.

The script was kept in the session scratchpad and is not in the repo. The
16x16 grid policy is `results/grid16.npz`, viewer-ready through
`../temporal_drone/grid_view.py`.
