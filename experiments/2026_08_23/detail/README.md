# `detail/` — stride-1 dense code over the batch-2 rig

(2026-08-23, from the day log.) Script: `run_detail.py`. Results: `results/` (figures, `report.md`, `weights.npz`).

## Detail build — predictions (before running, 2026-08-23)

`run_detail.py` (MNIST): batch-2 architecture with a stride-1 dense code.
625 window positions (was 169); the dictionary LEARNS only at the
stride-2 subgrid (identical dynamics to the validated rig) but REPORTS
at stride 1. K1=36, K2=200, lam=0.5. Weights saved. Code dim 22500.

1. Generation visibly smoother and finer than batch 2 — the point of
   the build (each pixel = consensus of up to 16 windows; constellations
   carry ~4x spatial say-so).
2. Probe >= 0.946 (denser code cannot carry less).
3. Prototype readouts >= batch 2 (hard 0.743 / top10 0.782) — denser
   codes should overlap more between same-class variants.
4. Retrieval 10/10.

Recorded design for the separate depth fix (from the 4-layer binding
failure + the user's sparsity-gradient intuition, matching cortex's
rising selectivity with height): multi-winner top-k in the upward CODE
(communication), top-1 in LEARNING (plasticity), with k decreasing per
level. Untested; queued.

### Outcome (same day) — all four predictions confirmed; new champion rig

- Generation clarity transformed: smooth, continuous, natural-handwriting
  digits — no seams, patches, or roughness. Reconstruction is near-
  photographic to each instance. Density was the detail lever.
- **hard 0.8008** (first readout above 0.80; batch 2: 0.743),
  top10 0.8084, **probe 0.9516** (new high), retrieval 10/10.
- The learn-sparse/report-dense decoupling worked as designed: training
  dynamics identical to the validated rig, all gains from the denser
  code alone. Weights saved (results/weights.npz).

The detail rig (stride-1 code, stride-2 learning, K1=36, K2=200) is the
new baseline. Single-seed — owes the 8-seed treatment before its
numbers are quoted as facts.
