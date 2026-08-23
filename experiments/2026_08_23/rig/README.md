# `rig/` — the shared code of 2026-08-23

Not an experiment of its own: the modules that this day's experiments (and later days) import.
They sit together so one `sys.path.insert(0, ".../experiments/2026_08_23/rig")` finds all of them.

- `gain_feedback.py` — the basic unit (`ZenithLayer`, `ClassTemplates`, `center_norm`, the gain
  field). Written for [`../gamma_sweep/`](../gamma_sweep/README.md); every script of the day imports it.
- `run_4layer_topk.py` — the 4-layer stack with per-level output modes
  (`GF_REPORT=topk|reluall|contrast|signed|sparselearn`), its encode/expand/render helpers and the
  CPU evaluation. Run on its own it writes `../four_layer/results/<mode>/`
  (see [`../four_layer/README.md`](../four_layer/README.md)).
- `run_gpu_minibatch.py` — the mini-batch GPU trainer for the consolidated unit (`GF_XP=cupy|numpy`,
  `GF_B`, `GF_W1_STR`). Run on its own it writes `../gpu_minibatch/results/stride<N>/`
  (see [`../gpu_minibatch/README.md`](../gpu_minibatch/README.md)).
- [`ConsolidatedUnit.md`](ConsolidatedUnit.md) — the spec of the settled unit these runs arrived at
  (was `notes/Zenith/ConsolidatedUnit.md`).

Used here by: gamma_sweep, ab_test, concat_generation, warmup_test, rebuild, batch2, four_layer,
detail, unnormalized_l1, celeba_faces, celeba_big, pyramid, inpaint, wide3. Used later by
`2026_08_24`, `2026_08_26` and `2026_08_27`.
