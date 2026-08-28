# `drone_continual/` — the continual drone, sliced read (Exp 20)

2026-08-28, late. No writeup was kept for this run; this is written from the script's
docstring and its log.

## What it tests

The cascaded drone of Exp 19 (`../drone/`), but as one continuous life instead of DAgger
rounds: templates store the sensory channels and the command together (no tally), the outer
hypercolumn is read in two slices (lateral channels pick the lean, vertical channels pick the
collective thrust, possibly from different minicolumns), hypercolumns have a fixed size from
tick zero, and every hypercolumn learns from the oracle's answer at every tick while the pupil
flies. Episodes reset; weights never do.

## Files

- `run_drone_continual.py` — the run. Imports `run_recon3` from `../recon_ladder/` and
  `run_drone` / `run_drone_cascade` from `../drone/`; reads the teacher gains from
  `../drone/results/cascade/expert.json`.
- `results/` — checkpoint.npz/json, metrics.json, `run_drone_continual.log`.

    .venv/bin/python experiments/2026_08_28/drone_continual/run_drone_continual.py
    Env:  DN_TICKS (600000), DN_WARM (40000), DN_REPORT (20000), DN_K, DN_TAG

## What was recorded

The log and metrics.json stop at tick 40000, still inside the warm-up (the oracle flying,
success 1.00). The pupil never took over in the saved record, so there is no result. The
single-layer design in `../single_layer_drone/` followed it the same evening.
