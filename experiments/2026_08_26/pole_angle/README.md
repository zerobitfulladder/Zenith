# `pole_angle/` — recovering the pole from its angle alone

Scripts: [`run_pole_ema.py`](run_pole_ema.py) (imported by `../pole_swingup/`, `../pole_tracks/` and by 2026-08-28), [`run_pole_dagger.py`](run_pole_dagger.py) (DAgger rounds, rewrites the weights), [`pole_viewer.py`](pole_viewer.py) (live Dear PyGui viewer). Results: [`results/`](results/) — `report.md`, `weights.npz`.

## Angle-only pole recovery (run_pole_ema.py, run_pole_dagger.py,
## pole_viewer.py)

User spec: sense ONLY the pole angle (population bumps; instantaneous
reading is direction-ambiguous — the EMA trails must carry velocity
themselves); force continuous in [-1,1], 9 levels, thermometer-coded;
trails on both angle and executed force (efference); pure imitation;
demos = recovery episodes from random recoverable states, ending at
recovery; live Dear PyGui viewer (Reset / Run buttons).

Three-act debugging, each act measured:
1. ENCODING TRAP: 0/1 thermometer's "full left" = all-zeros —
   correlation-invisible AND aliases the empty query half; full-left
   units dominated retrieval (constant hard-left, fell in 16 steps).
   LAW: in a correlation-matching architecture, "nothing" must never
   be a valid symbol. Fix: bipolar thermometer (-1/+1 bits).
2. DRIFT DIAGNOSED: teacher-forced agreement 0.81 exact / 0.97
   within-1 near upright, yet closed-loop 5/100 — the pupil KNOWS the
   policy; its own +-1 slips create efference-trail contexts no demo
   contains (covariate shift through its own force history).
3. CURE: DAgger — expert labels the pupil's own visited contexts
   (pupil's trail, expert's correction). Recovery 4 -> 43 -> 46 -> 68
   /100 over four rounds, corrected-sample yield rising each round as
   the pupil survives longer (competence compounding). Not saturated;
   more rounds + capacity queued. Pending control: no-trail arm +
   DAgger (the pure trail-ablation; note the no-trail arm's efference
   channel still leaks direction, so the ablation needs care).
   weights.npz saved; pole_viewer.py (dearpygui, installed via uv) is
   ready to run from a display for live Reset/Run viewing.
