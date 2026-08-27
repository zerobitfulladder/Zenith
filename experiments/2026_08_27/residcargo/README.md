# Exp 4: run_square_residcargo.py — residual AS cargo (user's self-computed next-slot)

User's synthesis: store [trail ; GC*residual] jointly — the
subtraction's output plays the next-frame role, but computed by the
network's own top-down completion at storage time (no oracle).
Structurally this IS a within-vector time offset (arrow law
satisfied in form) and self-match becomes a feature (my row's cargo
= my future). Phase 1 trains b1/b2 as before; phase 2 stores the
joint vector; playback queries cargo-empty (lag-advance form),
cargo names the pool, trail picks the soonest.

Outcome: the cargo INHERITS the residual's contamination almost
exactly (TF forward 0.38 / backward 0.56 vs the read-time residual's
0.41/0.56): storing a backward-leaking guess yields a bank of
backward-leaking guesses. Free-run: hard two-point oscillation
(13 <-> 2), err 6.65. LESSON, the day's cleanest: the cargo is only
as good as its writer. A cargo INFERRED at time t (top-down
completion, causal-skewed) is noisy; a cargo OBSERVED one tick later
(write the actual next percept into the vector whose keys were
frozen at t) is exact — and that observed-cargo form is precisely
lag-advance. Neither is oracle; the difference is inference vs
observation. Inference-as-cargo might still pay where observation
is unavailable (planning, counterfactuals), not for replay.

---

Files: `run_square_residcargo.py` (imports the rig from `../completion/`).
Results in `results/` (`generated.gif`, `gen_frames.npz`, `report.md`).

Run:

    .venv/bin/python experiments/2026_08_27/residcargo/run_square_residcargo.py
