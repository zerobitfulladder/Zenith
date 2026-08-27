# Exp 5: run_square_lagcargo.py — observed cargo (delayed write)

Same row layout as Exp 4; the cargo slot is filled by OBSERVATION:
keys = trail as it stood before the frame arrived, cargo = the frame
when it arrives (storage lags perception by one tick; keys older,
cargo newer — the offset IS the arrow). Playback queries cargo-empty
and emits the retrieved cargo. Result: **0.00, max 0** — perfect
100-frame free run through both bounces. Day closed: inference-cargo
6.65 vs observation-cargo 0.00 on the identical chassis is the
cleanest single-variable demonstration of the arrow law the project
has. Standard temporal unit going forward: same-rate decay-ladder
chassis + delayed-write (lag-advance) storage; subtractive feedback
banked separately as anti-freeze / set-completion machinery.

---

Files: `run_square_lagcargo.py` (imports the rig from `../completion/`),
`show_lagcargo_templates.py` (draws `results/templates.png` from the saved
weights). Results in `results/` (`generated.gif`, `gen_frames.npz`,
`report.md`, `weights.npz`, `templates.png`).

Run:

    .venv/bin/python experiments/2026_08_27/lagcargo/run_square_lagcargo.py
    .venv/bin/python experiments/2026_08_27/lagcargo/show_lagcargo_templates.py
