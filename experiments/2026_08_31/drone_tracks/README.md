# Sensory track, motor track, and a table between them learned by TD

2026-08-31. The failures in `../drone_rl` came from one subspace doing three
jobs at once (territory, code and policy). This splits them: a sensory track of
N experts over the 6 sensor channels, trained unsupervised by fit competition; a
motor track of M experts over the 2 thrust channels; and a plain table
`Q[n, m]` learned by TD, the only thing reward may edit.

No separate writeup was kept. The result is recorded in
[`../drone_imitate/README.md`](../drone_imitate/README.md) and the day README
([`../README.md`](../README.md), section on the drone): like the `drone_rl`
runs it scored 0.000 success (`results/metrics.json`: 256 sensory experts, 25
motor experts, 24,000 episodes).

Run: `.venv/bin/python experiments/2026_08_31/drone_tracks/tracks.py`

Results: `results/metrics.json`, `results/tracks.json`, `results/tracks.npz`,
`results/01_tracks.png`.
