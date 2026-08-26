# `digits_bridge/` — digit carousel that sees with one vocabulary and draws with another

Script: [`run_digits_bridge.py`](run_digits_bridge.py). Results: [`results/`](results/) —
[`cycles.png`](results/cycles.png), `report.md`.

Run: `.venv/bin/python experiments/2026_08_26/digits_bridge/run_digits_bridge.py`

No separate writeup was kept for this run; this README is written from the script's
docstring, its `report.md`, and the one mention in the day log.

**What it tests.** The digit carousel of [`../temporal_digits/`](../temporal_digits/), but
the eye and the hand have separate template vocabularies: an encoder L1 and a decoder L1
with the same shape and different seeds. The temporal Layer stores
[encoder trails ; label ; next frame in the decoder's vocabulary ; next label]. In
free-run, the emitted hand code is rendered to pixels by the decoder, re-encoded by the
encoder, and the trails update from that re-perception — no code skips the pixel world.

**What came out.** From the day log (the hierarchical-carousel section, in
[`../hier_time/README.md`](../hier_time/README.md)): it completed one full clean 0-9 lap
by re-perceiving its own handwriting across the two vocabularies, then froze on the 4.
The judged label sequence in `report.md` (first 80 frames) is
`00123456789000112234444444…` — one lap, then 4 forever.
