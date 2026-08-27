# static — the pre-temporal rig, rebuilt clean

No trails, no arrows, no time. One script, self-contained, importing
nothing from the temporal work: `run_static3.py`.

## Architecture

    L1   8x8 pixel windows, stride 2  -> 11x11 grid, K=64    strokes
    L2   3x3 windows over L1, stride 2 ->  5x5 grid, K=64    motifs
    L3   3x3 windows over L2, stride 1 ->  3x3 grid, K=100   parts
    TOP  memory over [L3 code ; 0.5 * one-hot(10)], K=200

Every layer is the same unit: mean-center and L2-normalize the window,
correlate against the bank, top-1 winner rotates toward the input on the
unit sphere. Nothing else learns.

Three settled choices, each previously validated and carried over intact:
- DENSE COMMUNICATION — a layer speaks every positive correlation at full
  magnitude, not a top-k selection.
- SKELETON LEARNING at L2/L3 — they learn from a per-position top-1 view
  of their window while still speaking densely (three-regime law: dense
  targets poison a layer only where the shared mass is task-irrelevant,
  so L1 and TOP learn dense, L2/L3 learn skeleton).
- TOP-1 GENERATION — the label queries the top memory with the code half
  blank; the retrieved code is hardened to one unit per position at every
  level on the way down, then rendered with squelch + feathered
  overlap-add.

## Results (20k train, 2 epochs, 5k test)

| | 8x8 L1 (`results/w8/`) | 4x4 L1 control (`results/w4/`) | previous project best |
|---|---|---|---|
| probe L1 | **0.9594** | 0.9516 | 0.953 |
| probe L2 | **0.9606** | 0.9582 | 0.959 |
| probe L3 | 0.9548 | 0.9578 | 0.959 |
| hard readout | **0.8762** | 0.8706 | 0.8748 |
| round trip corr | **0.651** | 0.487 | — |
| label -> own memory | 10/10 | 10/10 | 10/10 |

All units used at every level (64/64/100/200) — no dead units, no
monopoly. The probe ladder is flat, so no depth sag.

8x8 AT L1 IS A WIN over the 4x4 geometry the earlier rigs used: better
on every measure, and decisively better on reconstruction (0.651 vs
0.487). Bigger first-layer windows carry more of the picture through the
round trip at no cost to the probes or the readout.

## Artifacts

In `results/w8/` (8x8 L1) and `results/w4/` (4x4 L1 control):


- `templates_L1.png` — 64 stroke units, raw 8x8 windows
- `templates_L2.png` — 64 motif units, expanded down to pixels
- `templates_L3.png` — 100 part units, expanded down to pixels
- `generation_labels.png` — one picture per label, from the label alone
- `generation_variants.png` — six memories per label (style variety)
- `roundtrip.png` — real image vs its round trip through L1-L3
- `metrics.json`, `report.md`, `weights.npz`

By eye: L1 is a clean oriented-stroke dictionary; L2 is corners, arcs and
bar-pairs; L3 is recognizable digit fragments (hooks, cups, 7-corners,
the double bar of an 8's sides). Generation gives ten readable digits;
0/1/3/5/6/7/9 are clean, 2/4/8 are the weak ones — the same three that
were always hardest in the earlier rigs.

Run: `.venv/bin/python experiments/2026_08_27/static/run_static3.py`
(4x4 control: `run_static3_w4.py`, identical except the L1 window)
Env: `ST_TRAIN ST_EPOCHS ST_K1 ST_K2 ST_K3 ST_KTOP ST_TAG` (`ST_TAG` names the
folder under `results/`; default `w8` for `run_static3.py`, `w4` for
`run_static3_w4.py`)
