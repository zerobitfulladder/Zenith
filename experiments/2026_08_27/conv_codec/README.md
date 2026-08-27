# Exp 9: the conv L1 CODEC — row-balance sweep (run_conv_codec.py)

Re-opened Exp 8's conv failure with instruments before accepting its
diagnosis. Two corrections, both measured on the saved weights:

1. THE 0.15 WAS SCORED THROUGH A BROKEN DECODER. Hand the renderer the
   TRUE code map of the actual frame and classify it: **0.26**, against
   a judge that scores **0.87** on the real frames themselves. Nothing
   downstream could have scored above 0.26, so "the global code-trail
   match is non-discriminative" was never actually measured. (Reproduced
   the report exactly — TF 0.15, 9 distinct winners — only at
   CC_VIEW=skeleton, CC_G2=0.5; that is what the saved run used.)
2. THE CODE TRAIL IS FINE. Carousel phase recovered from the code trail
   alone: 0.71 graded / **0.85 skeleton**, vs 0.82 from the raw pixel
   trail — the code beats pixels under the skeleton view. The failure
   was never in the code, it was in the decode.

ROOT CAUSE, measured: the conv unit's row is **74% lagged trail / 26%
arriving window**. On the carousel a position's local history is a blur
of ten different digits, so units get indexed by an uninformative
history and each one's cargo becomes an average over unrelated windows.
Per-window cargo fidelity **0.232**, whole-frame reconstruction 0.516 —
templates_L1 shows it: crisp comet KEYS over muddy near-empty CARGO.
Note L1 here is never queried with an empty half (it only encodes and
renders; the cargo-empty successor read is at L2), so the
key-must-dominate-the-row law does NOT bind at L1 — that split was
inherited from the uniform-unit template, not required by any read.

Sweep, one axis (the trail's share of the L1 row) plus window geometry.
Gate measured FIRST, with no memory in the loop:

| arm | L1 row | window fid | frame recon | decoder ceiling | TF read (code space) | L2 top-row share | bounce advance | bounce winners |
|---|---|---|---|---|---|---|---|---|
| base | 8x8 K64, trail 74% | 0.232 | 0.502 | 0.27 | 0.18 | 0.943 | 0.00 | 2 |
| gc2 | 8x8 K64, trail 34% | 0.654 | 0.936 | 0.81 | 0.57 | 0.527 | 0.58 | 21 |
| gc4 | 8x8 K64, trail 22% | 0.660 | 0.937 | 0.82 | 0.51 | 0.622 | 0.57 | 30 |
| notrail | 8x8 K64, no trail | 0.770 | 0.938 | 0.83 | **0.62** | **0.311** | **0.75** | 9 |
| w4k36 | 4x4 K36, trail 39% | 0.443 | **0.960** | **0.87** | 0.10 | 0.989 | **0.75** | 31 |
| w4notrail | 4x4 K36, no trail | **0.846** | **0.962** | **0.87** | 0.39 | 0.406 | 0.00 | 2 |

(judge on real frames 0.87 throughout — w4k36/w4notrail sit AT the
measurement ceiling. Bounce numbers are the committed read.)

RESULT: the codec diagnosis is confirmed and fixed. Every arm that gives
the current window the majority of the L1 row lifts the decoder ceiling
0.27 -> 0.81-0.87 and turns generation from frozen (advance 0.00, one
emission repeated 100x) into a counting carousel in real handwriting.
recon_panel.png tells it at a glance: base decodes every digit to the
same scribble; w4k36/w4notrail decode to clean digits. notrail's
template sheet is a proper oriented-stroke dictionary.

THEN THE FIX UNMASKED THE REAL BLOCKER. Three arms with near-identical
codecs generate completely differently:
- notrail (8x8): 9 spread L2 winners, healthiest L2 (top share 0.311)
  — and yet the WEAKEST real counter of the working arms (see the
  matched-judge table below: advance 0.364, only 8 classes, a scrambled
  repeating loop 5134367-8). The mean-judge had flattered it to 0.747.
- w4k36: counts all ten, best-looking digits of the project on this
  task — yet its L2 is MONOPOLIZED (one row took 98.9% of updates) and
  teacher-forced it reads that one row every time (TF 0.10, 1 winner).
  Free-run it uses 31 winners. The generation works because the
  self-encoded render is OFF-DISTRIBUTION enough to miss the monopoly
  row's basin, so the untouched BOOTSTRAP rows — crisp single-moment
  snapshots — serve as the transition table. It works by accident.
- w4notrail: the BEST codec of all (fidelity 0.846) and a healthier L2
  (0.406, TF 0.39, 12 winners) — and it FREEZES, one 5/8 attractor for
  100 ticks, 0.99 winner share. Closing the generate->recognize loop
  removed the accidental escape hatch and dropped it straight into the
  monopoly row.

LAW (new, and it inverts an old assumption): on this rig a BETTER codec
makes free-run generation WORSE, because self-drive quality is riding
on the render being off-distribution. The 08-26 finding "the network
cannot recognize its own drawings" was filed as a defect; here it is
load-bearing. Anything that closes that loop must fix the L2 allocation
monopoly first or generation collapses.

## Exp 9b: matched judge (judge_generation.py) — the gate was the instrument, and it reordered the results

The nearest-class-mean gate scores 0.87 on real MNIST but only
0.155-0.868 on RENDERS of those same frames — it was never a valid
instrument for emissions. Fair judge: a logistic probe trained on
renders of real frames through each arm's own L1 round trip (the same
distribution the emissions come from), validated 0.818-0.922.

| arm | advance | timeline | classes | mean-judge said |
|---|---|---|---|---|
| base | 0.000 | 0.110 | 2 | 0.000 |
| gc2 | 0.828 | 0.190 | 10 | 0.576 |
| gc4 | **0.899** | 0.200 | 10 | 0.566 |
| notrail | 0.364 | 0.200 | 8 | 0.747 |
| w4k36 | 0.879 | **0.920** | 10 | 0.747 |
| w4notrail | 0.000 | 0.100 | 2 | 0.000 |

WINNER, unambiguous: **w4k36** — free run
`9999436789012345678901234567890123436789`, PHASE-LOCKED to the real
carousel after a 4-tick stumble (timeline 0.920 vs <=0.200 for every
other arm), all ten classes, and the only arm whose GRADED read works
as well as its committed one (0.879/0.920 both ways). gc4 counts just
as well (advance 0.899, clean `...678901234567890123456789`) but a
constant one tick behind, so it never recovers phase.

TWO RANKINGS REVERSED by the fix: notrail 0.747 -> 0.364 (it was never
counting — 8 classes in a scrambled loop), and gc2/gc4 0.57 -> 0.83/0.90
(both were near the top all along). Standing caution: w4k36 is still
the accidental mechanism (L2 monopoly 98.9%, TF reads one row), so the
best generator in the sweep is the one whose success is least robust;
gc4 is the runner-up on a slightly healthier L2 (0.622).

LAW (instrument edition, and it cost two wrong conclusions today —
Exp 8's decoder ceiling and Exp 9's notrail claim): NEVER JUDGE
GENERATED OUTPUT WITH A GATE CALIBRATED ON REAL INPUT. Renders are
off-distribution to anything trained on the real thing (the 08-26
"cannot recognize its own drawings" finding, striking again as a
measurement bug rather than an architectural one). Train the judge on
the round trip, and validate the judge before reading the result.

NEXT LEVER, changed by this: not the codec, and not "conv L2 over the
code" as queued last night — the L2 ALLOCATION MONOPOLY. Candidates,
untried: per-row win caps / adopt-until-full over a longer bootstrap,
K2 sweep, learning-rate decay by win count, and the interleave lesson
from 08-26 v3 (curriculum order IS allocation). The queued conv-L2 +
global-L3 split-view build should sit behind that gate, since every
number above is measured through whatever L2 does with its rows.

Artifacts per arm in results/<arm>/: templates_L1.png,
recon_panel.png (input / committed / graded), recon.gif,
generated_{graded,committed}.gif, generated_panel_{graded,committed}.png,
metrics.json, weights.npz. Re-run eval on saved weights with
CODEC_LOAD=1 CODEC_ARM=<arm>.

---

Files: `run_conv_codec.py` (imports the rig from `../completion/`; arm chosen
by `CODEC_ARM`), `judge_generation.py` (the matched judge; imports
`run_conv_codec`, writes `judged.json` into the arm's folder). Results in
`results/<arm>/` for base, gc2, gc4, notrail, w4k36, w4notrail.

Run:

    CODEC_ARM=gc2 .venv/bin/python experiments/2026_08_27/conv_codec/run_conv_codec.py
    CODEC_ARM=w4k36 .venv/bin/python experiments/2026_08_27/conv_codec/judge_generation.py
