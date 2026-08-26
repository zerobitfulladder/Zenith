# `bounce/` — generation by bouncing between image and code

Scripts: [`run_bounce.py`](run_bounce.py) (self vs consult arms on `../base/results/weights.npz`) and [`run_bounce_vs_oneshot.py`](run_bounce_vs_oneshot.py) (head-to-head against one-shot generation, judged by a pixel classifier). Results: [`results/`](results/) — [`bounce_self.png`](results/bounce_self.png), [`bounce_consult.png`](results/bounce_consult.png), [`oneshot_vs_bounce.png`](results/oneshot_vs_bounce.png), `report.md`.

## Bouncing generation (run_bounce.py)

User's mechanism: generation as associative settling. Label queries
the top; the code renders to pixels; the pixels re-encode up; repeat —
top-to-bottom-to-top. Annealed resolution: start dense, each bounce
keeps fewer templates per position (dense -> 32 -> 8 -> 2 -> 1) so the
architecture eliminates candidates gradually instead of hardening in
one step. Arms: self (bounce only, top consulted once) vs consult
(each bounce re-queries the top with the label CLAMPED, blends the
matched memory's code back in). Base weights, CPU, no retraining.

Predictions (before running):
1. Peakiness rises across bounces — resolution as designed.
2. The consult arm holds class identity 10/10; the self arm may drift
   (re-encoded mush can wander off-class) — the clamp is what makes
   settling safe.
3. Final k=1 images at least as clean as one-shot generation; per the
   F-series law (feedback pays where the forward pass is lossy), the
   gain on the strong dense base rig is modest — visible in stroke
   cleanliness, not identity. The real payoff is expected later on
   lossy/sparse rigs.

### Outcome

P2 CONFIRMED, vividly: the self arm (no anchor) drifts into repeating
stripe/blob textures by k=32-8 and fragments at k=1 — the render/encode
loop has its own non-digit attractors (classic spurious attractors,
live on our rig). The consult arm (label clamped + memory code blended
each bounce, BLEND=0.5) SHARPENS through the anneal: k=2 and k=1 rows
are cleaner than one-shot generation for several digits (3, 4, 5, 9).
The user's settling mechanism works — but only anchored; the clamp is
load-bearing. P1 was mis-instrumented (peakiness measured on the dense
re-encode, which the anneal never touches — metric says nothing).

SURPRISE FINDING (from debugging the self-classification metric): the
network cannot recognize its own drawings. Encode+hard-match scores
45/50 on real test images but 1/10 on the rig's own one-shot
generations — the round trip generate -> encode -> match is broken at
the code level even though the drawings look fine to a human. The
bounce didn't lose self-recognition; it never existed. Renders are
off-distribution to the code pipeline (feathered strokes, halos,
different contrast statistics). Explains F3's design choice in
retrospect (searchlight compared renders to inputs in PIXELS, not
codes). Open thread with legs: closing the generate/recognize loop —
if the network could recognize its own drawings, sample-verify (F4)
and the bounce could anchor without the label.
