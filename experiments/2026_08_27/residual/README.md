# Exp 3: run_square_residual.py — same-rate chassis + subtractive feedback (completion read)

User's chassis: every layer ticks AND speaks every input tick;
timescale from decay alone (G1=.5, G2=.9); one-hot speech upward
(orthogonal inputs -> slow leaky integration is interference-free —
user's observation, mathematically exact). L2 starved to K2=8 so rows
must span eras; L1 full K1=64. Arrow attempted as a READ (user's
"subtract the up-to-now"): retrieved arc-whole minus explained
history = the era's future, used as a gate; then pick within pool.

RESULTS LADDER (teacher-forced forward/self/backward):
- CHASSIS WIN: starved L2 self-organizes CLEAN CONTIGUOUS ARCS
  tiling the cycle (8 rows, 4-7 consecutive phases each) — the
  completion substrate exists, decay-only timescales suffice, no
  strides needed. Keeper. (K1=32 arm degrades the arcs — L1 grain
  couples upward; full L1 capacity helps L2.)
- unit-space subtraction alone (lam sweep to 5): forward 0.00 — never
  fires forward. Leak found: PHASE-TWIN CLONES (K1=64 > 40 phases)
  alternate across laps; the not-recently-fired twin survives the
  fired-history subtraction at offset 0 and hijacks the trail match.
- dual subtraction (fired-history lam=2 + content-explained mu=0.5):
  twins die (self .80 -> .03), FIRST FORWARD RETRIEVALS OF THE DAY
  (fwd 0.41, emission-ahead 0.52) but backward still 0.56.
- recovery ordering (most-recovered = next-due on a loop; the
  adaptation-channel idea as a read): no improvement — drowned.

DIAGNOSIS, structural: the stored whole is itself a CAUSAL average —
recency-weighted toward its past — so its future-part mass is weak,
and after subtraction the surviving pool is past-heavy; no read rule
can ORDER the survivors because order was never written
asymmetrically (trail match symmetric; recovery signal drowned by
the same causal skew). Subtraction is an ANTI-FREEZE, not an arrow:
free-run errors improved monotonically with mechanism
(10.02 plain -> 7.49 refractory -> 5.56 dual subtraction; real
sustained motion appears, direction wobbles) but never approach the
lag forms' 0.00.

DAY'S LAW (arrow law, 4th confirmation — completion, refractory,
subtractive gate, recovery order): a library of causal snapshots
read by its own causal state can tell WHERE it is and WHAT REMAINS
of the current era (a set), but not WHAT COMES NEXT (a sequence);
sequential order must be written at storage time as a within-vector
time offset. NEW TOOL BANKED: subtractive (explaining-away) feedback
— never previously tried in the feedback program (F1-F6 were
gain/additive/rescoring) — cleanly removes the spent part of an era
from a candidate pool; natural fits: anti-freeze in playback loops,
and SET-COMPLETION (the composition thread's partial reinstatement).

Synthesis available (all user-owned pieces, no next-slots): same-rate
decay-ladder chassis + subtractive-feedback gating + lag-advance
storage ([small-gain present ; dominant lagged keys], validated 0.00)
= proposed standard temporal unit going forward.

## Exp 3b: dense upward communication (SR_COMM=dense, user request)

Upward speech = relu correlation profile instead of the winner
one-hot, same rig. MIXED, instructive:
- WINS: at lam=2 backward retrievals go to 0.00-0.04 and mean offset
  turns POSITIVE for the first time (+1.4-1.5); free-run 4.98 =
  best of the completion family (10.02 -> 7.49 -> 5.56 -> 4.98).
- LOSSES: self-match resurges (0.74-0.83) — graded echoes keep the
  present alive in T2 past the subtraction; and L2 arcs MIRROR-ALIAS
  (rows own phase pairs from opposite travel directions, one row
  swallows 16 phases): grading washes out direction, which lived in
  exactly WHICH comet-tail unit fired. Read-mode law, temporal
  edition: dense speech smooths content (better loop stability),
  hard speech carries decisions (direction identity).
- Family asymptote: each mechanism dose shaves free-run error but
  the curve is flattening far above the lag forms' 0.00 — consistent
  with the day's law that this family lacks a written arrow.

---

Files: `run_square_residual.py` (imports the rig from `../completion/`). The
dense-speech arm (Exp 3b) is the same script with `SR_COMM=dense`; other knobs
`SR_K1`, `SR_SEL`. Only one run's artifacts are on disk, in `results/`
(`filmstrip.png`, `generated.gif`, `gen_frames.npz`, `report.md`).

Run:

    .venv/bin/python experiments/2026_08_27/residual/run_square_residual.py
