# td_balance — the validation rung: stop and level, position free

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

User's call (2026-08-29, late): before harder training, teach it to
BALANCE — zero velocity, zero angle, position unconstrained. This is
the Aug-28 cascade's inner loop rebuilt in the bound form, and a world
where the position task's two measured killers are absent by design:

  * hold data exists — fixed-length 400-tick episodes, so most
    training ticks ARE the balanced hover the gate demands;
  * the blind start is aligned — the TEACHER wears the pupil's
    handicap (plain hover while the windows fill), so "a second of
    uncorrected tumble, then recovery" is in the data.

`../temporal_drone/td_balance.py`, `run_td_balance.py`, `results/`. Tracks:
**vx, vy, tilt** — velocity sensed directly with its own encoding (the
speedometer law; position is not sensed, so velocity cannot be
anybody's slope). Teacher = fast_oracle's cascade with commanded
velocity pinned to zero (audition required >= 0.98 before training).
Success = at_goal minus its position term: speed < 0.3 m/s and |tilt|
< 0.17 rad held 10 ticks. K3 = 4096 (d ~ 4), everything else identical
to td_bound. Motors receive (left, right) levels as always;
coll/diff exist only inside the template, for the measured
difference-of-two-reads reason.

Predictions, recorded before the full run:

1. Teacher audition 1.00 (smoke: confirmed, median 176 ticks).
2. Pupil EVAL >= 0.9. Both measured failure causes are removed and
   d ~ 4; if this rung fails, the mechanism itself is at fault — that
   is what makes it a validation rung.
3. On-path reads well above the position task's (diff > 0.8 vs 0.59):
   the cue now encodes exactly the quantities the teacher's law uses.
4. Failure mode to watch: velocity saturation runaway — beyond
   VSCALE = 6 m/s the velocity channels clip and the policy goes
   state-blind (the 8 m clip lesson, velocity edition).

## Balance results: the hold works; recovery is the missing skill

Run 1 (vx, vy, tilt; 120k ticks, 2.3 min): EVAL 0.00-0.04 while
on-path agreement sat at 0.83 — high mimicry, closed-loop divergence.
The probe decomposition (40 eps/arm) and two encoding arms:

| arm (weights_best) | A tumble init | B gentle init | C hold from balanced |
|---|---|---|---|
| vx, vy, tilt (run 1) | 0.03 | 0.30 | **0.95** |
| + gyro track | 0.00 | 0.35 | 0.97 |
| + gyro + arcsinh velocity fovea | 0.00 | 0.35 | **0.97** |

**THE HOLD WORKS — 0.95-0.97.** First reliably successful closed-loop
skill in this folder; the dwell-data fix did exactly what it promised.
`eval_traces.png` showed the failure shape: sawtooth tumbling — tilt
pulled to zero and through, spin never killed (P without D).

Two encoding hypotheses tested and REFUTED as the binding cause: the
gyro as its own track (the slope-resolution argument) and the velocity
fovea (17/40 tumble episodes engaged with a saturated +-6 clip). Both
improved the on-path read (steering corr 0.66 -> 0.81, project best)
and neither moved recovery. The read was not the constraint.

What is left standing, by elimination and consistent with the
noise-injection control on the position task: **recovery transients
are off-manifold.** The pupil's own imperfect flying creates window
HISTORIES no teacher demonstration contains (the teacher corrects from
tick one, so "flailing for 300 ms, now at this state" never occurs in
its data); the read degrades on those cues; error compounds. The hold
basin is dense in the data (most training ticks), so hold survives —
transients cover a large state volume thinly.

Phase 2 launched (`results/dagger/`, 120k + 120k ticks): odd
episodes are PUPIL-flown with teacher labels — every stored moment is
then [a pupil-style history ; the right answer]. K3 stays 4096 (fixed
counts; the memory reallocates, it does not grow). Prediction: B >=
0.8, A up substantially, C stays ~0.97. If this fails too, the next
suspect is loop lag (the cue's 720 ms footprint as delayed feedback),
testable by shortening the top window.

## CORRECTION (2026-08-30): the 0.95 "hold" was a short-window artifact

The user watched the pupil in the viewer and saw no stable hover.
The user was right; the probe was wrong. Re-measured over 800 ticks
instead of probe C's 150: the pupil holds for a median of **54-80
ticks (~1.3 s)** after a settled handover, then a small command bias
integrates through the plant's chain of integrators — vx creeps, the
state leaves the gentle envelope, the broken recovery takes over,
tumble. Balanced fraction over 16 s: 0.06-0.12. The 10-tick success
gate inside a 150-tick window certified a transient as a skill. EVAL
in run_td_balance is now strict: balanced through the episode's final
150 ticks, plus a balanced-fraction column.

Two more refutations, measured on the same checkpoint:

- **Graded top-k command read: does not help.** top-4/8/16/32 across
  the l3 winners all land at balanced-frac 0.07-0.09 vs top-1's 0.12.
  The drift is not single-winner quantisation bias.
- **DAgger phase 2: made it worse.** 50/50 pupil-flown episodes at
  fixed K3=4096 flooded the memory with wild states (the pupil visits
  50 m/s spins; the envelope cap was too generous): steering read
  collapsed 0.81 -> 0.22-0.33, agreement 0.84 -> 0.68, EVAL 0.00.
  Correction data needs an on-policy envelope far tighter than the
  labels the teacher can produce, or a growing memory — with K fixed
  it reallocates AWAY from the basin that mattered.

## The standing diagnosis: the success band is smaller than one cell

Arithmetic, no experiment needed once looked at: cell spacing on every
track axis is 0.087 normalised. Tilt: 0.27 rad/cell against a success
band of +-0.17 rad. Velocity (even arcsinh-warped at scale 1.5): +-0.3
m/s spans 0.7 cells. Gyro: 0.35 rad/s per cell. **The entire hold
basin sits inside one cell of every axis** — the cue is constant
across it, the read cannot correlate command with state there, the
loop has no small-signal gain, and neutral stability plus any bias
walks the state out in ~1 s. The teacher's hold corrections respond
to velocity differences of 0.05-0.3 m/s — all sub-cell. This also
explains why the gyro track helped the READ but not the LOOP.

Fix in flight (`results/fovea/`): the fovea law applied to
the SENSORS — arcsinh warps centred on zero, scales chosen so the
success band spans 3-5 cells per axis (v: scale 0.15 m/s, rim 20;
tilt: 0.05 rad, rim pi; gyro: 0.1 rad/s, rim 10). The far/recovery
range gets coarser in trade. Prediction: strict EVAL from tumbles may
or may not move, but balanced-frac after a settled handover must jump
— if it does not, small-signal gain was not the binding cause and the
stabiliser-plus-residual split moves to the front of the queue.

## Fovea result: prediction FAILED — but the failure mode changed

Balanced-frac after settled handover: 0.12 / 0.07 (was 0.12 / 0.06).
By the recorded criterion, small-signal gain was not the binding
cause. What DID change: the freefall tumble became bounded slow
OSCILLATION — vx swinging +-10 m/s over seconds with attitude and
vertical velocity roughly kept. Gain that turns divergence into
oscillation is the signature of feedback DELAY.

## The unified diagnosis: the present is 3% of the cue

The same arithmetic that convicted the motor tracks, now on the
sensor side. The current tick is one tap of five in an L1 code
(240 ms), and that code is one tap of seven in the L2 code the top
layer cues on (720 ms): the present state carries ~3% of the cue's
energy. On the teacher's path, windows and present agree, so every
open-loop read metric looks fine (0.84 agreement) — but the teacher
is a function of NOW alone, and in closed loop, the moment history
and present diverge (any transient), the read follows the HISTORY:
feedback delayed by hundreds of ms. Delayed feedback = drift when
gain is low (pre-fovea), oscillation when gain is high (post-fovea).
Every observation fits, including why each encoding fix improved
reads without moving the loop.

The record's own proof of the alternative: the SOLVED cart-pole
(87/100) keyed on [now, -4, -16] delay slices — the present at ~1/3
of the key's energy, not 3%. The candidate fix inside the bound
architecture: give the PRESENT sense bumps their own dedicated cells
in the L3 metric at a healthy energy share, with the track L2 codes
as lagged context beside them — [present ; lagged keys], sensor
edition. The standing alternative remains the stabiliser + residual
split. Decision pending discussion.

## Route 1 chosen (user): present cells in the metric

`results/pres/`. The L3 vector becomes

    [ present sense bumps | 4 tracks' L2 codes | coll | diff ]
        4 x 24 (37.5%)        4 x 128 (37.5%)     48+48 (25%)

Each sensed channel's CURRENT value as a bump beside the track codes
— the pole's key structure ([now, -4, -16], present at a healthy
share) inside the bound architecture. Implemented as `pres_share` on
`td_bound.Stack` (0 = off, drone rig untouched).

Side effect adopted with it: **the blind period is gone.** Present
bumps exist from tick one, the masked read handles the missing track
context, so the pupil acts immediately — and the teacher handicap is
dropped in training AND eval (distributions stay aligned; no more
one-second uncontrolled tumble anywhere).

Predictions, recorded before the run: (1) if the delayed-feedback
diagnosis is right, the oscillation damps — balanced-frac after a
settled handover jumps past 0.5 and the strict tumble EVAL leaves
zero for the first time; (2) on-path agreement rises (the teacher is
a function of the present, which the cue now carries at 37.5%);
(3) if balanced-frac stays ~0.1 despite the present-dominant cue, the
delay story is wrong too and stabiliser+residual is next. Smoke:
agreement 0.876 (highest yet), bal_frac already nonzero at toy scale.

## Route 1 result: direction confirmed, magnitude short

Settled-handover hold (800-tick horizon): balanced-frac 0.28 / 0.21
(was 0.12 / 0.06), median last-balanced tick **306 / 181** (was
80 / 54) — the hold went from ~1.3 s to 4-6 s. Tumble episodes no
longer die in freefall (end-speed 45 -> 14 m/s); agreement 0.88, the
project's best. Every metric moved exactly as the delayed-feedback
diagnosis predicted, so the mechanism was real — but the recorded bar
was balanced-frac > 0.5 and it reached 0.28. Strict tumble EVAL still
0.00: recovery is still missing.

What remains is a SLOW walk-out after several seconds of clean hold —
the residual state-locked bias of the read, integrated over the vx
loop's multi-second time constant. Levers, none yet tried:
(a) training duration — 300 episodes is thin coverage of the basin's
    bias field; data density is the one capacity that is not frozen;
(b) the stabiliser + residual split (standing);
(c) accept the balance rung's verdict as-is and carry the present-
    cells law back to the position task, where the same fix applies.

## The glide, diagnosed exactly (user's question: "does it not SEE
## that it is gliding?")

Probed on the pres checkpoint, 4031 gliding ticks (|v| > 0.8, level):

    actual vx on the axis          0.59 warped ≈ 7 cells from zero
    winner's own stored vx         0.12  — near-stopped moments
    teacher's command there        |diff| ≈ 0.99 N, a decisive brake
    winner's stored command        0.018 N, sign right 64%
    teacher's occupancy of the
    state "level+calm but moving"  11 / 20000 ticks

So NOT encoding resolution (the glide is loudly visible) and NOT
small-signal bias (the needed command is ~1 N). It is MIS-RETRIEVAL:
a gliding-but-level cue matches "hovering, stopped" on 7 of its 8
channel groups, and the one channel that matters holds ~9% of the
vote. Underneath: the teacher NEVER occupies "level while moving"
(when it moves, it is tilted into the brake), so no stored moment
holds the right answer — the state is manufactured only by the pupil.
Missing data + democratic similarity = a stable trap. The last-mile
"bias field" framing above was wrong in the specific: the error is
not small and not subtle.

Fix in flight (`results/dagger2/`): DAgger phase 2 on the
pres rig with a TIGHT envelope — pupil episodes terminate at speed
> 5 or |gyro| > 5, so stored pupil ticks are glide->brake pairs
rather than the 50 m/s junk that collapsed the first DAgger try.
Predictions: winner's stored vx during glides rises toward actual;
stored-command sign agreement > 0.9; settled balanced-frac > 0.5;
strict tumble EVAL leaves zero.

## Tight-envelope DAgger: THE HOLD IS SOLVED, recovery climbing

Phase 2 (120k ticks, 50/50 pupil/teacher episodes, envelope speed<5
|gyro|<5): strict tumble EVAL **0.00 -> 0.04 -> 0.16 -> 0.28, still
climbing at budget's end**; tumble bal_frac 0.02 -> 0.38. On-path
steering corr DROPPED (0.67 -> 0.36, the metric now includes hard
pupil-path ticks) while the closed loop improved — mimicry !=
competence, at last in the right direction.

Probes on `weights_best` (settled handover, 800-tick horizon):

    balanced-frac        0.76 - 0.79   (was 0.21 - 0.28; bar was 0.5)
    median last-balanced tick 799/800  — HOLDS TO THE END, both arms
    gliding ticks per 15 episodes: 4031 -> 92  (44x fewer; it brakes)
    winner's stored vx on residual glides: 0.12 -> 0.34 (actual 0.45)

Scorecard: predictions 1-3 confirmed; sign-agreement (>0.9) not met
on the 92 residual glide ticks, but the population it described has
collapsed. The glide trap was missing data, as diagnosed. CONFOUND,
honestly held: the failed first DAgger and this one differ in TWO
ways — the tight envelope AND the present cells landed in between —
so the credit split between them is unmeasured; an ablation (pres rig
+ loose envelope) would settle it if it ever matters.

`results/dagger3/` (600k ticks): strict tumble EVAL
**0.28 -> 0.52 -> 0.60 -> 0.72**, bal_frac 0.86, mean end-speed
0.41 m/s — still climbing at budget's end. Recovery from random
tumbles now mostly works; the balance rung is substantially solved
with fixed template counts, everything inside the hypercolumns.
Open: the user's ensemble idea (several hypercolumns over the SAME
input, phase-offset learning so each stores a disjoint fifth of the
moments, per-channel median of their top-1 decodes) — to be measured
against the strongest DAgger baseline, with a single big memory of
equal total templates as the capacity control.

## Where the results are

`results/` is run 1 (vx, vy, tilt); the other runs are subfolders:
`gyro/`, `gyro_warp/` (the two encoding arms), `dagger/`, `fovea/`, `pres/`,
`dagger2/`, `dagger3/` (the 600k run whose `weights_best.npz` is the frozen base
for `../critic/`, `../tally/` and `../split/`), `dagger4/` (no writeup kept;
its log is `results/balance_dagger4.log`), and the smoke tests `smoke/`,
`pres_smoke/`. Each run's console output is `results/balance*.log`.
`balance_viz.py` draws the figures.
