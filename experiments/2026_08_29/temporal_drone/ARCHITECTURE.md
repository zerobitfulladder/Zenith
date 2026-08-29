# The Drone: Architecture, Learning, and What It Means

*2026-08-29 → 2026-08-31. Two days from "nothing here flies" to a
drone that flies at 100%, refines itself by reward, and — in its
final form — teaches itself from zero.*

---

## 1. The world

A planar drone: two rotors, thrust snapped to 13 foveated levels per
rotor, 20 ms ticks. State = (x, y, vx, vy, tilt, spin). Task: from a
random tumbling spawn, reach a target and hold it — within 0.25 m,
slower than 0.3 m/s, level, for 10 consecutive ticks. Evaluation is
always **frozen, on fresh episodes** (the training-time metric was
caught lying once — a one-tick-delayed oracle repeater scoring 0.79
that flew at 0.00 — and never trusted again).

## 2. The champion architecture (the cascade)

    sensors      dx, dy — target-relative position, arcsinh-FOVEATED
                 (centimetre resolution near the target, coarse at
                 30 m; the success band spans cells — a law paid for
                 twice)
    OUTER        every 5th tick (10 Hz):
      recognise  256 templates over the present position bumps —
                 competitive prototypes, winner-take-all, geodesic
                 rotation learning (a recency-weighted average, so
                 templates track drifting input natively)
      consult    the CETELE: a 256 x 32 count table. Row = situation,
                 column = a word of the VELOCITY VOCABULARY (32
                 prototypes over commands, learned early from the
                 command stream). Entry = how often that word was
                 used on that situation's watch. Read = the
                 DUTY-CYCLE MEAN: proportion-weighted average of the
                 words' velocities. (Mode belongs at decision
                 boundaries; regulation lives in the mixture — the
                 skew IS the control law.)
      output     one velocity command: "fly this fast, this way"
    smoothing    vc += 0.15 * (command - vc) each tick — steps become
                 ramps; the self-inflicted "windy day" wobble was the
                 servo faithfully executing each 10 Hz step edge
    INNER        every tick (50 Hz): a fixed nested-P reflex —
                 velocity error -> desired tilt -> torque, plus
                 collective for the vertical rate. Five lines. Not
                 learned, by decision: "low level precise motor
                 control is a dumb processor either way."

The two loops touch through exactly one wire — the velocity command —
and they are CO-ADAPTED: the memory was trained on demonstrations
executed by this very reflex, and swapping in a stiffer servo
post-hoc collapsed the same memory 0.98 -> 0.10.

### What was tried and shed on the way

- **Temporal track hierarchies** (240/720 ms per-signal L1/L2): the
  founding design. Essential in partially observed worlds (they carry
  velocity as slope); measured as a *tax* here once velocity was
  sensed directly — the present state was ~3% of the cue's energy and
  the read followed history: feedback delayed by hundreds of ms
  (drift at low gain, oscillation at high gain). The champion's top
  layer sees **zero milliseconds of past**: this world is fully
  observed, and every rig's improvement tracked the present's share
  of the cue rising.
- **Motor tracks / command-in-the-template**: commands as history
  channels and as jointly-learned cargo; each form paid measured
  costs (identity-chain decode capped steering at 0.43; the joint
  attribution trap cost 0.72 -> 0.00 when a tally was bolted on).
  Final law: **description and selection are separate organs**.
- **The full flat memory** (6 channels, 8192 templates, one layer):
  approach solved, terminal never (closest ~1 m plateau, gate shut at
  4M ticks). The cascade decomposition solved in 3 minutes what the
  flat rig couldn't in 40: the outer problem is d~2.

## 3. The brain map

| organ here                          | counterpart | notes |
|---|---|---|
| foveated sensory bumps              | sensory cortex / retina | resolution where the task lives; the fovea law, applied to position, velocity, tilt, command axes alike |
| (temporal tracks, when used)        | sensory cortex hierarchy | earn their keep only under partial observability — measured both ways |
| outer templates (situations)        | PFC / association cortex | recognise "where am I relative to what I want"; goal-relative by construction |
| velocity/motor vocabulary banks     | motor cortex | a repertoire of action prototypes; the champion's 32 words, the raw rig's innate 5x5 grid. **The vocabulary audition law**: a bank must prove "zero decodes to zero" before anything speaks through it — the drunk-pilot bug was a dictionary with no word for "stay" (its training window ended before any drone ever dwelled) |
| the cetele (count table)            | corticostriatal weights | per-(situation, action) structure — the ONLY conditioning that can carry action preference (per-template scalars provably cannot: no per-tick baseline reorders a candidate set) |
| value tables V[j], W[j,u]; TD error | basal ganglia / dopamine | selection bias only; **records are never edited by reward** — content learns what happened, goals bias what is chosen |
| smoothing integrator                | brainstem neural integrator | steps to ramps; one neuron's worth |
| the fixed reflex                    | brainstem / spinal servo | innate, fast, not dopamine-trained; the muscle-level loop is unemployed here (actuators are assignment statements) |
| (absent)                            | cerebellum | the adaptive predictive floor — unemployed until the world gets unkind (wind, payload, wear) |
| attention (share selection)         | thalamo-cortical gating | greedy conditional-MI over cue blocks; conditioning strips correlational free-riders; validated on a known-answer case, then run as a live co-adaptive loop (leaky cetele + never-frozen templates + slow mask — three timescales) |

## 4. The three ways it learned

**4.1 Imitation (oracle) + DAgger.** Teacher = a hand-made cascade
controller. Watch, notch, imitate. Alone: approach yes, finish no.
Two data laws closed the gap: **dwell** (training episodes must run
through the hold, or the skill being tested never appears in the
data) and **tight-envelope DAgger** (the pupil's own flailing,
teacher-labelled — the only source of the states only an imperfect
flyer produces; with a loose envelope the same idea floods fixed
capacity with junk and collapses the read).

**4.2 Reward refinement.** On the working base, teacher off: TD
values per (situation, word), tilt the duty weights by exp(beta*W).
Naive version: monotone harm. With two fixes — TERMINAL PROTECTION
(hover rows read pure habit) and DIRECTED EXPLORATION (try the
fastest same-direction word the row knows) — and enough training:
the project's first **dominating** reward improvement, and at high
beta the policy flies at the measured interface ceiling, proving the
values found exactly the priced slack.

**4.3 Tabula rasa.** No teacher anywhere. Innate: the reflex, a
7x7 velocity-word grid (with "stay"). Learned from reward alone
(distance progress + time tax + success bonus, eps-greedy,
SARSA-lambda): situations self-organise from its own flailing, and
stillness is discovered as *the profitable way to end an episode*.
A final variant removed even the reflex — raw motor levels at 50 Hz,
six sensed channels, an innate vestibular potential. VERDICT: after
400k decisions, indistinguishable from random flailing (upright 20%
vs 17%). With the reflex, the same machinery reached 0.63 in fewer
decisions. **The decomposition is not an efficiency trick — it is a
learnability requirement**: reward cannot build the reflex floor at
this budget, because credit must flow through an unstable plant at
50 Hz before any state worth valuing exists. Brainstems are innate
(or trained by prediction error — the cerebellar route), and reward
learns on top of them. The night's thesis, measured from its
failure side.

## 5. Results

| policy | strict success | closest | median ticks |
|---|---|---|---|
| flat memory, 4M ticks | 0.00 | ~1.2 m plateau | — |
| tuned full PID (reference) | 1.00 | — | 110 |
| **cascade champion** (imitation) | **1.00** (30/30, 100/100) | **0.063 m** | 266 |
| champion + reward tilt (beta 0.1) | 0.98 | — | **246** |
| reward frontier (beta 0.5) | 0.70 | — | 206 (= vocab ceiling) |
| co-adaptive attention loop | **1.00** | 0.06 m | 265 |
| pure RL, reflex kept | 0.63 | 0.03 m | 445 |
| pure RL, raw motors, no reflex | **0.00 — indistinguishable from random** (upright 20% vs 17%, survival 81 vs 70 ticks) | | |

Side quests: balance rung 0.72 (where dwell, present-share, fovea and
DAgger laws were forged); attention rung 1 — the architecture
rediscovered its own hand-built cue (pres-dx 0.287, pres-dy 0.155,
angle and history refused after conditioning) and ran it live at
1.00; MNIST transfer — the same cetele with labels as actions:
76.2% classification in ~6 s of training, and read backwards (tally
inversion: label -> P(template|label) -> graded descent) it draws
legible digits for every class.

## 5b. The partition ablation (the deflation, courtesy of an outside critic)

Replace the champion's 256 learned templates with a FIXED 16x16 grid
over the foveated (dx, dy) — 256 cells, zero learning — everything
else identical (same vocabulary, cetele, reads, reflex, budget):

    templates 0.95 / 0.081 m / 291    grid16 0.99 / 0.033 m / 229
    grid8 0.64                         grid32 0.60

The grid WINS. On this problem the learned partition is not
load-bearing: the champion's outer loop is tabular behaviour cloning
over a foveated grid. What IS load-bearing and shared by both arms:
the fovea (raw-coordinate grids fail — the warp is the intelligence),
the cetele + duty-mean, the cascade, the reflex. Non-monotone in
cells (32x32 loses: thinner notches); the grid's edge is plausibly
its CRISP boundaries — purer count-table conditionals than soft
cosine borders. Learned partitions must earn their keep where cue
spaces cannot be enumerated (the 4-channel balance rig, the 900-dim
MNIST cue) — an ablation owed there too.

## 6. The laws (earned, each with a scar)

1. Evaluate frozen; training metrics on live trajectories lie.
2. Foveate where the task lives; a success band inside one cell is
   invisible to the read.
3. Derived quantities get their own encoding — never the difference
   of two reads, never a mixed reference inside one window.
4. The present must dominate a control cue; history is delay.
5. Choose between answers, average within an answer: mode at decision
   boundaries, duty-mean for regulation.
6. Description and selection are separate organs; a tally must be
   conditioned on the same partition that owns its rows.
7. Records are never edited by outcome; value lives at selectors,
   as per-(situation, action) structure or not at all.
8. The skill being tested must exist in the data (dwell); the states
   the pupil creates need labels (DAgger, tight envelope).
9. Commander and reflex co-adapt; tune them together or not at all.
10. Audition everything that speaks: teachers, harnesses, and
    vocabularies ("zero must decode to zero").
11. A layer's cue must carry exactly what its decision depends on —
    and *what is relevant can itself be learned* (conditional
    information gain; attention as selection turned inward).
12. Hierarchy pays only where the task's law needs what it extracts;
    the drone taught us when layers matter by refusing to need them.
13. Ablate the learned part against the dumbest fixed alternative —
    then sweep the budget. At 256 cells the hand grid beat the
    learned partition (0.99 vs 0.95: crisp edges, cells to spare).
    At 64 cells the learned partition equalled the 256-grid (0.99,
    best-ever 0.022 m) while the 64-grid collapsed to 0.64. THE LAW:
    adaptive partitions buy ALLOCATION EFFICIENCY (4x here), not
    asymptotic quality — hand grids win when cells are free, learned
    wins when they are dear relative to the world. (This is why the
    d=4 balance rig, deep in the scarcity regime, was never a
    passenger.)

## 7. Open

Per-row reward tilt (spend beta only where V says time is wasted);
wind (the world that employs the tracks, the wind-skin channels, and
the cerebellar floor at once); goal-encoding in the top layer
(goal-conditioned cetele rows — dissimilar goals partition apart, so
skills should coexist); the MNIST coupling curve (does better
selection make better generation); crisper descent (hardened reads,
residual voices). All queued with their instruments ready.

---

# Part II — The Computations

## A. Encoding: value -> fovea -> bump

Every scalar the system senses or commands becomes a small population:

    u = clip( asinh(v / s) / asinh(R / s), -1, 1 )      # fovea warp
    w_i = 0.5 * (1 + cos(pi * |c_i - u| / r))  where |c_i - u| < r

24 cells c_i uniformly spaced on [-1,1]; bump radius r = 3 spacings
(~7 active cells). The warp scale s places the resolution: s = 0.25 m
for position (centimetres near the target), 0.15 m/s velocity,
0.05 rad tilt, 0.1 rad/s spin. WHY: overlap between bumps IS the
similarity metric — nearby values share cells, distant ones do not —
and the fovea decides which differences the architecture can see at
all. (The law's scar: a success band inside one cell is invisible.)

## B. The template layer (hypercolumn)

**Scoring** (masked cosine with a contrast floor):

    supp   = cells where the query x > 0
    n_j    = || w_j restricted to supp ||
    s_j    = (x . w_j) / ( max(n_j, 0.25 * max_k n_k) * ||x|| )

Both norms matter: dividing only by the template's gives a
non-correlation that ranged to 3.3; the floor stops a template with
almost no mass in the queried cells from bidding absurdly high
(tiny overlap / tiny norm). This one formula is the read everywhere.

**Learning** (boot, then geodesic rotation). First K inputs become
rows directly (centred, tiny jitter, unit-normalised). After that,
for input x with centred norm nn and winner w (by raw dot):

    c  = (w . x) / nn                       # cosine to the input
    th = eta * c                            # step size, eta = 0.05
    tn = sqrt(1 - c^2)
    a  = cos(th) - c * sin(th) / tn
    b  = sin(th) / tn
    w <- a*w + (b/nn)*x - (b*mean(x)/nn)    # rotate toward x
    (re-centre + renormalise every 128 wins)

This is rotation ON THE UNIT SPHERE toward the input — an exponential
moving average of directions. Consequences used all night: templates
track drifting inputs natively (the co-adaptive loop needs no
rebuild), and they BLEND everything they absorb (why the cetele, which
keeps alternatives countable, must exist beside them).

**The code up**: relu the scores, keep the top-8 with magnitudes
(graded sparse speech — overlap preserved, unlike one-hot).

## C. The cetele (count table) and its reads

    notch:   C[winner_row, word] += 1
             word = argmax over the vocabulary bank's scores of the
             executed command's bump code (HARD assignment — soft
             profiles smear the tally and the mean read over-smooths)
    leak:    C *= (1 - B/15000) per decision epoch (co-adaptive form;
             multi-skill version should decay only visited rows)

    duty-mean read (regulation):   v = sum_u (C[j,u]/sum C[j]) * val(u)
    mode read (rival decisions):   u* = argmax_u C[j,u]
    value-tilted read (RL):        weights ~ C[j,:] * exp(beta * W[j,:])
                                   (+ terminal protection: rows whose
                                   duty-mean speed < 0.5 read pure
                                   habit)

## D. The vocabulary bank and its decode

The bank is just another template layer over command bump codes
(2 x 24 cells). Decoding a prototype back to numbers is a centroid
per half, then the inverse warp:

    h   = clip(row_half, 0)
    u*  = sum(h_i * c_i) / sum(h_i)
    v   = s * sinh( u* * asinh(R / s) )

**The audition law**: before anything speaks through a bank, assert
decode(assign(0)) ~ 0. The drunk-pilot bug was this assertion failing
silently: the bank's training window (20k ticks / 128 worlds = 156
ticks each) ended before any drone ever dwelled, so the vocabulary
contained no word for "stay" and stillness was filed under cruising.

## E. Attention: relevance as conditional information

Each cue block gets a free quantiser (a present block: its bump cell
index; a context block: its L2 winner). Over a window of decisions
with action labels U:

    gain(b | S) = H(U | q_S) - H(U | q_S, q_b)      # plug-in counts
    corrected   = gain - gain_with_shuffled_labels    # bias control

Greedy: pick the best block, CONDITION on it, re-score the rest,
repeat while corrected gain clears a threshold. The conditioning is
the whole trick — raw MI credits history and even the angle (they
correlate with position, which determines the action); conditioned on
the chosen present blocks, the free-riders collapse to ~0 and are
refused. Shares ~ gains, applied as sqrt(share) scaling per block
(or hard pruning of dims). Live version: shares slewed slowly
(0.3 per 100k ticks) — mask slowest, templates middle (eta), leaky
tally fastest. Three timescales, strictly ordered, or the loop
chases its own tail.

## F. Reinforcement: the exact updates

SARSA(lambda) on the decision chain (10 Hz outer, or 50 Hz raw):

    delta = r + gamma * Q[j', u'] - Q[j, u]
    e    <- gamma * lambda * e;   e[j, u] = 1     # replacing traces
    Q    <- Q + alpha * delta * e                  # (per world, summed)
    traces cut at episode boundaries; no bootstrap across teleports

Reward (potential-based shaping, policy-invariant in the limit):

    r = (d_prev - d_now)                # distance progress
      + 0.3 * (|tilt|_prev - |tilt|_now)  # innate vestibular term
      - time_tax                        # -0.05 (10 Hz) / -0.01 (50 Hz)
    terminal: +20 on success (hold 10 ticks), -5 on blowup/timeout

Exploration: eps-greedy annealed 0.3 -> 0.03; refined form adds
DIRECTED exploration — with p = 0.2, take the fastest word whose
direction projects onto the row's duty-mean (proj = VCU . v_hat,
masked to words with duty > 0), so speed evidence accumulates where
it could matter. Stillness is never rewarded directly: at the target,
progress = 0 while the tax bleeds, and the only unpunished exit is
the 10-tick hold — staying is learned as the profitable way to die.

## G. The reflex and the plant

    phi_des = clip( 0.30 * (vx - vcx),  +-0.6 )
    coll    = clip( (4.9 + 1.8 * (vcy - vy)) / max(cos(phi), 0.5),
                    0.3, 8.0 )
    diff    = clip( 2.0 * (phi_des - phi) - 0.6 * omega,  +-2.5 )
    levels  = nearest of 13 foveated thrusts to coll -+ diff
    smoothing upstream: vc <- vc + 0.15 * (vc_cmd - vc) each tick

Physics per 20 ms tick (Euler): a = (-T sin(phi), T cos(phi) - g),
alpha = arm * (t_R - t_L) / I, with T the two rotor thrusts summed.
The 13 levels are themselves foveated (|u|^1.7 warp around hover =
4.9 N per rotor), so the finest thrust distinctions live exactly at
hover — the same fovea law, on the actuator.

## H. Schedules and hygiene

Every-5-tick learning (staggered per world) decorrelates near-
duplicate 20 ms neighbours. Staged freezing where partitions must
stay meaningful to what is stored over them; the co-adaptive
alternative replaces freezing with leak + EMA + slow mask. Batched
GPU learning applies each winner's rotation individually (winners
deduped within a batch — never averaged: the B=2 parity lesson).
Evaluation: frozen, fresh episodes, 64-100 at a time, the strict
gate only.

## I. The layered sensory track (the temporal hierarchy)

The project's founding organ — shed by the champion, load-bearing in
the balance rig, and the right tool wherever state is hidden. One
track per signal; a track is two stacked template layers over time:

    tick stream    v(t), foveated -> 24-cell bump per tick
    L1 window      the taps [t, t-3, t-6, t-9, t-12] (240 ms),
                   each tap its own 24-cell block -> a 120-dim
                   vector. 64 templates. A template is a SHORT
                   MOTION SHAPE of that signal: its level AND its
                   slope (velocity implicit — no derivative is ever
                   computed; the shape carries it).
    L1 code        masked-cosine scores -> relu -> top-8 with
                   magnitudes (graded sparse speech)
    the ring       ONE L1 code is computed per tick and kept in a
                   37-deep ring; L2 reads its taps from the ring
                   instead of recomputing (the 11 ms -> 1.7 ms
                   per-tick lesson)
    L2 window      the L1 codes at [t, t-6, ..., t-36] (720 ms),
                   7 x 64 = 448-dim. 128 templates. A template is a
                   TRAJECTORY OF SHAPES — trend, acceleration, "how
                   this signal has been carrying itself lately."
    up             graded top-8 again; per-track code normalised to
                   sqrt(its energy share) before joining the top cue

Design invariants, each a paid-for law: every window ENDS AT NOW
(the present is the last tap — and still needs its own dedicated
present bumps at ~50% share, or the cue is 97% history and the loop
gets delayed feedback); references inside a window must be computed
from the same taps at train and read (the reference bug, four
appearances); masked partial reads let early-episode windows (still
filling) degrade gracefully instead of poisoning the match.

When tracks PAY: hidden state (velocity unsensed -> it exists only
as slope; constant-per-episode wind -> drift history is the only
wind gauge), and pattern-over-time worlds (the wave rig: two layers
0.086 vs one layer's 0.817). When tracks TAX: fully-observed control
— measured on the chase outer loop as 1.00 (present-only) -> 0.52
(+history) -> 0.05 (+an irrelevant channel). Same organ, opposite
signs; the difference is whether the task's law needs what the
layers extract. Attention (Part II.E) is how the architecture is
learning to make that call itself.
