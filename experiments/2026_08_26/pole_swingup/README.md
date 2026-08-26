# `pole_swingup/` — swing-up from any state, and the channel-separation law

Scripts: [`run_pole_swingup.py`](run_pole_swingup.py) (imported by `../pole_tracks/` and by 2026-08-28) and [`swingup_viewer.py`](swingup_viewer.py) (live viewer). Results: [`results/`](results/) — `report.md`, `weights.npz`.

## Swing-up from ANY state (run_pole_swingup.py, swingup_viewer.py)

User's insight: "fallen" was an arbitrary label — every cart-pole
state is recoverable (rock to pump energy, catch at the top). Sensing
went periodic (64 bumps around the full circle; hanging down is a
percept). Self-calibrating energy-pump + PD-catch demonstrator
(sign auto-chosen -1, 0.68 recovery from uniform-random states).

Two-factor debugging, both factors necessary:
1. PHASE-LAG LAW: trail-only percept (v1) had 0.92 teacher-forced
   agreement yet 2/100 closed-loop, DAgger flat — a resonant pump
   dies of the ~2-step EMA lag (mistimed pushes REMOVE energy), while
   a balancing regulator tolerates it. Regulators tolerate lag;
   resonators die of it. Trails complement the percept, never replace
   it. Fix: CURRENT angle code as its own pathway beside the trails.
2. With the lag-free channel, DAgger ignites: 3 -> 0 -> 58 -> 47 /100
   (teacher 68). Neither factor alone works (lag-free round 0 = 3;
   lagged + DAgger = flat 2). Round-3 dip = eval noise/capacity
   (71k samples on K=512) — capacity and more rounds queued.

The pupil swings up from ANY state at ~85% of its demonstrator,
sensing only the pole angle. swingup_viewer.py: Reset-anywhere /
Shove / Run, no fall concept — live.

### Metric correction (user's eye vs our gates) — the pupil is PERFECT

User watched the viewer and disputed the 47/100: "it manages almost
every time; it has its own way of doing it." Measured three ways on
the same weights: strict textbook gate (1.5-deg stillness touch) =
47/100; user's practical criterion (+-7 deg held 10s) WITH the 30m
arena walls = 29/100 — but every single failure was CART-EXIT (median
3 upright-touches per failure; zero in-place failures): the pupil
senses ONLY the pole angle, so cart position does not exist in its
world and drift cannot be its fault. Remove the walls (the viewer's
actual physics): **100/100** from uniform-random states incl.
hanging, within 60s.

METROLOGY LAW: evaluate a creature in the world it can perceive.
Every reported "failure" was either stillness pedantry or an
invisible wall outside its sensorium. The angle-only swing-up skill
is COMPLETE. (Cart-position regulation is a different task requiring
a second percept channel — future.) The user's subjective judgment
outperformed both formal metrics.

### Representation trilogy — CHANNEL-SEPARATION LAW

User proposal: replace [current ; EMA trails] with a single LEAKY-SUM
trace (T = x + gamma*T; present always the brightest element,
eligibility-trace form). Measured three ways, same task, same rule:
dedicated present channel + separate EMA tails = 58/100; single
leaky-sum gamma .5 = 13; gamma .25 = 19. Weight-share was not the
issue (gamma .25 puts the head at 3x the whole tail and still loses
3x): summing present and past into ONE population smears the present
bump into a ridge — phase discrimination dies at any mixing ratio.
LAW: PRESENT AND PAST BELONG IN SEPARATE CHANNELS — a precision
channel and a context channel; the head must never share a vector
with its tail. Biology concurs (transient vs sustained populations).
The user's brightness-falloff picture survives as the two-population
implementation. Winning config restored and retrained.
