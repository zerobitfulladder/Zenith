# Exp 8 (late night): carousel + bounce + convolutional unit

run_mnist_carousel_bounce.py — MNIST carousel (0..9 loop, FRESH
exemplar per class per lap), whole-frame same-rate stack, dense
speech, up-down bounce. TOP=L1: PERFECT (advance 1.00, timeline
1.00) — the two-day-old carousel problem solved clean; galleries
show L1 = a transition table (keys = previous class blur, cargo =
canonical next digit, ~12 style variants per transition). TOP=L2/L3:
frozen 8s (mean-digit mush attractor) — L2 galleries show WHY: its
keys are IDENTICAL mush across units because gamma .9 horizon ==
period 10, so its trail is phase-invariant. LAW SHARPENED: a layer
helps only at timescales where the world actually varies; this task
has single-scale structure, so the top has nothing to add. Also:
free-running is self-drive (own canonical museum) — the top's real
test is prediction under REAL drive or two-scale structure
(alternating-lap carousel designed, queued).

run_square_conv_temporal.py — the user's UNIFICATION: no separate
spatial encoder; convolution IS the layer (8x8/s1 shared dictionary,
per-position leaky trails, delayed-write rows per position, blank
windows skipped). Square: tracks with constant offset then wall-
sticks (8.58) — local windows alone drift; prediction belongs over
the code (as the user said). Templates = mini comet units.

run_mnist_carousel_conv.py — conv L1 + code-level L2 bounce.
L1 gallery: proper 8x8 STROKE SHEET (success — the conv temporal
unit learns the V1-like dictionary). L2 over the global code map:
FAILED twice, diagnosed twice: (1) co-training a tiny global bank on
a forming code -> stale bootstrap + rich-get-richer MONOPOLY (1
distinct read winner; one row absorbed all updates) — FIXED by
staged training (fresh L2 on frozen L1: 32/32 rows win, top share
.21, 9 distinct read winners); (2) remaining 0.10 TF: the global
code-trail match is non-discriminative — shared stroke mass swamps
the class-specific fraction. Root cause vs the working static
stack: the static architecture NEVER had this layer — it had conv
3x3 L2/L3 over code (local, high signal) and the global memory at
L4 with SKELETON learning views. The shortcut (dictionary ->
global memory directly, single-mode views) is what failed, not the
leaky integration (bounce inference keeps full leaky integration at
every layer, learning off — verified).

NEXT SESSION (priority order): (1) conv L2 over the code map +
global L3 with split views (skeleton learn / graded speak) — the
full uniform-unit architecture, on the carousel; (2) two-scale
carousel (alternating lap orders A/B) — the fair test of "the top
should help"; (3) three-movies on the wired/bounce arms.

---

Files (all import the rig from `../completion/`):

- `run_mnist_carousel_bounce.py` -> `results/mnist_carousel_bounce/`;
  `show_carousel_templates.py` draws its `templates_L{1,2,3}.png`
- `run_square_conv_temporal.py` -> `results/square_conv_temporal/`
- `run_mnist_carousel_conv.py` -> `results/mnist_carousel_conv/`;
  `show_conv_carousel_templates.py` draws its `templates_L2.png`

Run:

    .venv/bin/python experiments/2026_08_27/carousel/run_mnist_carousel_bounce.py
    .venv/bin/python experiments/2026_08_27/carousel/show_carousel_templates.py
    .venv/bin/python experiments/2026_08_27/carousel/run_square_conv_temporal.py
    .venv/bin/python experiments/2026_08_27/carousel/run_mnist_carousel_conv.py
    .venv/bin/python experiments/2026_08_27/carousel/show_conv_carousel_templates.py
