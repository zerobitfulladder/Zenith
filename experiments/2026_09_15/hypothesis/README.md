# Step 1 — new action set                                  2026-09-15   GATE FAILED

Spec: `SPEC.md`. One axis moved: what the machine can do and notice. Learning, credit,
splitting, the answer-counterfactual, the charges and the read-out are byte-identical
in spirit to the reference line (`../programs/machine.py`).

    python run.py ref      the hand-written program in the new vocabulary
    python run.py learn    6 seeds

## Numbers

| | | |
|---|---|---|
| hand-written, new vocabulary | **1.000** at 4.9 steps, score **+0.901** | ceiling is higher than the old +0.709 |
| learned, 6 seeds, 20k | 0.333 0.342 0.442 0.443 0.480 0.490, median **0.443**, 2/6 | |
| reference line, 6 seeds, 20k | 0.517 0.545 0.735 0.843 0.853 0.917, median **0.789**, 6/6 | |

The gate in SPEC.md said stop if it does not match. It does not match. Stopping.

## Why — it is arithmetic, not a worse machine

The action set is cleaner but the **shortest correct program got longer**, and blind
search is exponential in program length.

| | old | new |
|---|---|---|
| actions | 28 | 38 |
| shortest program | `GOTO 1; PEEK s0 DOWN; SAY s0` = **3** | `GOTO 1; WRITE s1; STEP DOWN; WRITE s0; SAY s0` = **5** |
| chance of stumbling on it in one episode | ~28 x (1/28)^3 = **1.3e-3** | ~26 x (1/38)^5 = **3.3e-7** |
| expected stumbles in 20k episodes | ~26 | ~0.007 |

**More episodes will not fix this.** At 3.3e-7 it would take ~3 million episodes to
expect a single stumble. The machine reached 0.44 without any search at all, by the one
route that needs none: `[edge LEFT & under cursor=3] -> SAY 3`, i.e. "say what is under
the cursor", which the answer-counterfactual hands it directly without a program.

## The thing worth seeing

`PEEK sK <direction>` was a **hand-built chunk** -- "step there and write it" fused into
one action. Removing it was right on design grounds (it is an eye movement disguised as
a memory write) and it cost two steps of program length, which cost four orders of
magnitude of findability. The old machine worked partly because I had pre-chunked its
subroutine by hand without noticing.

The old line also only worked at all because of the answer-counterfactual: a local signal
for the **last** action. Nothing gives the middle of a program a signal of its own. That
is the same open item as before, and it is now the blocker rather than a footnote.

## Choices

1. **Propagation.** A step's value inherits from the state it leads to, so a step that
   merely sets up a later step has an outcome. Still counting, no gradients. Changes the
   credit story from "the worked program takes the outcome" to "each step inherits from
   the next". **Needs a go.**
2. **Chunking.** Let the machine name a recurring pair of actions as one action -- the
   automatic version of what `PEEK` was by hand. Shortens programs as it learns, which is
   exactly the lever the arithmetic says matters.
3. Put `PEEK` back, and admit we are hand-chunking.

Not recommended: tuning step 1, or adding belief and prediction on top of a step that did
not pass its gate.

---

# Teaching instead of searching                                 2026-09-15   WORKS

I had claimed you cannot name a program you have never executed. That is wrong, and the
user was right: a program can also be **shown**. What survives of the claim is only that
*if blind search is the only source, it must be findable by blind search*. Demonstration
is a second source and it removes the exponential outright.

The division that makes it work, in the user's words -- a demonstration *"shows actions
that are correct but doesn't say why; the why is the thing the student needs to figure
out by doing itself"*. A trace is not a program. The teacher supplies the half this
machine is bad at (which actions, in which order) and the student supplies the half it is
good at (under which conditions). **The conditions are never shown.**

The target must be **contrast, not approval**: at each demonstrated step the teacher's
action is right and all 37 others are wrong. Without that nothing is learnable -- if
everything the student sees is correct, no fact ever separates anything and no split is
ever worth making.

## Numbers — tested alone, on grids it has never seen

| | median | best |
|---|---|---|
| hand-written program (the teacher) | 1.000 at 4.9 steps | |
| **taught, 20k demonstrations, no duplicate rules** | **0.863** at 4.5 steps | **0.910** |
| taught, 5k demonstrations, no duplicate rules | 0.851 | 0.905 |
| taught, 20k demonstrations, duplicates allowed | 0.762 | 0.782 |
| learned by search, same machine | 0.443 | 0.490 |
| chance | 0.333 | |

## What it worked out for itself

Never shown a single condition:

    [s1 empty & under cursor=1]             -> WRITE s1        v=+1.000
    [s0 empty & s1=1 & standing on s1]      -> STEP DOWN       v=+1.000
    [s0 filled & s1=1]                      -> SAY s0          v=+1.000
    [s1 empty & under cursor=0]             -> GOTO nearest 1  v=+1.000

It found `standing on s1` unaided -- the exact cursor-to-slot relation I was about to add
by hand to rescue the search version. Under teaching, the expressiveness gap closed
itself, because the student only needed the relations that actually discriminate.

## Duplicate rules are a search device and poison teaching

Copies of a rule are extra tickets in the selection lottery, which is why they were
load-bearing for search. Under teaching there is no lottery to win during training, and
they spend the 160-rule budget on eight copies of a handful of perfect rules instead of
on covering the remaining cases. With copies, more demonstrations make it *worse*
(0.818 -> 0.762). Without, more demonstrations make it better (0.851 -> 0.863).

Also worth recording: the first version of this comparison returned identical numbers to
three decimals in both arms because a string replace had silently matched nothing and both
arms ran at 8 copies. Assert the pattern exists before replacing.

## Open

* The last ~0.14 to the ceiling: conditions that are over-specific, e.g. `GOTO nearest 1`
  tied to `under cursor=0` because it correlated at the start, when the right condition is
  just `s1 empty`. Coverage, not learning.
* Fading the teacher. Trigger it on **failure**, not on a timetable -- the student tries,
  the teacher steps in on the ones it got wrong. No schedule, and it fades itself.
* Naming: a demonstrated trace that recurs is the natural thing to name, and teaching is
  what finally makes first successes plentiful enough to name anything at all.

---

# Practise, and ask when you fail                               2026-09-15   CEILING

The user's protocol, built as stated: demonstrate, let the student try, show the solution
when it fails, repeat. `practise.py`.

One learning rule throughout -- contrast on a trace. The only question was where traces
come from.

| | median | steps | |
|---|---|---|---|
| hand-written program (the teacher) | 1.000 | 4.9 | |
| **practises, asks on failure, never copies itself** | **0.997** (best 1.000) | 5.0 | asked on 9% of grids by the end |
| taught only, no practice | 0.862 | 4.7 | |
| practises, asks on failure, copies its short wins | 0.800 | 4.8 | |
| practises alone, copies its short wins | 0.206 | 23.6 | |
| practises alone, copies any win | 0.107 | 26.8 | |
| learned by search, same machine | 0.443 | | |
| chance | 0.333 | | |

## Why correcting beats demonstrating

Taught-only plateaus at 0.862 with coverage gaps -- conditions that are over-specific
because they happened to correlate, e.g. `GOTO nearest 1` tied to `under cursor=0`.
Practice finds exactly the grids where those gaps bite, and the teacher's demonstrations
then land only there. The teacher's effort goes where the student is weak instead of
being spread over random grids. **The fading needs no schedule**: the teacher is summoned
by failure, so it disappears as the student improves (9% by the end).

## Self-imitation is harmful, and no filter fixed it

Feeding the student's own successful runs back as demonstrations destroys it: 0.997 ->
0.800 when added to correction, and 0.206 -> 0.107 alone. It is a feedback spiral --
slight noise gets copied, which makes more noise.

The reason is structural. **A correct answer does not make a run worth imitating.**
Judging a trace only by its outcome gives no way to tell the necessary steps from the
incidental ones inside it, and the dependency walk cannot help, because a lucky wander
into the right cell is genuinely part of the causal chain. Two filters were tried and
neither rescued it: the dependency walk, and a cap on trace length.

For the student to teach itself, something independent of the outcome would have to vouch
for the run. Prediction is the only candidate on the table that could -- a run whose
predictions came true is a run the machine understood, regardless of whether it also
happened to get the answer.

## A knob chosen wrongly, and the arithmetic that found it

The first practice run used a selection fuzziness of 0.40. After teaching the right rule
sits at +1 and the others at -1, so at 0.40 the right rule wins e^(2/0.4)=148 tickets
against 37 others, i.e. 80% of steps, and a five-step program survives 0.80^5 = 0.33.
Measured "right while practising" was 0.34. Pick this knob from that arithmetic, not by
guess. (0.25 gives ~0.99 per step. It did not rescue self-imitation, which was the real
fault.)


---

# Reference re-check                                            2026-09-15, late

`machine.py` gained opt-in facts (region, count, next-empty-slot), a coarse region
jump, and knobs (slots, relations, maxcond, cap) for the digits line. `Grid` opts into
none of them. Re-measured: hand-written 1.000 at 4.9 steps; practise + ask on failure
0.997 / 0.998. Unchanged.

After the `grow()` rewrite (split thresholds as knobs, defaults unchanged): ceiling 1.000 at 4.9, practise + ask 0.997 / 0.997. Unchanged.
