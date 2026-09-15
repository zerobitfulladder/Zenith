# Inventing programs (first look)                                    2026-09-15

A machine that writes its own programs in a language given to it, and pays for
every step it takes. One axis changes here versus everything before: **what the
learner produces is a program, not a code.**

    python run.py ref      score the hand-written program (the ceiling)
    python run.py grid     learn on BELOW
    python run.py mnist    learn on three digits, one cell seen at a time
    python run.py all

## The language

An instruction is `(condition, action)`. A condition is a set of tests on working
memory. **There is no if, no jump, no loop and no halt**, so every program that can
be written runs:

* sequence — happens because working memory changed
* a loop — a rule whose condition still holds
* a conditional — a rule whose condition names what to check

Working memory: a cursor (x, y, scale), 2 slots holding `(symbol, x, y)`, 2 tags.
Actions: move / raster-advance / jump to the nearest given symbol / centre on the
ink / zoom; attend here or one cell over, into a slot; set a tag; say a constant;
say what a slot holds. Nothing returns a value — every action writes to the shared
memory and every action reads it.

Credit is **dataflow, not time**. Every slot, tag and the cursor record which step
wrote them. When the machine says an answer, walk backwards through what that step
read: those steps are the *worked program* and take the outcome. Steps outside the
walk are charged their time and credited nothing. Learning is counting only —
a running average per rule, plus a tally per (rule, test) so a rule splits when one
test separates its good firings from its bad ones. No gradients, no schedules.

## BELOW — the reference task

5x5 grid, one cell holds the marker `1`, the answer is the symbol directly below it.
The marker moves, so no fixed route works. The shortest correct program is three
rules: scan, `if the marker is under the cursor, attend one cell down`, `if that
slot is filled, say what it holds`. Answer copied out of memory, not a constant.

| | acc | chance | steps | score | rules |
|---|---|---|---|---|---|
| hand-written program | **1.000** | 0.333 | 14.6 | +0.709 | 3 |
| learned, no answer-credit | 0.000 | 0.333 | 30.0 | -0.600 | 32 |
| learned, 6 seeds (read out as learned) | 0.517 / 0.545 / 0.735 / 0.843 / 0.853 / **0.917** | 0.333 | | | 160 |
| best seed, greedy read-out | **1.000** | 0.333 | 2.9 | **+0.941** | 160 |

The hand-written program proves the language can *say* the answer. 6 of 6 seeds get
above chance, median 0.789; the best seed beats the hand-written program's score, because it
replaces the 14-step scan with one `GOTO nearest 1`. It invents the conditionals
exactly:

    [s0 empty & under cursor=1]  -> PEEK s0 DOWN     v=+1.197  n=368
    [s0 filled & under cursor=1] -> SAY s0           v=+0.973  n=707

## MNIST — 14x14, four ink levels, digits 0/1/7, one cell at a time

0.663 greedy / 0.712 read out as learned, against 0.345 chance, 2.6-4.5 steps. What it learned is a one-glance heuristic,
not a program: `[under cursor=3] -> SAY 1` — the cursor starts at the centre, a 1
has ink there and a 0 has a hole. Honest reading: with three classes and a price per
step, a one-step guess is a good deal, so there is no pressure for a program. This
is the same objection as ever — a task that one glance can half-solve will not
make programs appear.

Final verified run of `run.py all 20000`: BELOW hand-written 1.000, best learned seed
1.000 greedy / 0.917 soft at 2.9 steps; MNIST 0.663 / 0.712.

## Six ways this broke, each a finding

1. **Stalling beat answering.** `TAG` costs one step and can never be blamed for a
   wrong answer, so the machine learned to tag forever. A step limit reached without
   an answer has to be blamed on every step of the episode.
2. **Bystanding beat contributing.** With off-graph steps charged only their time
   (-0.02) and in-graph steps risking -1, the rules that survived were the ones that
   never touched the answer. Dataflow credit breeds free-riders unless the outcome is
   compared against a baseline. Running mean of the reward fixes it.
3. **The chain is too rare to stumble on.** `GOTO 1; PEEK DOWN; SAY s0` is three
   specific actions in order out of ~43, about 1e-4 per episode. No amount of value
   learning finds that. What fixed it: saying is the one action whose outcome can be
   known *without doing it*, so after each episode we ask what each way of saying
   would have scored at moments it could have fired. That turns "when may I answer"
   from a search problem into a counting problem — and it is exactly the signal a
   verification loop would provide.
4. **That credit then drowned the rest** — answering rules got ~30 updates an episode
   against 1 for everything else, so blunt `SAY` rules outranked the scan. Sampling
   two moments per episode restores the balance.
5. **A frozen greedy policy loops where a soft one does not.** `GOTO 1 -> MOVE DOWN
   -> GOTO 1 ...` forever. Two fixes: charge every step since working memory last
   returned to a state it had already been in (provably idle), and read the machine
   out the way it was learned. Seed 0: 0.590 greedy, **0.958 soft** — same rules.
6. **Duplicate rules are vote weight.** Splitting made identical copies; each copy
   enters the selection pool separately, so copies amplify a rule. De-duplicating
   *cost* accuracy (1 of 6 seeds took off instead of 2 of 4). Copies are capped at 8
   and left in, but this is an accident standing in for something that should be
   explicit — a rule's strength.

## What the language cannot say

A condition can test *what* a slot holds, never *how it got there*. The hand-written
program works partly because only one rule ever fills a slot, so "s0 filled" implies
"filled correctly"; with ten filling actions that test is uninformative. Tags are the
expressible workaround — the machine has to mark "I came from the marker" itself — and
that only works once the credit walk follows tag provenance too (it did not, at first).

## Open

* **No propagation.** A rule whose only worth is that it enables a later rule — the
  scan — cannot be valued by episode averages. This is why 2 of 6 seeds never take
  off: they find the peek and the say and never the scan. The fix is one-step value
  inheritance, still counting, no gradients, but it changes the credit story from
  "the worked program takes the outcome" to "each step inherits from the next".
  **Needs a go — it is a change of model, not a knob.**
* **No chunking.** Naming a recurring worked program as a new action, with a hole
  where its copies disagree, is designed but not built.
* A task where one glance cannot win, and where the same sub-program is needed
  several times inside one input.


---

# Thinking (2026-09-15, second session)                             NEGATIVE

The idea: real successes are ~1 in 10,000, so let the machine re-run episodes **in
its head** on inputs it remembers, and learn from what works there. Imagining costs
no real experience, so it can practise 8 times per real attempt.

Built: working memory records every cell it reads; the last 64 inputs are kept as
`Remembered` (only the cells actually read -- it can only imagine where it has been);
`think()` re-runs the machine on those. Four supports were added when it did not work:

1. **It had nothing to think about.** 1.2 cells of 25 remembered, because the machine
   had collapsed to answering at step 1. Added look-around episodes (answering
   forbidden, higher temperature, charged against the same episode budget) -> 7.8 of 25.
2. **Imagination dreamt the same collapse** -- 64,000 imagined runs, all 1.6 steps.
   Made imagining bolder than acting, since nothing is at stake in the head.
3. **Selection was biased by headcount** -- splitting writes far more rules about
   answering than anything else, so "answer now" won on rule count. Changed selection
   to be over actions, one voice each. **This broke the reference** (see below).
4. **Imagined failures teach the wrong thing** -- a failure in the head may only mean
   it does not remember that cell. Imagined failures now teach nothing.

## Result

| | seeds | median | above chance |
|---|---|---|---|
| reference (restored) | 0.517 0.545 0.735 0.843 0.853 0.917 | **0.789** | 6/6 |
| reference, under change 3 | 0.300 0.320 0.357 0.358 | 0.338 | 0/4 |
| think8+look0.5, under change 3 | 0.062 0.080 0.085 0.135 | 0.083 | 0/4 |

**Change 3 was the damage, and it was mine, not the idea's.** Selecting over actions
destroyed a reference that works 6/6. The duplicate-rule copies that I had written off
as an accident are load-bearing: they are the only thing expressing *how strongly* a
rule is held, and flattening them to one-voice-per-action removed that. That is worth
keeping as a finding -- a rule needs a strength, and right now multiplicity is it.

Everything after change 3 was measured against a broken baseline and says nothing.
The fair re-measurement of thinking against the restored reference is the only number
that counts.

## What I did wrong

Five changes in a row with no clean result between them, against a baseline I had
silently broken on change 3. The rule from the last reset holds and I broke it: one
axis, fixed reference, re-measure the reference whenever anything shared changes.
