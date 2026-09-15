# The hypothesis machine — spec                                     2026-09-15

Second line in this folder. The first (`../programs/machine.py`) is the reference: it reached
6/6 seeds above chance on BELOW, median 0.789, against a hand-written ceiling of 1.000.
That machine stays untouched and stays the baseline.

What changes here: the machine gets a **belief it can hold and test**, and the eye
becomes relative-only. One axis per build step, reference re-run at each.

---

## 1. State

| | |
|---|---|
| cursor | `(x, y)`. No zoom. |
| slots | `S0..S3` — each empty, or holds `(symbol, x, y)`. |
| tags | `T0..T3` — each set or clear. Settable **and** clearable. |
| belief | empty, or one answer value. *(step 2)* |
| prediction | empty, or one pending symbol. *(step 3)* |
| read-set | every cell read this episode. **Not noticeable** — used only to decide whether a prediction was honest. |

## 2. What it notices

Facts true right now. A rule fires only if every fact in its condition is here.

* what is under the cursor — one per palette symbol, plus one for *outside the grid*
* per slot — empty / filled / holds symbol *s*
* per tag — set
* per pair of filled slots — same symbol / different symbol; and *a* is above / below /
  left of / right of *b*
* cursor is standing on the place held in slot *k*
* cannot step further in direction *d* (the four boundaries)
* belief is empty / belief is *c* *(step 2)*
* a prediction is pending *(step 3)*

Roughly 80 facts on BELOW with 4 slots and 4 tags. Conditions are up to 3 of them.

**Comparison is noticed, not done.** Relations between filled slots are free facts, not
actions. Reason: noticing is already free for everything else, and what matters is not
comparing but *branching on what the comparison says* — which only works if it is a fact.

## 3. What it can do

**Eye — relative only.** No absolute addressing anywhere.

* `STEP d` — one cell, eight directions. Further is reached by repeating, so distance
  costs time.
* `GOTO nearest s` — the closest cell holding symbol *s*. The one action that lets the
  world decide where to look.
* `RECALL sK` — stand on the place held in slot K. Going back to somewhere it has been.

**Memory**

* `WRITE sK` — put what is under the cursor, and where it is, into slot K
* `CLEAR sK`

**Its own state**

* `TAG j` / `UNTAG j`

**Answer**

* `SAY c` — a constant. Ends the episode.
* `SAY sK` — whatever slot K holds. Ends the episode.

**Belief** *(step 2)*

* `BELIEVE c` — write the belief. Cheap, revisable, overwritable.
* **Pays nothing by itself.** If a correct belief were rewarded it would be a free answer
  with no downside and the machine would farm it. A belief is only ever paid for through
  the predictions it licenses coming true.

**Predict** *(step 3)*

* `PREDICT s` — *the next cell I read holds s*. Resolves on the very next read.
* **Pays only if that cell has not been read this episode and is not held in a slot.**
  Predicting what it is already holding earns nothing. This is decidable from the
  read-set, not a judgement call — it separates predicting-to-test from reciting.
* Payoff scales with how much the prediction narrowed things down: being right about a
  rare symbol pays, being right about the common one barely does. Closes the
  predict-the-usual-thing leak.
* A prediction left unresolved at the end of the episode expires at a small cost, so it
  does not spray predictions it never intends to check.

Dropped from the old machine, deliberately: zoom, raster scan, `CENTER`, absolute jumps,
and `PEEK <direction>` (an eye movement disguised as a memory write).

## 4. Learning — unchanged

Counting. A rule keeps a running average of what it got and a tally per candidate test;
a test that separates its good firings from its bad ones becomes a new, narrower rule.
No gradients, no schedules. **Counting is not the weak link and is not being replaced.**

Credit is still the dataflow walk: every slot, tag and the cursor records which step wrote
it; when the machine answers, walk back through what that step read, and those steps are
the worked program. Steps outside the walk are charged their time. Steps after working
memory returns to a state it has already been in are charged as idle.

Prediction adds the thing this has never had: **a signal at a step, not only at the end.**

## 5. Score

`+1` right, `-1` wrong, `-1` if the step limit is reached without an answer, minus a small
charge per step, plus prediction payoffs.

## 6. Build order — one axis each, reference re-run every time

1. **New action set.** BELOW stays solvable (`GOTO nearest 1`, `STEP down`, `WRITE s0`,
   `SAY s0`), so rewrite the hand-written program in the new vocabulary first and confirm
   it still scores 1.000 — that is the ceiling. Then re-measure learning and compare
   against 6/6, median 0.789. **If it does not match, stop and find out why before
   anything else is added.**
2. **Belief.** Needs a task where holding a hypothesis pays, which BELOW does not:
   add *two markers are hidden; say whether they hold the same symbol*. Same sub-program
   twice in one input, which is also what makes a reusable piece worth naming later.
3. **Predict.**

## 7. Still not built, and known to be missing

* No naming of a recurring worked program as a new action, so no library.
* No arguments taken from memory — every argument is still baked into the action, so
  "punch a hole where copies disagree" remains unbuilt.
* No propagation: a step whose only worth is enabling a later step still cannot be valued
  by an episode average. Prediction is the first thing that touches this, by giving some
  steps an outcome of their own.
* `SAY` still ends the episode, so the never-halting verify-and-revise loop only exists
  once belief is in.
