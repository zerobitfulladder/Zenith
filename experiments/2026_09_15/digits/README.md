# Digits — teaching the hypothesis machine to read                2026-09-15

Uses `../hypothesis/machine.py` unchanged in kind; the additions are opt-in so BELOW
keeps the exact fact alphabet its 0.997 was measured with (`Grid` sets none of them):

* `regions` — a fact for which of nine coarse patches a sample came from, combined with
  what was seen, so `[s0 is full ink in the top band]` is one fact, not two
* a `LOOK region r` action — coarse absolute jump, legitimate because images are centred
* `count_fact` — `k looks taken`, so "where am I in the routine" is one fact not k
* `slots`, `relations`, `maxcond`, `cap` as knobs

## What the task can give

| | |
|---|---|
| all 196 ink cells, decision tree depth 6 | 0.980 |
| nine coarse cells, 4 ink levels | **0.854** |
| nine coarse cells, 10 k-means 4x4 patch templates, depth 6 | 0.929 |
| nine coarse cells, patch templates, depth 3 | 0.817 |
| most common class | 0.345 |

Patches raise the ceiling, but only with deep conditions: a one-hot template identity
says nothing until several are conjoined, while an ink level is ordinal and one test
already splits usefully. **Vocabulary and condition depth are one change, not two.**

## What the machine got

| | |
|---|---|
| stares at one cell (the centre) | **0.690** |
| fixed nine-place probe | 0.341 |
| random teacher (jump anywhere, write, then answer) | 0.389 |
| tree teacher (a real look-and-branch program) | 0.207 |
| chance | 0.345 |

**Every arm that needs a routine loses to the arm that needs none.**

## Why — execution reliability, and it is exact

The fixed probe learns *correct* answering rules:

    [s4 is 0 in region 4 & s5=3 & s8 is 0 in region 8]                -> SAY 0   v=+0.979 n=946
    [s1 is 1 in region 1 & s4 is 3 in region 4 & s8 is 0 in region 8] -> SAY 1   v=+0.978 n=269

341 answering rules, well supported, sensible. And they never fire, because they assume
slot k holds region k's sample, and on the student's own runs:

    slots filled at the end             0.61
    slot k actually holding region k    0.22

Per-step reliability ~0.92 over the probe's 18 decisions: **0.92^18 = 0.22**. Exactly the
measured figure. A program's reliability is per-step reliability raised to its length, so
the shortest routine wins every time.

Per-step reliability is 0.92 and not ~1 because of selection: every action is always
available, hundreds of rules match at once, and the pick is a weighted lottery. A
well-learned rule does not win, it out-votes -- and loses 8% of the time. Reading out
greedily instead is worse (0.256, loops to 45 steps).

## Two causes found and fixed along the way, both "the condition was unsayable"

* sequencing past the third look needed `s0 filled & s1 filled & s2 filled & s3 empty`
  = 4 facts against a cap of 3. The `k looks taken` fact fixed it: the probe went from
  14.5 to 19.0 of 19 steps.
* on BELOW the same class of problem blocked `WRITE s0` (see `../hypothesis/README.md`).

## Execution fixed: the teacher labels the student's own states     (dagger.py)

The 0.92-per-step story was wrong. Greedy readout was *worse* and looped, so the
best-valued rule is genuinely wrong in some states -- values, not lottery noise. And 22%
right against 61% filled means one slip poisons everything after it: errors cascade, they
are not independent.

Cause: values are estimated only on the teacher's states. One slip and the student is
somewhere no demonstration reached, where its rules carry values learned elsewhere. The
`k looks taken` count made it unrecoverable -- after a slip the count lies. (Ross, Gordon,
Bagnell 2011 is the diagnosis and the fix.)

| fixed nine-place probe, 2000 demos | acc | slot k holds region k |
|---|---|---|
| count fact | 0.341 | 0.22 |
| `next empty slot` fact (self-correcting) | 0.422 | 0.22 |
| **+ 2000 rounds of the teacher labelling the student's own states** | **0.704** | **1.00** |

The fact alone did nothing. Labelling the student's states took slot correctness from
0.22 to 1.00 -- the routine now runs flawlessly. What remains, with perfect features, is
the classifier: 0.704 against a 0.854 tree on the same nine cells.

## With execution perfect, what decides the answer?

|  | lottery | best rule wins | answering rules matching at the end | best of them right |
|---|---|---|---|---|
| cap 600 | 0.709 | 0.685 | 16 | 0.663 |
| cap 1500 | 0.683 | 0.702 | 81 | 0.693 |

Not the lottery (greedy is the same), not the budget (cap does nothing). The single
best-valued matching rule is right only ~0.67 of the time, though its training value says
~0.98. Suspect: selecting by max value over 16-81 matching rules picks the most overfit
narrow conjunction. Tested next by reading the answer off the matching rules several ways,
on train and test.

## Not overfitting, not readout: the rule set carries 0.70

Same trained machine, final state built by the teacher's plan, answer read off the
matching answering rules several ways:

| readout | train | test |
|---|---|---|
| max value | 0.690 | 0.718 |
| max value, n >= 100 | 0.690 | 0.718 |
| lower bound v - 2/sqrt(n) | 0.690 | 0.718 |
| vote, sum n*v per class | 0.650 | 0.655 |
| vote, count per class | 0.677 | 0.713 |

Train = test, so the rules are not overfit; no readout beats ~0.70. With execution at
1.00, budget irrelevant and readout irrelevant, **the split mechanism is the bottleneck**:
greedy one-fact-at-a-time growth gives a rule set worth 0.70 where a tree on the same
nine cells gets 0.854. Next tested: the split thresholds (gain 0.15 / min 25 / one child
per pass were set for a five-rule program).

## What a counted answer step would get from the same nine slots

| nine cells | counted per-cell tables (naive Bayes) | full nine-symbol table, backoff | tree depth 6 |
|---|---|---|---|
| ink levels (4) | **0.852** | 0.851 (95% of test keys seen) | 0.854 |
| patch templates (10) | **0.954** | 0.954 (69% seen) | 0.917 |
| greedy splitting, ink, execution perfect | 0.70 | | |

Counting per cell reaches the whole ceiling on ink and beats the tree on patches. The
program already gathers the nine symbols perfectly; only the answering is weak. This is
the project's standing result -- counting beats depth, per-cell readout -- showing up
again at the selector of a program.

## The patch vocabulary, measured                                   (patchtask.py)

4x4 patches of the centred image at stride 2, k-means k=10, cell = nearest template.
Fixed nine-place probe, 2000 demos + 2000 rounds of teacher-labels-student, cap 1500,
conditions to 6, execution perfect in both:

| | acc (2 seeds) | mean | counted ceiling |
|---|---|---|---|
| ink levels (4) | 0.711 0.691 | 0.701 | 0.852 |
| patch templates (10) | 0.716 0.741 | **0.728** | **0.954** |

The vocabulary is worth +0.10 at the ceiling and delivers +0.03 through greedy splitting.
The answer step masks it, as it masks everything else.

## Calibration: the same ceilings on ALL TEN digits

Everything above is 0/1/7 (chance 0.345). Counted per-cell tables, 8000/2000, centred:

| | 3x3 looks | 5x5 looks | 7x7 looks | every cell |
|---|---|---|---|---|
| ink levels (4) | 0.392 | 0.635 | 0.753 | 0.825 |
| patches k=10 | 0.708 | 0.825 | 0.874 | 0.891 |
| patches k=32 | 0.757 | 0.857 | 0.899 | 0.914 |

Tree depth 12 on every ink cell: 0.805. The project's own whole-image results on full
MNIST: two-rung k-means 0.949, per-cell readout 0.987. So nine coarse looks on ten
digits is 0.39-0.76 depending on vocabulary; a glimpse machine needs ~49 looks and a
richer vocabulary to approach 0.90, and even then sits under the whole-image readouts.
**The 0.954 on 0/1/7 is not "0.95 on MNIST."**

## Next

**Specific suppresses general** -- when a matching rule with a narrower condition exists,
the unconditional ones stay silent, instead of merely being outvoted. Aimed straight at
per-step reliability, which the arithmetic above says is the binding constraint. Nothing
else -- longer probes, tree demonstrations, the patch vocabulary -- can pay off while an
18-step routine survives 22% of the time.

Note: retiring unconditional rules was tried on BELOW and did not help, because it
required a narrower rule to beat them first. Suppression is the untried one.
