# 2026-09-15 — inventing programs

## `programs/` — a machine that writes its own programs

One axis changes versus everything before: what the learner produces is a
program, not a code. An instruction is `(condition, action)`; there is no if,
jump, loop or halt, so every program that can be written runs. Credit follows
dataflow, not time, and learning is counting only. On BELOW (5x5 grid, the answer
is the symbol under the marker) the hand-written program scores 1.000; 6 of 6
learned seeds get above chance, median 0.789, and the best seed beats the
hand-written program's score by replacing the 14-step scan with one
`GOTO nearest 1`. On three MNIST digits seen one cell at a time it learns a
one-glance heuristic, not a program (0.663 greedy / 0.712 soft). Six ways it broke
along the way are each a finding. A second session tried "thinking" (re-running
episodes in the head on remembered inputs): negative, and the damage came from a
selection change that broke the reference, not from the idea — duplicate-rule
copies turned out to be load-bearing as a rule's strength.

Full writeup: [`programs/README.md`](programs/README.md).

---

## `hypothesis/` — new action set, then teaching, then practise and ask

Step 1 moved one axis — what the machine can do and notice (relative steps,
`GOTO`, recall, relations as facts). The hand-written program in the new
vocabulary scores 1.000 at +0.901, but learning by search reached a median of
only 0.443 against the reference's 0.789, so the gate failed and step 1 stopped.
Then teaching instead of searching: the teacher shows which actions in which
order, never the conditions, and the student works out the conditions itself —
median 0.863. Then practising and asking the teacher only on failure: **0.997**
(best 1.000). Feeding the student's own successful runs back as demonstrations is
harmful (0.997 falls), and no filter fixed it.

Full writeup: [`hypothesis/README.md`](hypothesis/README.md) (spec in
[`hypothesis/SPEC.md`](hypothesis/SPEC.md)).

---

## `digits/` — teaching the hypothesis machine to read

The taught machine on digits 0/1/7 through nine coarse looks. Every arm that
needs a routine lost to staring at the centre cell (0.690), because an 18-step
routine ran correctly too rarely. Having the teacher label the student's own
states (`dagger.py`) made execution perfect (slot correctness 0.22 -> 1.00), and
what remains is the classifier: 0.704 against a 0.854 tree on the same nine
cells. Calibrated on all ten digits, nine coarse looks give 0.39-0.76 depending
on vocabulary, so the 0.954 on 0/1/7 is not "0.95 on MNIST."

Full writeup: [`digits/README.md`](digits/README.md).
