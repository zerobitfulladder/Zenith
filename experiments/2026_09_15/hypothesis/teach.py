"""Learning by being shown, instead of by stumbling.

A demonstration is a trace, not a program: it says what was done, never why. So the
teacher supplies the half this machine is bad at (which actions, in which order) and
the student supplies the half it is good at (under which conditions) -- the conditions
are never shown and have to be worked out.

The target is contrast, not approval. At every demonstrated step the action the teacher
took is right and all 37 others are wrong. Without that there is nothing to split on:
if everything the student ever sees is correct, no fact ever separates anything.

    python teach.py [demonstrations]
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from machine import (MAX_COND, N_MAX, WM, Machine, Rule, true_tests)
from run import evaluate, reference
from tasks import Grid


class Teacher:
    """The hand-written program. It need not be clever -- it only has to be right."""

    def __init__(self, task):
        self.rules = reference(task).rules

    def act(self, ts):
        m = [r for r in self.rules if r.cond <= ts]
        return max(m, key=lambda r: r.v).act if m else None


def show(student, teacher, g):
    """one demonstration: the student watches, and is asked what it would have done"""
    wm, task = WM(student.task), student.task
    for i in range(student.max_steps):
        ts = set(true_tests(wm, task, g))
        a = teacher.act(ts)
        if a is None:
            break
        for r in student.rules:
            if not r.cond <= ts:
                continue
            target = 1.0 if r.act == a else -1.0
            r.n += 1
            r.v += (target - r.v) / min(r.n, N_MAX)
            if student.split_on and len(r.cond) < MAX_COND:
                for t in ts:
                    if t not in r.cond:
                        q = r.tally[t]; q[0] += target; q[1] += 1
        if student.apply(a, wm, g, i) is not None:
            break
    student.grow()


def taught(n, seed, task, copies=1):
    s = Machine(task, seed=seed, bootstrap=False, retire=False, copies=copies)
    t = Teacher(task)
    rng = random.Random(seed + 1)
    for _ in range(n):
        show(s, t, task.sample(rng))
    return s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    task = Grid()
    print(f"taught by demonstration, {n} demonstrations, tested alone on unseen grids")
    for label, soft in (("chooses the best rule", False), ("chooses by value, softly", True)):
        out, t0 = [], time.time()
        for sd in range(4):
            s = taught(n, sd, task)
            out.append(evaluate(s, task, soft=soft))
        a = [x for x, _ in out]
        print(f"  {label:<26} " + "  ".join(f"{x:.3f}" for x in sorted(a)) +
              f"   median {st.median(a):.3f}   steps {st.mean([k for _, k in out]):.1f}"
              f"   {time.time()-t0:.0f}s")
    print("  for comparison: hand-written 1.000, learned by search median 0.443, chance 0.333")
    s = taught(n, 0, task)
    print("    what it worked out (its own conditions, never shown):")
    for r in s.top(8):
        print("      ", r.show())
