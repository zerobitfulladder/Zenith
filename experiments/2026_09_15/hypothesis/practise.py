"""After being taught: the student works on the training grids by itself.

One learning rule throughout -- contrast on a trace: at each moment of the trace, the
action taken there is right and every other action is wrong. Two sources of traces:

    it got the answer    learn from its own run, but only the steps the answer actually
                         depended on (the dependency walk), so lucky detours teach nothing
    it got it wrong      ask the teacher, and learn from the teacher's run on that grid

No schedule anywhere. The teacher appears exactly when the student fails, so it fades by
itself as the student improves. Choosing is made fuzzier during practice so alternatives
get tried at all.

    python practise.py [demos] [practices]
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from machine import MAX_COND, N_MAX, WM, Machine, true_tests
from run import evaluate, reference
from tasks import Grid

PRACTICE_TEMP = 0.40      # fuzzier than the 0.15 it is read out with


class Teacher:
    def __init__(self, task):
        self.rules = reference(task).rules

    def trace(self, task, g, student):
        """what the teacher does on this grid, as (facts, action) moments"""
        wm, out = WM(task), []
        for i in range(student.max_steps):
            ts = set(true_tests(wm, task, g))
            m = [r for r in self.rules if r.cond <= ts]
            if not m:
                break
            a = max(m, key=lambda r: r.v).act
            out.append((ts, a))
            if student.apply(a, wm, g, i) is not None:
                break
        return out


def learn_from(student, moments):
    """contrast: here, this was done and the other 37 were not"""
    for ts, a in moments:
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
    student.grow()


def taught(n, seed, task, copies=1):
    s = Machine(task, seed=seed, bootstrap=False, retire=False, copies=copies)
    t = Teacher(task)
    rng = random.Random(seed + 1)
    for _ in range(n):
        learn_from(s, t.trace(task, task.sample(rng), s))
    return s, t


def practise(student, teacher, task, n, rng, ask=True, mimic=True, cap=None):
    got, asked = 0, 0
    for _ in range(n):
        g = task.sample(rng)
        said, steps, waste = student.run(g, learn=False, temp=PRACTICE_TEMP, report=True)
        if said == task.answer:
            got += 1
            if not mimic:
                continue                      # practice only to find what it gets wrong
            inside = Machine.worked(steps, said)
            keep = [(ts, r.act) for i, (r, _, _, ts, noop) in enumerate(steps)
                    if i in inside and not noop and i not in waste]
            if cap is not None and len(keep) > cap:
                continue                      # it got there by wandering: not worth copying
            learn_from(student, keep)
        elif ask:
            asked += 1
            learn_from(student, teacher.trace(task, g, student))
    return got / n, asked / n


if __name__ == "__main__":
    demos = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    prac = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    task = Grid()
    print(f"{demos} demonstrations, then {prac} practices. Tested alone, unseen grids.")

    for label, arm in (("taught only", None),
                       ("then practises alone", False),
                       ("then practises, asks when it fails", True)):
        out, t0, info = [], time.time(), []
        for sd in range(4):
            s, t = taught(demos, sd, task)
            if arm is not None:
                info.append(practise(s, t, task, prac, random.Random(sd + 99), ask=arm))
            out.append(evaluate(s, task, soft=True))
        a = [x for x, _ in out]
        extra = ""
        if info:
            extra = (f"   while practising: right {st.mean([g for g, _ in info]):.2f}, "
                     f"asked {st.mean([q for _, q in info]):.2f}")
        print(f"  {label:<36} " + "  ".join(f"{x:.3f}" for x in sorted(a)) +
              f"   median {st.median(a):.3f}   steps {st.mean([k for _, k in out]):.1f}"
              f"   {time.time()-t0:.0f}s{extra}", flush=True)
    print("  ceiling 1.000   by search 0.443   chance 0.333")
