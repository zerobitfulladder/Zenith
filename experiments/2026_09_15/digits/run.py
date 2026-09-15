"""Teaching the hypothesis machine to read digits, by a teacher that knows nothing
about digits, and by one that does.

    python run.py [demonstrations]
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hypothesis"))

from machine import MAX_COND, N_MAX, WM, Machine, true_tests
from task import Digits
from teachers import CentreTeacher, FixedTeacher, RandomTeacher, TreeTeacher

ROOT = HERE.parents[2]
CAP = 600          # a classifier needs more room than a five-rule program


def learn_from(student, moments):
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


def demo(student, task, plan, g):
    """run a planned action sequence, collecting (facts, action) moments"""
    wm, out = WM(task), []
    for i, a in enumerate(plan):
        if i >= student.max_steps:
            break
        out.append((set(true_tests(wm, task, g)), a))
        if student.apply(a, wm, g, i) is not None:
            break
    return out


def taught(task, teacher, n, seed, last_only=False):
    s = Machine(task, seed=seed, bootstrap=False, retire=False, copies=1,
                cap=CAP, max_steps=60, maxcond=6)
    rng = random.Random(seed + 1)
    for _ in range(n):
        i = rng.choice(task.train)
        g = task.take(i)
        m = demo(s, task, teacher.plan(g, task.answer, rng), g)
        learn_from(s, m[-1:] if last_only else m)
    return s


def check(student, task, soft=True):
    ok = tot = 0
    for i in task.test:
        g = task.take(i)
        said, k = student.run(g, learn=False, soft=soft)
        ok += (said == task.answer)
        tot += k
    return ok / len(task.test), tot / len(task.test)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    task = Digits(ROOT / "data")
    base = task.majority(task.test)
    tree = TreeTeacher(task)
    print(f"digits {task.answers}, 14x14 centred, four ink levels, one cell seen at a time")
    print(f"  most common class {base:.3f}   the tree teacher itself scores {tree.score:.3f}")
    print(f"  ({n} demonstrations, 2 seeds, tested alone on held-out digits)")

    for label, teacher, last in (("fixed probe: same nine places", FixedTeacher(task), False),
                                 ("stares at the centre only", CentreTeacher(task), False),
                                 ("random teacher, whole trace", RandomTeacher(task), False),
                                 ("random teacher, last step only", RandomTeacher(task), True),
                                 ("tree teacher, whole trace", tree, False)):
        out, t0 = [], time.time()
        for sd in range(2):
            out.append(check(taught(task, teacher, n, sd, last_only=last), task))
        a = [x for x, _ in out]
        print(f"  {label:<32} " + "  ".join(f"{x:.3f}" for x in a) +
              f"   mean {st.mean(a):.3f}   steps {st.mean([k for _, k in out]):.1f}"
              f"   {time.time()-t0:.0f}s", flush=True)
