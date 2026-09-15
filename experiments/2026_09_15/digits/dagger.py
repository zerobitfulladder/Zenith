"""The teacher labels the student's own states.

Demonstrations only ever cover the teacher's states. One slip puts the student somewhere
no demonstration reached, and from there its rules are unconstrained; errors cascade.
Fix (Ross, Gordon, Bagnell 2011): let the student run, and at every state it actually
visits, learn what the teacher would have done there. Same contrast rule as before.

    python dagger.py [demos] [rounds]
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "hypothesis")); sys.path.insert(0, str(HERE))
import run as R
from machine import WM, region_centre
from task import Digits
from teachers import FixedTeacher

PRACTICE_TEMP = 0.25


def relabel(student, teacher, task, n, rng):
    """student acts, teacher says what it should have done at each state"""
    got = 0
    for _ in range(n):
        g = task.take(rng.choice(task.train))
        said, steps, _ = student.run(g, learn=False, temp=PRACTICE_TEMP, report=True)
        got += (said == task.answer)
        R.learn_from(student, [(ts, teacher.act(ts, task.answer)) for (_, _, _, ts, _) in steps])
    return got / n


def slots_right(student, task, n=200):
    right = tot = 0
    for i in task.test[:n]:
        g = task.take(i)
        wm = WM(task)
        import math
        for step in range(student.max_steps):
            from machine import true_tests
            ts = set(true_tests(wm, task, g))
            c = [r for r in student.rules if r.cond <= ts]
            mx = max(x.v for x in c)
            r = student.rng.choices(c, weights=[math.exp((x.v - mx) / 0.15) for x in c])[0]
            if student.apply(r.act, wm, g, step) is not None:
                break
        for k, it in enumerate(wm.slots):
            tot += 1
            right += (it is not None and (it[1], it[2]) == region_centre(task, k))
    return right / tot


if __name__ == "__main__":
    demos = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    task = Digits(R.ROOT / "data")
    t = FixedTeacher(task)
    print(f"fixed nine-place probe, 'next empty slot' fact, {demos} demos" +
          f" then {rounds} rounds of the teacher labelling the student's own states")
    print(f"  before this change:  acc 0.341   slot k holds region k 0.22")
    for label, do_dagger in (("demonstrations only", False), ("+ teacher labels student's states", True)):
        accs, slots, t0 = [], [], time.time()
        for sd in range(2):
            s = R.taught(task, t, demos, sd)
            rng = random.Random(sd + 7)
            if do_dagger:
                relabel(s, t, task, rounds, rng)
            accs.append(R.check(s, task)[0]); slots.append(slots_right(s, task))
        print(f"  {label:<36} acc " + "  ".join(f"{a:.3f}" for a in accs) +
              f"   mean {st.mean(accs):.3f}   slot k holds region k {st.mean(slots):.2f}"
              f"   {time.time()-t0:.0f}s", flush=True)
    print("  ceiling from the nine cells 0.854   staring at the centre 0.690   chance 0.345")
