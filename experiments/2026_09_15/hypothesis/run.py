"""Step 1 of SPEC.md: the new action set alone.

    python run.py ref      the hand-written program in the new vocabulary (the ceiling)
    python run.py learn    6 seeds, against the reference line's 6/6 median 0.789
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from machine import Machine, Rule, STEP_COST, build_actions, true_tests, WM
from tasks import Grid


def evaluate(mach, task, n=600, seed=999, soft=True):
    rng = random.Random(seed)
    ok = tot = 0
    for _ in range(n):
        g = task.sample(rng)
        said, k = mach.run(g, learn=False, soft=soft)
        ok += (said == task.answer)
        tot += k
    return ok / n, tot / n


def reference(task):
    """find the marker, remember it, step down, take what is there, say it"""
    m = Machine(task, seed=0, split=False)
    m.rules = [
        Rule([("empty", 1)], ("goto", 1), 0.1),                      # go find the marker
        Rule([("cur", 1), ("empty", 1)], ("write", 1), 0.4),         # remember it and where
        Rule([("slot", 1, 1), ("on", 1)], ("step", 2), 0.7),         # while standing on it, step down
        Rule([("slot", 1, 1), ("empty", 0)], ("write", 0), 0.5),     # take what is here
        Rule([("full", 0)], ("saysl", 0), 0.9),                      # say it
    ]
    return m


def vocab(task):
    acts = build_actions(task)
    rng = random.Random(0)
    m = Machine(task, seed=0)
    alpha, sizes = set(), []
    for _ in range(300):
        g = task.sample(rng)
        wm = WM(task)
        for i in range(10):
            ts = true_tests(wm, task, g)
            alpha |= set(ts); sizes.append(len(ts))
            m.apply(rng.choice(acts), wm, g, i)
    return len(acts), len(alpha), sum(sizes) / len(sizes)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    task = Grid()
    na, nf, act = vocab(task)
    print(f"vocabulary: {na} actions, {nf} possible facts, {act:.1f} true at a time")

    if what in ("ref", "all"):
        m = reference(task)
        acc, steps = evaluate(m, task, soft=False)
        print(f"  hand-written, new vocabulary   acc {acc:.3f}   steps {steps:.1f}   "
              f"score {acc - STEP_COST*steps:+.3f}   rules {len(m.rules)}")
        tr = []
        m.run(task.sample(random.Random(3)), learn=False, trace_out=tr)
        for i, r in tr:
            print(f"      {i}  {r.show().split('   (v=')[0]}")

    if what in ("learn", "all"):
        out, t0 = [], time.time()
        for sd in range(6):
            mm = Machine(task, seed=sd)
            rng = random.Random(sd + 1)
            for _ in range(n):
                mm.run(task.sample(rng))
            out.append(evaluate(mm, task))
        a = [x for x, _ in out]
        print(f"  learned, 6 seeds ({n//1000}k)      " + "  ".join(f"{x:.3f}" for x in sorted(a)) +
              f"   median {st.median(a):.3f}   above chance {sum(x>0.45 for x in a)}/6"
              f"   steps {st.mean([k for _, k in out]):.1f}   {time.time()-t0:.0f}s")
        print("     reference line was:          0.517  0.545  0.735  0.843  0.853  0.917"
              "   median 0.789   6/6")
        best = max(range(6), key=lambda i: out[i][0])
        mm = Machine(task, seed=best); rng = random.Random(best + 1)
        for _ in range(n):
            mm.run(task.sample(rng))
        print("    rules the best seed kept:")
        for r in mm.top(6):
            print("      ", r.show())
