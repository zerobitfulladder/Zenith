"""Search for programs in the tiny language.

    python run.py grid [episodes]      learn on BELOW
    python run.py ref                  score the hand-written program (the ceiling)
    python run.py mnist [episodes]     learn on 3-digit MNIST, one cell at a time
    python run.py all

Everything is scored frozen: learning off, no exploration, unseen inputs.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import machine as M
from machine import Machine, Rule, STEP_COST
from tasks import Grid, Mnist

ROOT = Path(__file__).parents[3]


def report(tag, acc, steps, mach, base, extra=""):
    conds = [r for r in mach.rules if r.cond] if mach else []
    score = acc - STEP_COST * steps
    print(f"  {tag:<26} acc {acc:.3f}  (chance {base:.3f})   steps {steps:5.1f}   "
          f"score {score:+.3f}   rules {len(mach.rules) if mach else 0:4d} "
          f"({len(conds)} with conditions) {extra}")


# ------------------------------------------------------------------------- BELOW
def eval_grid(mach, task, n=600, seed=999, soft=False):
    rng = random.Random(seed)
    ok = tot = 0
    for _ in range(n):
        st = task.sample(rng)
        said, k = mach.run(st, learn=False, soft=soft)
        ok += (said == task.answer)
        tot += k
    return ok / n, tot / n


def reference(task):
    """scan; if the marker is under the cursor attend one cell down; then say it."""
    m = Machine(task, seed=0, split=False)
    m.rules = [Rule([], ("next",), 0.1),
               Rule([("cur", task.marker)], ("peek", 0, 2), 0.5),
               Rule([("full", 0)], ("saysl", 0), 0.9)]
    return m


def grid(episodes=12000, jump=True, seed=0, quiet=False, hindsight=True, think=0,
         look=0.0):
    task = Grid()
    m = Machine(task, n_slots=2, n_tags=2, content_jump=jump, seed=seed, max_steps=30,
                hindsight=hindsight)
    rng = random.Random(seed + 1)
    t0 = time.time()
    for _ in range(episodes):
        if look and rng.random() < look:
            m.run(task.sample(rng), learn=False, record=True, noanswer=True)
        else:
            m.run(task.sample(rng), record=think > 0)
        if think:
            m.think(think)
    curve = [sum(m.hits[i:i + episodes // 6]) / max(1, len(m.hits[i:i + episodes // 6]))
             for i in range(0, episodes, max(1, episodes // 6))]
    print("    while learning: " + "  ".join(f"{c:.3f}" for c in curve))
    acc, steps = eval_grid(m, task)
    sacc, ssteps = eval_grid(m, task, soft=True)
    tag = ("learned" if hindsight else "learned, no hindsight") + ("" if jump else " (no jump)")
    tag += (f" +think{think}+look{look:g}" if think else " (no thinking)")
    report(tag, acc, steps, m, 1 / 3, f" {time.time()-t0:.0f}s "
           f"in-graph {m.in_graph/max(m.tot_steps,1):.0%}  |  same rules, soft picks: "
           f"acc {sacc:.3f} steps {ssteps:.1f}")
    if not quiet:
        print("    rules it kept:")
        for r in m.top(8):
            print("      ", r.show())
    return m, acc


# ------------------------------------------------------------------------- MNIST
def mnist(episodes=6000, seed=0):
    task = Mnist(root=ROOT / "data")
    tr, te = task.split(3000)
    m = Machine(task, n_slots=2, n_tags=2, content_jump=True, seed=seed, max_steps=30)
    rng = random.Random(seed + 1)
    t0 = time.time()
    for _ in range(episodes):
        m.run(task.take(rng.choice(tr)))
    def ev(soft):
        ok = tot = 0
        for i in te:
            said, k = m.run(task.take(i), learn=False, soft=soft)
            ok += (said == task.answer)
            tot += k
        return ok / len(te), tot / len(te)
    acc, steps = ev(False)
    sacc, ssteps = ev(True)
    base = max(sum(1 for i in te if task.Y[i] == d) for d in task.answers) / len(te)
    curve = [sum(m.hits[i:i + episodes // 6]) / max(1, len(m.hits[i:i + episodes // 6]))
             for i in range(0, episodes, max(1, episodes // 6))]
    print("    while learning: " + "  ".join(f"{c:.3f}" for c in curve))
    report("learned", acc, steps, m, base,
           f" {time.time()-t0:.0f}s in-graph {m.in_graph/max(m.tot_steps,1):.0%}"
           f"  |  same rules, soft picks: acc {sacc:.3f} steps {ssteps:.1f}")
    print("    rules it kept:")
    for r in m.top(8):
        print("      ", r.show())
    return m


def trace(mach, task, st):
    out = []
    said, _ = mach.run(st, learn=False, trace_out=out)
    print(f"    trace (answer was {task.answer}, it said {said}):")
    for i, r, ts in out[:14]:
        print(f"      {i:2d}  {r.show()}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if what in ("ref", "all", "grid"):
        t = Grid()
        m = reference(t)
        acc, steps = eval_grid(m, t)
        print("BELOW  5x5 grid, answer = the symbol below the marker")
        report("hand-written program", acc, steps, m, 1 / 3, " <- can the language say it?")
    if what in ("grid", "all"):
        grid(n or 12000, jump=True, hindsight=False, quiet=True)
        grid(n or 12000, jump=True)
        grid(n or 12000, jump=False)
        t = Grid()
        mm, _ = grid(n or 12000, jump=True, seed=1, quiet=True)
        trace(mm, t, t.sample(random.Random(7)))
    if what in ("mnist", "all"):
        print("\nMNIST  14x14, four ink levels, digits 0/1/7, one cell seen at a time")
        mnist(n or 6000)
