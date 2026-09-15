"""The hypothesis machine, step 1: new action set only.

Everything about learning is identical to ../programs/machine.py -- counting, dataflow credit,
splitting, the answer-counterfactual, the baseline, the idle charges, copies as vote
weight, soft selection. The ONE axis that moves is what the machine can do and notice:

  eye        relative only. STEP (8 ways), GOTO nearest s, RECALL a remembered place.
             No absolute addressing, no zoom, no raster scan, no peeking sideways.
  memory     4 slots, write and clear. 4 tags, set and clear.
  noticing   relations between filled slots (same / different / above / below / left /
             right) are facts, not actions, and so is standing on a remembered place.

Belief and prediction are NOT in this step. See SPEC.md.
"""

import math
import random
from collections import defaultdict

# eight ways to step; the first four are also the boundary facts
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
DIRNAME = ["UP", "RIGHT", "DOWN", "LEFT", "UP-RIGHT", "DOWN-RIGHT", "DOWN-LEFT", "UP-LEFT"]

STEP_COST = 0.02
NOOP_COST = 0.05
BASE_RATE = 0.01
TEMP = 0.15
N_MAX = 300
MIN_SPLIT = 25
SPLIT_GAIN = 0.15
MAX_COND = 3
RULE_CAP = 160
COPY_CAP = 8
GAMMA = 0.95       # how much of where-it-lands carries back one step
N_SLOTS, N_TAGS = 4, 4


class WM:
    def __init__(self, task):
        self.x, self.y = task.W // 2, task.H // 2
        n = getattr(task, "slots", N_SLOTS)
        self.slots = [None] * n
        self.tags = [False] * N_TAGS
        self.wrote_slot = [None] * n
        self.wrote_tag = [None] * N_TAGS
        self.wrote_cur = None
        self.seen = set()          # cells read this episode (for the prediction step)

    def sym(self, task, g):
        self.seen.add((self.x, self.y))
        return task.sym(g, self.x, self.y)


def region(task, x, y):
    """which coarse patch of a centred image this is: thirds each way"""
    return min(y * 3 // task.H, 2) * 3 + min(x * 3 // task.W, 2)


def finger(wm):
    return (wm.x, wm.y, tuple(wm.slots), tuple(wm.tags))


def true_tests(wm, task, g):
    ts = [("cur", wm.sym(task, g))]
    for k, it in enumerate(wm.slots):
        if it is None:
            ts.append(("empty", k))
        else:
            ts.append(("full", k))
            ts.append(("slot", k, it[0]))
            if (it[1], it[2]) == (wm.x, wm.y):
                ts.append(("on", k))
    for a in range(len(wm.slots) if getattr(task, "relations", True) else 0):
        ia = wm.slots[a]                           # relations are noticed, not done
        if ia is None:
            continue
        for b in range(a + 1, len(wm.slots)):
            ib = wm.slots[b]
            if ib is None:
                continue
            ts.append(("same", a, b) if ia[0] == ib[0] else ("diff", a, b))
            if ia[2] < ib[2]:
                ts.append(("above", a, b))
            elif ia[2] > ib[2]:
                ts.append(("below", a, b))
            if ia[1] < ib[1]:
                ts.append(("lefto", a, b))
            elif ia[1] > ib[1]:
                ts.append(("righto", a, b))
    if getattr(task, "count_fact", False):
        ts.append(("done", sum(x is not None for x in wm.slots)))
    if getattr(task, "next_fact", False):          # self-correcting: a skipped slot comes back
        empties = [k for k, x in enumerate(wm.slots) if x is None]
        ts.append(("next", empties[0]) if empties else ("next", -1))
    R = getattr(task, "regions", 0)
    if R:
        for k, it in enumerate(wm.slots):
            if it is not None:
                ts.append(("at", k, region(task, it[1], it[2]), it[0]))
        ts.append(("here", region(task, wm.x, wm.y)))
    for j, t in enumerate(wm.tags):
        if t:
            ts.append(("tag", j))
    for d in range(4):
        dx, dy = DIRS[d]
        if not (0 <= wm.x + dx < task.W and 0 <= wm.y + dy < task.H):
            ts.append(("edge", d))
    return ts


def region_centre(task, r):
    ry, rx = divmod(r, 3)
    return (rx * 2 + 1) * task.W // 6, (ry * 2 + 1) * task.H // 6


def build_actions(task):
    a = [("step", d) for d in range(8)]
    if getattr(task, "regions", 0):
        a += [("region", r) for r in range(9)]      # one coarse jump; images are centred
    a += [("goto", s) for s in range(1, task.palette)]
    ns = getattr(task, "slots", N_SLOTS)
    a += [("recall", k) for k in range(ns)]
    a += [("write", k) for k in range(ns)]
    a += [("clear", k) for k in range(ns)]
    a += [("tag", j) for j in range(N_TAGS)]
    a += [("untag", j) for j in range(N_TAGS)]
    a += [("say", c) for c in task.answers]
    a += [("saysl", k) for k in range(ns)]
    return a


def act_name(a):
    k = a[0]
    return {"step": lambda: f"STEP {DIRNAME[a[1]]}",
            "goto": lambda: f"GOTO nearest {a[1]}",
            "region": lambda: f"LOOK region {a[1]}",
            "recall": lambda: f"RECALL s{a[1]}",
            "write": lambda: f"WRITE s{a[1]}",
            "clear": lambda: f"CLEAR s{a[1]}",
            "tag": lambda: f"TAG {a[1]}",
            "untag": lambda: f"UNTAG {a[1]}",
            "say": lambda: f"SAY {a[1]}",
            "saysl": lambda: f"SAY s{a[1]}"}[k]()


def test_name(t):
    k = t[0]
    return {"cur": lambda: f"under cursor={t[1]}",
            "slot": lambda: f"s{t[1]}={t[2]}",
            "full": lambda: f"s{t[1]} filled",
            "empty": lambda: f"s{t[1]} empty",
            "on": lambda: f"standing on s{t[1]}",
            "same": lambda: f"s{t[1]} same as s{t[2]}",
            "diff": lambda: f"s{t[1]} differs from s{t[2]}",
            "above": lambda: f"s{t[1]} above s{t[2]}",
            "below": lambda: f"s{t[1]} below s{t[2]}",
            "lefto": lambda: f"s{t[1]} left of s{t[2]}",
            "righto": lambda: f"s{t[1]} right of s{t[2]}",
            "tag": lambda: f"tag{t[1]}",
            "edge": lambda: f"edge {DIRNAME[t[1]]}",
            "at": lambda: f"s{t[1]} is {t[3]} in region {t[2]}",
            "here": lambda: f"cursor in region {t[1]}",
            "done": lambda: f"{t[1]} looks taken",
            "next": lambda: (f"next empty slot is s{t[1]}" if t[1] >= 0 else "all slots full")}[k]()


def flow(a):
    """what an action reads and writes, for the credit walk"""
    k = a[0]
    rc = wc = False
    rs, ws = set(), set()
    if k == "step":
        rc = wc = True
    elif k in ("goto", "region"):
        wc = True
    elif k == "recall":
        rs.add(a[1]); wc = True
    elif k in ("write",):
        rc = True; ws.add(a[1])
    elif k == "clear":
        ws.add(a[1])
    elif k == "saysl":
        rs.add(a[1])
    return rc, wc, rs, ws


class Rule:
    __slots__ = ("cond", "act", "v", "n", "tally", "id")
    _next = 0

    def __init__(self, cond, act, v=0.0):
        self.cond, self.act, self.v, self.n = frozenset(cond), act, v, 0
        self.tally = defaultdict(lambda: [0.0, 0])
        self.id = Rule._next
        Rule._next += 1

    def show(self):
        c = " & ".join(sorted(test_name(t) for t in self.cond)) or "always"
        return f"[{c}] -> {act_name(self.act)}   (v={self.v:+.3f} n={self.n})"


class Machine:
    def __init__(self, task, seed=0, max_steps=30, split=True, bootstrap=True, retire=True, copies=COPY_CAP, cap=RULE_CAP, maxcond=MAX_COND,
                 gain=SPLIT_GAIN, minsplit=MIN_SPLIT, kids=1):
        self.task, self.rng = task, random.Random(seed)
        self.max_steps = max_steps
        self.actions = build_actions(task)
        self.rules = [Rule([], a) for a in self.actions]
        self.split_on = split
        self.bootstrap = bootstrap
        self.retire = retire
        self.copies = copies
        self.cap = cap
        self.maxcond = maxcond
        self.gain, self.minsplit, self.kids = gain, minsplit, kids
        self.retired = self.stranded = 0
        self.base = 0.0
        self.hits = []
        self.in_graph = self.tot_steps = 0

    def run(self, g, learn=True, soft=False, trace_out=None, temp=None, report=False):
        task, wm = self.task, WM(self.task)
        steps, said, vals = [], None, []
        seen, waste = {finger(wm): -1}, set()
        for i in range(self.max_steps):
            ts = set(true_tests(wm, task, g))
            cands = [r for r in self.rules if r.cond <= ts]
            if not cands:
                cands = self.rules; self.stranded += 1
            mx = max(c.v for c in cands)
            w = [math.exp((c.v - mx) / TEMP) for c in cands]
            tw = sum(w)
            # what this state is worth: what the machine expects to get from here,
            # under the very policy it is about to follow
            vals.append(sum(wi * c.v for wi, c in zip(w, cands)) / tw)
            if temp is not None:
                w = [math.exp((c.v - mx) / temp) for c in cands]
            if learn or soft or temp is not None:
                r = self.rng.choices(cands, weights=w)[0]
            else:
                r = self.rng.choice([c for c in cands if c.v >= mx - 1e-9])
            rc, wc, rs, ws = flow(r.act)
            dep_cur = wm.wrote_cur if rc else None
            dep = [wm.wrote_slot[k] for k in rs]
            for t in r.cond:                       # a condition is a read too
                if t[0] in ("slot", "full", "empty"):
                    dep.append(wm.wrote_slot[t[1]])
                elif t[0] in ("same", "diff", "above", "below", "lefto", "righto"):
                    dep.append(wm.wrote_slot[t[1]]); dep.append(wm.wrote_slot[t[2]])
                elif t[0] == "on":
                    dep.append(wm.wrote_slot[t[1]]); dep_cur = wm.wrote_cur
                elif t[0] in ("cur", "edge"):
                    dep_cur = wm.wrote_cur
                elif t[0] == "tag":
                    dep.append(wm.wrote_tag[t[1]])
            f0 = finger(wm)
            said = self.apply(r.act, wm, g, i)
            f1 = finger(wm)
            if f1 in seen:
                waste.update(range(seen[f1] + 1, i + 1))
            else:
                seen[f1] = i
            steps.append((r, dep_cur, [d for d in dep if d is not None], ts, f1 == f0))
            if trace_out is not None:
                trace_out.append((i, r))
            if said is not None:
                break
        if report:
            return said, steps, waste
        if not learn:
            return said, len(steps)
        self.credit(steps, said, waste, vals)
        return said, len(steps)

    def apply(self, a, wm, g, step):
        task, k = self.task, a[0]
        if k == "step":
            dx, dy = DIRS[a[1]]
            wm.x = min(max(wm.x + dx, 0), task.W - 1)
            wm.y = min(max(wm.y + dy, 0), task.H - 1)
            wm.wrote_cur = step
        elif k == "region":
            wm.x, wm.y = region_centre(task, a[1])
            wm.wrote_cur = step
        elif k == "goto":
            p = task.nearest(g, a[1], wm.x, wm.y)
            if p is not None:
                wm.x, wm.y = p
            wm.wrote_cur = step
        elif k == "recall":
            it = wm.slots[a[1]]
            if it is not None:
                wm.x, wm.y = it[1], it[2]
            wm.wrote_cur = step
        elif k == "write":
            wm.slots[a[1]] = (wm.sym(task, g), wm.x, wm.y)
            wm.wrote_slot[a[1]] = step
        elif k == "clear":
            wm.slots[a[1]] = None
            wm.wrote_slot[a[1]] = step
        elif k == "tag":
            wm.tags[a[1]] = True
            wm.wrote_tag[a[1]] = step
        elif k == "untag":
            wm.tags[a[1]] = False
            wm.wrote_tag[a[1]] = step
        elif k == "say":
            return a[1]
        elif k == "saysl":
            it = wm.slots[a[1]]
            return -1 if it is None else it[0]
        return None

    @staticmethod
    def worked(steps, said):
        """the steps the answer actually depended on -- the rest was passing time"""
        n = len(steps)
        if said is None:
            return set(range(n))
        inside, stack, seen = set(), [n - 1], set()
        while stack:
            i = stack.pop()
            if i is None or i in seen:
                continue
            seen.add(i); inside.add(i)
            _, dc, ds, _, _ = steps[i]
            for d in ([dc] if dc is not None else []) + ds:
                if d not in seen:
                    stack.append(d)
        return inside

    def credit(self, steps, said, waste=frozenset(), vals=None):
        n = len(steps)
        if said is None:
            inside = set(range(n))
        else:
            inside, stack, seen = set(), [n - 1], set()
            while stack:
                i = stack.pop()
                if i is None or i in seen:
                    continue
                seen.add(i); inside.add(i)
                _, dc, ds, _, _ = steps[i]
                for d in ([dc] if dc is not None else []) + ds:
                    if d not in seen:
                        stack.append(d)
        self.in_graph += len(inside)
        self.tot_steps += n
        right = (said is not None and said == self.task.answer)
        self.hits.append(right)
        reward = 1.0 if right else -1.0
        adv = reward - self.base
        self.base += BASE_RATE * (reward - self.base)
        for i, (r, _, _, ts, noop) in enumerate(steps):
            noop = noop or i in waste
            if self.bootstrap:
                # a step is worth what it costs plus what it leads to; only the last
                # link is judged by the world, and that is what keeps the rest honest
                after = (reward if i == n - 1 else vals[i + 1])
                target = -STEP_COST + GAMMA * after
            else:
                target = (adv if i in inside and not noop else 0.0) - STEP_COST
            if noop:
                target -= NOOP_COST
            r.n += 1
            r.v += (target - r.v) / min(r.n, N_MAX)
            if self.split_on and len(r.cond) < self.maxcond:
                for t in ts:
                    if t not in r.cond:
                        q = r.tally[t]; q[0] += target; q[1] += 1
        self.counterfactual(steps)
        if self.split_on:
            self.grow()

    def counterfactual(self, steps):
        """Saying is the one action whose outcome is known without doing it."""
        ans = self.task.answer
        for j in self.rng.sample(range(len(steps)), min(2, len(steps))):
            ts = steps[j][3]
            held = {t[1]: t[2] for t in ts if t[0] == "slot"}
            for r in self.rules:
                a = r.act
                if a[0] == "say":
                    out = a[1]
                elif a[0] == "saysl":
                    out = held.get(a[1], -1)
                else:
                    continue
                if not r.cond <= ts:
                    continue
                hit = 1.0 if out == ans else -1.0
                target = (-STEP_COST + GAMMA * hit) if self.bootstrap \
                    else (hit - self.base - STEP_COST)
                r.n += 1
                r.v += (target - r.v) / min(r.n, N_MAX)
                if self.split_on and len(r.cond) < MAX_COND:
                    for t in ts:
                        if t not in r.cond:
                            q = r.tally[t]; q[0] += target; q[1] += 1

    def grow(self):
        for r in list(self.rules):
            if len(r.cond) >= self.maxcond or r.n < self.minsplit:
                continue
            good = sorted(((s / c, t) for t, (s, c) in r.tally.items()
                           if c >= self.minsplit and s / c - r.v >= self.gain), reverse=True)
            for m, bt in good[: self.kids]:          # the best few tests, not only the best
                cond = frozenset(set(r.cond) | {bt})
                if sum(1 for q in self.rules if q.cond == cond and q.act == r.act) < self.copies:
                    self.rules.append(Rule(cond, r.act, m))
            if good:
                r.tally.clear()
        if self.retire:
            # an unconditional rule may go once a narrower rule for the same action
            # has beaten it -- the action stays reachable, it just stops firing blindly
            for r in [q for q in self.rules if not q.cond and q.n >= MIN_SPLIT]:
                if any(c.act == r.act and c.cond and c.n >= MIN_SPLIT
                       and c.v >= r.v + SPLIT_GAIN for c in self.rules):
                    self.rules.remove(r)
                    self.retired += 1
        wide = [r for r in self.rules if not r.cond]
        narrow = sorted((r for r in self.rules if r.cond), key=lambda r: -r.v)
        self.rules = wide + narrow[: max(0, self.cap - len(wide))]

    def top(self, k=10):
        rs = [r for r in self.rules if r.n >= 20]
        rs.sort(key=lambda r: -r.v)
        return rs[:k]
