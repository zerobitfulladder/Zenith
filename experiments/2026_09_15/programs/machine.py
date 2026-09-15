"""A tiny programming language, and a machine that searches for programs in it.

language   an instruction is (condition, action). A condition is a set of tests on
           working memory. There is no if, no jump, no loop, no halt:
             sequence      happens because working memory changed
             a loop        is a rule whose condition still holds
             a conditional is a rule whose condition names what to check
           So every syntactically possible program runs.
memory     a cursor (x, y, scale), K slots each holding (symbol, x, y), T tags.
           attending writes what is under the cursor into a slot.
credit     every slot and the cursor carry the step that wrote them. When the machine
           says an answer, walk backwards through what that step read: those steps are
           the worked program and take the reward. Steps outside the walk are charged
           for their time and credited nothing -- that is the whole minimality pressure.
cost       a step costs STEP_COST. Rules are capped, so the program pays for size too.
learning   running averages per rule (counts, no gradients). A rule splits when one
           test separates its good firings from its bad ones.
"""

import math
import random
from collections import defaultdict, deque

from tasks import Remembered

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIRNAME = ["UP", "RIGHT", "DOWN", "LEFT"]

STEP_COST = 0.02
TEMP = 0.15        # selection softness; fixed, no schedule
N_MAX = 300           # value averages stop moving after this many firings
MIN_SPLIT = 25        # firings under a test before it may be split off
SPLIT_GAIN = 0.15     # the test must be worth this much more than the parent
NOOP_COST = 0.05   # a step that changed nothing in working memory
BASE_RATE = 0.01   # how fast the reward baseline follows
MAX_COND = 3
RULE_CAP = 160
COPY_CAP = 8       # copies of a rule are its vote weight (accident, but load-bearing)
BUFFER = 64        # how many inputs it keeps in mind
THINK = 8          # hypothetical runs per real one
TEMP_LOOK = 0.6    # looking around is deliberately undecided
TEMP_THINK = 0.6   # imagining is free, so it is bolder than acting       # how many copies of one rule may stand (a rule's vote weight)


# ---------------------------------------------------------------- working memory
class WM:
    def __init__(self, task, n_slots, n_tags):
        self.x, self.y, self.scale = task.W // 2, task.H // 2, 0
        self.slots = [None] * n_slots
        self.tags = [False] * n_tags
        self.wrote_slot = [None] * n_slots   # step index that wrote it
        self.wrote_cur = None
        self.wrote_tag = [None] * n_tags
        self.rec = None

    def sym(self, task, st, dx=0, dy=0):
        v = task.sym(st, self.x + dx, self.y + dy, self.scale)
        if self.rec is not None:
            self.rec[(self.x + dx, self.y + dy, self.scale)] = v
        return v


def finger(wm):
    return (wm.x, wm.y, wm.scale, tuple(wm.slots), tuple(wm.tags))


def true_tests(wm, task, st):
    ts = [("cur", wm.sym(task, st)), ("scale", wm.scale)]
    for k, it in enumerate(wm.slots):
        if it is None:
            ts.append(("empty", k))
        else:
            ts.append(("full", k))
            ts.append(("slot", k, it[0]))
    for j, t in enumerate(wm.tags):
        if t:
            ts.append(("tag", j))
    for d, (dx, dy) in enumerate(DIRS):
        if not (0 <= wm.x + dx < task.W and 0 <= wm.y + dy < task.H):
            ts.append(("edge", d))
    return ts


# ---------------------------------------------------------------------- actions
def build_actions(task, n_slots, n_tags, content_jump=True):
    a = []
    for d in range(4):
        a.append(("move", d))
    a.append(("next",))                                   # raster advance
    for k in range(n_slots):
        a.append(("write", k))                            # attend here
        for d in range(4):
            a.append(("peek", k, d))                      # attend one cell over
    for j in range(n_tags):
        a.append(("tag", j))
    for c in task.answers:
        a.append(("say", c))
    if task.say_slot:
        for k in range(n_slots):
            a.append(("saysl", k))                        # say what a slot holds
    a.append(("norm",))                                   # jump to the middle of the ink
    a.append(("zoom", 1))
    a.append(("zoom", -1))
    if content_jump:
        for s in range(1, task.palette):
            a.append(("goto", s))                         # jump to the nearest s
    return a


def act_name(a):
    k = a[0]
    if k == "move":  return f"MOVE {DIRNAME[a[1]]}"
    if k == "next":  return "NEXT"
    if k == "write": return f"WRITE s{a[1]}"
    if k == "peek":  return f"PEEK s{a[1]} {DIRNAME[a[2]]}"
    if k == "tag":   return f"TAG {a[1]}"
    if k == "say":   return f"SAY {a[1]}"
    if k == "saysl": return f"SAY s{a[1]}"
    if k == "norm":  return "CENTER"
    if k == "zoom":  return f"ZOOM {'IN' if a[1] > 0 else 'OUT'}"
    if k == "goto":  return f"GOTO nearest {a[1]}"
    return str(a)


def test_name(t):
    if t[0] == "cur":   return f"under cursor={t[1]}"
    if t[0] == "slot":  return f"s{t[1]}={t[2]}"
    if t[0] == "full":  return f"s{t[1]} filled"
    if t[0] == "empty": return f"s{t[1]} empty"
    if t[0] == "tag":   return f"tag{t[1]}"
    if t[0] == "edge":  return f"edge {DIRNAME[t[1]]}"
    if t[0] == "scale": return f"scale={t[1]}"
    return str(t)


# reads/writes of each action, for the credit walk
def flow(a):
    k = a[0]
    rc = wc = False          # reads / writes the cursor
    rs, ws = set(), set()    # reads / writes slots
    if k in ("move", "next"):
        rc = wc = True
    elif k in ("norm", "goto"):
        wc = True
    elif k == "zoom":
        rc = wc = True
    elif k in ("write", "peek"):
        rc = True
        ws.add(a[1])
    elif k == "saysl":
        rs.add(a[1])
    return rc, wc, rs, ws


# ------------------------------------------------------------------------ rules
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
    def __init__(self, task, n_slots=2, n_tags=2, content_jump=True, seed=0,
                 max_steps=30, split=True, hindsight=True):
        self.task, self.rng = task, random.Random(seed)
        self.n_slots, self.n_tags, self.max_steps = n_slots, n_tags, max_steps
        self.actions = build_actions(task, n_slots, n_tags, content_jump)
        self.rules = [Rule([], a) for a in self.actions]
        self.split_on = split
        self.hindsight = hindsight
        self.in_graph = self.tot_steps = 0
        self.base = 0.0
        self.hits = []
        self.buffer = deque(maxlen=BUFFER)
        self.dreamt = self.dreamt_ok = self.dreamt_steps = 0

    # ---------------------------------------------------------------- one episode
    def run(self, st, learn=True, trace_out=None, soft=False, record=False, real=True,
            noanswer=False, temp=None):
        task, wm = self.task, WM(self.task, self.n_slots, self.n_tags)
        if record:
            wm.rec = {}
        steps = []          # per step: rule, reads/writes, cursor-writer before
        said = None
        seen, waste = {finger(wm): -1}, set()
        for i in range(self.max_steps):
            ts = set(true_tests(wm, task, st))
            cands = [r for r in self.rules if r.cond <= ts]
            if noanswer:
                cands = [c for c in cands if c.act[0] not in ('say', 'saysl')] or cands
            if learn or soft:
                t = temp or (TEMP_LOOK if noanswer else TEMP)
                w = [math.exp(c.v / t) for c in cands]
                r = self.rng.choices(cands, weights=w)[0]
            else:
                best = max(c.v for c in cands)
                r = self.rng.choice([c for c in cands if c.v >= best - 1e-9])
            rc, wc, rs, ws = flow(r.act)
            dep_cur = wm.wrote_cur if rc else None
            dep_slots = [wm.wrote_slot[k] for k in rs]
            if r.cond:
                for t in r.cond:                     # a condition is a read too
                    if t[0] in ("slot", "full", "empty"):
                        dep_slots.append(wm.wrote_slot[t[1]])
                    elif t[0] in ("cur", "edge"):
                        dep_cur = wm.wrote_cur
                    elif t[0] == "tag":
                        dep_slots.append(wm.wrote_tag[t[1]])
            f0 = finger(wm)
            said = self.apply(r.act, wm, st, i)
            f1 = finger(wm)
            if f1 in seen:                    # back where it was: everything since was idle
                waste.update(range(seen[f1] + 1, i + 1))
            else:
                seen[f1] = i
            steps.append((r, dep_cur, [d for d in dep_slots if d is not None], ts,
                          f1 == f0))
            if trace_out is not None:
                trace_out.append((i, r, sorted(ts)))
            if said is not None:
                break
        if record:
            self.buffer.append(Remembered(wm.rec, task.W, task.H, task.answer))
        if learn == (0, 1) and said != task.answer:
            return said, len(steps)          # imagined failure: proves nothing, teaches nothing
        if not learn:
            return said, len(steps)
        self.credit(steps, said, st, waste, real)
        return said, len(steps)

    def think(self, n=THINK):
        """Run the machine on inputs it remembers, without acting on anything.
        A success found in the head counts exactly like one found in the world --
        which is the point: real successes are far too rare to learn from."""
        if not self.buffer:
            return
        keep = self.task.answer
        for _ in range(n):
            mem = self.rng.choice(self.buffer)
            self.task.answer = mem.answer
            # learn from what worked in the head, never from what failed there:
            # a failure may only mean it does not remember that part of the input
            said, k = self.run(mem, learn=(0, 1), real=False, temp=TEMP_THINK)
            self.dreamt += 1
            self.dreamt_ok += (said == mem.answer)
            self.dreamt_steps += k
        self.task.answer = keep

    def apply(self, a, wm, st, step):
        task, k = self.task, a[0]
        if k == "move":
            dx, dy = DIRS[a[1]]
            wm.x = min(max(wm.x + dx, 0), task.W - 1)
            wm.y = min(max(wm.y + dy, 0), task.H - 1)
            wm.wrote_cur = step
        elif k == "next":
            p = wm.y * task.W + wm.x + 1
            p %= task.W * task.H
            wm.x, wm.y = p % task.W, p // task.W
            wm.wrote_cur = step
        elif k == "write":
            wm.slots[a[1]] = (wm.sym(task, st), wm.x, wm.y)
            wm.wrote_slot[a[1]] = step
        elif k == "peek":
            dx, dy = DIRS[a[2]]
            wm.slots[a[1]] = (wm.sym(task, st, dx, dy), wm.x + dx, wm.y + dy)
            wm.wrote_slot[a[1]] = step
        elif k == "tag":
            wm.tags[a[1]] = True
            wm.wrote_tag[a[1]] = step
        elif k == "norm":
            wm.x, wm.y = task.centre(st)
            wm.wrote_cur = step
        elif k == "zoom":
            wm.scale = min(max(wm.scale + a[1], 0), task.scales - 1)
            wm.wrote_cur = step
        elif k == "goto":
            p = task.nearest(st, a[1], wm.x, wm.y)
            if p is not None:
                wm.x, wm.y = p
            wm.wrote_cur = step
        elif k == "say":
            return a[1]
        elif k == "saysl":
            it = wm.slots[a[1]]
            return -1 if it is None else it[0]
        return None

    # ------------------------------------------------- walk back, pay, then split
    def credit(self, steps, said, st, waste=frozenset(), real=True):
        n = len(steps)
        inside = set()
        if said is None:
            inside = set(range(n))     # ran out of time: the whole episode is to blame
        else:
            stack, seen = [n - 1], set()
            while stack:
                i = stack.pop()
                if i in seen or i is None:
                    continue
                seen.add(i)
                inside.add(i)
                r, dep_cur, dep_slots, _, _ = steps[i]
                for d in ([dep_cur] if dep_cur is not None else []) + dep_slots:
                    if d not in seen:
                        stack.append(d)
        self.in_graph += len(inside)
        self.tot_steps += n
        right = (said is not None and said == self.task.answer)
        reward = 1.0 if right else -1.0
        if real:
            self.hits.append(right)
        adv = reward - self.base            # a baseline, so bystanding is not the safe bet
        self.base += BASE_RATE * (reward - self.base)
        for i, (r, _, _, ts, noop) in enumerate(steps):
            noop = noop or i in waste
            target = (adv if i in inside and not noop else 0.0) - STEP_COST
            if noop:
                target -= NOOP_COST
            r.n += 1
            r.v += (target - r.v) / min(r.n, N_MAX)
            if self.split_on and len(r.cond) < MAX_COND:
                for t in ts:
                    if t in r.cond:
                        continue
                    a = r.tally[t]
                    a[0] += target
                    a[1] += 1
        if self.hindsight:
            self.counterfactual(steps)
        if self.split_on:
            self.grow()
        return inside

    def counterfactual(self, steps):
        """Saying is the one action whose outcome can be known without doing it.
        After the episode, for every moment and every way of saying, ask what it
        would have scored, and let the answering rules count that. This is how a
        rule learns *when* it is allowed to answer -- otherwise the whole chain
        has to be stumbled on at once."""
        ans = self.task.answer
        look = self.rng.sample(range(len(steps)), min(2, len(steps)))
        for j in look:
            ts = steps[j][3]
            held = {t[1]: t[2] for t in ts if t[0] == "slot"}
            for r in self.rules:
                a = r.act
                if a[0] == "say":
                    said = a[1]
                elif a[0] == "saysl":
                    said = held.get(a[1], -1)
                else:
                    continue
                if not r.cond <= ts:
                    continue
                target = (1.0 if said == ans else -1.0) - self.base - STEP_COST
                r.n += 1
                r.v += (target - r.v) / min(r.n, N_MAX)
                if self.split_on and len(r.cond) < MAX_COND:
                    for t in ts:
                        if t not in r.cond:
                            q = r.tally[t]
                            q[0] += target
                            q[1] += 1

    def grow(self):
        for r in list(self.rules):
            if len(r.cond) >= MAX_COND or r.n < MIN_SPLIT:
                continue
            best, bt = None, None
            for t, (s, c) in r.tally.items():
                if c < MIN_SPLIT:
                    continue
                m = s / c
                if m - r.v >= SPLIT_GAIN and (best is None or m > best):
                    best, bt = m, t
            if bt is not None:
                cond = frozenset(set(r.cond) | {bt})
                copies = sum(1 for q in self.rules if q.cond == cond and q.act == r.act)
                if copies < COPY_CAP:
                    self.rules.append(Rule(cond, r.act, best))
                r.tally.clear()
        if len(self.rules) > RULE_CAP:
            base = len(self.actions)
            extra = self.rules[base:]
            extra.sort(key=lambda r: -r.v)
            self.rules = self.rules[:base] + extra[: RULE_CAP - base]

    # ------------------------------------------------------------------ reporting
    def top(self, k=12):
        rs = [r for r in self.rules if r.n >= 20]
        rs.sort(key=lambda r: -r.v)
        return rs[:k]
