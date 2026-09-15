"""Two teachers for the same student.

random   walks somewhere at random, writes down what is there, four times, then says the
         right digit. It knows nothing about digits except the answer. The looking teaches
         nothing consistent -- but the last step is never random, and the conditions under
         which "say 7" is right are whatever happened to be in the slots.
tree     a decision tree over the cells, which IS a look-here-then-branch program: each
         node says go to this cell and branch on what is there, each leaf says an answer.
         A teacher that actually knows where to look, for contrast.
"""

import numpy as np
from machine import region_centre
from sklearn.tree import DecisionTreeClassifier

DIRS8 = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]


def walk(x, y, tx, ty):
    """relative steps from here to there; the machine has no absolute jump"""
    out = []
    while (x, y) != (tx, ty):
        dx = (tx > x) - (tx < x)
        dy = (ty > y) - (ty < y)
        out.append(("step", DIRS8.index((dx, dy))))
        x, y = x + dx, y + dy
    return out


class RandomTeacher:
    def __init__(self, task, looks=4, spread=4):
        self.task, self.looks, self.spread = task, looks, spread

    def plan(self, g, ans, rng):
        """jump somewhere at random, write down what is there, four times, then answer"""
        t = []
        for k in range(self.looks):
            t.append(("region", rng.randrange(9)))
            t.append(("write", k))
        t.append(("say", ans))
        return t


class TreeTeacher:
    def __init__(self, task, depth=6, seed=0):
        F = task.G[task.train].reshape(len(task.train), -1)
        self.t = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(
            F, task.Y[task.train])
        self.task = task
        te = task.G[task.test].reshape(len(task.test), -1)
        self.score = self.t.score(te, task.Y[task.test])

    def plan(self, g, ans, rng):
        tr, f = self.t.tree_, g.reshape(-1)
        out, node, x, y, k = [], 0, self.task.W // 2, self.task.H // 2, 0
        while tr.children_left[node] != -1:
            p = tr.feature[node]
            ty, tx = divmod(int(p), self.task.W)
            r = min(ty * 3 // self.task.H, 2) * 3 + min(tx * 3 // self.task.W, 2)
            out.append(("region", r))
            x, y = region_centre(self.task, r)
            out += walk(x, y, tx, ty)
            out.append(("write", k % 4))
            x, y, k = tx, ty, k + 1
            node = (tr.children_left[node] if f[p] <= tr.threshold[node]
                    else tr.children_right[node])
        out.append(("say", int(self.t.classes_[np.argmax(tr.value[node])])))
        return out


class CentreTeacher:
    """Looks at the middle and nowhere else. The accidental baseline."""

    def __init__(self, task, looks=4):
        self.task, self.looks = task, looks

    def plan(self, g, ans, rng):
        return [("write", k) for k in range(self.looks)] + [("say", ans)]


class FixedTeacher:
    """Always the same nine places, in the same order, then the answer.
    Nothing to work out about where to look -- all the learning goes into what to say."""

    def __init__(self, task):
        self.task = task

    def plan(self, g, ans, rng):
        t = []
        for r in range(9):
            t += [("region", r), ("write", r)]
        return t + [("say", ans)]

    def act(self, ts, ans):
        """what now, from wherever the student is: fill the lowest empty slot from its
        region; standing there already, write; nothing empty, answer"""
        nxt = next((t[1] for t in ts if t[0] == "next"), -1)
        if nxt < 0:
            return ("say", ans)
        here = next((t[1] for t in ts if t[0] == "here"), None)
        return ("write", nxt) if here == nxt else ("region", nxt)
