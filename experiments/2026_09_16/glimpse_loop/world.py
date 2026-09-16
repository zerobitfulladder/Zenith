"""Glimpse world  (DESIGN.md XI.2).

A hidden library the agent never sees:
    primitives  - the sensory alphabet (a, b, c, ...); each cell holds one or is empty
    subs        - sub-assemblies: 3 connected cells of primitives
    objects     - two subs placed touching, plus sometimes one extra primitive

Each episode places one object at a random translation on a canvas.  The agent may
look at one cell at a time and gets back that cell's primitive code (or EMPTY).
Ground truth exists only for scoring.
"""
import numpy as np
from memory import normalize

NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class World:
    def __init__(self, alg, rows=12, cols=12, n_prims=6, n_subs=5, n_objects=6, seed=0):
        self.alg, self.rows, self.cols = alg, rows, cols
        self.rng = np.random.default_rng(seed)
        self.prim_labels = [chr(ord("a") + i) for i in range(n_prims)]
        self.prim_codes = [alg.random() for _ in range(n_prims)]
        self.empty_code = alg.random()
        self.subs = []
        while len(self.subs) < n_subs:
            s = self._make_sub()
            if s not in self.subs:
                self.subs.append(s)
        self.objects = []
        self.object_subs = []
        self.add_objects(n_objects)
        self.glimpses = 0
        self.canvas = {}
        self.current = None

    # ---------------- hidden library ----------------
    def _make_sub(self, size=3):
        cells = {(0, 0)}
        while len(cells) < size:
            r, c = list(cells)[self.rng.integers(len(cells))]
            dr, dc = NEIGH4[self.rng.integers(4)]
            cells.add((r + dr, c + dc))
        return normalize((int(self.rng.integers(len(self.prim_codes))), rc) for rc in cells)

    def add_objects(self, n, max_tries=2000):
        """Objects reuse the subs: pairs are drawn so that every sub appears at least twice."""
        made = 0
        tries = 0
        while made < n and tries < max_tries:
            tries += 1
            counts = np.zeros(len(self.subs))
            for pair in self.object_subs:
                for s in pair:
                    counts[s] += 1
            order = np.argsort(counts + self.rng.random(len(self.subs)) * 0.1)
            a = int(order[0])
            b = int(self.rng.choice([s for s in range(len(self.subs)) if s != a]))
            fp = self._compose(self.subs[a], self.subs[b])
            if fp is None or fp in self.objects or fp in self.subs:
                continue
            self.objects.append(fp)
            self.object_subs.append((a, b))
            made += 1

    def _compose(self, A, B):
        cellsA = {rc: p for p, rc in A}
        for _ in range(50):
            off = (int(self.rng.integers(-3, 4)), int(self.rng.integers(-3, 4)))
            cellsB = {(r + off[0], c + off[1]): p for p, (r, c) in B}
            if set(cellsA) & set(cellsB):
                continue
            touching = any((r + dr, c + dc) in cellsB for (r, c) in cellsA for dr, dc in NEIGH4)
            if not touching:
                continue
            cells = dict(cellsA); cells.update(cellsB)
            if self.rng.random() < 0.5:                       # one extra primitive, attached
                ring = [(r + dr, c + dc) for (r, c) in cells for dr, dc in NEIGH4
                        if (r + dr, c + dc) not in cells]
                rc = ring[self.rng.integers(len(ring))]
                cells[rc] = int(self.rng.integers(len(self.prim_codes)))
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            if max(rs) - min(rs) > 5 or max(cs) - min(cs) > 5:
                continue
            return normalize((p, rc) for rc, p in cells.items())
        return None

    # ---------------- episodes ----------------
    def start(self, obj_id=None):
        if obj_id is None:
            obj_id = int(self.rng.integers(len(self.objects)))
        fp = self.objects[obj_id]
        rs = [r for _, (r, _) in fp]; cs = [c for _, (_, c) in fp]
        r0 = int(self.rng.integers(-min(rs), self.rows - max(rs)))
        c0 = int(self.rng.integers(-min(cs), self.cols - max(cs)))
        self.canvas = {(r + r0, c + c0): p for p, (r, c) in fp}
        self.current = (obj_id, (r0, c0))
        self.glimpses = 0
        return obj_id

    def cue(self):
        """Attention is drawn to one cell of the object (the first glimpse lands on it)."""
        cells = list(self.canvas)
        return cells[self.rng.integers(len(cells))]

    def look(self, cell):
        assert 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols
        self.glimpses += 1
        p = self.canvas.get(cell)
        return self.empty_code if p is None else self.prim_codes[p]

    def truth(self):
        return dict(self.canvas)

    def truth_fp(self):
        return self.objects[self.current[0]]
