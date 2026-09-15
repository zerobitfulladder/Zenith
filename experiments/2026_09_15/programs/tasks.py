"""Two inputs for the machine.

BELOW   a 5x5 grid. Exactly one cell holds the marker 1. The answer is the symbol
        in the cell directly underneath it. The shortest correct program is three
        rules -- scan, then "if the marker is under the cursor, attend one cell down",
        then "if that slot is filled, say what it holds" -- so a right answer needs a
        found marker, a relative attend, and an answer copied out of memory. No fixed
        route can do it, because the marker moves.
MNIST   28x28 pooled to 14x14 and quantised to four ink levels, three digits.
        The machine only ever sees one cell at a time, so one glance cannot answer.
"""

import numpy as np

OOB = 9       # a read outside the grid
UNKNOWN = 8   # a read, in the head, of somewhere it never actually looked


class Remembered:
    """What the machine kept of one input: the cells it actually read, nothing else.
    Thinking runs on these, so it can only imagine places it has been."""
    __slots__ = ("cells", "W", "H", "answer")

    def __init__(self, cells, W, H, answer):
        self.cells, self.W, self.H, self.answer = cells, W, H, answer


class Grid:
    name = "BELOW"
    W = H = 5
    palette = 4
    answers = [0, 2, 3]
    say_slot = True
    scales = 2
    marker = 1

    def __init__(self):
        self.answer = None

    def sample(self, rng):
        g = np.zeros((self.H, self.W), dtype=np.int8)
        for y in range(self.H):
            for x in range(self.W):
                g[y, x] = 0 if rng.random() < 0.6 else rng.choice([2, 3])
        mx, my = rng.randrange(self.W), rng.randrange(self.H - 1)
        g[my, mx] = self.marker
        ans = rng.choice(self.answers)
        g[my + 1, mx] = ans
        self.answer = int(ans)
        return pyramid(g)

    sym = staticmethod(lambda st, x, y, s: read(st, x, y, s))
    centre = staticmethod(lambda st: centre_of(st))
    nearest = staticmethod(lambda st, s, x, y: nearest_of(st, s, x, y))


class Mnist:
    name = "MNIST"
    W = H = 14
    palette = 4
    say_slot = False
    scales = 2

    def __init__(self, digits=(0, 1, 7), n=4000, seed=0, root="data"):
        from pathlib import Path
        r = Path(root)
        X = np.load(r / "mnist/digits/train_images.npy").astype(np.float32)
        Y = np.load(r / "mnist/digits/train_labels.npy")
        keep = np.isin(Y, digits)
        X, Y = X[keep][:n], Y[keep][:n]
        X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))          # 28 -> 14
        q = np.digitize(X, [0.15, 0.40, 0.70]).astype(np.int8)     # -> 0..3
        self.X, self.Y = q, Y.astype(int)
        self.answers = sorted(int(d) for d in digits)
        self.answer = None
        self.pool = [pyramid(self.X[i]) for i in range(len(self.X))]

    def split(self, n_train):
        return list(range(n_train)), list(range(n_train, len(self.pool)))

    def take(self, i):
        self.answer = int(self.Y[i])
        return self.pool[i]

    sym = staticmethod(lambda st, x, y, s: read(st, x, y, s))
    centre = staticmethod(lambda st: centre_of(st))
    nearest = staticmethod(lambda st, s, x, y: nearest_of(st, s, x, y))


# ------------------------------------------------------------------ grid helpers
def pyramid(g):
    h, w = g.shape
    lv1 = np.zeros(((h + 1) // 2, (w + 1) // 2), dtype=np.int8)
    for y in range(lv1.shape[0]):
        for x in range(lv1.shape[1]):
            lv1[y, x] = int(round(g[2 * y:2 * y + 2, 2 * x:2 * x + 2].mean()))
    return (g, lv1)


def read(st, x, y, s):
    if isinstance(st, Remembered):
        if not (0 <= x < st.W and 0 <= y < st.H):
            return OOB
        return st.cells.get((x, y, s), UNKNOWN)
    g = st[0]
    if not (0 <= x < g.shape[1] and 0 <= y < g.shape[0]):
        return OOB
    a = st[min(s, len(st) - 1)]
    return int(a[y >> s, x >> s]) if s else int(g[y, x])


def centre_of(st):
    if isinstance(st, Remembered):
        p = [(x, y) for (x, y, sc), v in st.cells.items() if sc == 0 and v not in (0, UNKNOWN)]
        if not p:
            return st.W // 2, st.H // 2
        return int(sum(a for a, _ in p) / len(p)), int(sum(b for _, b in p) / len(p))
    g = st[0]
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return g.shape[1] // 2, g.shape[0] // 2
    return int(xs.mean()), int(ys.mean())


def nearest_of(st, s, x, y):
    if isinstance(st, Remembered):
        p = [(a, b) for (a, b, sc), v in st.cells.items() if sc == 0 and v == s]
        return min(p, key=lambda q: abs(q[0] - x) + abs(q[1] - y)) if p else None
    g = st[0]
    ys, xs = np.nonzero(g == s)
    if len(xs) == 0:
        return None
    d = np.abs(xs - x) + np.abs(ys - y)
    i = int(np.argmin(d))
    return int(xs[i]), int(ys[i])
