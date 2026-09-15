"""BELOW, unchanged from the reference line: 5x5 grid, one cell holds the marker 1,
the answer is the symbol directly underneath it. The marker moves, so no fixed route
works. Kept identical so the two machines are comparable."""

import numpy as np

OOB = 9        # a read outside the grid
UNKNOWN = 8    # reserved for the prediction step


class Grid:
    name = "BELOW"
    W = H = 5
    palette = 4
    answers = [0, 2, 3]
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
        return g

    def sym(self, g, x, y):
        if not (0 <= x < self.W and 0 <= y < self.H):
            return OOB
        return int(g[y, x])

    def nearest(self, g, s, x, y):
        ys, xs = np.nonzero(g == s)
        if len(xs) == 0:
            return None
        d = np.abs(xs - x) + np.abs(ys - y)
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])
