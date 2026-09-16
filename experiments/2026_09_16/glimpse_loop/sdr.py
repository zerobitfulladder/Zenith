"""Sparse block-code algebra  (DESIGN.md Parts IV, VI, Appendix).

A code is an int array of shape (B,): B blocks of L slots, one active slot per block,
so D = B*L bits with exactly k = B of them on.

    bind(a, b)   = per-block addition mod L      -> exact inverse, poses compose by binding
    bundle       = per-slot integer counts        -> superposition, exact subtraction ("undo")
    unbind(W, c) = shift every block back by c    -> whatever was bound with c lines up
    presence     = min count over the B slots of a stored code  (0 = absent)

The only failure mode of exact unbinding is a *ghost*: a code whose B slots are all
occupied by other items' bits.  P(ghost) = density**B per probe.  That number is the
whole "over-stimulation" drive of Part VIII, in closed form.
"""
import numpy as np


class Algebra:
    def __init__(self, B=12, L=41, seed=0):
        self.B, self.L = B, L
        self.rng = np.random.default_rng(seed)
        self.ar = np.arange(B)
        self.slots = np.arange(L)

    # ---------------- codes ----------------
    def random(self, nonzero=False):
        return self.rng.integers(1 if nonzero else 0, self.L, self.B)

    def bind(self, a, b):     return (a + b) % self.L
    def inverse(self, a):     return (-a) % self.L
    def power(self, a, n):    return (a * n) % self.L          # a ⊗ a ⊗ ... (n times, n may be < 0)
    def overlap(self, a, b):  return int((a == b).sum())        # blocks in agreement, 0..B

    # ---------------- bundles: (B, L) counts ----------------
    def empty(self):                 return np.zeros((self.B, self.L), np.int32)
    def add(self, W, code, m=1):     W[self.ar, code] += m
    def sub(self, W, code, m=1):     W[self.ar, code] -= m
    def density(self, W):            return float((W > 0).mean())

    def unbind(self, W, code):
        """(B, L) -> (B, L): U[b, s] = W[b, (s + code[b]) % L]."""
        return W[self.ar[:, None], (self.slots[None, :] + code[:, None]) % self.L]

    def unbind_many(self, W, codes):
        """codes (N, B) -> (N, B, L)."""
        idx = (self.slots[None, None, :] + codes[:, :, None]) % self.L
        return W[self.ar[None, :, None], idx]

    def presence(self, U, vocab):
        """U (..., B, L) unbound bundle, vocab (V, B) -> (..., V) min count over blocks."""
        if U.ndim == 2:
            return U[self.ar[None, :], vocab].min(-1)
        return U[:, self.ar[None, :], vocab].min(-1)

    def cleanup(self, code, vocab):
        """Nearest stored code by block overlap -> (index, overlap)."""
        ov = (vocab == code[None, :]).sum(1)
        i = int(ov.argmax())
        return i, int(ov[i])

    def ghost_estimate(self, W, n_probes):
        """Expected number of false items in a read-out of n_probes (tag, cell, name) probes."""
        return n_probes * self.density(W) ** self.B


class Poses:
    """pose(r, c) = r·U_r  ⊗  c·U_c   so that   pose(a) ⊗ pose(b) = pose(a + b).   (DESIGN VI.4)

    Translation composition IS binding; the same codes serve as absolute cell positions and as
    relative offsets.  Block-shift binding is commutative, so this covers translations only.
    """
    def __init__(self, alg, rows, cols):
        self.alg, self.rows, self.cols = alg, rows, cols
        self.ur = alg.random(nonzero=True)
        self.uc = alg.random(nonzero=True)
        self.cells = [(r, c) for r in range(rows) for c in range(cols)]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.cell_codes = np.stack([self(cell) for cell in self.cells])     # (n_cells, B)

    def __call__(self, rc):
        return self.alg.bind(self.alg.power(self.ur, rc[0]), self.alg.power(self.uc, rc[1]))

    def on_canvas(self, rc):
        return 0 <= rc[0] < self.rows and 0 <= rc[1] < self.cols

    def check_composition(self, trials=200):
        """VI.4: composing two transformations must equal binding their codes."""
        rng = np.random.default_rng(1)
        for _ in range(trials):
            a = tuple(rng.integers(-6, 7, 2)); b = tuple(rng.integers(-6, 7, 2))
            lhs = self.alg.bind(self(a), self(b))
            rhs = self((a[0] + b[0], a[1] + b[1]))
            if not np.array_equal(lhs, rhs):
                return False
        # distinctness of canvas cells
        return len({tuple(c) for c in self.cell_codes}) == len(self.cells)


def measure_capacity(alg, n_cells=144, n_tags=3, V=40, trials=40, max_n=60, seed=0):
    """DESIGN XI.1 as it applies here: bundle n bound items, do a full read-out over
    n_tags x n_cells poses against V names, count false items.  Returns a list of
    (n, measured ghosts, predicted ghosts)."""
    rng = np.random.default_rng(seed)
    tags = rng.integers(0, alg.L, (n_tags, alg.B))
    poses = rng.integers(0, alg.L, (n_cells, alg.B))
    vocab = rng.integers(0, alg.L, (V, alg.B))
    out = []
    for n in range(1, max_n + 1):
        ghosts = 0.0
        for _ in range(trials):
            W = alg.empty()
            written = set()
            for _ in range(n):
                t, p, v = rng.integers(n_tags), rng.integers(n_cells), rng.integers(V)
                written.add((t, p, v))
                alg.add(W, alg.bind(alg.bind(tags[t], poses[p]), vocab[v]))
            found = 0
            for t in range(n_tags):
                U = alg.unbind_many(W, alg.bind(tags[t][None, :], poses))   # (n_cells, B, L)
                pres = alg.presence(U, vocab)                                 # (n_cells, V)
                found += int((pres > 0).sum())
            ghosts += found - len(written)
        pred = alg.ghost_estimate(W, n_tags * n_cells * V)
        out.append((n, ghosts / trials, pred))
        if ghosts / trials > 8:
            break
    return out
