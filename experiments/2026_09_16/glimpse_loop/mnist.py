"""MNIST as a glimpse world with a learned patch vocabulary  (DESIGN.md Part III, Part XIV).

    image  ->  4x4 grid of 7x7 patches  ->  each patch cleaned up to a prototype (or EMPTY)

Prototypes are allocated online and gated (X.2): a patch far from every prototype opens a
provisional one, which is kept only if it recurs.  Prototype codes are block codes whose block
groups carry different predictive roles (III.4, a factored code):

    sim blocks - winner-take-all random projections of the patch itself: similar strokes overlap
    ctx blocks - the same hashing applied to the prototype's context statistics (what occurs
                 next to it): strokes with the same neighbours overlap   (III.6, self-supervised)
    lab blocks - the same hashing of the prototype's label statistics: same-outcome strokes overlap

No gradients anywhere: the "learning" is counting, and hashing what was counted.
"""
import gzip, struct
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "mnist"
PATCH, GRID = 7, 4
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def load(split):
    with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(n, r, c)
    with gzip.open(DATA / f"{split}-labels-idx1-ubyte.gz") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), np.uint8)
    return X.astype(np.float32) / 255.0, y.astype(int)


def patches(img):
    """(28, 28) -> (16, 49), row-major over the 4x4 grid."""
    return img.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3).reshape(GRID * GRID, PATCH * PATCH)


class Vocabulary:
    """Gated online allocation of patch prototypes (running means)."""

    def __init__(self, ink=0.06, tau_new=1.6, recur=4, patience=3000):
        self.ink, self.tau_new, self.recur, self.patience = ink, tau_new, recur, patience
        self.means = np.zeros((0, PATCH * PATCH), np.float32)
        self.counts = np.zeros(0, int)
        self.status = []              # "provisional" | "consolidated"
        self.last_hit = np.zeros(0, int)
        self.t = 0
        self.EMPTY = -2               # id of the blank patch in grids; -1 = unobserved

    def blank(self, patch):
        return (patch > 0.5).mean() < self.ink            # fewer than ~3 of 49 pixels clearly on

    def nearest(self, patch):
        if len(self.means) == 0:
            return -1, np.inf
        d = np.sqrt(((self.means - patch) ** 2).sum(1))
        i = int(d.argmin())
        return i, float(d[i])

    def learn(self, patch):
        self.t += 1
        if self.blank(patch):
            return self.EMPTY
        i, d = self.nearest(patch)
        if d > self.tau_new:                                   # retrieval failed: allocate, provisionally
            self.means = np.vstack([self.means, patch[None]])
            self.counts = np.append(self.counts, 1)
            self.status.append("provisional")
            self.last_hit = np.append(self.last_hit, self.t)
            i = len(self.means) - 1
        else:
            self.counts[i] += 1
            self.last_hit[i] = self.t
            self.means[i] += (patch - self.means[i]) / min(self.counts[i], 200)   # running mean
            if self.status[i] == "provisional" and self.counts[i] >= self.recur:
                self.status[i] = "consolidated"                # it recurred: keep it
        return i

    def sweep(self):
        """Forget provisional prototypes that did not recur."""
        keep = [k for k in range(len(self.means))
                if self.status[k] == "consolidated" or self.t - self.last_hit[k] < self.patience]
        self.means, self.counts = self.means[keep], self.counts[keep]
        self.status = [self.status[k] for k in keep]
        self.last_hit = self.last_hit[keep]

    def finalize(self):
        keep = [k for k in range(len(self.means)) if self.status[k] == "consolidated"]
        self.means, self.counts = self.means[keep], self.counts[keep]
        self.status = ["consolidated"] * len(keep)
        self.last_hit = self.last_hit[keep]

    def sense(self, patch):
        """Cleanup: the nearest prototype, or EMPTY."""
        if self.blank(patch):
            return self.EMPTY
        return self.nearest(patch)[0]

    def grid(self, img):
        return np.array([self.sense(p) for p in patches(img)])

    def __len__(self):
        return len(self.means)


def build_vocabulary(X, n_images=8000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    voc = Vocabulary(**kw)
    order = rng.permutation(len(X))[:n_images]
    for t, i in enumerate(order):
        for p in patches(X[i]):
            voc.learn(p)
        if t % 1000 == 999:
            voc.sweep()
    voc.finalize()
    # one refinement pass: reassign and re-average (one EM step, Part I style)
    sums = np.zeros_like(voc.means); cnt = np.zeros(len(voc), int)
    for i in order[:4000]:
        for p in patches(X[i]):
            if not voc.blank(p):
                k = voc.nearest(p)[0]; sums[k] += p; cnt[k] += 1
    ok = cnt > 0
    voc.means[ok] = sums[ok] / cnt[ok][:, None]
    voc.counts = cnt
    return voc


def kmeans(Xf, K, rng, iters=15):
    """Plain k-means (EM with counting).  Returns assignments and centroids."""
    K = min(K, len(Xf))
    cent = Xf[rng.choice(len(Xf), K, replace=False)].copy()
    for _ in range(iters):
        d = ((Xf[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in range(K):
            if (a == k).any():
                cent[k] = Xf[a == k].mean(0)
            else:
                cent[k] = Xf[rng.integers(len(Xf))]
    d = ((Xf[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return d.argmin(1), cent


class Codes:
    """Block codes for prototypes.  Every block is a *partition* of the vocabulary into L groups,
    learned by k-means in a role-specific feature space; a prototype's slot in a block is its
    group.  Two prototypes overlap in the blocks whose feature space puts them together:

        sim blocks - pixel space (random pixel subsets per block)      -> similar strokes overlap
        ctx blocks - neighbour statistics (self-supervised, III.6)      -> same-context strokes overlap
        lab blocks - label statistics (the outcome)                     -> same-outcome strokes overlap

    A factored code (III.4): different block groups carry different predictive roles, and pixel
    blocks keep visually distinct things apart, which is what stops collapse (III.3)."""

    def __init__(self, alg, voc, grids, labels, B_sim=8, B_ctx=4, B_lab=4, K_coarse=12, seed=0):
        self.alg, self.voc = alg, voc
        self.B_sim, self.B_ctx, self.B_lab = B_sim, B_ctx, B_lab
        assert alg.B == B_sim + B_ctx + B_lab
        rng = np.random.default_rng(seed)
        P, L = len(voc), alg.L
        self.coarse, _ = kmeans(voc.means, K_coarse, rng)          # coarse stroke groups, for context features
        ctx = np.zeros((P, 4, K_coarse + 1))
        lab = np.zeros((P, 10))
        for g, y in zip(grids, labels):
            g2 = g.reshape(GRID, GRID)
            for i in range(GRID):
                for j in range(GRID):
                    p = g2[i, j]
                    if p < 0:
                        continue
                    lab[p, y] += 1
                    for d, (di, dj) in enumerate(NEIGH4):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < GRID and 0 <= jj < GRID:
                            q = g2[ii, jj]
                            ctx[p, d, self.coarse[q] if q >= 0 else K_coarse] += 1
        self.ctx = np.sqrt(ctx.reshape(P, -1) / (ctx.reshape(P, -1).sum(1, keepdims=True) + 1e-9))
        self.lab = np.sqrt(lab / (lab.sum(1, keepdims=True) + 1e-9))
        self.codes = np.zeros((P + 1, alg.B), int)                # last row = EMPTY, random
        self.sim_masks, self.sim_cent = [], []
        b = 0
        for _ in range(B_sim):
            mask = rng.random(PATCH * PATCH) < 0.6
            a, cent = kmeans(voc.means[:, mask], L, rng)
            self.codes[:P, b] = a; self.sim_masks.append(mask); self.sim_cent.append(cent); b += 1
        for _ in range(B_ctx):
            mask = rng.random(self.ctx.shape[1]) < 0.7
            a, _ = kmeans(self.ctx[:, mask], L, rng)
            self.codes[:P, b] = a; b += 1
        for _ in range(B_lab):
            a, _ = kmeans(self.lab + rng.normal(0, 0.02, self.lab.shape), L, rng)
            self.codes[:P, b] = a; b += 1
        self.codes[P] = alg.random()
        self.EMPTY_ROW = P
        self.overlap = (self.codes[:, None, :] == self.codes[None, :, :]).sum(-1)
        self.overlap_sim = (self.codes[:, None, :B_sim] == self.codes[None, :, :B_sim]).sum(-1)

    def sim_slots(self, patch):
        """Hash a raw patch on the sim blocks: nearest group centroid in each block's subspace."""
        return np.array([((self.sim_cent[b] - patch[self.sim_masks[b]][None, :]) ** 2).sum(1).argmin()
                         for b in range(self.B_sim)])

    def cleanup_patch(self, patch):
        """Sense through the code: hash the raw patch, nearest prototype by sim-block overlap."""
        s = self.sim_slots(patch)
        ov = (self.codes[:-1, :self.B_sim] == s[None, :]).sum(1)
        return int(ov.argmax()), int(ov.max())


# ------------------------------------------------------------------ measurements (Part III, Part I on real data)
def measure_codes(voc, C, X_test, rng, n=3000):
    P = len(voc)
    out = {}
    # 1. similar strokes share bits: sim-overlap vs pixel distance, over prototype pairs
    d = np.sqrt(((voc.means[:, None] - voc.means[None]) ** 2).sum(-1))
    iu = np.triu_indices(P, 1)
    dd, oo = d[iu], C.overlap_sim[iu]
    bins = np.quantile(dd, [0, .1, .3, .6, 1.0])
    out["sim_by_distance"] = [(float(bins[k]), float(bins[k + 1]), float(oo[(dd >= bins[k]) & (dd <= bins[k + 1])].mean()))
                              for k in range(4)]
    # 2. same-outcome strokes share bits: lab/ctx overlap for same- vs different-dominant-label pairs
    dom = C.lab.argmax(1)
    same = dom[iu[0]] == dom[iu[1]]
    o_lab = (C.codes[:-1, None, C.B_sim + C.B_ctx:] == C.codes[None, :-1, C.B_sim + C.B_ctx:]).sum(-1)[iu]
    o_ctx = (C.codes[:-1, None, C.B_sim:C.B_sim + C.B_ctx] == C.codes[None, :-1, C.B_sim:C.B_sim + C.B_ctx]).sum(-1)[iu]
    out["lab_overlap_same_vs_diff"] = (float(o_lab[same].mean()), float(o_lab[~same].mean()), C.B_lab)
    out["ctx_overlap_same_vs_diff"] = (float(o_ctx[same].mean()), float(o_ctx[~same].mean()), C.B_ctx)
    out["sim_overlap_same_vs_diff"] = (float(oo[same].mean()), float(oo[~same].mean()), C.B_sim)
    # 3. noisy sensing: does the noisy patch still clean up to the same prototype?
    idx = rng.permutation(len(X_test))[:n // 8]
    ps = np.vstack([patches(X_test[i]) for i in idx])
    ps = ps[[not voc.blank(p) for p in ps]][:n]
    clean = np.array([voc.nearest(p)[0] for p in ps])
    rows = {}
    for name, noisy in [("sigma 0.1", ps + rng.normal(0, 0.1, ps.shape)), ("sigma 0.25", ps + rng.normal(0, 0.25, ps.shape)),
                        ("shift 1px", np.vstack([np.roll(p.reshape(PATCH, PATCH), 1, axis=rng.integers(2)).reshape(-1) for p in ps]))]:
        noisy = np.clip(noisy, 0, 1)
        pix = np.array([voc.nearest(p)[0] for p in noisy])
        code = np.array([C.cleanup_patch(p)[0] for p in noisy])
        exact = np.array([voc.nearest(p)[1] < 1e-6 for p in noisy])   # the dict: identical pixels or nothing
        rows[name] = (float((pix == clean).mean()), float((code == clean).mean()), float(exact.mean()))
    out["noise"] = rows
    return out
