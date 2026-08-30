"""A three-layer convolutional stack with the label concatenated at the top.

    layer 1   4x4 patches of the image, 7x7 = 49 positions, ONE shared
              hypercolumn -> a code per position
    layer 2   3x3 windows of layer-one codes, stride 2, 3x3 = 9 positions,
              ONE shared hypercolumn -> a code per position
    layer 3   all 9 layer-two codes, with the one-hot label concatenated
              on the end. One hypercolumn, winner-take-all, so a stored
              template is a whole (image, label) pair.

Reading it is the partial-cue read this project uses everywhere: write
one half of layer three's input, leave the other half blank, let the
templates compete on the cells that are actually present, and read the
missing half back off the winner.

    classification   write the image half, read the label half
    generation       write the label half, read the image half, and push
                     it back down through layer two and layer one to pixels

Two choices worth naming.

**Energy.** After concatenation the label is 10 cells against 1152, so
left alone it is 0.3% of the code's length and the competition ignores
it entirely. The label block is therefore rescaled to `rho` times the
length of the image block before the two are joined. rho is the knob the
whole thing turns on and it is swept, not guessed.

**Normalize, but do not mean-centre, at layers one and two.** The rest of
this project centres every unit before normalizing. That is right when
only the code matters, but it throws away each patch's mean, and there
is nowhere to put it -- so the way back down can only ever recover a
patch up to an unknown offset, and a generated digit comes out blotchy.
Dividing by the norm and carrying that norm forward as the code's gain
keeps the path invertible: `decode(encode(x)) == x` whenever the
dictionary spans the patch. Layer three still centres, because nothing
is ever inverted past it.
"""

import numpy as np

EPS = 1e-8
SIDE, PATCH, GRID = 28, 4, 7            # 49 layer-one positions
WIN, STRIDE = 3, 2                      # layer-two window over that grid
L2_GRID = 1 + (GRID - WIN) // STRIDE    # 3 -> 9 layer-two positions


def unit_rows(X):
    """Scale each row to unit length; return the rows and their norms."""
    n = np.linalg.norm(X, axis=1)
    ok = n > EPS
    U = np.zeros_like(X)
    U[ok] = X[ok] / n[ok, None]
    return U, n


# --------------------------------------------------------------- layers ---
class DenseColumn:
    """All templates score, all learn from the leftover (the `weighted` rule).

    tau_i = s_i (e - g_i w_i) with e the shared residual and g_i = e . w_i;
    the update is rank-1, so a whole batch of patches costs two matmuls.
    """

    kind = "dense"

    def __init__(self, k, dim, eta, rng):
        W = rng.standard_normal((k, dim))
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)
        self.k, self.dim, self.eta = k, dim, eta

    def learn(self, U):
        for x in U:
            W = self.W
            s = W @ x
            e = x - s @ W
            g = W @ e
            tn = np.sqrt(np.maximum(float(e @ e) - g * g, 0.0))
            live = (tn > EPS) & (np.abs(s) > EPS)
            theta = self.eta * tn * np.abs(s)
            step = np.zeros_like(tn)
            step[live] = np.sin(theta[live]) * np.sign(s[live]) / tn[live]
            a = np.where(live, np.cos(theta) - step * g, 1.0)
            W = a[:, None] * W + step[:, None] * e[None, :]
            self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)

    def encode(self, U, norms):
        return (U @ self.W.T) * norms[:, None]

    def decode(self, C):
        return C @ self.W


class WTAColumn:
    """One winner per input; only the winner learns, and only it speaks.

    The codebook reading of the same layer, kept as the control: the code
    is a single (template, strength) pair instead of k numbers.
    """

    kind = "wta"

    def __init__(self, k, dim, eta, rng):
        self.k, self.dim, self.eta, self.rng = k, dim, eta, rng
        self.W = np.zeros((k, dim))
        self.n_boot = 0
        self.wins = np.zeros(k, dtype=np.int64)

    def learn(self, U):
        for x in U:
            if self.n_boot < self.k:                  # adopt the first k
                w = x + 0.01 * self.rng.standard_normal(self.dim)
                self.W[self.n_boot] = w / (np.linalg.norm(w) + EPS)
                self.n_boot += 1
                continue
            c = self.W @ x
            i = int(np.argmax(c))
            self.wins[i] += 1
            w = self.W[i]
            tau = x - float(c[i]) * w
            tn = float(np.linalg.norm(tau))
            if tn < EPS:
                continue
            th = self.eta * tn
            w = w * np.cos(th) + (tau / tn) * np.sin(th)
            self.W[i] = w / (np.linalg.norm(w) + EPS)

    def encode(self, U, norms):
        C = np.zeros((len(U), self.k))
        s = U @ self.W.T
        win = s.argmax(axis=1)
        r = np.arange(len(U))
        C[r, win] = np.maximum(s[r, win], 0.0) * norms
        return C

    def decode(self, C):
        return C @ self.W


def make_column(kind, k, dim, eta, rng):
    return (DenseColumn if kind == "dense" else WTAColumn)(k, dim, eta, rng)


# ------------------------------------------------------- the two conv rungs ---
def patches_of(X):
    """(n, 28, 28) -> (n*49, 16) in row-major position order."""
    n = len(X)
    P = X.reshape(n, GRID, PATCH, GRID, PATCH).transpose(0, 1, 3, 2, 4)
    return P.reshape(n * GRID * GRID, PATCH * PATCH)


def unpatch(P, n):
    """(n*49, 16) -> (n, 28, 28)."""
    return (P.reshape(n, GRID, GRID, PATCH, PATCH)
             .transpose(0, 1, 3, 2, 4).reshape(n, SIDE, SIDE))


def l2_windows(F):
    """(n, 7, 7, k1) layer-one field -> (n*9, 3*3*k1) windows."""
    n, _, _, k1 = F.shape
    out = np.empty((n, L2_GRID, L2_GRID, WIN * WIN * k1))
    for a in range(L2_GRID):
        for b in range(L2_GRID):
            r, c = a * STRIDE, b * STRIDE
            out[:, a, b] = F[:, r:r + WIN, c:c + WIN].reshape(n, -1)
    return out.reshape(n * L2_GRID * L2_GRID, -1)


def unwindow(Wv, n, k1):
    """(n*9, 3*3*k1) -> (n, 7, 7, k1), averaging where windows overlap."""
    acc = np.zeros((n, GRID, GRID, k1))
    cnt = np.zeros((1, GRID, GRID, 1))
    Wv = Wv.reshape(n, L2_GRID, L2_GRID, WIN, WIN, k1)
    for a in range(L2_GRID):
        for b in range(L2_GRID):
            r, c = a * STRIDE, b * STRIDE
            acc[:, r:r + WIN, c:c + WIN] += Wv[:, a, b]
            cnt[:, r:r + WIN, c:c + WIN] += 1
    return acc / np.maximum(cnt, 1)


class ConvStack:
    """Layers one and two: image in, the 9 layer-two codes out."""

    def __init__(self, kind, k1, k2, eta, rng):
        self.k1, self.k2 = k1, k2
        self.l1 = make_column(kind, k1, PATCH * PATCH, eta, rng)
        self.l2 = make_column(kind, k2, WIN * WIN * k1, eta, rng)

    def l1_field(self, X):
        U, nrm = unit_rows(patches_of(X))
        return self.l1.encode(U, nrm).reshape(len(X), GRID, GRID, self.k1)

    def forward(self, X):
        F = self.l1_field(X)
        U, nrm = unit_rows(l2_windows(F))
        return self.l2.encode(U, nrm).reshape(len(X), L2_GRID * L2_GRID * self.k2)

    def train(self, X, epochs, rng, chunk=256):
        for _ in range(epochs):                      # layer one, on patches
            order = rng.permutation(len(X))
            for s in range(0, len(X), chunk):
                B = X[order[s:s + chunk]]
                U, _ = unit_rows(patches_of(B))
                self.l1.learn(U[np.linalg.norm(U, axis=1) > EPS])
        for _ in range(epochs):                      # layer two, on frozen L1
            order = rng.permutation(len(X))
            for s in range(0, len(X), chunk):
                F = self.l1_field(X[order[s:s + chunk]])
                U, _ = unit_rows(l2_windows(F))
                self.l2.learn(U[np.linalg.norm(U, axis=1) > EPS])

    def backward(self, code, n):
        """9 layer-two codes -> pixels, back down through both rungs."""
        Wv = self.l2.decode(code.reshape(n * L2_GRID * L2_GRID, self.k2))
        F = unwindow(Wv, n, self.k1)
        P = self.l1.decode(F.reshape(n * GRID * GRID, self.k1))
        return unpatch(P, n)


# ------------------------------------------------------------- layer three ---
class BoundLayer:
    """Layer three: whole (image code, label) pairs, winner-take-all.

    The concatenated vector is [image block ; label block], the label
    block rescaled so its length is `rho` times the image block's. The
    join is then centred and normalized, and a template is one stored
    pair.

    Both reads are the same partial-cue read, differing only in which
    half is written. Scoring divides each template's overlap by ITS OWN
    norm over the cells actually present -- the plain dot product lets a
    template that is vague about the missing half keep all its length in
    the queried half and outbid the templates that genuinely match.
    `floor` caps that amplification for templates holding almost nothing
    there.
    """

    def __init__(self, k, img_dim, n_lab, rho, eta, rng):
        self.k, self.img_dim, self.n_lab, self.rho = k, img_dim, n_lab, rho
        self.dim = img_dim + n_lab
        self.eta, self.rng = eta, rng
        self.W = np.zeros((k, self.dim))
        self.n_boot = 0
        self.wins = np.zeros(k, dtype=np.int64)
        self.tally = np.zeros((k, n_lab), dtype=np.int64)
        self.img_ix = np.arange(img_dim)
        self.lab_ix = np.arange(img_dim, self.dim)

    def join(self, img, lab=None):
        """(n, img_dim) [+ (n, n_lab)] -> centred unit rows of length dim."""
        n = len(img)
        V = np.zeros((n, self.dim))
        V[:, :self.img_dim] = img
        if lab is not None:
            gain = self.rho * np.linalg.norm(img, axis=1) / \
                np.maximum(np.linalg.norm(lab, axis=1), EPS)
            V[:, self.img_dim:] = lab * gain[:, None]
        V -= V.mean(axis=1, keepdims=True)
        return unit_rows(V)[0]

    def learn(self, img, lab, y):
        for x, cls in zip(self.join(img, lab), y):
            if self.n_boot < self.k:
                w = x + 0.01 * self.rng.standard_normal(self.dim)
                self.W[self.n_boot] = w / (np.linalg.norm(w) + EPS)
                self.tally[self.n_boot, cls] += 1
                self.n_boot += 1
                continue
            c = self.W @ x
            i = int(np.argmax(c))
            self.wins[i] += 1
            self.tally[i, cls] += 1
            w = self.W[i]
            tau = x - float(c[i]) * w
            tn = float(np.linalg.norm(tau))
            if tn < EPS:
                continue
            th = self.eta * tn
            w = w * np.cos(th) + (tau / tn) * np.sin(th)
            self.W[i] = w / (np.linalg.norm(w) + EPS)

    def _bids(self, Q, idx, floor=0.25):
        """Every template's bid for a query written only at cells `idx`."""
        sub = self.W[:self.n_boot][:, idx]
        raw = Q @ sub.T
        nrm = np.linalg.norm(sub, axis=1)
        return raw / np.maximum(nrm, floor * float(nrm.max()) + EPS)

    # -- write the image, read the label ----------------------------------
    def classify(self, img, topm=1):
        Q = unit_rows(img - img.mean(axis=1, keepdims=True))[0]
        S = self._bids(Q, self.img_ix)
        win = S.argmax(axis=1)
        if topm <= 1:
            prof = self.W[win][:, self.lab_ix]
        else:
            g = np.clip(S, 0.0, None)
            cut = np.argpartition(-g, topm, axis=1)[:, topm:]
            np.put_along_axis(g, cut, 0.0, axis=1)
            prof = g @ self.W[:self.n_boot][:, self.lab_ix]
        return prof.argmax(axis=1), win, prof

    # -- write the label, read the image ----------------------------------
    def generate(self, lab, topm=1):
        Q = unit_rows(lab - lab.mean(axis=1, keepdims=True))[0]
        S = self._bids(Q, self.lab_ix)
        if topm <= 1:
            win = S.argmax(axis=1)
            return self.W[win][:, self.img_ix], win
        g = np.clip(S, 0.0, None)
        cut = np.argpartition(-g, topm, axis=1)[:, topm:]
        np.put_along_axis(g, cut, 0.0, axis=1)
        g /= np.maximum(g.sum(axis=1, keepdims=True), EPS)
        return g @ self.W[:self.n_boot][:, self.img_ix], S.argmax(axis=1)

    def purity(self):
        """How class-pure the stored pairs are: the winner's share per template."""
        used = self.tally.sum(axis=1) > 0
        t = self.tally[used]
        return float((t.max(axis=1) / t.sum(axis=1)).mean()), int(used.sum())
