"""A hypercolumn where k minicolumns speak, in sequence, and only they learn.

Nonnegative matching pursuit. The residual starts as the whole input; the
best-matching template takes its share, that share is subtracted, and the
next template answers what is left. Each unit explains alone, on its own
turn -- never a committee handed a shared leftover, which is the 08-30
dense rule that went to speckle.

    r = x
    repeat up to kmax times:
        i   = argmax_i (w_i . r),  positive matches only
        tau = r - (w_i . r) w_i
        rotate w_i toward relu(tau);  r = tau
        stop when ||r|| < eps ||x||

`tau` is the same vector twice over: the tangent at w_i pointing toward r,
which is what the geodesic step rotates along, AND the residual passed to
the next winner. Setting kmax=1 recovers the winner-take-all rule exactly.

Everything is batched. The kmax loop cannot be vectorised -- step t+1 needs
step t's choice -- but it is the ONLY python-level loop: kmax iterations of
(n,d)@(d,K), not n*kmax of anything. The per-template update is a one-hot
matmul, so the whole rule is matmul / argmax / gather / scatter-add and
ports to a GPU unchanged.
"""

import numpy as np

EPS = 1e-12
MAX_STEP = np.pi / 4          # cap on one rotation, radians


class SparseColumn:
    """k-of-K nonnegative matching pursuit, with geodesic learning."""

    def __init__(self, n_t, dim, kmax=4, eta=0.5, eps_stop=0.05,
                 nonneg_w=False, rng=None):
        """`nonneg_w` clamps the template WEIGHTS nonnegative. Off by
        default: it makes every template point into the same orthant, the
        dictionary becomes coherent, and the pursuit stalls after one pick
        (measured in the README). Coefficients are nonnegative either way --
        the rule refuses a negative match -- so the message leaving this
        column is nonnegative regardless."""
        self.n_t, self.dim, self.kmax = n_t, dim, kmax
        self.eta, self.eps_stop, self.nonneg = eta, eps_stop, nonneg_w
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.W = np.zeros((n_t, dim))
        self.n_boot = 0
        self.wins = np.zeros(n_t)

    # -- the pursuit ------------------------------------------------------
    def pursue(self, U, tangents=False):
        """(n, dim) unit rows -> code, residual, how many each one used.

        `tangents` also returns the per-template accumulated learning
        target and how many samples contributed to it.
        """
        n, K = len(U), self.n_t
        rows = np.arange(n)
        R = np.array(U, dtype=np.float64, copy=True)
        C = np.zeros((n, K))
        used = np.zeros(n, dtype=np.int32)
        e0 = np.maximum((U * U).sum(1), EPS)
        Tau = np.zeros((K, self.dim)) if tangents else None
        cnt = np.zeros(K) if tangents else None
        alive = np.ones(n, dtype=bool)

        for _ in range(self.kmax):
            if not alive.any():
                break
            S = R @ self.W.T                       # (n, K)  one GEMM
            i = S.argmax(1)
            c = S[rows, i]
            step = alive & (c > 0.0)               # refuse negative matches
            if not step.any():
                break
            tau = R - c[:, None] * self.W[i]       # tangent AND next residual

            if tangents:
                oh = np.zeros((n, K))
                oh[rows[step], i[step]] = 1.0
                tgt = np.maximum(tau, 0.0) if self.nonneg else tau
                Tau += oh.T @ tgt                  # scatter as a GEMM
                cnt += oh.sum(0)

            C[rows[step], i[step]] += c[step]
            R[step] = tau[step]
            used += step
            alive = step & ((R * R).sum(1) > (self.eps_stop ** 2) * e0)

        return C, R, used, Tau, cnt

    # -- learning ---------------------------------------------------------
    def learn(self, U):
        """One batch. Bootstraps from the data, then rotates the winners."""
        if self.n_boot < self.n_t:
            take = min(self.n_t - self.n_boot, len(U))
            w = U[:take] + 0.01 * self.rng.standard_normal((take, self.dim))
            if self.nonneg:
                w = np.maximum(w, 0.0)
            self.W[self.n_boot:self.n_boot + take] = (
                w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS))
            self.n_boot += take
            U = U[take:]
            if not len(U):
                return
        _, _, _, Tau, cnt = self.pursue(U, tangents=True)
        self._rotate(Tau, cnt)
        self.wins += cnt

    def _rotate(self, Tau, cnt):
        """One geodesic step per template, toward its mean tangent."""
        W = self.W
        T = Tau / np.maximum(cnt, 1.0)[:, None]
        T = T - (T * W).sum(1, keepdims=True) * W      # relu broke tangency
        tn = np.linalg.norm(T, axis=1)
        live = (tn > EPS) & (cnt > 0)
        th = np.where(live, np.clip(self.eta * tn, 0.0, MAX_STEP), 0.0)
        D = np.zeros_like(W)
        D[live] = T[live] / tn[live, None]
        W = W * np.cos(th)[:, None] + D * np.sin(th)[:, None]
        if self.nonneg:
            W = np.maximum(W, 0.0)
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)

    # -- the other way to pick k: let them settle ------------------------ #
    def settle(self, U, lam, n_iter=64, step=0.2):
        """LCA (Rozell et al. 2008). Every template scores at once, they
        inhibit each other in proportion to how much they overlap, and the
        loop settles. `G` off-diagonal IS the lateral inhibition, and it is
        explaining-away computed in parallel instead of in turns.

            u <- u + step * (Wx - u - (WW' - I) a),   a = relu(u - lam)

        Unlike the pursuit this can revise: a template that grabbed early
        can be pushed back down as the picture settles. Every iteration is
        one (n,K)@(K,K) GEMM -- no argmax, no gather, no scatter.
        """
        G = self.W @ self.W.T
        np.fill_diagonal(G, 0.0)                   # rows are unit, so this is G-I
        b = U @ self.W.T
        u = np.zeros_like(b)
        for _ in range(n_iter):
            u += step * (b - u - np.maximum(u - lam, 0.0) @ G)
        return np.maximum(u - lam, 0.0)

    def learn_settled(self, U, lam, n_iter=64, step=0.2):
        """Settle, then every active template rotates toward the residual,
        weighted by how loudly it spoke. That is the 08-30 `weighted` rule
        -- the true gradient of the rebuild error -- now with a sparse code
        under it, so the rotation symmetry it used to suffer is broken."""
        if self.n_boot < self.n_t:
            SparseColumn.learn(self, U)
            return
        A = self.settle(U, lam, n_iter, step)
        R = U - A @ self.W
        self._rotate(A.T @ R, A.sum(0))            # scatter is ONE gemm
        self.wins += (A > 0).sum(0)

    def revive(self, U, floor=1e-3):
        """Re-seed templates nobody ever picks onto the worst-explained input."""
        share = self.wins / max(self.wins.sum(), 1.0)
        dead = np.nonzero(share < floor / self.n_t)[0]
        if not len(dead):
            return 0
        _, R, _, _, _ = self.pursue(U)
        worst = np.argsort(-(R * R).sum(1))[:len(dead)]
        w = U[worst] + 0.01 * self.rng.standard_normal((len(dead), self.dim))
        if self.nonneg:
            w = np.maximum(w, 0.0)
        self.W[dead] = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)
        self.wins[dead] = self.wins.mean()
        return len(dead)

    # -- reading ----------------------------------------------------------
    def encode(self, U, norms=None):
        C = self.pursue(U)[0]
        return C if norms is None else C * norms[:, None]

    def decode(self, C):
        return C @ self.W


# --------------------------------------------------------------- naive ---
class NaiveColumn(SparseColumn):
    """Identical rule, one sample at a time. Kept only to time against."""

    def learn(self, U):
        if self.n_boot < self.n_t:
            SparseColumn.learn(self, U)
            return
        for x in U:
            r = x.copy()
            e0 = max(float(x @ x), EPS)
            for _ in range(self.kmax):
                s = self.W @ r
                i = int(s.argmax())
                if s[i] <= 0.0:
                    break
                tau = r - s[i] * self.W[i]
                tgt = np.maximum(tau, 0.0) if self.nonneg else tau
                tgt = tgt - (tgt @ self.W[i]) * self.W[i]
                tn = float(np.linalg.norm(tgt))
                if tn > EPS:
                    th = min(self.eta * tn, MAX_STEP)
                    w = self.W[i] * np.cos(th) + (tgt / tn) * np.sin(th)
                    if self.nonneg:
                        w = np.maximum(w, 0.0)
                    self.W[i] = w / (np.linalg.norm(w) + EPS)
                self.wins[i] += 1
                r = tau
                if float(r @ r) <= (self.eps_stop ** 2) * e0:
                    break
