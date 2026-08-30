"""One hypercolumn whose templates try to explain the input *together*.

Every template sees every input. Each one learns from the part of the input
that the others left unexplained -- a leave-one-out residual -- and moves by
rotating on the unit sphere (the repo's geodesic step, experiments/2026_03_19/srl/learning4.py).

Preprocessing (same as everywhere else here): subtract the image's own mean,
then scale to unit length.

    x_hat = (x - mean(x)) / ||x - mean(x)||

Scores, reconstruction, and the leftover for template i:

    s      = W x_hat                  one score per template, signed
    recon  = W^T s                    everyone's contribution added up
    e      = x_hat - recon            what nobody explained
    others = recon - s_i w_i          what the *others* built
    r_i    = x_hat - others = e + s_i w_i

Three ways to turn r_i into a move (this is the thing under test):

  raw       rotate toward r_i as-is.
            The tangent projection kills the s_i w_i term outright:
                tau_i = r_i - (r_i . w_i) w_i = e - g_i w_i,  g_i = e . w_i
            so every template chases the *same* vector e and differs only by
            where it already sits. Averaged over data the drift is
            (I - w_i w_i^T)(I - W^T W) mu -- it only ever sees the mean image.
            Cheap, and the honest reading of "learn from the leftover".

  weighted  rotate toward s_i * r_i, i.e. tau_i = s_i (e - g_i w_i).
            Same direction, scaled by how loudly this template claimed the
            input -- and it *reverses* for s_i < 0. This is the actual
            gradient of ||x_hat - W^T s||^2 in w_i, so it is driven by second
            moments and cannot degenerate to the mean the way `raw` does.

  unit      the repo's rule (learning4.py): normalize the others' construction
            to unit length before subtracting, then flip by sign(s_i):
                r_i = sign(s_i) (x_hat - others / ||others||)
            Scale-free. The sign and the 1/||others|| both carry s_i, so this
            is data-coupled too.

`raw` and `weighted` need no (k, dim) temporaries at all -- the update is
    W <- diag(a) W + b e^T
which is why this runs at a few hundred microseconds per digit.
"""

import numpy as np

EPS = 1e-8
RULES = ("raw", "weighted", "unit")


def center_norm(v):
    """Subtract the mean, scale to unit length. Returns (x_hat, original norm)."""
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return (np.zeros_like(v), 0.0) if n < EPS else (v / n, n)


def center_norm_batch(X):
    """center_norm over rows of (n, dim). Returns (X_hat, norms)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1)
    ok = n > EPS
    Xh = np.zeros_like(Xc)
    Xh[ok] = Xc[ok] / n[ok, None]
    return Xh, n


class Hypercolumn:
    """`k` templates that reconstruct the input jointly.

    No competition and no winner: all templates score, all contribute to the
    reconstruction, and all learn on every input.
    """

    def __init__(self, k, dim, eta, rng, rule="weighted"):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        self.k, self.dim, self.eta, self.rule = k, dim, eta, rule
        W = rng.standard_normal((k, dim))
        W -= W.mean(axis=1, keepdims=True)
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)
        self.seen = 0

    # -- reading -----------------------------------------------------------
    def scores(self, Xh):
        """(n, dim) -> (n, k) scores. Plain dot products."""
        return Xh @ self.W.T

    def rebuild(self, S):
        """(n, k) scores -> (n, dim) reconstruction, everyone added up."""
        return S @ self.W

    # -- learning ----------------------------------------------------------
    def _rotate(self, tau, tau_norm):
        """Geodesic step: rotate each w_i by eta*||tau_i|| toward tau_i.

        tau is passed as (coefficients, basis vectors) so no (k, dim) array is
        ever built for the rank-1 rules. Here it arrives already assembled.
        """
        theta = self.eta * tau_norm
        live = tau_norm > EPS
        tau_hat = np.zeros_like(tau)
        tau_hat[live] = tau[live] / tau_norm[live, None]
        W = self.W * np.cos(theta)[:, None] + tau_hat * np.sin(theta)[:, None]
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)

    def learn_one(self, x_hat):
        """One input, one geodesic step for every template."""
        W = self.W
        s = W @ x_hat
        e = x_hat - s @ W                       # the shared leftover

        if self.rule in ("raw", "weighted"):
            g = W @ e                           # e . w_i
            # tau_i = e - g_i w_i is orthogonal to w_i, so its length is exact:
            #     ||tau_i||^2 = ||e||^2 - g_i^2
            ee = float(e @ e)
            tn_raw = np.sqrt(np.maximum(ee - g * g, 0.0))
            scale = s if self.rule == "weighted" else np.ones_like(s)
            live = (tn_raw > EPS) & (np.abs(scale) > EPS)
            theta = self.eta * tn_raw * np.abs(scale)
            # w_i <- cos(t) w_i + sin(t) * sign(scale_i) (e - g_i w_i)/||e - g_i w_i||
            step = np.zeros_like(tn_raw)
            step[live] = np.sin(theta[live]) * np.sign(scale[live]) / tn_raw[live]
            a = np.where(live, np.cos(theta) - step * g, 1.0)
            W = a[:, None] * W + step[:, None] * e[None, :]
            self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + EPS)
        else:                                   # unit -- learning4.py, verbatim
            recon = s @ W
            others = recon[None, :] - s[:, None] * W
            others_hat = others / (np.linalg.norm(others, axis=1, keepdims=True) + EPS)
            r = np.sign(s)[:, None] * (x_hat[None, :] - others_hat)
            tau = r - np.sum(r * W, axis=1, keepdims=True) * W
            self._rotate(tau, np.linalg.norm(tau, axis=1))

        self.seen += 1

    def learn_batch(self, Xh):
        """`weighted` only: one geodesic step per batch, summed over samples.

        sum_b s_b (e_b - g_b w_i) = M_i - (M_i . w_i) w_i  with  M = S^T E,
        so the batch move is just the tangent projection of a single matmul.
        """
        if self.rule != "weighted":
            raise ValueError("batching is only exact for rule='weighted'")
        W = self.W
        S = Xh @ W.T
        E = Xh - S @ W
        M = (S.T @ E) / len(Xh)
        tau = M - np.sum(M * W, axis=1, keepdims=True) * W
        self._rotate(tau, np.linalg.norm(tau, axis=1))
        self.seen += len(Xh)


class Codebook:
    """Control: the usual winner-take-all layer, where each template acts alone.

    Same template budget, same geodesic step -- but only the best-matching
    template learns, and reconstruction is that one template alone. This is
    the "codebook" reading the collective rule is being tested against.
    """

    def __init__(self, k, dim, eta, rng):
        self.k, self.dim, self.eta, self.rng = k, dim, eta, rng
        self.W = np.zeros((k, dim))
        self.n_boot = 0
        self.win_counts = np.zeros(k, dtype=np.int64)

    def learn_one(self, x_hat):
        if self.n_boot < self.k:                # adopt the first k inputs
            w = x_hat + 0.01 * self.rng.standard_normal(self.dim)
            w -= w.mean()
            self.W[self.n_boot] = w / (np.linalg.norm(w) + EPS)
            self.n_boot += 1
            return
        c = self.W @ x_hat
        i = int(np.argmax(c))
        self.win_counts[i] += 1
        w = self.W[i]
        tau = x_hat - float(c[i]) * w
        tn = float(np.linalg.norm(tau))
        if tn < EPS:
            return
        theta = self.eta * tn
        w = w * np.cos(theta) + (tau / tn) * np.sin(theta)
        self.W[i] = w / (np.linalg.norm(w) + EPS)

    def rebuild(self, Xh):
        """Winner only -- one template has to carry the whole digit."""
        C = Xh @ self.W.T
        win = C.argmax(axis=1)
        return np.maximum(C[np.arange(len(Xh)), win], 0.0)[:, None] * self.W[win]
