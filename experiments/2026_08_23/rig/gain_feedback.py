"""Standalone two-layer Zenith with Variant A gain feedback (no AxonForge).

L1 is the Zenith unit (top-1 WTA, geodesic rotation on the zero-mean unit
sphere), processing a gain-modulated pixel input. L2 holds one class template
per label over L1's code space; the label selects which L2 template learns
(the "feedback from a hypothetical solved layer above" boundary condition).

Feedback (Variant A, from experiments/2026_03_20/feedback/Thoughts2.md "Hierarchical Gain-Feedback"):
the true class's L2 template, projected through L1's own weights, becomes a
multiplicative gain field over pixel space, applied to the mean-centered input
before normalization. The label never touches L1's weights directly — its only
influence on L1 is through what L1 sees.
"""

import numpy as np

EPS = 1e-8


def center_norm(v):
    """Mean-center and L2-normalize. Returns (v_hat, norm); v_hat is zeros if degenerate."""
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v), 0.0
    return v / n, n


class ZenithLayer:
    """Top-1 WTA spherical quantizer (standalone port of experiments/2026_03_29/zenith_node/learning6.py).

    Two deviations from learning6.py, both required for raw MNIST:
    - Bootstrap init: the first k inputs are adopted directly as templates
      (Forgy init) instead of random Gaussians. MNIST lives in a narrow cone
      on the sphere; random templates lose the race to the first template
      that reaches the cone (the single-template monopoly seen live in
      AxonForge). Bootstrapping starts every template inside the cone.
    - Winner by argmax(c) rather than argmax(|c|): with bootstrap init all
      real matches are positive, and the anti-correlated winner channel is
      itself a monopoly route on cone-shaped data.
    """

    def __init__(self, k, dim, eta, rng):
        self.k = k
        self.dim = dim
        self.eta = eta
        self.rng = rng
        self.W = np.zeros((k, dim))
        self.n_boot = 0
        self.win_counts = np.zeros(k, dtype=np.int64)

    def forward(self, x_hat):
        """Correlations of x_hat with all templates (zeros for unadopted rows)."""
        return self.W @ x_hat

    def learn(self, x_hat, c, step_gain=None, k_active=1):
        """Adopt (bootstrap phase) or rotate the k_active best matches toward x_hat.

        step_gain: optional (k,) plasticity multipliers (Variant B). Selection
        always comes from raw correlations — feedback never decides who fires,
        only how much a firing template learns (the V12 rule: perception
        honest, plasticity gated).

        k_active=1 is classic Zenith (only the winner learns). k_active>1 is
        graded learning: the top-k positively-correlated templates each rotate
        with step eta * c_i (* step_gain[i]) — this is what gives a plasticity
        gate a population to redistribute over. Usage stats still count only
        the argmax winner, so entropy is comparable across modes.

        Returns the winner index.
        """
        if self.n_boot < self.k:
            w = x_hat + 0.01 * self.rng.standard_normal(self.dim)
            w -= w.mean()
            w /= np.linalg.norm(w) + EPS
            i = self.n_boot
            self.W[i] = w
            self.n_boot += 1
            self.win_counts[i] += 1
            return i

        order = np.argsort(c)[::-1][:k_active]
        for i in order:
            c_i = float(c[i])
            if c_i <= 0.0:
                break
            w = self.W[i]
            tau = x_hat - c_i * w
            tau_n = float(np.linalg.norm(tau))
            if tau_n > EPS:
                theta = self.eta * c_i
                if step_gain is not None:
                    theta *= float(step_gain[i])
                self.W[i] = w * np.cos(theta) + (tau / tau_n) * np.sin(theta)
        winner = int(order[0])
        self.win_counts[winner] += 1
        return winner


class ClassTemplates:
    """L2: one unit-norm template per class over L1's code space.

    Supervised assignment — the label selects the row, no competition. The
    selected row rotates toward the current L1 code with step proportional
    to misalignment (1 - c): far templates move fast, settled ones barely
    move. Rows stay zero-mean unit-norm by the same rotation argument as L1.
    """

    def __init__(self, n_classes, dim, eta, rng):
        W = rng.standard_normal((n_classes, dim))
        W -= W.mean(axis=1, keepdims=True)
        W /= np.linalg.norm(W, axis=1, keepdims=True) + EPS
        self.W = W
        self.eta = eta

    def forward(self, h_hat):
        return self.W @ h_hat

    def learn(self, h_hat, y):
        w = self.W[y]
        c_y = float(w @ h_hat)
        tau = h_hat - c_y * w
        tau_n = float(np.linalg.norm(tau))
        if tau_n > EPS:
            theta = self.eta * (1.0 - c_y)
            self.W[y] = w * np.cos(theta) + (tau / tau_n) * np.sin(theta)


def gain_field(w2_row, W1, gamma, floor=0.05):
    """Variant A gain field for one class: project the class template through
    L1's weights into pixel space, z-score, center the multiplier at 1.

    Deviation from the note's formula: normalized by std instead of vector
    norm, so gamma reads directly as rms fractional modulation (gamma=0.5
    means pixels are brightened/dimmed by ~50% rms). gamma=0 is exactly
    neutral (all-ones). The floor keeps the gain nonnegative.
    """
    g = w2_row @ W1
    z = (g - g.mean()) / (g.std() + EPS)
    return np.clip(1.0 + gamma * z, floor, None)


def plasticity_gain(source, gamma, floor=0.05):
    """Variant B gain: z-score a relevance vector over L1 template indices and
    center the multiplier at 1. Same recipe as gain_field but with no
    projection to pixel space — the gain scales learning steps and never
    touches what L1 sees, so neither the train/test input mismatch nor the
    collapse-through-W1 channel can exist."""
    z = (source - source.mean()) / (source.std() + EPS)
    return np.clip(1.0 + gamma * z, floor, None)


def encode_batch(X, W1, W2):
    """Feedforward pass for evaluation, neutral gain (no label available).

    X: (N, D) raw images. Returns (A1 relu codes, C2 class correlations).
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + EPS
    A1 = np.maximum(Xc @ W1.T, 0.0)
    H = A1 - A1.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    C2 = H @ W2.T
    return A1, C2


def class_gain_distinctness(W2, W1):
    """How different the 10 downward class projections are from each other:
    mean pairwise cosine distance of the centered pixel-space gain directions.
    ~0 means the gain says the same thing for every class (uninformative);
    grows as the feedback becomes class-specific (the self-bootstrap claim).
    """
    G = W2 @ W1
    G = G - G.mean(axis=1, keepdims=True)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + EPS
    S = G @ G.T
    iu = np.triu_indices(G.shape[0], 1)
    return float(1.0 - S[iu].mean())
