"""Zenith — Pearson correlation with top-1 winner-take-all on the unit hypersphere."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer, Bool
from axonforge.core.descriptors.displays import Heatmap, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State
from axonforge.core.descriptors import branch
from .utilities import to_display_grid


@branch("Hypercolumn/Zenith")
class Zenith(Node):
    """Zenith — top-1 winner-take-all on the unit hypersphere.

    Templates are competing whole-input hypotheses. The single template with the
    highest absolute Pearson correlation wins; all others receive no learning signal.
    Geodesic rotation moves the winner toward (positive) or away (negative) from the input.

    Each tick:
        1. L2-normalize input (assumed already mean-centered by LGN)
        2. Pearson correlations: c = W x_hat
        3. Top-1 by |c|: winner gets c_winner, all others zeroed
        4. Learning: winner rotates geodesically, scaled by c_winner
    """

    inputs = InputPort("Input", np.ndarray)

    amount = Integer("Amount", default=25)
    step_fraction = Range(
        "Step (η)", default=0.05, min_val=0.0, max_val=0.5, step=0.001
    )
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    raw_activations = OutputPort("Raw Activations", np.ndarray)
    activations = OutputPort("Activations", np.ndarray)
    input_shape = OutputPort("Input Shape", np.ndarray)

    w = State("Weights")

    weights_preview = Heatmap(
        "Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    activations_preview = Heatmap(
        "Activations", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    info = Text("Info", default="idle")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._input_shape = None
        self._dim = None
        self.w = None

    def _on_reset(self, _params=None):
        if self._dim is not None:
            self._init_weights(self._dim)
        return {"status": "ok"}

    def _init_weights(self, dim):
        self._dim = dim
        k = int(self.amount)
        w = np.random.randn(k, dim)
        w -= np.mean(w, axis=1, keepdims=True)
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self.w = w
        self.weights_preview = to_display_grid(self.w, patch_shape=self._input_shape)

    def process(self):
        eps = 1e-8

        if self.inputs is None:
            if self.w is not None:
                self.weights = self.w
            return

        raw_input = np.asarray(self.inputs, dtype=np.float64)
        if raw_input.ndim == 2:
            self._input_shape = raw_input.shape
        x = raw_input.ravel()

        if self.w is None or self._dim != x.size:
            self._init_weights(x.size)

        w = np.asarray(self.w, dtype=np.float64)
        k = w.shape[0]

        # 1. Mean-center and L2-normalize input
        x = x - np.mean(x)
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.weights = self.w
            self.activations = np.zeros(k)
            return
        x_hat = x / x_norm

        # 2. Pearson correlations
        c = w @ x_hat  # (k,)

        # 3. Top-1 by magnitude
        winner = np.argmax(np.abs(c))
        a = np.zeros_like(c)
        a[winner] = c[winner]

        # 4. Learning — geodesic rotation for winner only
        if self.is_learning:
            eta = float(self.step_fraction)
            c_w = c[winner]

            w_win = w[winner]  # (D,)
            tau = x_hat - c_w * w_win  # (D,)
            tau_norm = np.linalg.norm(tau)
            tau_hat = tau / tau_norm if tau_norm > eps else np.zeros_like(tau)

            theta = eta * c_w
            w[winner] = w_win * np.cos(theta) + tau_hat * np.sin(theta)

            self.w = w

        # 5. Outputs
        self.weights = self.w
        self.raw_activations = c
        self.activations = a
        if self._input_shape is not None:
            self.input_shape = np.array(self._input_shape, dtype=np.int64)

        self.weights_preview = to_display_grid(self.w, patch_shape=self._input_shape)
        self.activations_preview = to_display_grid(a, patch_shape=self._input_shape)
        w_cur = np.asarray(self.w, dtype=np.float64)
        norms = np.linalg.norm(w_cur, axis=1)
        means = np.mean(w_cur, axis=1)
        self.info = (
            f"k={k}  dim={self._dim}  ||x||={x_norm:.3f}\n"
            f"template norms  — min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}\n"
            f"template means  — min={means.min():.6f}  max={means.max():.6f}  mean={means.mean():.6f}"
        )

