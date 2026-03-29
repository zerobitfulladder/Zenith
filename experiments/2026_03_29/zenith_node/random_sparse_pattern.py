"""RandomSparsePattern — cycles through a bank of random sparse binary vectors."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import OutputPort
from axonforge.core.descriptors.fields import Integer, Range
from axonforge.core.descriptors.displays import Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State


@branch("Inputs")
class RandomSparsePattern(Node):
    """Generates N sparse binary vectors and emits them one per tick, cycling."""

    output = OutputPort("Output", np.ndarray)
    pattern_index = OutputPort("Pattern Index", int)

    dim = Integer("Dim", default=200)
    amount = Integer("Amount", default=10)
    sparsity = Range(
        "Sparsity", default=0.10, min_val=0.0001, max_val=1.0, step=0.01, scale="log"
    )

    bank = State("Bank")
    idx = State("Index")

    info = Text("Info", default="idle")

    regenerate = Action("Regenerate", lambda self, params=None: self._on_regenerate(params))

    def init(self):
        self.bank = None
        self.idx = 0

    def _on_regenerate(self, _params=None):
        self._generate_bank()
        return {"status": "ok"}

    def _generate_bank(self):
        d = int(self.dim)
        n = int(self.amount)
        s = float(self.sparsity)
        n_active = max(1, int(round(d * s)))

        patterns = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            idx = np.random.choice(d, size=n_active, replace=False)
            patterns[i, idx] = 1.0

        self.bank = patterns
        self.idx = 0
        self.info = f"Generated {n} patterns, dim={d}, active={n_active}"

    def process(self):
        d = int(self.dim)
        n = int(self.amount)

        if self.bank is None or self.bank.shape[0] != n or self.bank.shape[1] != d:
            self._generate_bank()

        bank = np.asarray(self.bank, dtype=np.float64)
        i = int(self.idx) % n

        self.output = bank[i]
        self.pattern_index = i
        self.idx = (i + 1) % n
        self.info = f"Pattern {i}/{n}  dim={d}  sparsity={float(self.sparsity):.2f}"
