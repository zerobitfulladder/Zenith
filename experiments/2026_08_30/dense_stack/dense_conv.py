"""The 2026-08-29 convolutional stack with a DENSE layer one.

Same rig as `2026_08_29/conv_stack.py` in every respect but
one: layer one's hypercolumn no longer picks a winner. All 144 templates
score every patch, all of them learn from what the others left over
(the `weighted` rule from collective.py), and the code that goes upward
is 144 signed numbers instead of a short list of shape names.

That is the layer measured on 2026-08-30 as near-optimal at rebuilding
its input and completely meaningless template by template: an arbitrary
orthonormal basis of the patch subspace, reproducible only up to an
arbitrary rotation. This asks the question that matters for the
architecture -- whether anything can be built ON such a layer.

Layer two, the decode and the task are untouched, so any difference in
the result belongs to layer one.
"""

import sys
from pathlib import Path

import numpy as np

_A29 = Path(__file__).resolve().parents[2] / "2026_08_29"
for _sub in ("regress", "two_layer", "conv_stack"):
    sys.path.insert(0, str(_A29 / _sub))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collective"))

import pop_regress as P                                  # noqa: E402
import conv_stack as C                                   # noqa: E402
from collective import Hypercolumn as DenseColumn        # noqa: E402

EPS = 1e-8


class DensePatchLayer(C.PatchLayer):
    """conv_stack.PatchLayer with the winner-take-all column swapped out.

    The patch encoding, the taps, the per-patch mean reference and the
    cell positions are inherited unchanged -- only the column differs.
    """

    def __init__(self, k=144, taps=5, step=2, nb=64, halfw=6, dy=2.0,
                 size=1024, eta=0.5, seed=0, train_l1=True):
        super().__init__(k=k, taps=taps, step=step, nb=nb, halfw=halfw,
                         dy=dy, size=size, eta=eta, seed=seed)
        self.k = k
        self.train_l1 = train_l1
        self.dense = DenseColumn(k, size, eta, np.random.default_rng(seed + 1),
                                 rule="weighted")

    # -- the code ---------------------------------------------------------
    def _dense_vec(self, vals, mask=None, ref_mask=None):
        """A patch as a full centred unit vector, ready for the column."""
        idx, w, ref = self.encode_sparse(vals, mask, ref_mask)
        buf = np.zeros(self.size)
        buf[idx] = w
        buf -= buf.mean()
        n = np.linalg.norm(buf)
        return (buf / n if n > EPS else buf), ref

    def train(self, grid, val, known, n=20000, seed=0, ref_mask=None):
        """Every template learns from every patch -- no competition."""
        ok = self.valid(grid, known)
        rng = np.random.default_rng(seed)
        if not self.train_l1:               # ablation: leave them random
            return ok
        for i in rng.choice(ok, size=n):
            x_hat, _ = self._dense_vec(val[self.taps_at(int(i))],
                                       ref_mask=ref_mask)
            self.dense.learn_one(x_hat)
        return ok

    def code(self, grid, val, i, topm=None):
        """(indices, magnitudes, ref) -- all 144 of them, signed.

        `topm` is accepted and ignored: this layer is dense by definition,
        and trimming it would be the sparsity change this run is not making.
        """
        x_hat, ref = self._dense_vec(val[self.taps_at(int(i))])
        return np.arange(self.k), (self.dense.W @ x_hat).astype(np.float32), ref

    # -- reading a code back out ------------------------------------------
    def patch_from_code(self, c):
        """A dense code -> the 5 relative sample values it stands for.

        This is the layer's own rebuild, `recon = W^T c`, which is the one
        thing it is genuinely good at. Taking an argmax over the code and
        using that single template instead -- what the winner-take-all
        stack does -- would be meaningless here, because no single
        template stands for anything.
        """
        recon = np.asarray(c, dtype=np.float64) @ self.dense.W
        return np.array([P.sharpen(recon[self.pos[t]], self.centers)[1]
                         for t in range(self.T)])

    def template_patch(self, j):
        """One template on its own. Kept so the vocabulary can be pictured."""
        return np.array([P.sharpen(self.dense.W[j][self.pos[t]],
                                   self.centers)[1] for t in range(self.T)])


class DenseCodeWindowLayer(C.CodeWindowLayer):
    """Layer two, unchanged, with a decode that respects a dense code below.

    `CodeWindowLayer.template_shapes` reads each tap by argmax -- it asks
    "which template does layer one name here". That question has no answer
    when layer one is dense, so the tap slice is handed back whole and
    rebuilt through layer one's dictionary instead.
    """

    def tap_code(self, row, t):
        return row[self.pos[t]]

    def complete_dense(self, codes, ok, centre, l1):
        """Fill the taps with no layer-one code. Returns a shape per tap."""
        ti = self.taps_at(int(centre))
        mask = ok[ti]
        idx, w = self.encode_sparse(
            [codes[j] if ok[j] else None for j in ti], mask)
        s = self.hc.scores(idx, w, mode="masked")
        j = int(np.argmax(s))
        row = self.hc.row(j)
        shapes = {int(ti[t]): l1.patch_from_code(self.tap_code(row, t))
                  for t in np.nonzero(~mask)[0]}
        return shapes, j, float(s[j])

    def template_shapes(self, j, l1):
        row = self.hc.row(j)
        return [l1.patch_from_code(self.tap_code(row, t)) for t in range(self.T)]
