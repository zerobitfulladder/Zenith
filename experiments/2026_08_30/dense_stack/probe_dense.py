"""Why does a layer of meaningless templates still feed a working layer two?

Three things to separate:
  1. did layer one's learning actually do anything at layer one?
  2. how far did the templates move from where they started?
  3. does the code preserve the geometry of the patches -- i.e. do two
     similar patches still produce two similar codes?

(3) is the candidate mechanism. Layer two never reads a template's name;
it reads the whole 144-vector. Any rotation of the templates preserves
every inner product between codes, so overlap survives even though
identity does not.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "2026_08_29" / "regress"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_29" / "two_layer"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_29" / "conv_stack"))
import pop_regress as P, conv_stack as C, two_layer as T          # noqa: E402
from dense_conv import DensePatchLayer                            # noqa: E402

P.configure(x_range=(-12, 12), y_range=(-6, 6), gap=(1.2, 2.8),
            target_name="wave_plus_trend")
xtr, ytr, *_ = P.make_dataset(n_train=1600, seed=0)
g = T.tap_grid(0.2)
val, known = C.sample_raw(g, xtr, ytr)

tr = DensePatchLayer(k=144, seed=0, train_l1=True);  tr.train(g, val, known, n=20000, seed=0)
rd = DensePatchLayer(k=144, seed=0, train_l1=False); rd.train(g, val, known, n=20000, seed=0)
pc = DensePatchLayer(k=144, seed=0, train_l1=False)

pos = tr.valid(g, known)
X = np.array([tr._dense_vec(val[tr.taps_at(int(i))])[0] for i in pos])
print(f"{len(X)} patches, {X.shape[1]} cells; patch subspace holds "
      f"{np.linalg.matrix_rank(X, tol=1e-6)} real dimensions\n")

_Xp = np.array([pc._dense_vec(val[pc.taps_at(int(i))])[0]
                for i in pc.valid(g, known)])
pc.dense.W = np.linalg.svd(_Xp, full_matrices=False)[2][:144]

for name, l1 in (("trained", tr), ("untrained", rd), ("pca", pc)):
    W = l1.dense.W
    S = X @ W.T
    rec = np.linalg.norm(X - S @ W, axis=1).mean()
    G = W @ W.T; off = np.abs(G - np.diag(np.diag(G)))
    lam = np.linalg.eigvalsh(G)
    # does code-space similarity track patch-space similarity?
    Cx = np.corrcoef(X); Cs = np.corrcoef(S)
    iu = np.triu_indices(len(X), 1)
    geo = np.corrcoef(Cx[iu], Cs[iu])[0, 1]
    print(f"{name:10s} rebuild error {rec:.4f}   effective rank "
          f"{lam.sum()**2/(lam**2).sum():5.1f}/144   max pair similarity "
          f"{off.max():.3f}")
    rms = np.sqrt((S ** 2).mean(axis=0))
    rms = np.sort(rms)[::-1]
    print(f"{'':10s} patch similarity -> code similarity: {geo:.4f}")
    print(f"{'':10s} code energy across the 144 entries: strongest "
          f"{rms[0]:.4f}, median {np.median(rms):.4f}, weakest {rms[-1]:.2e}"
          f"   (spread {rms[0]/max(np.median(rms),1e-12):.0f}x)")

moved = np.degrees(np.arccos(np.clip(
    np.sum(tr.dense.W * rd.dense.W, axis=1), -1, 1)))
print(f"\ntemplates rotated from where they started: median "
      f"{np.median(moved):.1f} deg, max {moved.max():.1f} deg")
