"""Two questions about Part three: why is generation dithered, and why do
layer-two's templates all look alike?"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stack3 import (ConvStack, BoundLayer, unit_rows, patches_of, l2_windows,
                    SIDE, GRID, PATCH, L2_GRID, WIN, EPS)
ROOT = HERE.parents[2]

X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
p = np.random.default_rng(0).permutation(len(X)); X, y = X[p], y[p]
Xtr, ytr, Xte = X[:20000], y[:20000], X[20000:25000]

rng = np.random.default_rng(0)
cs = ConvStack("dense", 64, 128, 0.5, rng)
cs.train(Xtr, 2, rng)


def stats(name, W, D):
    G = W @ W.T
    off = np.abs(G - np.diag(np.diag(G)))
    lam = np.linalg.eigvalsh(G)
    sv = np.linalg.svd(D, compute_uv=False)
    dim = float(sv.sum() ** 2 / (sv ** 2).sum())
    print(f"{name}: {len(W)} templates in {W.shape[1]} cells")
    print(f"   the data they see really occupies ~{dim:.0f} dimensions")
    print(f"   pairwise similarity  max {off.max():.3f}   mean {off.mean():.3f}")
    print(f"   effective rank of the layer {lam.sum()**2/(lam**2).sum():.1f}"
          f" / {len(W)}")
    n = len(W); k = dim
    print(f"   floor for {n} unit vectors in {k:.0f} dims (Welch): "
          f"{np.sqrt(max((n-k)/(k*(n-1)),0)):.3f}")


P1 = unit_rows(patches_of(Xtr[:2000]))[0]
P1 = P1[np.linalg.norm(P1, axis=1) > EPS]
F = cs.l1_field(Xtr[:2000])
P2 = unit_rows(l2_windows(F))[0]
P2 = P2[np.linalg.norm(P2, axis=1) > EPS]
stats("layer 1", cs.l1.W, P1[:4000])
print()
stats("layer 2", cs.l2.W, P2[:4000])

print("\n--- is W^T W the identity, as the decode assumes? ---")
for nm, L, d in (("layer 1", cs.l1, 16), ("layer 2", cs.l2, 576)):
    M = L.W.T @ L.W
    print(f"{nm}: decode(encode(x)) multiplies x by about "
          f"{np.trace(M)/d:.2f} (would be 1.0 for a perfect basis)")

print("\n--- where does the dither come from? ---")
Hte = cs.forward(Xte[:64])
rec_true = cs.backward(Hte, 64)                       # no layer three at all
img = unit_rows(Xte[:64].reshape(64, -1))[0]
r = unit_rows(rec_true.reshape(64, -1))[0]
print(f"image -> code -> image, layer 3 not involved: error "
      f"{np.linalg.norm(img - r, axis=1).mean():.3f}")
hf = lambda A: float(np.mean(np.abs(np.diff(A.reshape(-1, SIDE, SIDE), axis=2))))
print(f"   high-frequency content: original {hf(Xte[:64]):.4f}, "
      f"rebuilt {hf(rec_true):.4f}")

# what a flat offset alone decodes to -- layer three centres its input, so
# every retrieved code carries one.
ones = np.ones((1, L2_GRID * L2_GRID * 128))
off_img = cs.backward(ones, 1)[0]
print(f"   a CONSTANT code offset decodes to a pattern of size "
      f"{np.abs(off_img).max():.3f} with high-frequency content {hf(off_img):.4f}")
np.save("/tmp/claude-1000/-home-lavender-Projects-Geodesique/b740704d-75db-4cdd-bdff-0ecb9063c424/scratchpad/offset.npy", off_img)
