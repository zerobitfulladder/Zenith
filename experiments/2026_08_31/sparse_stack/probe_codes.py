"""Are the codes themselves classifiable, or is layer three the problem?

Bypasses layer three entirely: 1-nearest-neighbour on the layer-two codes.
If dense codes are far more classifiable than sparse ones, the information
really is missing. If they are close, layer three is what loses the points.
Also asks the question directly: do two images of the same digit produce
similar codes?
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_30" / "stack3"))
from stack3 import ConvStack                                     # noqa: E402
from sparse_stack import SparseStack, calibrate                  # noqa: E402
from run_sparse_stack import load, K1, K2, ETA1, EPOCHS, SEED    # noqa: E402


def unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-12)


def stats(Htr, ytr, Hte, yte, tag, chunk=500):
    U, V = unit(Htr), unit(Hte)
    hit = 0
    for s in range(0, len(V), chunk):                 # 1-NN, cosine
        S = V[s:s + chunk] @ U.T
        hit += int((ytr[S.argmax(1)] == yte[s:s + chunk]).sum())
    nn = hit / len(V)
    rng = np.random.default_rng(0)                    # same/different class cos
    a, b = rng.integers(0, len(V), 20000), rng.integers(0, len(V), 20000)
    ok = a != b
    a, b = a[ok], b[ok]
    cos = (V[a] * V[b]).sum(1)
    same = yte[a] == yte[b]
    w, x = float(cos[same].mean()), float(cos[~same].mean())
    print(f"{tag:<14} 1-NN {nn:.4f}   same-class cos {w:.4f}   "
          f"other-class {x:.4f}   separation {w - x:+.4f}   "
          f"lit {float((Hte > 0).mean()):.3f}")
    return {"nn_accuracy": nn, "same_class_cos": w, "other_class_cos": x,
            "separation": w - x, "fraction_lit": float((Hte > 0).mean())}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    out = {}
    for tag in ("dense", "wta", "sparse k=4", "sparse k=4 settled"):
        rng = np.random.default_rng(SEED)
        if tag in ("dense", "wta"):
            cs = ConvStack(tag, K1, K2, ETA1, rng)
        else:
            cs = SparseStack(K1, K2, 4, ETA1, rng,
                             read="settle" if "settled" in tag else "pursue")
        cs.train(Xtr, EPOCHS, rng)
        Htr, Hte = cs.forward(Xtr), cs.forward(Xte)
        out[tag] = stats(Htr, ytr, Hte, yte, tag)
        if isinstance(cs, SparseStack):
            for nm, W in (("l1", cs.l1.W), ("l2", cs.l2.W)):
                G = W @ W.T
                out[tag][f"{nm}_coherence"] = float(
                    G[~np.eye(len(W), dtype=bool)].mean())
            print(f"{'':<14} dictionary coherence  l1 "
                  f"{out[tag]['l1_coherence']:+.4f}   l2 "
                  f"{out[tag]['l2_coherence']:+.4f}")
    (HERE / "results" / "probe_codes.json").write_text(json.dumps(out, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
