"""One arm: sparse k=4, layer three scoring in the dictionary's geometry.

Everything else identical to `run_sparse_stack.py`, so the number is
directly comparable to that run's 0.7494 (same substrate, cell-wise score)
and 0.9232 (dense).
"""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_30" / "stack3"))
from stack3 import BoundLayer                                    # noqa: E402
from sparse_stack import GramStack                               # noqa: E402
from run_sparse_stack import (load, onehot, norm_img, OUT, K1, K2, K3,
                              ETA1, ETA3, EPOCHS, EPOCHS3, RHO, SEED)  # noqa

KMAX = 4
BASELINE = {"sparse k=4, cell-wise score": 0.7494, "dense": 0.9232,
            "wta": 0.5708}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    judge = LogisticRegression(max_iter=2000).fit(norm_img(Xtr), ytr)
    rng = np.random.default_rng(SEED)
    cs = GramStack(K1, K2, KMAX, ETA1, rng)
    cs.train(Xtr, EPOCHS, rng)

    Htr, Hte = cs.forward_gram(Xtr), cs.forward_gram(Xte)
    print(f"layers 1-2 done in {time.time()-t0:.0f}s; "
          f"layer-three input {Htr.shape[1]} cells "
          f"({(Hte > 0).mean():.3f} lit)")

    rng3 = np.random.default_rng(SEED + 1)
    l3 = BoundLayer(K3, Htr.shape[1], 10, RHO, ETA3, rng3)
    for _ in range(EPOCHS3):
        o = rng3.permutation(len(Htr))
        for s in range(0, len(o), 256):
            b = o[s:s + 256]
            l3.learn(Htr[b], onehot(ytr[b]), ytr[b])

    pred, _, _ = l3.classify(Hte)
    acc = float((pred == yte).mean())
    pur, used = l3.purity()
    gen = {}
    for tag, m in (("top-1", 1), ("graded", 32)):
        code, _ = l3.generate(onehot(np.arange(10)), topm=m)
        imgs = cs.backward_gram(code, 10)
        jl = judge.predict(norm_img(imgs))
        gen[tag] = (float((jl == np.arange(10)).mean()), imgs, jl)

    print(f"\n  gram-space score   accuracy {acc:.4f}   purity {pur:.3f}   "
          f"gen top-1 {gen['top-1'][0]:.1f}  graded {gen['graded'][0]:.1f}")
    for k, v in BASELINE.items():
        print(f"  {k:<28} {v:.4f}   ({acc - v:+.4f})")

    fig, axes = plt.subplots(2, 10, figsize=(10, 2.3))
    for r, tag in enumerate(("top-1", "graded")):
        _, imgs, jl = gen[tag]
        for d in range(10):
            a = axes[r, d]
            a.imshow(imgs[d], cmap="gray"); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#2ca02c" if jl[d] == d else "#d62728")
                sp.set_linewidth(1.8)
            if d == 0:
                a.set_ylabel(tag, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"gram-space layer three, sparse k=4 — accuracy {acc:.4f}",
                 fontsize=10)
    fig.subplots_adjust(left=.08, right=.99, top=.85, bottom=.02,
                        wspace=.05, hspace=.06)
    fig.savefig(OUT / "03_gram_generated.png", dpi=135); plt.close(fig)

    (OUT / "gram.json").write_text(json.dumps(
        {"kmax": KMAX, "l3_input_dim": int(Htr.shape[1]),
         "l3_fraction_lit": float((Hte > 0).mean()),
         "accuracy": acc, "purity": pur, "templates_used": used,
         "gen_top1": gen["top-1"][0], "gen_graded": gen["graded"][0],
         "baselines": BASELINE, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
