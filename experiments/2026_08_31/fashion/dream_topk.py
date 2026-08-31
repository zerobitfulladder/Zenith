"""Is the graininess the templates, or the directions nobody uses?

Sample the same matched Gaussian but only along the coefficient directions that
actually carry variance for that expert; hold the rest at their mean. If the
dreams get clean, most of the 36 templates are still initialisation noise and
the real generative dimension is much smaller.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG, KTOP = 784, 8
RNG = np.random.default_rng(1)
z = np.load(OUT / "both_w_lam4.0.npz")
W, wins = z["W"].astype(np.float64), z["wins"]
alive = wins.sum(1) > 0
claim = np.where(alive, wins.argmax(1), -1)
Xtr, ytr, _, _ = load("fashion_mnist")
proto = center_norm(np.stack([Xtr[ytr == c].mean(0) for c in range(10)]))
proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), EPS)

picks = [int(np.where(claim == c)[0][wins[np.where(claim == c)[0]].sum(1).argmax()])
         for c in range(10) if (claim == c).any()]
fig, axes = plt.subplots(len(picks), 8, figsize=(8.4, 1.15 * len(picks)))
cos, eff = [], []
for r, h in enumerate(picks):
    c = claim[h]
    S = join(Xtr[ytr == c][:2000]) @ W[h].T
    mu, var = S.mean(0), S.var(0)
    pr = float(var.sum() ** 2 / np.maximum((var ** 2).sum(), EPS))   # effective rank
    eff.append(pr)
    top = np.argsort(var)[::-1][:KTOP]
    v = np.zeros_like(var); v[top] = np.sqrt(var[top])
    a = axes[r, 0]
    a.imshow(proto[c].reshape(28, 28), cmap="gray"); a.set_xticks([]); a.set_yticks([])
    a.set_title("class mean", fontsize=5.5, pad=1.5)
    a.set_ylabel(f"e{h}\n{CLASSES[c][:9]}\nrank {pr:.1f}", fontsize=5.5,
                 rotation=0, ha="right", va="center")
    for k in range(7):
        g = ((mu + v * RNG.standard_normal(len(var))) @ W[h])[:N_IMG]
        cs = float(g @ proto[c] / (np.linalg.norm(g) + EPS))
        cos.append(cs)
        a = axes[r, 1 + k]
        a.imshow(g.reshape(28, 28), cmap="gray"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{cs:+.2f}", fontsize=5.5, pad=1.5)
fig.suptitle(f"dreams sampled along the top {KTOP} coefficient directions only",
             fontsize=9)
fig.subplots_adjust(left=.11, right=.995, top=.91, bottom=.005, wspace=.06, hspace=.30)
fig.savefig(OUT / "72_dream_topk.png", dpi=140); plt.close(fig)
print(f"top-{KTOP} dream cosine {np.mean(cos):+.3f}   "
      f"(all 36 directions was +0.460, pure noise -0.024)")
print(f"effective rank of the coefficient variance: "
      f"{np.mean(eff):.1f} of 36 templates  {[round(e,1) for e in eff]}")
json.dump({"topk": KTOP, "cos": float(np.mean(cos)),
           "effective_rank": eff}, open(OUT / "dream_topk.json", "w"), indent=2)
