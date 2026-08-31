"""Did the repulsion damage the model? (the nanmin the first pass got wrong)

For every test image:
  own    = error of the best expert that claims the true class   <- generative quality
  other  = error of the best expert that claims something else   <- how much it
                                                                    trespasses
A healthy change makes `own` stay put and `other` go up.
"""
import json, warnings
from pathlib import Path
import numpy as np
from common import load, join, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG = 784
warnings.filterwarnings("ignore", message="All-NaN")


def errs(W, X, chunk=2000):
    h, k, d = W.shape
    E = np.empty((len(X), h))
    for s in range(0, len(X), chunk):
        Q = join(X[s:s + chunk])
        S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
        R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
        E[s:s + chunk] = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG],
                                        axis=2) / np.maximum(
            np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return E


z = np.load(OUT / "dopamine_weights.npz")
_, _, Xte, yte = load("fashion_mnist")
out = {}
for tag in ("none", "B"):
    W = z[f"W_{tag}"].astype(np.float64); wins = z[f"wins_{tag}"]
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = errs(W, Xte)
    own = (claim[None, :] == yte[:, None]) & alive[None]
    oth = (~own) & alive[None]
    o = np.nanmin(np.where(own, E, np.nan), axis=1)
    t = np.nanmin(np.where(oth, E, np.nan), axis=1)
    allm = np.nanmean(np.where(alive[None], E, np.nan), axis=1)
    per = {CLASSES[c]: [float(np.nanmean(o[yte == c])), float(np.nanmean(t[yte == c]))]
           for c in range(10)}
    out[tag] = {"own": float(np.nanmean(o)), "other": float(np.nanmean(t)),
                "gap": float(np.nanmean(t - o)),
                "own_beats_other": float(np.nanmean(o < t)),
                "mean_over_all_experts": float(np.nanmean(allm)),
                "per_class_own_other": per}
    print(f"[{tag}]  own-class rebuild {out[tag]['own']:.4f}   "
          f"best other-class {out[tag]['other']:.4f}   "
          f"gap {out[tag]['gap']:+.4f}   "
          f"own is best for {out[tag]['own_beats_other']*100:.1f}% of images   "
          f"mean over all experts {out[tag]['mean_over_all_experts']:.4f}")
d = json.loads((OUT / "dopamine_metrics.json").read_text())
d["rebuild_cost"] = out
(OUT / "dopamine_metrics.json").write_text(json.dumps(d, indent=2))
print("\nper class  own / best-other:")
for c in CLASSES:
    a, b = out["none"]["per_class_own_other"][c]
    x, y = out["B"]["per_class_own_other"][c]
    print(f"  {c:<12} none {a:.3f} / {b:.3f}      B {x:.3f} / {y:.3f}")

# ---- the cost figure (accuracy lives only in the board, see board.py) ------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
res = d["results"]
arms = ["none", "A", "B", "A+B"]
content = {"none": "none", "A": "none", "B": "B", "A+B": "B"}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
w = .38
ax[0].bar(np.arange(4) - w/2, [out[content[a]]["own"] for a in arms], width=w,
          color="#2f8f4e", label="own class (lower = model intact)")
ax[0].bar(np.arange(4) + w/2, [out[content[a]]["other"] for a in arms], width=w,
          color="#c1462d", label="best other class (higher = less trespassing)")
for i, a in enumerate(arms):
    ax[0].text(i, out[content[a]]["other"] + .01,
               f"gap {out[content[a]]['gap']:+.3f}", ha="center", fontsize=6.5)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(arms); ax[0].set_ylim(0, 1.0)
ax[0].set_ylabel("relative rebuild error"); ax[0].legend(fontsize=6.5, loc="lower left")
ax[0].grid(alpha=.25, axis="y")
ax[0].set_title("what repulsion cost the model", fontsize=9)
o = np.argsort([res["none"]["per_class"][c] for c in CLASSES])
for j_, a in enumerate(arms):
    ax[1].barh(np.arange(10) + j_ * .22, [res[a]["per_class"][CLASSES[c]] for c in o],
               height=.21, label=a)
ax[1].set_yticks(np.arange(10) + .33)
ax[1].set_yticklabels([CLASSES[c] for c in o], fontsize=7)
ax[1].set_xlim(0, 1); ax[1].legend(fontsize=7); ax[1].grid(alpha=.25, axis="x")
ax[1].set_title("per class", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "40_cost.png", dpi=140); plt.close(fig)
print("wrote 40_cost.png")
