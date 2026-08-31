"""What we are actually spending, and whether any of it is idle.

Shape of the model: H hypercolumns (we have been calling them experts), each a
group of K minicolumns, each minicolumn one template vector of D = 784 + 10.

Three questions answered without retraining anything:
  1. parameter count against the baselines
  2. redundant hypercolumns -- keep only the N most used, does accuracy hold?
  3. idle minicolumns   -- keep only the k highest-variance templates per
                          hypercolumn, does accuracy hold?
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG = 784

z = np.load(OUT / "coef_stats.npz")
W, claim, wins, cov = (z["W"].astype(np.float64), z["claim"], z["wins"],
                       z["cov"].astype(np.float64))
H, K, D = W.shape
Xtr, ytr, Xte, yte = load("fashion_mnist")


def errs(Wx, X, chunk=2000):
    h, k, d = Wx.shape
    E = np.empty((len(X), h))
    for s in range(0, len(X), chunk):
        Q = join(X[s:s + chunk])
        S = (Q @ Wx.reshape(h * k, d).T).reshape(len(Q), h, k)
        R = np.matmul(S.transpose(1, 0, 2), Wx).transpose(1, 0, 2)
        E[s:s + chunk] = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG],
                                        axis=2) / np.maximum(
            np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return E


def acc_from(E, keep):
    e = np.where(keep[None], E, np.inf)
    return float((claim[e.argmin(1)] == yte).mean())


# ---- 1. parameters --------------------------------------------------------
n_tr = len(Xtr)
params = {
    "ours (40 hypercolumns x 36 minicolumns x 794)": H * K * D,
    "  + reluctance (one per hypercolumn)": H,
    "  + coefficient mean & covariance": H * (K + K * K),
    "logistic": 784 * 10 + 10,
    "linear SVM": 784 * 10 + 10,
    "MLP 256": 784 * 256 + 256 + 256 * 10 + 10,
    "MLP 512-256": 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10,
    "kNN k=3 (stores the training set)": n_tr * 784,
}
print("parameters")
for k_, v in params.items():
    print(f"  {k_:<46} {v:>12,}")

# ---- 2. are hypercolumns redundant? --------------------------------------
E = errs(W, Xte)
alive = wins.sum(1) > 0
E = np.where(alive[None], E, np.inf)
use = np.bincount(E.argmin(1), minlength=H) / len(yte)
order = np.argsort(use)[::-1]
full = acc_from(E, alive)
curve_h = []
for n in range(1, int(alive.sum()) + 1):
    keep = np.zeros(H, bool); keep[order[:n]] = True
    curve_h.append(acc_from(E, keep & alive))
print(f"\nhypercolumns: {int(alive.sum())} alive of {H}, "
      f"busiest takes {use.max()*100:.1f}% of the test set")
for n in (5, 10, 15, 20, 25, 30, int(alive.sum())):
    if n <= len(curve_h):
        print(f"  keep the {n:>2} most used -> {curve_h[n-1]:.4f}"
              f"   ({curve_h[n-1]-full:+.4f})")

ov = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
            np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                    (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
      for i in np.where(alive)[0] for j in np.where(alive)[0] if i < j]
same = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
              np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                      (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
        for i in np.where(alive)[0] for j in np.where(alive)[0]
        if i < j and claim[i] == claim[j]]
print(f"  pairwise subspace overlap: all pairs {np.mean(ov):.4f}, "
      f"pairs claiming the SAME class {np.mean(same):.4f}   (1.0 = duplicates)")

# ---- 3. are minicolumns idle? --------------------------------------------
curve_k = []
for k in (2, 4, 6, 9, 12, 18, 24, 30, 36):
    Wk = np.zeros_like(W)
    for h in range(H):
        top = np.argsort(np.diag(cov[h]))[::-1][:k] if claim[h] >= 0 else np.arange(k)
        Wk[h, :k] = W[h, top]
    a = acc_from(np.where(alive[None], errs(Wk[:, :k], Xte), np.inf), alive)
    curve_k.append((k, a))
    print(f"  keep the {k:>2} highest-variance minicolumns of each -> {a:.4f}")

res = {"params": params, "alive": int(alive.sum()), "full_gate": full,
       "keep_top_hypercolumns": curve_h,
       "keep_top_minicolumns": curve_k,
       "overlap_all": float(np.mean(ov)), "overlap_same_class": float(np.mean(same)),
       "use_share": use.tolist()}
(OUT / "capacity.json").write_text(json.dumps(res, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].plot(range(1, len(curve_h) + 1), curve_h, "o-", color="#1b6ca8")
ax[0].axhline(full, color="#999", ls="--")
ax[0].set_xlabel("hypercolumns kept (most used first)"); ax[0].set_ylabel("accuracy")
ax[0].set_title("are hypercolumns redundant?", fontsize=9)
ax[1].plot([c[0] for c in curve_k], [c[1] for c in curve_k], "o-", color="#2f8f4e")
ax[1].axhline(full, color="#999", ls="--")
ax[1].set_xlabel("minicolumns kept per hypercolumn (highest variance first)")
ax[1].set_title("are minicolumns idle?", fontsize=9)
ax[2].bar(range(H), np.sort(use)[::-1], color="#6a3d9a")
ax[2].set_xlabel("hypercolumn (sorted)"); ax[2].set_ylabel("share of test set won")
ax[2].set_title("who does the work", fontsize=9)
for a in ax:
    a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "80_capacity.png", dpi=140); plt.close(fig)
print("\n-> 80_capacity.png")
