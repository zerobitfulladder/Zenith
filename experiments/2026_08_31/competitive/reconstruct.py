"""Let the winning hypercolumn rebuild the digit, and draw one from a label."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compete import load, join, center_norm, predict, H, K, OUT, EPS

Xtr, ytr, Xte, yte = load(); n_img = Xtr.shape[1]
z = np.load(OUT / "weights.npz")
Q = join(Xte); n = len(Xte)
rows, res = [], {}

for tag in ("W_competitive", "W_control"):
    W = z[tag].astype(np.float64)
    S = (Q @ W.reshape(H * K, -1).T).reshape(n, H, K).transpose(1, 0, 2)
    R = np.matmul(S, W)
    err = np.linalg.norm(Q[None, :, :n_img] - R[:, :, :n_img], axis=2) / np.maximum(
        np.linalg.norm(Q[None, :, :n_img], axis=2), EPS)
    win = err.argmin(0)
    ix = np.arange(n)
    res[tag] = {
        "winner_rebuild_err": float(err[win, ix].mean()),
        "mean_over_all_columns": float(err.mean()),
        "worst_column": float(err.max(0).mean()),
        "best_possible": float(err.min(0).mean()),
        "pooled_all_columns": float(np.linalg.norm(
            Q[:, :n_img] - R[:, :, :n_img].mean(0), axis=1).mean() /
            np.linalg.norm(Q[:, :n_img], axis=1).mean()),
    }
    print(f"[{tag}] image->code->image, winner {res[tag]['winner_rebuild_err']:.4f}   "
          f"mean column {res[tag]['mean_over_all_columns']:.4f}   "
          f"pooled average of all {res[tag]['pooled_all_columns']:.4f}")
    rows.append((tag, R[win, ix][:, :n_img], win))

# ---- draw a digit from the label alone -------------------------------------
gen = {}
for tag in ("W_competitive", "W_control"):
    W = z[tag].astype(np.float64)
    L = np.zeros((10, n_img + 10)); L[np.arange(10), n_img + np.arange(10)] = 1.0
    Lq = center_norm(L)
    S = (Lq @ W.reshape(H * K, -1).T).reshape(10, H, K).transpose(1, 0, 2)
    R = np.matmul(S, W)
    fit = -np.linalg.norm(Lq[None, :, n_img:] - R[:, :, n_img:], axis=2)
    pick = fit.argmax(0)
    gen[tag] = (R[pick, np.arange(10)][:, :n_img], pick)
    print(f"[{tag}] label->image, hypercolumns chosen: {pick}")

fig, axes = plt.subplots(6, 12, figsize=(11, 5.6))
sel = np.arange(12)
for j in sel:
    t = Q[j, :n_img].reshape(28, 28); m = np.abs(t).max()
    axes[0, j].imshow(t, cmap="gray_r", vmin=-m, vmax=m)
    axes[0, j].set_title(str(yte[j]), fontsize=7)
axes[0, 0].set_ylabel("truth", fontsize=6, rotation=0, ha="right", va="center")
for r, (tag, rec, win) in enumerate(rows):
    for j in sel:
        t = rec[j].reshape(28, 28); m = np.abs(t).max()
        axes[1 + r, j].imshow(t, cmap="gray_r", vmin=-m, vmax=m)
        axes[1 + r, j].set_title(f"h{win[j]}", fontsize=6)
    axes[1 + r, 0].set_ylabel(tag.replace("W_", "") + "\nwinner", fontsize=6,
                              rotation=0, ha="right", va="center")
axes[3, 0].set_ylabel("label ->\nimage", fontsize=6, rotation=0, ha="right", va="center")
for r, tag in enumerate(("W_competitive", "W_control")):
    img, pick = gen[tag]
    for d in range(10):
        t = img[d].reshape(28, 28); m = np.abs(t).max()
        axes[3 + r, d].imshow(t, cmap="gray_r", vmin=-m, vmax=m)
        axes[3 + r, d].set_title(f"{d} (h{pick[d]})", fontsize=6)
    for d in (10, 11):
        axes[3 + r, d].axis("off")
    axes[3 + r, 0].set_ylabel(tag.replace("W_", ""), fontsize=6,
                              rotation=0, ha="right", va="center")
for r in (5,):
    for a in axes[r]:
        a.axis("off")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("top: held-out digits, rebuilt by whichever hypercolumn fits best.  "
             "bottom: drawn from the label alone.", fontsize=10)
fig.subplots_adjust(left=.06, right=.995, top=.90, bottom=.01, wspace=.06, hspace=.30)
fig.savefig(OUT / "03_reconstructions.png", dpi=135); plt.close(fig)
(OUT / "reconstruction.json").write_text(json.dumps(res, indent=2))
print(f"\n-> {OUT/'03_reconstructions.png'}")
