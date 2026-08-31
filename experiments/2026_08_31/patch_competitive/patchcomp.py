"""Unsupervised competitive experts on 8x8 patches. No labels anywhere.

50 experts x 8 templates. The winner is whoever REBUILDS the patch best --
fit-only competition -- and only the winner learns. A win-frequency
conscience keeps anyone from starving.

The question: do specialists appear without supervision? The control already
exists in ../patch_ensemble, where every expert learned from every patch and
the pairwise subspace overlap came out at exactly 1.0000 -- ten copies of one
subspace. If competition is the ingredient, this should come out far lower
and the experts should be things you can look at and name.
"""

import json, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
H, K, PATCH, ETA = 50, 8, 8, 0.5
N_IMG, EPOCHS, BATCH, CHUNK, GAMMA, MIN_S = 8000, 3, 2048, 96, 1.0, 4
SEED = 0


def center_norm(V):
    Vc = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Vc, axis=1, keepdims=True)
    return Vc / np.maximum(n, EPS), n[:, 0]


def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    if X.max() > 1.5:
        X = X / 255.0
    return X[np.random.default_rng(SEED).permutation(len(X))][:N_IMG]


def patches(imgs):
    P = sliding_window_view(imgs, (PATCH, PATCH), axis=(1, 2)).reshape(-1, PATCH * PATCH)
    U, nrm = center_norm(P)
    keep = nrm > 1e-6
    return U[keep], P[keep]


def errors(W, B):
    S = (B @ W.reshape(H * K, -1).T).reshape(len(B), H, K).transpose(1, 0, 2)
    R = np.matmul(S, W)
    return np.linalg.norm(B[None] - R, axis=2)          # rows are unit, so this is relative


def geo_step(Wh, B, eta):
    S = B @ Wh.T
    E = B - S @ Wh
    M = (S.T @ E) / len(B)
    tau = M - (M * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau)
    live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


def main():
    t0 = time.time()
    imgs = load()
    d = PATCH * PATCH
    rng = np.random.default_rng(SEED + 1)
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    f = np.full(H, 1.0 / H)
    seen = 0
    print(f"{H} experts x {K} templates on {PATCH}x{PATCH} patches, stride 1, "
          f"{N_IMG} images", flush=True)
    for ep in range(EPOCHS):
        order = rng.permutation(len(imgs))
        for c in range(0, len(order), CHUNK):
            U, _ = patches(imgs[order[c:c + CHUNK]])
            U = U[rng.permutation(len(U))]
            for s in range(0, len(U), BATCH):
                B = U[s:s + BATCH]
                if len(B) < MIN_S * 2:
                    continue
                e = errors(W, B) + GAMMA * (f - 1.0 / H)[:, None]   # conscience
                win = e.argmin(0)
                cnt = np.bincount(win, minlength=H)
                f = 0.995 * f + 0.005 * (cnt / max(cnt.sum(), 1))
                for h in range(H):
                    m = win == h
                    if m.sum() >= MIN_S:
                        W[h] = geo_step(W[h], B[m], ETA)
                seen += len(B)
        print(f"  epoch {ep+1}/{EPOCHS}  {seen/1e6:.2f}M patches  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- what did they claim? ------------------------------------------
    U, RAW = patches(imgs[:1500])
    sub = rng.choice(len(U), 120000, replace=False)
    U, RAW = U[sub], RAW[sub]
    e = errors(W, U)
    win = e.argmin(0)
    cnt = np.bincount(win, minlength=H)
    live = np.nonzero(cnt > len(U) * 0.002)[0]
    ind = np.array([e[h, win == h].mean() for h in live])
    oud = np.array([e[h, win != h].mean() for h in live])
    ov = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
                np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                        (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
          for i in live for j in live if i < j]
    print(f"\nlive experts {len(live)}/{H}   busiest claims "
          f"{cnt.max()/len(U)*100:.1f}% of patches   smallest {cnt[live].min()/len(U)*100:.2f}%")
    print(f"rebuild error: own patches {ind.mean()*100:.1f}%   others "
          f"{oud.mean()*100:.1f}%   GAP {(oud.mean()-ind.mean())*100:.2f} pts")
    print(f"pairwise subspace overlap {np.mean(ov):.4f}   "
          f"(all-learn control in ../patch_ensemble was 1.0000)")

    np.savez_compressed(OUT / "weights.npz", W=W.astype(np.float32), counts=cnt)
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "patch": PATCH, "images": N_IMG,
                    "epochs": EPOCHS, "gamma": GAMMA},
         "live": int(len(live)), "busiest_share": float(cnt.max() / len(U)),
         "own_err": float(ind.mean()), "other_err": float(oud.mean()),
         "gap": float(oud.mean() - ind.mean()),
         "subspace_overlap": float(np.mean(ov)),
         "seconds": round(time.time() - t0, 1)}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = live[np.argsort(-cnt[live])]
    # 1: what each expert claimed -- the mean of the patches it won
    n = len(order)
    cols = 10; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * .9, rows * 1.0))
    for i, a in enumerate(np.atleast_1d(axes).ravel()):
        a.set_xticks([]); a.set_yticks([])
        if i < n:
            h = order[i]
            a.imshow(RAW[win == h].mean(0).reshape(PATCH, PATCH), cmap="gray_r")
            a.set_title(f"h{h}  {cnt[h]/len(U)*100:.1f}%", fontsize=6)
        else:
            a.axis("off")
    fig.suptitle("what each expert claimed — mean of the patches it won", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_claimed.png", dpi=140); plt.close(fig)
    # 2: the templates themselves
    fig, axes = plt.subplots(n, K, figsize=(K * .55, n * .55))
    axes = np.atleast_2d(axes)
    for r, h in enumerate(order):
        for j in range(K):
            t = W[h, j].reshape(PATCH, PATCH); m = np.abs(t).max() + EPS
            axes[r, j].imshow(t, cmap="bwr", vmin=-m, vmax=m)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(f"h{h}", fontsize=5, rotation=0, ha="right", va="center")
    fig.suptitle(f"the {K} templates of each live expert", fontsize=10)
    fig.subplots_adjust(left=.06, right=.99, top=.97, bottom=.005, wspace=.06, hspace=.06)
    fig.savefig(OUT / "02_templates.png", dpi=140); plt.close(fig)
    # 3: real patches each expert won
    fig, axes = plt.subplots(n, 12, figsize=(12 * .5, n * .5))
    axes = np.atleast_2d(axes)
    for r, h in enumerate(order):
        idx = np.nonzero(win == h)[0][:12]
        for j in range(12):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if j < len(idx):
                axes[r, j].imshow(RAW[idx[j]].reshape(PATCH, PATCH), cmap="gray_r")
            else:
                axes[r, j].axis("off")
        axes[r, 0].set_ylabel(f"h{h}", fontsize=5, rotation=0, ha="right", va="center")
    fig.suptitle("actual patches each expert won", fontsize=10)
    fig.subplots_adjust(left=.06, right=.99, top=.97, bottom=.005, wspace=.06, hspace=.06)
    fig.savefig(OUT / "03_examples.png", dpi=140); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
