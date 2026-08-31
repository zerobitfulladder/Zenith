"""10 hypercolumns x 100 templates, 08-30 dense rule, 8x8 MNIST patches, stride 1.

Note the regime change: 8x8 = 64 numbers, 63 after centring, and 100
templates. This is OVERCOMPLETE, where every 08-30 dense run was
undercomplete (144 in 784). W'W is no longer a projection, so the rebuild
overshoots; `scale` below is the best single number to multiply it by, and
1/1.6 would be the tight-frame prediction.
"""

import json, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
H, K, PATCH, ETA = 10, 100, 8, 0.5
N_IMG, EPOCHS, BATCH, CHUNK = 20000, 2, 1024, 128
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
    """(n,28,28) -> unit rows of every 8x8 patch, stride 1, blanks dropped."""
    P = sliding_window_view(imgs, (PATCH, PATCH), axis=(1, 2))
    P = P.reshape(-1, PATCH * PATCH)
    U, nrm = center_norm(P)
    return U[nrm > 1e-6]


def step(W, B, eta):
    """One geodesic step per hypercolumn: the 08-30 `weighted` batch rule."""
    h, k, d = W.shape
    S = (B @ W.reshape(h * k, d).T).reshape(len(B), h, k).transpose(1, 0, 2)
    E = B[None] - np.matmul(S, W)
    M = np.matmul(S.transpose(0, 2, 1), E) / len(B)
    tau = M - (M * W).sum(-1, keepdims=True) * W
    tn = np.linalg.norm(tau, axis=-1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau)
    live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    W = W * np.cos(th)[..., None] + hat * np.sin(th)[..., None]
    return W / (np.linalg.norm(W, axis=2, keepdims=True) + EPS)


def main():
    t0 = time.time()
    imgs = load()
    d = PATCH * PATCH
    rng = np.random.default_rng(SEED + 1)
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS

    per_img = (28 - PATCH + 1) ** 2
    print(f"{H} hypercolumns x {K} templates, {PATCH}x{PATCH} patches stride 1: "
          f"{per_img} per image, {N_IMG} images = {per_img*N_IMG/1e6:.2f}M patches/epoch",
          flush=True)
    seen = 0
    for ep in range(EPOCHS):
        order = rng.permutation(len(imgs))
        for c in range(0, len(order), CHUNK):
            U = patches(imgs[order[c:c + CHUNK]])
            U = U[rng.permutation(len(U))]
            for s in range(0, len(U), BATCH):
                B = U[s:s + BATCH]
                if len(B) > 8:
                    W = step(W, B, ETA); seen += len(B)
        print(f"  epoch {ep+1}/{EPOCHS}  {seen/1e6:.2f}M patches  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- diagnostics -----------------------------------------------------
    T = patches(imgs[:600])
    T = T[rng.choice(len(T), 20000, replace=False)]
    S = (T @ W.reshape(H * K, d).T).reshape(len(T), H, K).transpose(1, 0, 2)
    R = np.matmul(S, W)
    raw = np.linalg.norm(T[None] - R, axis=2).mean(1)
    a = (T[None] * R).sum(2).sum(1) / np.maximum((R * R).sum(2).sum(1), EPS)
    best = np.array([np.linalg.norm(T - a[h] * R[h], axis=1).mean() for h in range(H)])
    rank = np.array([float((np.linalg.svd(W[h], compute_uv=False) ** 2).sum() ** 2 /
                           (np.linalg.svd(W[h], compute_uv=False) ** 4).sum()) for h in range(H)])
    coh = np.array([float(np.abs((W[h] @ W[h].T)[~np.eye(K, dtype=bool)]).mean())
                    for h in range(H)])
    ov = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() / K)
          for i in range(H) for j in range(i + 1, H)]

    print(f"\nrebuild error  raw {raw.mean():.4f}   best-scaled {best.mean():.4f}"
          f"   optimal scale {a.mean():.4f}  (tight-frame prediction {63/K:.4f})")
    print(f"effective rank {rank.mean():.1f} / {min(K, d-1)}   "
          f"within-hypercolumn coherence {coh.mean():.4f}")
    print(f"pairwise subspace overlap between the {H} hypercolumns: "
          f"{np.mean(ov):.4f}  (1.0 = identical span)")

    np.savez_compressed(OUT / "weights.npz", W=W.astype(np.float32))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(10, 20, figsize=(11, 5.8))
    for h in range(H):
        for j in range(20):
            t = W[h, j].reshape(PATCH, PATCH)
            m = np.abs(t).max() + EPS
            ax[h, j].imshow(t, cmap="bwr", vmin=-m, vmax=m)
            ax[h, j].set_xticks([]); ax[h, j].set_yticks([])
        ax[h, 0].set_ylabel(f"h{h}", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle(f"{H} hypercolumns x {K} templates on 8x8 patches — first 20 of each",
                 fontsize=10)
    fig.subplots_adjust(left=.03, right=.995, top=.93, bottom=.005, wspace=.05, hspace=.05)
    fig.savefig(OUT / "01_templates.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(10, 10, figsize=(6, 6))
    for j, a_ in enumerate(ax.ravel()):
        t = W[0, j].reshape(PATCH, PATCH)
        m = np.abs(t).max() + EPS
        a_.imshow(t, cmap="bwr", vmin=-m, vmax=m)
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("hypercolumn 0 — all 100 templates", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "02_hypercolumn0.png", dpi=130); plt.close(fig)

    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "patch": PATCH, "stride": 1, "eta": ETA,
                    "images": N_IMG, "epochs": EPOCHS,
                    "patches_per_epoch": int(per_img * N_IMG)},
         "rebuild_raw": float(raw.mean()), "rebuild_best_scaled": float(best.mean()),
         "optimal_scale": float(a.mean()), "tight_frame_prediction": 63 / K,
         "effective_rank": float(rank.mean()),
         "within_coherence": float(coh.mean()),
         "pairwise_subspace_overlap": float(np.mean(ov)),
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
