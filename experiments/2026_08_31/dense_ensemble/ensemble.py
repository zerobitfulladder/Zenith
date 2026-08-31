"""100 dense hypercolumns, trained jointly with the label. Do they vote well?

The 08-30 `weighted` rule verbatim -- 144 templates, whole MNIST images,
centre-then-unit-normalise -- but the vector is [image ; label * gain] with
the label scaled to MATCH the image block's energy (rho = 1.0). 100 of them,
differently seeded, trained on the same data at the same time.

Reading a label out of one of them is: score with the image cells, rebuild,
read the label block --

    label_hat = W_lab^T W_img x  =  M x

a linear map. Note M is EXACTLY rotation-invariant: (RW)_lab^T (RW)_img =
W_lab^T R^T R W_img = M. So if these 100 converge to one subspace up to
rotation -- which 08-30 measured at 0.974 overlap with the top-144 -- they
compute the same classifier and voting buys nothing. That is the prediction
this run tests.

Everything is batched over hypercolumns AND samples: W is (H, k, d).
"""

import json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
H, K, RHO, ETA = 30, 144, 1.0, 0.5
N_TRAIN, N_TEST, EPOCHS, BATCH = 20000, 5000, 3, 64
SIZES = [1, 2, 3, 5, 10, 15, 20, 25, 30]
SEED = 0


def center_norm(V):
    """08-30's preprocessing: subtract the row mean, scale to unit length."""
    Vc = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Vc, axis=1, keepdims=True)
    return Vc / np.maximum(n, EPS)


def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    X = X.reshape(len(X), -1)
    if X.max() > 1.5:
        X = X / 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def join(X, y=None, n_lab=10):
    """[image ; label * gain], gain making the two blocks equal in energy."""
    V = np.zeros((len(X), X.shape[1] + n_lab))
    V[:, :X.shape[1]] = X
    if y is not None:
        L = np.zeros((len(X), n_lab)); L[np.arange(len(X)), y] = 1.0
        g = RHO * np.linalg.norm(X, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, X.shape[1]:] = L * g[:, None]
    return center_norm(V)


def train(Xh, rng):
    """H hypercolumns, one geodesic step per batch each. The 08-30 rule."""
    d = Xh.shape[1]
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xh))
        for s in range(0, len(order), BATCH):
            B = Xh[order[s:s + BATCH]]
            S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
            E = B[None] - np.einsum('hbk,hkd->hbd', S, W, optimize=True)
            M = np.einsum('hbk,hbd->hkd', S, E, optimize=True) / len(B)
            tau = M - (M * W).sum(-1, keepdims=True) * W
            tn = np.linalg.norm(tau, axis=-1)
            th = np.clip(ETA * tn, 0.0, np.pi / 4)
            hat = np.zeros_like(tau)
            live = tn > EPS
            hat[live] = tau[live] / tn[live, None]
            W = W * np.cos(th)[..., None] + hat * np.sin(th)[..., None]
            W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
        print(f"  epoch {ep+1}/{EPOCHS} done", flush=True)
    return W


def label_scores(W, Q, n_img):
    """(H, n, 10): each hypercolumn completes the missing label block."""
    S = np.einsum('bd,hkd->hbk', Q, W, optimize=True)
    R = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
    return R[:, :, n_img:]


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    Jtr = join(Xtr, ytr)
    print(f"{H} hypercolumns x {K} templates, dim {Jtr.shape[1]} "
          f"(image {n_img} + label 10, rho={RHO})", flush=True)

    W = train(Jtr, np.random.default_rng(SEED + 1))
    np.savez_compressed(OUT / "weights.npz", W=W.astype(np.float32),
                        config=np.array([H, K, RHO, ETA, N_TRAIN, EPOCHS]))
    print(f"weights -> {OUT/'weights.npz'}  ({time.time()-t0:.0f}s)", flush=True)

    Q = join(Xte)                                   # image written, label blank
    P = label_scores(W, Q, n_img)                   # (H, n, 10)
    Pn = P / np.maximum(np.linalg.norm(P, axis=2, keepdims=True), EPS)
    single = np.array([float((P[h].argmax(1) == yte).mean()) for h in range(H)])
    print(f"\nsingle hypercolumn: mean {single.mean():.4f}  "
          f"min {single.min():.4f}  max {single.max():.4f}  sd {single.std():.4f}")

    # how different are they, really?
    votes = P.argmax(2)
    agree = np.mean([(votes[i] == votes[j]).mean()
                     for i in range(0, H, 3) for j in range(i + 1, H, 3)])
    Wf = W.reshape(H, K, -1)
    ov = []
    for i in range(0, H, 4):
        for j in range(i + 1, H, 4):
            sv = np.linalg.svd(Wf[i] @ Wf[j].T, compute_uv=False)
            ov.append(float((sv ** 2).mean()))
    print(f"pairwise prediction agreement {agree:.4f}   "
          f"pairwise subspace overlap {np.mean(ov):.4f}  (1.0 = identical span)")

    rng = np.random.default_rng(7)
    curve = {}
    for n in SIZES:
        reps = 1 if n == H else 25
        soft, hard = [], []
        for _ in range(reps):
            idx = rng.choice(H, n, replace=False) if n < H else np.arange(H)
            soft.append(float((Pn[idx].mean(0).argmax(1) == yte).mean()))
            v = votes[idx]
            tal = np.zeros((len(yte), 10))
            for r in range(n):
                tal[np.arange(len(yte)), v[r]] += 1
            hard.append(float((tal.argmax(1) == yte).mean()))
        curve[n] = {"soft_mean": float(np.mean(soft)), "soft_sd": float(np.std(soft)),
                    "hard_mean": float(np.mean(hard)), "hard_sd": float(np.std(hard))}
        print(f"  {n:>3} hypercolumns:  soft {np.mean(soft):.4f} "
              f"(sd {np.std(soft):.4f})   majority {np.mean(hard):.4f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a, sc in zip(ax, ("linear", "log")):
        a.errorbar(SIZES, [curve[n]["soft_mean"] for n in SIZES],
                   yerr=[curve[n]["soft_sd"] for n in SIZES], marker="o",
                   color="#1b6ca8", label="soft vote (averaged label vectors)")
        a.errorbar(SIZES, [curve[n]["hard_mean"] for n in SIZES],
                   yerr=[curve[n]["hard_sd"] for n in SIZES], marker="s",
                   color="#c1462d", label="majority vote")
        a.axhline(single.mean(), ls=":", color="#888", label="one hypercolumn (mean)")
        a.set_xscale(sc); a.set_xlabel("hypercolumns voting")
        a.set_ylabel("classification accuracy"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle(f"{H} dense hypercolumns trained jointly with the label — "
                 f"does voting help?", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_vote_curve.png", dpi=130); plt.close(fig)

    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "rho": RHO, "eta": ETA, "train": N_TRAIN,
                    "test": N_TEST, "epochs": EPOCHS},
         "single_mean": float(single.mean()), "single_sd": float(single.std()),
         "single_min": float(single.min()), "single_max": float(single.max()),
         "pairwise_prediction_agreement": float(agree),
         "pairwise_subspace_overlap": float(np.mean(ov)),
         "curve": {str(k): v for k, v in curve.items()},
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
