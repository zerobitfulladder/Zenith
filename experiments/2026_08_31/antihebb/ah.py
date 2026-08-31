"""Anti-Hebbian lateral inhibition with geodesic learning, on MNIST patches.

Three local rules, nothing global anywhere:

    settle      y  <-  relu( W x  -  M y  -  t )        iterated
    feedforward each ACTIVE template rotates toward x along the sphere,
                at a rate proportional to its own activity (geodesic, so the
                template stays unit-norm and mean-centred)
    lateral     dM_ij  =  alpha ( <y_i y_j>  -  p^2 )   clipped at >= 0, no
                diagonal -- units that fire together start suppressing
                each other
    threshold   dt_i   =  beta ( <y_i>  -  p )          each unit is driven to
                its own target activity p

Competition here is not winner-take-all. Several templates can be active at
once; what is punished is REDUNDANT co-activation. That is the difference
between units that are mutually exclusive and units that are complementary.

Patches are mean-centred and L2-normalised, so W x is a Pearson correlation.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
PS, EPS = 5, 1e-12
K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
DECAY = 0.002          # without it the lateral weights run away when K > D
K_ACTIVE = 4.0         # aim for this MANY units per patch, not a fixed fraction
P_TARGET = K_ACTIVE / K
import os
ETA = 0.5
ALPHA = float(os.environ.get('AH_ALPHA', 0.4))   # how hard decorrelation pushes
BETA = 0.3
ITERS, DAMP, CAP = 20, 0.3, np.pi / 8
N_IMG, PER_IMG, EPOCHS, BATCH = 8000, 60, 8, 512
FLOOR, SEED = 0.05, 0


def load(name="mnist"):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    if X.max() > 1.5:
        X /= 255.0
    return X[np.random.default_rng(SEED).permutation(len(X))][:N_IMG]


def patches(X):
    V = sliding_window_view(X, (PS, PS), axis=(1, 2))
    return np.ascontiguousarray(V.reshape(len(X), -1, PS * PS), dtype=np.float64)


def prep(P):
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    return C / np.maximum(n, EPS)[..., None], n > FLOOR


def sample(X, rng):
    out = []
    for a in range(0, len(X), 256):
        Q, keep = prep(patches(X[a:a + 256]))
        for i in range(len(Q)):
            k = np.nonzero(keep[i])[0]
            if len(k):
                out.append(Q[i, rng.choice(k, min(PER_IMG, len(k)), False)])
    return np.concatenate(out)


def settle(Xb, W, M, t, iters=ITERS):
    """y <- relu(Wx - My - t), damped."""
    S = Xb @ W.T
    Y = np.maximum(S - t, 0.0)
    for _ in range(iters):
        Y = (1 - DAMP) * Y + DAMP * np.maximum(S - Y @ M.T - t, 0.0)
    return Y


def geo_toward(W, Xb, Y):
    """Each active unit rotates toward what the others did NOT explain.

    Toward the raw input is wrong: every active unit then chases the same
    patch and they all converge on each other. Toward the RESIDUAL, each one
    learns the part nobody covered -- which is what decorrelates them, and is
    the rule the rest of this project already uses.
    """
    tot = Y.sum(0)
    live = tot > EPS
    R = Xb - Y @ W                                        # what is left unexplained
    G = (Y.T @ R) / np.maximum(tot, EPS)[:, None]
    tau = G - (G * W).sum(1, keepdims=True) * W           # tangent at W
    tn = np.linalg.norm(tau, axis=1)
    th = np.where(live, np.clip(ETA * tn, 0.0, CAP), 0.0)
    hat = np.zeros_like(tau)
    ok = tn > EPS
    hat[ok] = tau[ok] / tn[ok, None]
    W = W * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    W -= W.mean(1, keepdims=True)                          # stay mean-centred
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), EPS)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    X = load()
    Q = sample(X, rng)
    print(f"{len(Q)} patches of {PS}x{PS}, K={K}, target activity {P_TARGET}",
          flush=True)

    W = rng.standard_normal((K, PS * PS))
    W -= W.mean(1, keepdims=True)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    M = np.zeros((K, K))
    t = np.zeros(K)
    hist = []
    for ep in range(EPOCHS):
        rng.shuffle(Q)
        for s in range(0, len(Q), BATCH):
            B = Q[s:s + BATCH]
            Y = settle(B, W, M, t)
            W = geo_toward(W, B, Y)
            Yb = (Y > 0).astype(np.float64)
            C = (Yb.T @ Yb) / len(B)
            M += ALPHA * (C - P_TARGET ** 2) - DECAY * M
            np.fill_diagonal(M, 0.0)
            np.maximum(M, 0.0, out=M)
            t += BETA * ((Y > 0).mean(0) - P_TARGET)
        Y = settle(Q[:4000], W, M, t)
        act = float((Y > 0).mean())
        R = Y @ W
        rn = np.maximum(np.linalg.norm(R, axis=1), EPS)
        err = float(np.mean((Q[:4000] * (R / rn[:, None])).sum(1)))   # as coded
        # how good was the SELECTION, ignoring whether the coefficients are right:
        # least-squares fit on just the units that fired
        sub = Q[:1500]
        Ys = settle(sub, W, M, t)
        fit = []
        for i in range(len(sub)):
            a = np.nonzero(Ys[i] > 0)[0]
            if not len(a):
                fit.append(0.0); continue
            Wa = W[a]
            c, *_ = np.linalg.lstsq(Wa.T, sub[i], rcond=None)
            r = Wa.T @ c
            fit.append(float(r @ sub[i] / max(np.linalg.norm(r), EPS)))
        ls = float(np.mean(fit))
        G = np.abs(W @ W.T); np.fill_diagonal(G, 0.0)
        hist.append({"epoch": ep + 1, "active_frac": act,
                     "active_units": act * K, "rebuild_cos": err, "lstsq_cos": ls,
                     "coherence": float(G.mean()), "M_mean": float(M.mean()),
                     "dead": int(((Y > 0).mean(0) < 1e-4).sum())})
        print(f"  epoch {ep+1}/{EPOCHS}  active {act*K:.2f}/{K} units  "
              f"coded {err:.4f}  lstsq {ls:.4f}  coherence {G.mean():.4f}  "
              f"lateral {M.mean():.4f}  dead {hist[-1]['dead']}", flush=True)

    np.savez_compressed(OUT / f"ah_K{K}_a{ALPHA}.npz", W=W, M=M, t=t)
    (OUT / f"ah_K{K}_a{ALPHA}.json").write_text(json.dumps(
        {"K": K, "p_target": P_TARGET, "eta": ETA, "alpha": ALPHA, "beta": BETA,
         "iters": ITERS, "history": hist,
         "seconds": round(time.time() - t0, 1)}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Y = settle(Q[:4000], W, M, t)
    use = (Y > 0).mean(0)
    o = np.argsort(use)[::-1]
    cols = 16; rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * .58, rows * .62))
    for i, a in enumerate(np.ravel(axes)):
        a.set_xticks([]); a.set_yticks([])
        if i >= K:
            a.axis("off"); continue
        w = W[o[i]].reshape(PS, PS)
        m = np.abs(w).max() + EPS
        a.imshow(w, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
        a.set_title(f"{use[o[i]]*100:.0f}%", fontsize=4, pad=1)
    fig.suptitle(f"anti-Hebbian + geodesic, K={K}, "
                 f"{hist[-1]['active_units']:.1f} units active per patch", fontsize=9)
    fig.subplots_adjust(left=.005, right=.995, top=1 - .55 / (rows * .62),
                        bottom=.005, wspace=.07, hspace=.30)
    fig.savefig(OUT / f"templates_K{K}_a{ALPHA}.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    im = ax[0].imshow(M[np.ix_(o, o)], cmap="magma")
    ax[0].set_title("learned lateral inhibition (sorted by usage)", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=.046)
    ax[1].plot([h["coherence"] for h in hist], "o-", label="template coherence")
    ax[1].plot([h["rebuild_cos"] for h in hist], "s-", label="rebuild cosine")
    ax[1].plot([h["active_units"] for h in hist], "^-", label="units active")
    ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "lateral.png", dpi=140); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s -> templates_K{K}.png")


if __name__ == "__main__":
    main()
