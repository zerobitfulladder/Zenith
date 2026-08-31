"""Does the L2 normalisation (and the mean-centring) earn its place?

`center_norm` does two separable things to every joined vector:
  centring       removes overall brightness -- structure only, no intensity
  L2 normalising removes contrast, and puts the vector on the unit sphere the
                 geodesic rule lives on

Four arms, otherwise the λ=4 rule exactly (grading on drawing + naming,
conscience, repulsion). Reference: 0.8170 raw / 0.8278 with the reluctance.

Step sizes are made comparable across arms: θ = clip(η‖τ‖, 0, π/4) would sit
pinned at the cap for raw vectors (‖x‖ ≈ 9 instead of 1), so η is divided by
the mean squared norm of that arm's training vectors. Only the geometry differs.
"""

import json, time
from pathlib import Path
import numpy as np
from common import load, EPS, geo_step, CLASSES
from dopamine import geo_step_neg

OUT = Path(__file__).resolve().parent / "results"
H, K, ETA, GAMMA, LAM = 40, 36, 0.5, 0.3, 4.0
EPOCHS, BATCH, MIN_S, SEED, N_IMG = 6, 128, 2, 0, 784
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
CAL_EPOCHS, CAL_LR, RHO = 6, 0.002, 1.0
ARMS = [("centre + L2 (current)", True, True), ("centre only", True, False),
        ("L2 only", False, True), ("raw", False, False)]


def make_join(centre, l2):
    def j(X, y=None):
        V = np.zeros((len(X), X.shape[1] + 10))
        V[:, :X.shape[1]] = X
        if y is not None:
            L = np.zeros((len(X), 10)); L[np.arange(len(X)), y] = 1.0
            g = RHO * np.linalg.norm(X, axis=1) / np.maximum(
                np.linalg.norm(L, axis=1), EPS)
            V[:, X.shape[1]:] = L * g[:, None]
        if centre:
            V = V - V.mean(axis=1, keepdims=True)
        if l2:
            V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)
        return V
    return j


def rebuild(W, Q):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return e, R


def run(Xtr, ytr, Xte, yte, name, centre, l2, rng):
    join = make_join(centre, l2)
    scale = float((np.linalg.norm(join(Xtr[:4000], ytr[:4000]), axis=1) ** 2).mean())
    eta, eta_neg = ETA / scale, ETA_NEG / scale
    W = rng.standard_normal((H, K, N_IMG + 10))
    if centre:
        W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(Xtr[b], ytr[b]), join(Xtr[b])
            err, R = rebuild(W, Q)
            L = R[:, :, N_IMG:]
            conf = L[np.arange(len(b)), :, ytr[b]] / np.maximum(
                np.linalg.norm(L, axis=2), EPS)
            score = err + LAM * (1.0 - conf) - GAMMA * (1.0 / H - f)[None]
            win = score.argmin(1)
            cnt = np.bincount(win, minlength=H)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            for h in range(H):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], eta)
                np.add.at(wins[h], ytr[b][m], 1)
            if ep >= WARMUP:
                claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                fw = err.argmin(1)
                same = claim[None, :] == ytr[b][:, None]
                er = np.where(same, err, np.inf).min(1)
                bad = ((claim[fw] != ytr[b]) & np.isfinite(er) &
                       (err[np.arange(len(b)), fw] / np.maximum(er, EPS) > WINDOW))
                for h in np.unique(fw[bad]):
                    p = bad & (fw == h)
                    if p.sum() >= MIN_S:
                        W[h] = geo_step_neg(W[h], Q[p], eta_neg, CAP_NEG)
        print(f"  {name:<22} epoch {ep+1}/{EPOCHS}", flush=True)

    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([np.where(alive[None], rebuild(W, join(Xte[s:s + 2000]))[0],
                                 np.inf) for s in range(0, len(Xte), 2000)])
    raw = float((claim[E.argmin(1)] == yte).mean())
    Etr = np.concatenate([np.where(alive[None], rebuild(W, join(Xtr[s:s + 2000]))[0],
                                   np.inf) for s in range(0, len(Xtr), 2000)])
    b = np.zeros(H)
    r2 = np.random.default_rng(0)
    for _ in range(CAL_EPOCHS):
        for i in r2.permutation(len(ytr)):
            s = Etr[i] + b
            w = int(s.argmin())
            if claim[w] == ytr[i]:
                continue
            same = np.where(claim == ytr[i])[0]
            if not len(same):
                continue
            b[w] += CAL_LR; b[int(same[s[same].argmin()])] -= CAL_LR
    acc_b = float((claim[(E + b[None]).argmin(1)] == yte).mean())

    T = W[alive].reshape(-1, W.shape[2])
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), EPS)
    G = np.abs(T @ T.T); np.fill_diagonal(G, 0.0)
    mimg = Xtr.mean(0); mimg = mimg / np.linalg.norm(mimg)
    dc = float(np.abs(T[:, :N_IMG] @ mimg).mean())
    pur = float((wins.max(1) / np.maximum(wins.sum(1), 1))[alive].mean())
    print(f"  -> {name:<22} raw {raw:.4f}   +reluctance {acc_b:.4f}   "
          f"coherence {G.mean():.4f}   DC alignment {dc:.4f}   "
          f"purity {pur:.3f}   alive {int(alive.sum())}", flush=True)
    return {"acc": raw, "acc_reluctance": acc_b, "coherence": float(G.mean()),
            "max_coherence": float(G.max()), "dc_alignment": dc, "purity": pur,
            "alive": int(alive.sum()), "eta_scale": scale}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    res = {}
    for name, centre, l2 in ARMS:
        res[name] = run(Xtr, ytr, Xte, yte, name, centre, l2,
                        np.random.default_rng(SEED + 1))
    (OUT / "norm_ablation.json").write_text(json.dumps(
        {"results": res, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
