"""Grade the drawing AND the naming, and sweep how much the naming counts.

The teacher shows an image. Every expert draws it and names it. The grade is

    score(h) = image_error(h)  +  LAM * (1 - confidence_of_h_in_the_true_label)

and the best-graded expert learns the whole joined vector. Both terms are O(1),
so LAM is literally how much energy the label carries:

    LAM = 0     the drawing rule   (../draw.py)
    LAM -> inf  the loud-child rule (../l1.py) -- name it right, draw however

Conscience is on throughout, because the sweep is meaningless if one expert can
starve the rest. Repulsion still targets the competition we READ (image only):
whoever would win the fit gate and is wrong gets pushed away.

Reported per LAM: accuracy, and two generation numbers, because the claim under
test is that these need not trade off.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from common import load, join, center_norm, EPS, geo_step, CLASSES
from dopamine import geo_step_neg, calibrate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
H, K, ETA, GAMMA = 40, 36, 0.5, 0.3
EPOCHS, BATCH, MIN_S = 6, 128, 2
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SEED, N_IMG = 0, 784


def rebuild(W, Q):
    """(n,H) image-half error and the FULL (n,H,d) reconstruction."""
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return e, R


def train(Xtr, ytr, lam, rng, reject=True):
    d = N_IMG + 10
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(Xtr[b], ytr[b]), join(Xtr[b])
            err, R = rebuild(W, Q)                       # (n,H) and (n,H,d)
            L = R[:, :, N_IMG:]
            conf = L[np.arange(len(b)), :, ytr[b]] / np.maximum(
                np.linalg.norm(L, axis=2), EPS)          # (n, H)
            score = err + lam * (1.0 - conf) - GAMMA * (1.0 / H - f)[None]
            win = score.argmin(1)
            cnt = np.bincount(win, minlength=H)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))

            bad = np.zeros(len(b), bool)
            if reject and ep >= WARMUP:                  # judged on the READ gate
                claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                fit_win = err.argmin(1)
                same = claim[None, :] == ytr[b][:, None]
                e_right = np.where(same, err, np.inf).min(1)
                bad = ((claim[fit_win] != ytr[b]) & np.isfinite(e_right) &
                       (err[np.arange(len(b)), fit_win] /
                        np.maximum(e_right, EPS) > WINDOW))
            for h in range(H):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
            if bad.any():
                fw = err.argmin(1)
                for h in np.unique(fw[bad]):
                    p = bad & (fw == h)
                    if p.sum() >= MIN_S:
                        W[h] = geo_step_neg(W[h], Q[p], ETA_NEG, CAP_NEG)
        print(f"  lam={lam:<5} epoch {ep+1}/{EPOCHS}", flush=True)
    return W, wins


def imagine(W, labels):
    V = np.zeros((len(labels), N_IMG + 10))
    V[np.arange(len(labels)), N_IMG + np.asarray(labels)] = 1.0
    _, R = rebuild(W, center_norm(V))
    return R[:, :, :N_IMG]


def report(W, wins, Xte, yte, proto, b, tag):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([rebuild(W, join(Xte[s:s + 2000]))[0]
                        for s in range(0, len(Xte), 2000)])
    E = np.where(alive[None], E, np.inf)
    got = claim[(E + b[None]).argmin(1)]
    acc = float((got == yte).mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    with np.errstate(all="ignore"):
        o = float(np.nanmean(np.nanmin(np.where(own, E, np.nan), axis=1)))
        t = float(np.nanmean(np.nanmin(np.where(~own & alive[None], E, np.nan), axis=1)))
    idx = np.where(alive)[0]
    G = imagine(W, [max(claim[h], 0) for h in idx])
    G = np.stack([G[i, h] for i, h in enumerate(idx)])
    G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), EPS)
    P = proto[[max(claim[h], 0) for h in idx]]
    gen = float(np.mean(np.sum(G * P, axis=1)))          # 1.0 = imagines its class
    pur = float((wins.max(1) / np.maximum(wins.sum(1), 1))[alive].mean())
    print(f"  {tag:<12} acc {acc:.4f}   own-class err {o:.4f}   other {t:.4f}   "
          f"gap {t-o:+.4f}   imagination {gen:+.4f}   purity {pur:.3f}   "
          f"alive {int(alive.sum())}   classes {len(set(claim[alive].tolist()))}/10")
    return {"acc": acc, "own_class_err": o, "other_class_err": t, "gap": t - o,
            "imagination": gen, "purity": pur, "alive": int(alive.sum()),
            "classes_covered": len(set(claim[alive].tolist())),
            "per_class": {CLASSES[c]: float((got[yte == c] == c).mean())
                          for c in range(10)}}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    proto = center_norm(np.stack([Xtr[ytr == c].mean(0) for c in range(10)]))
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), EPS)
    res = {}
    for lam in LAMS:
        W, wins = train(Xtr, ytr, lam, np.random.default_rng(SEED + 1))
        z = np.zeros(H)
        res[f"{lam}"] = report(W, wins, Xte, yte, proto, z, f"lam={lam}")
        bA = calibrate(W, wins, Xtr, ytr, np.random.default_rng(0))
        res[f"{lam}+A"] = report(W, wins, Xte, yte, proto, bA, f"lam={lam}+A")
        np.savez_compressed(OUT / f"both_w_lam{lam}.npz", W=W.astype(np.float32),
                            wins=wins, b=bA)
        (OUT / "both_metrics.json").write_text(json.dumps(
            {"lams": LAMS, "results": res, "gamma": GAMMA,
             "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
