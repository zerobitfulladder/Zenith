"""Does earned resistance cost anything when the world is stationary?

Same rule as the λ=4 run (grading on drawing + naming, conscience, repulsion),
with correction scaled by how active each hypercolumn still is:

    plasticity[h] = min(1, f[h] * H)

On a stationary dataset every live hypercolumn keeps winning its share, so
plasticity should sit near 1 and this should change nothing. If it does change
something, the mechanism is not free and the continual gain has a price here.
Reference: 0.8170 raw, 0.8278 with the reluctance.
"""

import json, time
from pathlib import Path
import numpy as np
from common import load, join, EPS, geo_step, CLASSES
from dopamine import geo_step_neg
from both import rebuild

OUT = Path(__file__).resolve().parent / "results"
H, K, ETA, GAMMA, LAM = 40, 36, 0.5, 0.3, 4.0
EPOCHS, BATCH, MIN_S, SEED, N_IMG = 6, 128, 2, 0, 784
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
CAL_EPOCHS, CAL_LR = 6, 0.002


def train(Xtr, ytr, consol, rng):
    W = rng.standard_normal((H, K, N_IMG + 10))
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
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
            if ep >= WARMUP:
                claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                fw = err.argmin(1)
                same = claim[None, :] == ytr[b][:, None]
                e_right = np.where(same, err, np.inf).min(1)
                bad = ((claim[fw] != ytr[b]) & np.isfinite(e_right) &
                       (err[np.arange(len(b)), fw] /
                        np.maximum(e_right, EPS) > WINDOW))
                pl = np.minimum(1.0, f * H) if consol else np.ones(H)
                for h in np.unique(fw[bad]):
                    p = bad & (fw == h)
                    if p.sum() >= MIN_S and pl[h] > 1e-3:
                        W[h] = geo_step_neg(W[h], Q[p], ETA_NEG * pl[h], CAP_NEG)
        print(f"  {'consolidated' if consol else 'plain':<13} epoch {ep+1}/{EPOCHS}",
              flush=True)
    return W, wins, np.minimum(1.0, f * H)


def calibrate(W, wins, Xtr, ytr, rng, plastic):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([rebuild(W, join(Xtr[s:s + 2000]))[0]
                        for s in range(0, len(Xtr), 2000)])
    E = np.where(alive[None], E, np.inf)
    b = np.zeros(H)
    for _ in range(CAL_EPOCHS):
        for i in rng.permutation(len(ytr)):
            s = E[i] + b
            w = s.argmin()
            if claim[w] == ytr[i]:
                continue
            same = np.where(claim == ytr[i])[0]
            if not len(same):
                continue
            c = same[s[same].argmin()]
            b[w] += CAL_LR * plastic[w]; b[c] -= CAL_LR * plastic[c]
    return b


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    res = {}
    W, wins, pl = train(Xtr, ytr, True, np.random.default_rng(SEED + 1))
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([np.where(alive[None], rebuild(W, join(Xte[s:s + 2000]))[0],
                                 np.inf) for s in range(0, len(Xte), 2000)])
    raw = float((claim[E.argmin(1)] == yte).mean())
    b = calibrate(W, wins, Xtr, ytr, np.random.default_rng(0), pl)
    withA = float((claim[(E + b[None]).argmin(1)] == yte).mean())
    print(f"\nconsolidated: raw {raw:.4f}   +reluctance {withA:.4f}"
          f"   (plain repulsion was 0.8170 / 0.8278)")
    print(f"plasticity of live hypercolumns: min {pl[alive].min():.2f}  "
          f"mean {pl[alive].mean():.2f}  "
          f"{int((pl[alive] > 0.9).sum())}/{int(alive.sum())} still fully plastic")
    res = {"raw": raw, "with_reluctance": withA,
           "plain_repulsion": {"raw": 0.8170, "with_reluctance": 0.8278},
           "plasticity_mean_live": float(pl[alive].mean()),
           "plasticity_min_live": float(pl[alive].min()),
           "fully_plastic": int((pl[alive] > 0.9).sum()), "alive": int(alive.sum())}
    (OUT / "consolidate.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "consolidate.npz", W=W.astype(np.float32),
                        wins=wins, b=b, plastic=pl)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
