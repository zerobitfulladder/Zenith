"""The tally again, with far more experts -- and a tiling that actually uses them.

The earlier sweep collapsed: at N=512 only 57 experts ever won, because random
seeds in 72-d space land off the data manifold and never win anything. Here
experts are seeded FROM DATA, so every one starts somewhere real, and the fit
error uses ||x - W'Wx||^2 = 1 - ||Wx||^2 (exact for orthonormal templates,
which the geodesic rule produces) so large N stays affordable.
"""
import json, time, numpy as np
from imitate import (pc, unit, sense, fly, demos, TMAX, LEVELS, HOVER, OUT)
from place_code import PlaceCode
from compete import geo_step

NS, K, M_MOT, SUB = [512, 1024, 2048, 4096], 3, 64, 120000
EPS = 1e-12


def tile(X, n, k, rng, epochs=4, batch=1024, eta=0.35):
    d = X.shape[1]
    W = X[rng.choice(len(X), n, replace=False)][:, None, :] \
        + 0.05 * rng.standard_normal((n, k, d))          # seeded FROM DATA
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    f = np.full(n, 1.0 / n)
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(o), batch):
            B = X[o[s:s + batch]]
            if len(B) < 16:
                continue
            S = (B @ W.reshape(n * k, d).T).reshape(len(B), n, k)
            e = 1.0 - (S ** 2).sum(2) + 2.0 * (f - 1.0 / n)[None]   # (B, n)
            win = e.argmin(1)
            c = np.bincount(win, minlength=n)
            f = 0.97 * f + 0.03 * (c / max(c.sum(), 1))
            for h in np.nonzero(c >= 2)[0]:
                W[h] = geo_step(W[h], B[win == h], eta)
    return W


def assign(W, Q, chunk=8192):
    n, k, d = W.shape
    out = np.empty(len(Q), np.int32)
    for s in range(0, len(Q), chunk):
        B = Q[s:s + chunk]
        S = (B @ W.reshape(n * k, d).T).reshape(len(B), n, k)
        out[s:s + chunk] = (1.0 - (S ** 2).sum(2)).argmin(1)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    Sit, Act = demos(rng, 1200)
    Q = unit(pc.encode(Sit))
    pcm = PlaceCode(2, nb=12, lo=0.0, hi=TMAX, halfw=1.5)
    Qm = unit(pcm.encode(Act))
    Wm = tile(Qm, M_MOT, 3, np.random.default_rng(2))
    midx = assign(Wm, Qm)
    acts = np.stack([Act[midx == m].mean(0) if (midx == m).any() else np.full(2, HOVER)
                     for m in range(M_MOT)])
    print(f"{len(Sit)} demo steps; motor track {len(np.unique(midx))}/{M_MOT} used\n",
          flush=True)
    sub = rng.choice(len(Q), min(SUB, len(Q)), replace=False)
    res = {}
    for n in NS:
        W = tile(Q[sub], n, K, np.random.default_rng(1))
        idx = assign(W, Q)
        used = len(np.unique(idx))
        tal = np.zeros((n, M_MOT)); np.add.at(tal, (idx, midx), 1.0)

        def pol(sit, W=W, tal=tal):
            q = unit(pc.encode(sit[None]))[0]
            S = (q @ W.reshape(n * K, -1).T).reshape(n, K)
            h = int((1.0 - (S ** 2).sum(1)).argmin())
            r = tal[h]
            return np.clip(acts[int(r.argmax())] if r.sum() else np.full(2, HOVER), 0, TMAX)
        sc, st = fly(pol, 7, 200)
        res[n] = {"success": sc, "steps": st, "used": int(used)}
        print(f"  N={n:>5}   experts used {used:>5}/{n}   TALLY success {sc:.3f}"
              f"   steps {st:5.1f}   [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(OUT / f"tallybig_{n}.npz", W=W.astype(np.float32),
                            tally=tal, acts=acts,
                            const=np.zeros((n, 2)), lin=np.zeros((n, 7, 2)),
                            has_lin=np.zeros(n, bool))
        (OUT / f"tallybig_{n}.json").write_text(json.dumps(
            {"kind": "module", "module": "tally_big",
             "dir": "experiments/2026_08_31/drone_imitate", "n_exp": n}))
    (OUT / "tallybig_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


def load_policy(npz, cfg):
    W, tal, acts = npz["W"].astype(np.float64), npz["tally"], npz["acts"]
    n, K_, _ = W.shape

    def act(s, tgt, lv_prev=None):
        q = unit(pc.encode(sense(s, tgt)[None]))[0]
        S = (q @ W.reshape(n * K_, -1).T).reshape(n, K_)
        r = tal[int((1.0 - (S ** 2).sum(1)).argmin())]
        a = np.clip(acts[int(r.argmax())] if r.sum() else np.full(2, HOVER), 0, TMAX)
        return (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
    return act


if __name__ == "__main__":
    main()
