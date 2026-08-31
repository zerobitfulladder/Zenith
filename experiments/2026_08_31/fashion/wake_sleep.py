"""Does sleeping replace punishing? Fashion-MNIST, stationary.

  wake        today's grading rule (image error + 4*(1-confidence)) with
              conscience, winner learns, NO repulsion
  wake+sleep  the same, and after every epoch the population dreams and the
              nearest rival of another class is pushed off each dream
  (for reference: wake + repulsion on real data was 0.8170 / 0.8278 with the
   reluctance, and it forgot 65 points on a stream)
"""

import json, time
from pathlib import Path
import numpy as np
from common import load, join, EPS, geo_step, CLASSES
from dopamine import calibrate
from both import rebuild
import sleepwake as sw

OUT = Path(__file__).resolve().parent / "results"
H, K, ETA, GAMMA, LAM = 40, 36, 0.5, 0.3, 4.0
EPOCHS, BATCH, MIN_S, SEED, N_IMG = 6, 128, 2, 0, 784
NIGHT = dict(n_dream=48, temp=0.8, eta_neg=0.1, cap=np.pi / 32)


def train(Xtr, ytr, do_sleep, rng):
    W = rng.standard_normal((H, K, N_IMG + 10))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)
    coef = sw.Coef(H, K)
    fixes = 0
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
                    coef.update(h, J[m] @ W[h].T)
                    coef.seen(h, err[m, h])
                np.add.at(wins[h], ytr[b][m], 1)
        if do_sleep:
            n, tot = sw.sleep(W, wins, coef, rng, **NIGHT)
            fixes += n
        print(f"  {'wake+sleep' if do_sleep else 'wake':<11} epoch {ep+1}/{EPOCHS}"
              f"   corrections {fixes}", flush=True)
    return W, wins, fixes


def report(W, wins, Xte, yte, b, tag):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([np.where(alive[None], rebuild(W, join(Xte[s:s + 2000]))[0],
                                 np.inf) for s in range(0, len(Xte), 2000)])
    got = claim[(E + b[None]).argmin(1)]
    acc = float((got == yte).mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    with np.errstate(all="ignore"):
        o = float(np.nanmean(np.nanmin(np.where(own, E, np.nan), 1)))
        t = float(np.nanmean(np.nanmin(np.where(~own & alive[None], E, np.nan), 1)))
    pur = float((wins.max(1) / np.maximum(wins.sum(1), 1))[alive].mean())
    print(f"  {tag:<12} acc {acc:.4f}   own-class err {o:.4f}   other {t:.4f}   "
          f"gap {t-o:+.4f}   purity {pur:.3f}   alive {int(alive.sum())}")
    return {"acc": acc, "own_class_err": o, "other_class_err": t, "gap": t - o,
            "purity": pur, "alive": int(alive.sum()),
            "per_class": {CLASSES[c]: float((got[yte == c] == c).mean())
                          for c in range(10)}}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    res = {}
    for do_sleep in (False, True):
        name = "wake+sleep" if do_sleep else "wake"
        W, wins, fixes = train(Xtr, ytr, do_sleep, np.random.default_rng(SEED + 1))
        res[name] = report(W, wins, Xte, yte, np.zeros(H), name)
        b = calibrate(W, wins, Xtr, ytr, np.random.default_rng(0))
        res[name + "+A"] = report(W, wins, Xte, yte, b, name + "+A")
        res[name]["sleep_corrections"] = fixes
        np.savez_compressed(OUT / f"ws_{name.replace('+','_')}.npz",
                            W=W.astype(np.float32), wins=wins, b=b)
    res["reference"] = {"wake+repulsion": 0.8170, "wake+repulsion+A": 0.8278}
    (OUT / "wake_sleep.json").write_text(json.dumps(
        {"results": res, "night": {k: (v if not isinstance(v, float) else v)
                                   for k, v in NIGHT.items()},
         "seconds": round(time.time() - t0, 1)}, indent=2, default=float))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
