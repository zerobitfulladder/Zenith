"""Stop asking who is loudest. Ask who draws it best.

Every run so far picked the training winner with the answer key in hand: the
expert most confident in the TRUE label learns. But at test time we pick by
who rebuilds the image best. Two different competitions, and we trained the
one we never read.

Here the training winner is the best DRAWER -- lowest reconstruction error on
the image alone, label not consulted -- and it then learns the whole joined
vector, image and label together. The bet is that whoever draws the picture
best will also draw the label best.

  draw     the winner always learns
  draw+B   the winner learns if its claimed class is right, and is pushed AWAY
           if it is wrong: reinforce what contributed, weaken what did not
"""

import json, time
from pathlib import Path
import numpy as np
from common import load, join, EPS, geo_step, CLASSES
from dopamine import rebuild, geo_step_neg, calibrate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
H, K, ETA = 40, 36, 0.5
EPOCHS, BATCH, MIN_S = 6, 128, 2
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
SEED, N_IMG, GAMMA = 0, 784, 0.3


def train(Xtr, ytr, reject, rng, conscience=False):
    d = N_IMG + 10
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)                       # running win fraction
    n_rep = 0
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(Xtr[b], ytr[b]), join(Xtr[b])
            err, _ = rebuild(W, Q)
            e_sel = err - GAMMA * (1.0 / H - f)[None] if conscience else err
            win = e_sel.argmin(1)                     # the best drawer, no label
            if conscience:
                cnt = np.bincount(win, minlength=H)
                f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            if reject and ep >= WARMUP:
                claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                bad = claim[win] != ytr[b]
                # only push when a right-class expert is genuinely nearby
                same = claim[None, :] == ytr[b][:, None]
                e_right = np.where(same, err, np.inf).min(1)
                bad &= (err[np.arange(len(b)), win] /
                        np.maximum(e_right, EPS) > WINDOW) & np.isfinite(e_right)
            else:
                bad = np.zeros(len(b), bool)
            for h in range(H):
                m = (win == h) & ~bad
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
                p = (win == h) & bad
                if p.sum() >= MIN_S:
                    W[h] = geo_step_neg(W[h], Q[p], ETA_NEG, CAP_NEG)
                    n_rep += int(p.sum())
        print(f"  {('draw' + ('+C' if conscience else '') + ('+B' if reject else '')):<9} epoch {ep+1}/{EPOCHS}"
              f"   repelled {n_rep}", flush=True)
    return W, wins, n_rep


def report(W, wins, Xte, yte, b, tag):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([rebuild(W, join(Xte[s:s + 2000]))[0]
                        for s in range(0, len(Xte), 2000)])
    E = np.where(alive[None], E, np.inf)
    got = claim[(E + b[None]).argmin(1)]
    acc = float((got == yte).mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    with np.errstate(all="ignore"):
        o = np.nanmin(np.where(own, E, np.nan), axis=1)
        t = np.nanmin(np.where(~own & alive[None], E, np.nan), axis=1)
    pur = wins.max(1) / np.maximum(wins.sum(1), 1)
    per = {CLASSES[c]: float((got[yte == c] == c).mean()) for c in range(10)}
    print(f"  {tag:<8} acc {acc:.4f}   own-class err {np.nanmean(o):.4f}   "
          f"other {np.nanmean(t):.4f}   gap {np.nanmean(t-o):+.4f}   "
          f"alive {int(alive.sum())}   classes covered {len(set(claim[alive]))}/10"
          f"   mean purity {pur[alive].mean():.3f}")
    return {"acc": acc, "own_class_err": float(np.nanmean(o)),
            "other_class_err": float(np.nanmean(t)),
            "gap": float(np.nanmean(t - o)), "alive": int(alive.sum()),
            "classes_covered": len(set(claim[alive].tolist())),
            "mean_purity": float(pur[alive].mean()),
            "purity": pur.tolist(), "per_class": per, "claim": claim.tolist()}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    res = {}
    W0, wins0, _ = train(Xtr, ytr, False, np.random.default_rng(SEED + 1))
    WB, winsB, nrep = train(Xtr, ytr, True, np.random.default_rng(SEED + 1))
    print(f"\n(draw+B repelled {nrep} sample-steps)\n\nreading:", flush=True)
    z = np.zeros(H)
    res["draw"] = report(W0, wins0, Xte, yte, z, "draw")
    bA = calibrate(W0, wins0, Xtr, ytr, np.random.default_rng(0))
    res["draw+A"] = report(W0, wins0, Xte, yte, bA, "draw+A")
    res["draw+B"] = report(WB, winsB, Xte, yte, z, "draw+B")
    bAB = calibrate(WB, winsB, Xtr, ytr, np.random.default_rng(0))
    res["draw+A+B"] = report(WB, winsB, Xte, yte, bAB, "draw+A+B")

    WC, winsC, _ = train(Xtr, ytr, False, np.random.default_rng(SEED + 1), True)
    res["draw+C"] = report(WC, winsC, Xte, yte, z, "draw+C")
    bC = calibrate(WC, winsC, Xtr, ytr, np.random.default_rng(0))
    res["draw+C+A"] = report(WC, winsC, Xte, yte, bC, "draw+C+A")
    WCB, winsCB, _ = train(Xtr, ytr, True, np.random.default_rng(SEED + 1), True)
    res["draw+C+B"] = report(WCB, winsCB, Xte, yte, z, "draw+C+B")
    bCB = calibrate(WCB, winsCB, Xtr, ytr, np.random.default_rng(0))
    res["draw+C+A+B"] = report(WCB, winsCB, Xte, yte, bCB, "draw+C+A+B")

    print("\nwho claimed what (draw):")
    for h in np.where(wins0.sum(1) > 0)[0]:
        w = wins0[h] / wins0[h].sum()
        top = np.argsort(w)[::-1][:3]
        print(f"  e{h:<3} {wins0[h].sum():>6} samples   " +
              "  ".join(f"{CLASSES[c][:9]} {w[c]:.2f}" for c in top if w[c] > .02))

    (OUT / "draw_metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "eta": ETA, "eta_neg": ETA_NEG,
                    "window": WINDOW, "warmup": WARMUP, "epochs": EPOCHS},
         "results": res, "repelled_steps": nrep,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / "draw_weights.npz", W_draw=W0.astype(np.float32),
                        wins_draw=wins0, W_drawB=WB.astype(np.float32),
                        wins_drawB=winsB, b_A=bA, b_AB=bAB)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
