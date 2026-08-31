"""The ceiling: put the teacher back, on top of the stack.

Everything the stack does is teacher-free -- the label enters only as a
co-occurring second stream in the grading. This adds the two mechanisms that
need an external verdict, to see what they are worth on top of composition:

    repulsion   during training, the hypercolumn that would win the READ gate
                and names it wrongly is rotated away from that input
    reluctance  at read time, a learned offset per hypercolumn, moved when it
                speaks and is wrong

Four arms on the best config from sweep2. For scale: one layer with both of
these scored 0.8278 on this split, teacher-free scored 0.6868, and the stack
without them scored 0.7190.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c
c.N_TRAIN, c.N_TEST = 20000, 5000
import l2                                                          # noqa: E402
from l2 import join, rebuild, errors_only                          # noqa: E402
from l2_identity import identity_map                               # noqa: E402
from common import EPS, geo_step                                   # noqa: E402
from dopamine import geo_step_neg                                  # noqa: E402

DS, L1KEY, SEED = "fashion_mnist", "H32_K8", 0
LAM, GAMMA, ETA, K2 = 4.0, 0.3, 0.5, 16
EPOCHS, BATCH, MIN_S = 6, 128, 2
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
CAL_EPOCHS, CAL_LR = 6, 0.002


def train(F, y, H, repel, rng, tag):
    nf = F.shape[1]
    W = rng.standard_normal((H, K2, nf + 10))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), np.int64)
    f = np.full(H, 1.0 / H)
    for ep in range(EPOCHS):
        order = rng.permutation(len(F))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(F[b], y[b]), join(F[b])
            e, R = rebuild(W, Q, nf)
            L = R[:, :, nf:]
            conf = L[np.arange(len(b)), :, y[b]] / np.maximum(
                np.linalg.norm(L, axis=2), EPS)
            score = e + LAM * (1.0 - conf) - GAMMA * (1.0 / H - f)[None]
            win = score.argmin(1)
            cnt = np.bincount(win, minlength=H)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            for h in range(H):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], y[b][m], 1)
            if repel and ep >= WARMUP:
                claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                fw = e.argmin(1)
                same = claim[None, :] == y[b][:, None]
                er = np.where(same, e, np.inf).min(1)
                bad = ((claim[fw] != y[b]) & np.isfinite(er) &
                       (e[np.arange(len(b)), fw] / np.maximum(er, EPS) > WINDOW))
                for h in np.unique(fw[bad]):
                    p = bad & (fw == h)
                    if p.sum() >= MIN_S:
                        W[h] = geo_step_neg(W[h], Q[p], ETA_NEG, CAP_NEG)
        print(f"  {tag:<14} epoch {ep+1}/{EPOCHS}", flush=True)
    return W, wins


def errs(W, F, alive):
    nf = F.shape[1]
    return np.where(alive[None], errors_only(W, join(F), nf), np.inf)


def calibrate(E, y, claim, rng, H):
    b = np.zeros(H)
    for _ in range(CAL_EPOCHS):
        for i in rng.permutation(len(y)):
            s = E[i] + b
            w = int(s.argmin())
            if claim[w] == y[i]:
                continue
            same = np.where(claim == y[i])[0]
            if not len(same):
                continue
            b[w] += CAL_LR; b[int(same[s[same].argmin()])] -= CAL_LR
    return b


def main():
    t0 = time.time()
    sw = json.loads((OUT / "sweep2.json").read_text())["results"]
    best = max(sw, key=lambda k: sw[k]["acc"])
    g = int(best.split("_")[0][4:]); H = int(best.split("_H")[1])
    print(f"best teacher-free config: {best} ({sw[best]['acc']:.4f}) -> "
          f"grid {g}, H2 {H}", flush=True)

    W1 = np.load(OUT / f"conv1_{DS}.npz")[L1KEY].astype(np.float64)
    Xtr, ytr, Xte, yte = c.load(DS)
    Ftr, Fte = identity_map(W1, Xtr, gridn=g), identity_map(W1, Xte, gridn=g)

    res = {}
    for repel in (False, True):
        W, wins = train(Ftr, ytr, H, repel, np.random.default_rng(SEED + 1),
                        "repulsion" if repel else "teacher-free")
        alive = wins.sum(1) > 0
        claim = np.where(alive, wins.argmax(1), -1)
        Ete, Etr = errs(W, Fte, alive), errs(W, Ftr, alive)
        base = "with repulsion" if repel else "teacher-free"
        res[base] = float((claim[Ete.argmin(1)] == yte).mean())
        b = calibrate(Etr, ytr, claim, np.random.default_rng(0), H)
        res[base + " + reluctance"] = float(
            (claim[(Ete + b[None]).argmin(1)] == yte).mean())
        for k in (base, base + " + reluctance"):
            print(f"  {k:<32} {res[k]:.4f}", flush=True)
    res["one layer, with teacher"] = 0.8278
    res["one layer, teacher-free"] = 0.6868
    res["logistic on pixels"] = 0.8512
    (OUT / "l2_teacher.json").write_text(json.dumps(
        {"config": best, "results": res, "seconds": round(time.time() - t0, 1)},
        indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
