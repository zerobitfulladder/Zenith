"""One unit at layer two, looking at the whole L1 map.

L1 is the trained 5x5 unit (H=32, K=8), slid with stride 1 over the image. At
each of the 24x24 positions it emits its winner's DENSE code -- 8 signed
coefficients, no indication of which hypercolumn produced them. So L2's input
is 576 x 8 = 4608 numbers, plus a 10-slot label block at matched energy.

L2 is a single unit: H2 hypercolumns of K2 minicolumns each, competing for the
whole map. The rule is the teacher-free base and nothing else --

    score = rebuild error of the L1 map + LAM*(1 - confidence in the label)
            - conscience

-- the winner rotates all its minicolumns toward the joined vector.

Control arm: the identical unit trained on raw pixels instead of L1 codes. If
L1 is doing anything, the two differ.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c                                                  # noqa: E402
from common import EPS, geo_step, center_norm                      # noqa: E402

L1KEY, DS = "H32_K8", "mnist"
H2, K2, LAM, GAMMA, ETA = 20, 16, 4.0, 0.3, 0.5
EPOCHS, BATCH, MIN_S, SEED = 6, 128, 2, 0
N_TR, N_TE = 12000, 3000


def l1_codes(W1, X, cache):
    """(n, 576*8) -- the winner's dense code at every position, identity dropped."""
    if cache.exists():
        z = np.load(cache)
        if len(z["F"]) == len(X):
            return z["F"].astype(np.float64)
    H, K, _ = W1.shape
    out = np.zeros((len(X), 576 * K), np.float32)
    for a in range(0, len(X), c.CHUNK_IMG):
        Q, keep = c.prep(c.grid(X[a:a + c.CHUNK_IMG]))
        npos = Q.shape[1]
        kf = keep.reshape(-1)
        e, S = c.errors(W1, Q.reshape(-1, c.PS * c.PS)[kf])
        win = e.argmin(1)
        code = S[np.arange(len(win)), win]                 # signed, (m, K)
        idx = np.nonzero(kf)[0]
        img, pos = a + idx // npos, idx % npos
        out[img[:, None].repeat(K, 1),
            pos[:, None] * K + np.arange(K)[None, :]] = code
    np.savez_compressed(cache, F=out)
    return out.astype(np.float64)


def join(F, y=None, rho=1.0):
    V = np.zeros((len(F), F.shape[1] + 10))
    V[:, :F.shape[1]] = F
    if y is not None:
        L = np.zeros((len(F), 10)); L[np.arange(len(F)), y] = 1.0
        g = rho * np.linalg.norm(F, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, F.shape[1]:] = L * g[:, None]
    return center_norm(V)


def errors_only(W, Q, nf, chunk=1000):
    """(n, H) rebuild error, without ever building the (n, H, D) reconstruction.

    One hypercolumn at a time, reduced to a scalar immediately. The naive form
    allocates 300x the size of the model and then throws it away.
    """
    h, k, d = W.shape
    E = np.empty((len(Q), h))
    for a in range(0, len(Q), chunk):
        B = Q[a:a + chunk]
        qn = np.maximum(np.linalg.norm(B[:, :nf], axis=1), EPS)
        for j in range(h):
            r = (B @ W[j].T) @ W[j]
            r[:, :nf] -= B[:, :nf]
            E[a:a + len(B), j] = np.linalg.norm(r[:, :nf], axis=1) / qn
    return E


def rebuild(W, Q, nf):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :nf] - R[:, :, :nf], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :nf], axis=1)[:, None], EPS)
    return e, R


def train(Ftr, ytr, rng, tag):
    nf = Ftr.shape[1]
    W = rng.standard_normal((H2, K2, nf + 10))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H2, 10), np.int64)
    f = np.full(H2, 1.0 / H2)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Ftr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(Ftr[b], ytr[b]), join(Ftr[b])
            e, R = rebuild(W, Q, nf)
            L = R[:, :, nf:]
            conf = L[np.arange(len(b)), :, ytr[b]] / np.maximum(
                np.linalg.norm(L, axis=2), EPS)
            score = e + LAM * (1.0 - conf) - GAMMA * (1.0 / H2 - f)[None]
            win = score.argmin(1)
            cnt = np.bincount(win, minlength=H2)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            for h in range(H2):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
        print(f"  {tag:<12} epoch {ep+1}/{EPOCHS}", flush=True)
    return W, wins


def report(W, wins, Fte, yte, tag):
    nf = Fte.shape[1]
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    e = np.where(alive[None], errors_only(W, join(Fte), nf), np.inf)
    pick = e.argmin(1)
    acc = float((claim[pick] == yte).mean())
    rec = float(e[np.arange(len(yte)), pick].mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    with np.errstate(all="ignore"):
        o = float(np.nanmean(np.nanmin(np.where(own, e, np.nan), 1)))
    pur = float((wins.max(1) / np.maximum(wins.sum(1), 1))[alive].mean())
    share = np.bincount(pick, minlength=H2) / len(yte)
    print(f"  [{tag}] accuracy {acc:.4f}   rebuild(input) {rec:.4f}   "
          f"own-class rebuild {o:.4f}   alive {int(alive.sum())}/{H2}   "
          f"purity {pur:.3f}   classes {len(set(claim[alive].tolist()))}/10   "
          f"busiest {share.max()*100:.1f}%")
    return {"acc": acc, "rebuild_input": rec, "own_class_rebuild": o,
            "alive": int(alive.sum()), "purity": pur, "busiest": float(share.max()),
            "classes_covered": len(set(claim[alive].tolist())),
            "claim": claim.tolist(), "wins": wins.tolist()}


def main():
    t0 = time.time()
    W1 = np.load(OUT / f"conv1_{DS}.npz")[L1KEY].astype(np.float64)
    Xtr, ytr, Xte, yte = c.load(DS)
    Xtr, ytr = Xtr[:N_TR], ytr[:N_TR]
    Xte, yte = Xte[:N_TE], yte[:N_TE]
    print(f"L1 {L1KEY} -> dense-only codes; L2 = {H2} hypercolumns x {K2} "
          f"minicolumns", flush=True)
    Ftr = l1_codes(W1, Xtr, OUT / f"codes_tr_{L1KEY}.npz")
    Fte = l1_codes(W1, Xte, OUT / f"codes_te_{L1KEY}.npz")
    print(f"  L2 input {Ftr.shape[1]} numbers "
          f"({float((Ftr != 0).mean())*100:.0f}% nonzero)", flush=True)

    res, Ws = {}, {}
    for tag, A, B in (("on L1 codes", Ftr, Fte),
                      ("on raw pixels", Xtr.reshape(len(Xtr), -1),
                       Xte.reshape(len(Xte), -1))):
        W, wins = train(A, ytr, np.random.default_rng(SEED + 1), tag)
        res[tag] = report(W, wins, B, yte, tag)
        Ws[tag] = (W, wins)
    (OUT / "l2.json").write_text(json.dumps(
        {"L1": L1KEY, "H2": H2, "K2": K2, "lam": LAM, "n_train": N_TR,
         "results": res, "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / "l2_weights.npz",
                        W_codes=Ws["on L1 codes"][0].astype(np.float32),
                        wins_codes=Ws["on L1 codes"][1],
                        W_pixels=Ws["on raw pixels"][0].astype(np.float32),
                        wins_pixels=Ws["on raw pixels"][1])
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
