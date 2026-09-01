"""Experts over [ink ; position ; label], with the templates left dense inside.

The intuition being tested is the user's original one: an expert should be a
specialist, and the templates inside it should cover the variations of what it
specialises in. That could never work on a bare 5x5 patch -- measured, a patch
carries essentially no class (0.1162 against 0.10 chance), so there was nothing
to specialise on. Adding WHERE gives it something.

The asymmetry that makes this safe, and that sank the earlier attempt:

    position   known at training AND at reading   -> may drive the competition
    label      known only at training             -> may NOT

Putting the label in the competition dropped things from 0.4143 to 0.1543,
because training built a partition that reading could not reproduce. Position
has no such problem.

    vector    [ ink 25 ; cell 16 ; label 10 ]
    compete   across experts, on ink error + position error
    inside    DENSE -- all K templates rebuild together, no in-expert competition
    learn     winner only, rotates toward the whole vector, label included
    read      label blank, the winner's rebuild of it is the answer

The thing to watch is not the accuracy. It is whether the experts finally show
class purity above 0.154, and whether the network's OWN answer (its label
rebuild, no external table) closes on the 0.9423 the counted table gets. That
would mean the table has moved inside the network.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E
import vote as V
import readouts as R

OUT = Path(__file__).resolve().parent / "results"
SIDE, GRID, NL = V.SIDE, V.GRID, 10
NCELL = GRID * GRID
D_INK, D_POS, D_LAB = E.PS * E.PS, NCELL, NL
DIM = D_INK + D_POS + D_LAB
H, K, EPOCHS, BATCH, MIN_S, ETA, GAMMA = 160, 8, 3, 2048, 4, 0.5, 1.0
RHO_LAB, SEED = 1.0, 0
RHOS = [0.0, 1.0, 2.0]
B_ = SIDE // GRID
CELL = ((np.arange(SIDE * SIDE) // SIDE) // B_) * GRID + ((np.arange(SIDE * SIDE) % SIDE) // B_)
EPS = 1e-12


def join(U, cell, lab, rho_pos):
    V_ = np.zeros((len(U), DIM), np.float32)
    V_[:, :D_INK] = U
    if rho_pos > 0:
        V_[np.arange(len(U)), D_INK + cell] = rho_pos
    if lab is not None:
        V_[np.arange(len(U)), D_INK + D_POS + lab] = RHO_LAB
    return V_ / np.maximum(np.linalg.norm(V_, axis=1, keepdims=True), EPS)


def sample(X, y, rng, per_img):
    U, L, C = [], [], []
    for a in range(0, len(X), 256):
        Q, keep = E.patches(X[a:a + 256])
        for i in range(len(Q)):
            idx = np.nonzero(keep[i])[0]
            if len(idx):
                p = rng.choice(idx, min(per_img, len(idx)), False)
                U.append(Q[i, p]); L.append(np.full(len(p), y[a + i])); C.append(CELL[p])
    return (np.concatenate(U).astype(np.float32), np.concatenate(L),
            np.concatenate(C))


def sel_err(W, B):
    """Error on the blocks that are known at read time: ink and position."""
    S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
    Rb = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
    D = B[None] - Rb
    return (D[:, :, :D_INK + D_POS] ** 2).sum(-1), Rb


def train(Vtr, rng):
    W = rng.standard_normal((H, K, DIM)).astype(np.float32)
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    f = np.full(H, 1.0 / H)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Vtr))
        for s in range(0, len(order), BATCH):
            B = Vtr[order[s:s + BATCH]]
            if len(B) < MIN_S * 2:
                continue
            e = sel_err(W, B)[0] + GAMMA * (f - 1.0 / H)[:, None]
            win = e.argmin(0)
            cnt = np.bincount(win, minlength=H)
            f = 0.995 * f + 0.005 * (cnt / max(cnt.sum(), 1))
            for j in np.nonzero(cnt >= MIN_S)[0]:
                W[j] = E.geo_step(W[j].astype(np.float64),
                                  B[win == j].astype(np.float64), ETA).astype(np.float32)
        print(f"    epoch {ep+1}/{EPOCHS}", flush=True)
    return W


def read(W, X, rho_pos, chunk=64):
    """Winner and its label rebuild at every position; label blank going in."""
    idx = np.full((len(X), SIDE * SIDE), -1, np.int16)
    sm = np.zeros((len(X), NL), np.float32)
    pat_hit = pat_n = 0
    ys = []
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        m = len(Q)
        B = join(Q.reshape(-1, D_INK), np.tile(CELL, m), None, rho_pos)
        e, Rb = sel_err(W, B)
        w = e.argmin(0)
        op = Rb[w, np.arange(len(B)), D_INK + D_POS:]
        k = keep.reshape(-1)
        idx[a:a + m] = np.where(keep.reshape(m, -1), w.reshape(m, -1), -1)
        sm[a:a + m] = (op.reshape(m, -1, NL) * keep.reshape(m, -1, 1)).sum(1)
        ys.append((w, op, k, m))
    return idx, sm


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(SEED + 1)
    U, L, C = sample(Xtr, ytr, rng, E.PER_IMG)
    print(f"{len(U):,} patches, joint dim {DIM} = {D_INK} ink + {D_POS} cell + {D_LAB} label",
          flush=True)
    res = {}
    for rho in RHOS:
        t1 = time.time()
        Vtr = join(U, C, L, rho)
        W = train(Vtr, np.random.default_rng(SEED + 1))
        itr, _ = read(W, Xtr, rho)
        ite, sm = read(W, Xte, rho)

        # what the network says on its own
        own = float((sm.argmax(1) == yte).mean())
        # per-patch class call
        cnt = np.zeros(H); ld = np.zeros((H, NL)); pd = np.zeros((H, NCELL))
        r, c = np.nonzero(ite >= 0)
        np.add.at(cnt, ite[r, c], 1.0)
        np.add.at(ld, (ite[r, c], yte[r]), 1.0)
        np.add.at(pd, (ite[r, c], CELL[c]), 1.0)
        live = cnt > 0
        lp = ld / np.maximum(ld.sum(1, keepdims=True), 1)
        pp = pd / np.maximum(pd.sum(1, keepdims=True), 1)
        # counted table on top, for comparison with 0.9423
        T1 = V.llr_table(itr, ytr, H, True)
        T0 = V.llr_table(itr, ytr, H, False)
        row = {"rho_pos": rho, "own_answer": own,
               "table_bag": float((V.llr_score(ite, T0, False).argmax(1) == yte).mean()),
               "table_by_cell": float((V.llr_score(ite, T1, True).argmax(1) == yte).mean()),
               "class_purity": float(lp.max(1)[live].mean()),
               "cell_purity": float(pp.max(1)[live].mean()),
               "live": int(live.sum()), "seconds": round(time.time() - t1, 1)}
        res[str(rho)] = row
        print(f"  rho_pos {rho:<4} live {row['live']}/{H}  OWN {own:.4f}  "
              f"table-bag {row['table_bag']:.4f}  table-cell {row['table_by_cell']:.4f}  "
              f"class-purity {row['class_purity']:.3f}  cell-purity {row['cell_purity']:.3f}  "
              f"({time.time()-t1:.0f}s)", flush=True)
        np.savez_compressed(OUT / f"weights_pos{rho}.npz", W=W, counts=cnt,
                            lab_dist=ld, cell_dist=pd)
    res["reference"] = {"shared experts, own answer": 0.4143,
                        "shared experts, table by cell": 0.9423,
                        "shared experts, class purity": 0.154}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "posexp.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
