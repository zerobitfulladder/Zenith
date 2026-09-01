"""Table and templates learning together, in one pass, from scratch.

The alternating version worked: freeze beliefs, train templates, rebuild table,
repeat -- six rounds, and match-against-average went 0.9100 -> 0.9460. But it
needed a trained vocabulary to start from and several passes.

This does it all at once, from random templates and an empty table:

    for each batch of images
        winners on ink alone            -> current table -> belief q
        winners again, with beta * q.T  -> the biased assignment
        counts += (winner, cell, label) for every patch
        winning templates rotate toward their patches
        rebuild the log-ratio table from the counts

Everything co-adapts continuously. The table shapes which patches a template
gets while the template shapes what the table counts.

The risk is a runaway: an early bad table biases assignment, which reinforces
the bad table. Early on that is self-limiting, since an empty table gives a flat
belief and therefore no bias -- but there is no guarantee it stays that way.
Accuracy is measured every epoch so a slide is visible while it happens.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, readouts as R, single as SG, pressure as PR

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = PR.GRID, 10, 1e-12
H, BETA, EPOCHS, IMG_BATCH, ETA, ALPHA = 180, 1.5, 3, 64, 0.5, 1.0
import sys


def _arg(i, default, cast):        # safe when imported from another script
    try:
        return cast(sys.argv[i])
    except (IndexError, ValueError, TypeError):
        return default


DECAY = _arg(1, 1.0, float)
ANNEAL = _arg(2, "", str) == "anneal"   # eta = max(1/n, floor), as km.py
ETA_MIN = 0.02
CELL = PR.CELL
NC = GRID * GRID


def table_from(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * len(N))
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * len(N) * NL))
    return (np.log(pc) - np.log(pm)).astype(np.float32)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(7)
    Wf = rng.standard_normal((H, 25)).astype(np.float32)
    Wf -= Wf.mean(1, keepdims=True)
    Wf /= np.linalg.norm(Wf, axis=1, keepdims=True) + EPS
    N = np.zeros((H, NC, NL))
    nwin = np.zeros(H)
    T = table_from(N)
    print(f"from scratch: {H} random templates, empty table, beta {BETA}, "
          f"decay {DECAY}, anneal {ANNEAL}\n", flush=True)

    order0 = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        rng.shuffle(order0)
        for s in range(0, len(order0), IMG_BATCH):
            ids = order0[s:s + IMG_BATCH]
            Q, keep = E.patches(Xtr[ids])
            m = len(Q)
            P = Q.reshape(-1, 25).astype(np.float32)
            k = keep.reshape(-1)
            wn = (Wf ** 2).sum(1)
            Sc = P @ Wf.T
            err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
            cells = np.tile(CELL, m)
            # pass 1: belief from ink-only winners
            w0 = err.argmin(1)
            ev = T[w0, cells]                                  # (m*P, NL)
            sc = (ev * k[:, None]).reshape(m, -1, NL).sum(1)
            z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)
            q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            # pass 2: biased winners
            bias = np.einsum('ik,ihk->ih', np.repeat(q, S.SIDE * S.SIDE, 0),
                             np.ascontiguousarray(T.transpose(1, 0, 2))[cells],
                             optimize=True)
            win = (err - BETA * bias).argmin(1)
            # count, then learn
            lab = np.repeat(ytr[ids], S.SIDE * S.SIDE)
            if DECAY < 1.0:
                N *= DECAY                      # forget at the rate templates drift
            np.add.at(N, (win[k], cells[k], lab[k]), 1.0)
            T = table_from(N)
            for j in np.unique(win[k]):
                sel = P[k][win[k] == j]
                if len(sel) >= 4:
                    nwin[j] += len(sel)
                    e = max(1.0 / nwin[j] * len(sel), ETA_MIN) if ANNEAL else ETA
                    Wf[j] = E.geo_step(Wf[j:j + 1].astype(np.float64),
                                       sel.astype(np.float64), min(e, ETA))[0].astype(np.float32)
        W = Wf[:, None, :].astype(np.float64)
        ite = SG.winners(W, Xte)
        acc = float((B.score(ite, T, GRID).argmax(1) == yte).mean())
        dead = int((N.sum((1, 2)) == 0).sum())
        print(f"  epoch {ep+1}/{EPOCHS}  table {acc:.4f}   dead {dead}/{H}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    W = Wf[:, None, :].astype(np.float64)
    ite = SG.winners(W, Xte)
    online = float((B.score(ite, T, GRID).argmax(1) == yte).mean())
    m, _, _, _ = __import__("pressure2").full_eval(W, Xtr, ytr, Xte, yte, BETA)
    m["online_table"] = online
    print(f"\n  ONLINE table (the one it actually accumulated): {online:.4f}")
    print(f"\n  co-adaptive, one pass:  table {m['table']:.4f}  "
          f"match-avg {m['nm']:.4f}  linear {m['lin']:.4f}  gap {m['lin']-m['nm']:.4f}")
    print(f"  alternating, 6 rounds:  table 0.9540  match-avg 0.9460  "
          f"linear 0.9810  gap 0.0350")
    np.savez_compressed(OUT / f"coadapt_d{DECAY}_{ANNEAL}.npz", W=W.astype(np.float32), N=N)
    (OUT / f"coadapt_d{DECAY}_{ANNEAL}.json").write_text(json.dumps(
        {"beta": BETA, "epochs": EPOCHS, **m,
         "alternating": {"table": 0.9540, "nm": 0.9460, "lin": 0.9810}}, indent=2))


if __name__ == "__main__":
    main()
