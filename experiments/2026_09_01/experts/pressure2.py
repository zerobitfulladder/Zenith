"""More rounds, to find where the pressure turns self-confirming, and the
weighting stacked on top.

Round 2 was still climbing -- nearest-mean 0.9100 -> 0.9380 -> 0.9460, gains
shrinking but not stopped. This pushes beta until it breaks, which is the number
worth knowing: the point where the templates stop learning from the data and
start learning from the classifier's own opinion.

And the two interventions should be independent. The PRESSURE reshapes what the
templates are; the WEIGHTING reshapes how their codes are compared. Same table,
different jobs, no overlap. Both reported every round.

    stop when table accuracy slides monotonically -- that is the loop
    confirming itself, and it is what the settling run did this afternoon
    (0.9423 -> 0.9257 -> 0.9083 -> 0.8857)
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, readouts as R, single as SG
import pressure as PR

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = PR.GRID, 10, 1e-12
BETAS = [0.0, 0.75, 1.5, 2.5, 4.0, 6.0]


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)


def ladder(Atr, ytr, Ate, yte):
    M = unit(np.stack([Atr[ytr == c].mean(0) for c in range(NL)]))
    nm = float(((unit(Ate) @ M.T).argmax(1) == yte).mean())
    net = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax",
                    hidden=0, epochs=30)
    lin = float((R.predict_net(net, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())
    return nm, lin


def full_eval(W, Xtr, ytr, Xte, yte, beta):
    h = len(W)
    itr0, ite0 = SG.winners(W, Xtr), SG.winners(W, Xte)
    T0 = B.table(itr0, ytr, h, GRID)
    _, qtr = PR.belief(itr0, T0); _, qte = PR.belief(ite0, T0)
    if beta > 0:
        itr = PR.biased_winners(W, Xtr, T0, qtr)
        ite = PR.biased_winners(W, Xte, T0, qte)
    else:
        itr, ite = itr0, ite0
    T = B.table(itr, ytr, h, GRID)
    acc = float((B.score(ite, T, GRID).argmax(1) == yte).mean())
    Atr, Ate = PR.pooled(itr, h), PR.pooled(ite, h)
    nm, lin = ladder(Atr, ytr, Ate, yte)
    w = T.std(axis=2).T[None]                                   # (1, ncell, h)
    n = len(Atr)
    Awtr = (Atr.reshape(n, GRID * GRID, h) * w).reshape(n, -1)
    Awte = (Ate.reshape(len(Ate), GRID * GRID, h) * w).reshape(len(Ate), -1)
    nmw, linw = ladder(Awtr, ytr, Awte, yte)
    return dict(table=acc, nm=nm, lin=lin, nm_w=nmw, lin_w=linw), itr, T, qtr


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(E.SEED + 1)
    U, I, C = PR.tagged(Xtr, ytr, rng, E.PER_IMG)
    W = np.load(OUT / "single_unlabelled.npz")["W"].astype(np.float64)
    print(f"{len(U):,} patches, {len(W)} templates\n", flush=True)
    print("  beta   table    nearest-mean  +weight   linear   +weight   gap", flush=True)

    res, prev = [], None
    for r, beta in enumerate(BETAS):
        if r > 0:
            _, _, T, q = full_eval(W, Xtr, ytr, Xte, yte, BETAS[r - 1])
            W = PR.train_biased(W, U, I, C, q, T, beta, np.random.default_rng(100 + r))
        m, _, _, _ = full_eval(W, Xtr, ytr, Xte, yte, beta)
        m.update(round=r, beta=beta, gap=m["lin"] - m["nm"],
                 gap_w=m["lin_w"] - m["nm_w"])
        res.append(m)
        print(f"  {beta:<5} {m['table']:.4f}   {m['nm']:.4f}      {m['nm_w']:.4f}   "
              f"{m['lin']:.4f}   {m['lin_w']:.4f}   {m['gap']:.4f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(OUT / f"pressure2_r{r}.npz", W=W.astype(np.float32))
        if prev is not None and m["table"] < prev - 0.02:
            print("  -- table accuracy dropped 2 points; stopping", flush=True)
            break
        prev = m["table"]

    best = max(res, key=lambda d: d["nm_w"])
    print(f"\n  best match-against-average: {best['nm_w']:.4f} at beta {best['beta']} "
          f"(started at 0.9100)")
    (OUT / "pressure2.json").write_text(json.dumps(
        {"rounds": res, "reference": {"no pressure, no weight": 0.9100,
                                      "weight only": 0.9283,
                                      "pressure only, beta 1.5": 0.9460}}, indent=2))


if __name__ == "__main__":
    main()
