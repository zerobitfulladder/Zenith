"""How much machinery does the code need before it classifies?

A readout ladder, weakest first. The rung where the accuracy arrives tells you
what kind of structure the code actually has:

    nearest class mean   no learning at all -- average each class, compare by
                         cosine. Works only if classes are BLOBS.
    1-nearest-neighbour  no learning, but no averaging either. Works if the
                         class is a manifold rather than a blob.
    linear probe         a learned hyperplane. Works if classes are separable
                         but not clustered -- needs a rotation first.
    mlp probe            a learned curved boundary.

If nearest-mean ~ linear, the code is clustered and nothing has to be
transformed. If nearest-mean << linear, the information is there but the code
is not a stable encoding of the class, it is a puzzle a classifier can solve.

The circularity check that matters: the shaped code is built from a belief that
is right 94% of the time, so of course it clusters. AGREEMENT measures whether
the code has become nothing BUT a re-encoding of that belief -- how often
nearest-mean returns exactly what was injected. Near 100% means shaping added
stability and no information.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, vote as V, settle as S, readouts as R, shaped as SH

OUT = Path(__file__).resolve().parent / "results"
NL, BETAS = 10, [0.0, 0.4, 1.0]


def codes(W, X, Tc, beta, h, chunk=500):
    C, bel = [], []
    for a in range(0, len(X), chunk):
        F, KP = S.fits(W, X[a:a + chunk])
        w0, s = SH.forward(F, KP, Tc)
        bel.append(s.argmax(1))
        C.append(S.code(w0 if beta == 0 else SH.reshape(F, KP, Tc, s, beta), h))
    return np.concatenate(C), np.concatenate(bel)


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def ladder(Atr, ytr, Ate, yte):
    out = {}
    M = unit(np.stack([Atr[ytr == c].mean(0) for c in range(NL)]))
    p_mean = (unit(Ate) @ M.T).argmax(1)
    out["nearest_mean"] = float((p_mean == yte).mean())
    sim = unit(Ate) @ unit(Atr).T
    out["1nn"] = float((ytr[sim.argmax(1)] == yte).mean())
    for name, hid in (("linear", 0), ("mlp", 256)):
        n = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax",
                      hidden=hid, epochs=30)
        out[name] = float(
            (R.predict_net(n, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())
    return out, p_mean


def main():
    t0 = time.time()
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()
    itr = np.concatenate([np.where(kp, f.argmax(1), -1) for f, kp in
                          [S.fits(W, Xtr[a:a + 1000]) for a in range(0, len(Xtr), 1000)]])
    Tc = V.llr_table(itr, ytr, h, True)[:, S.CELL, :]

    res = {}
    for beta in BETAS:
        Atr, _ = codes(W, Xtr, Tc, beta, h)
        Ate, bel = codes(W, Xte, Tc, beta, h)
        row, p_mean = ladder(Atr, ytr, Ate, yte)
        w, b, g, _ = S.simstats(Ate[:600], yte[:600])
        row["gap"] = g
        row["belief_acc"] = float((bel == yte).mean())
        row["agreement_with_belief"] = float((p_mean == bel).mean())
        ok = bel == yte
        row["nearest_mean_when_belief_right"] = float((p_mean[ok] == yte[ok]).mean())
        row["nearest_mean_when_belief_wrong"] = float((p_mean[~ok] == yte[~ok]).mean())
        res[str(beta)] = row
        print(f"  beta {beta:<4} nearest-mean {row['nearest_mean']:.4f}  "
              f"1-NN {row['1nn']:.4f}  linear {row['linear']:.4f}  mlp {row['mlp']:.4f}  "
              f"| gap {g:+.3f}  agrees with belief {row['agreement_with_belief']:.4f}",
              flush=True)

    print("\n  when the injected belief was RIGHT / WRONG, nearest-mean gets:")
    for beta in BETAS:
        r = res[str(beta)]
        print(f"    beta {beta:<4} {r['nearest_mean_when_belief_right']:.4f} / "
              f"{r['nearest_mean_when_belief_wrong']:.4f}")
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "probe.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
