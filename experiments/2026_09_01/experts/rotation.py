"""Does non-negativity plus overcompleteness break the rotation symmetry?

The claim, from early in the day: when every template is active and the only
objective is rebuilding, the layer pins down a SUBSPACE, not a basis. Any
rotation of the templates rebuilds equally well, so which particular directions
you get is decided by the initialisation, not by the data. Sparsity and
non-negativity are coordinate-dependent -- rotate a non-negative sparse code and
it stops being non-negative and sparse -- so they should force the data to
choose.

100 templates over 25-dimensional patches, four times overcomplete. All of them
active, coefficients found by non-negative ISTA, every template rotating toward
what the whole population failed to explain:

    a  <-  relu(a + s(B - aW)Wt - s*lam)      a few iterations
    W  <-  W + eta * aT (B - aW)              each template chases the leftover,
                                              weighted by its own activity

Three conditions:

    A  raw patches, non-negative templates AND coefficients   (parts, in theory)
    B  centred patches, signed templates, non-negative coefs
    C  centred patches, signed templates, UNCONSTRAINED coefs  <- the control,
                                                    which should be rotation-free

THE TEST is not what the templates look like -- it is whether two runs from two
different random starts find the SAME templates. Train each condition twice and
match the two sets up by cosine. If the data determines the templates, the two
sets agree. If only the subspace is determined, they agree no better than two
random sets of directions, which is measured here as the chance line.

No conscience, no minimum-sample gate, no revival. Just the rule.
"""

import json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E

OUT = Path(__file__).resolve().parent / "results"
N, EPOCHS, BATCH, ETA, LAM, ISTA = 100, 3, 2048, 0.05, 0.05, 20
EPS = 1e-12


def patches(X, y, rng, centred):
    U, _ = E.sample(X, y, rng, E.PER_IMG)          # centred + unit already
    if centred:
        return U
    P = np.empty_like(U)                            # raw, non-negative
    from numpy.lib.stride_tricks import sliding_window_view
    out = []
    for a in range(0, len(X), 256):
        V = sliding_window_view(X[a:a + 256], (5, 5), axis=(1, 2)).reshape(
            len(X[a:a + 256]), -1, 25)
        n = np.linalg.norm(V, axis=-1)
        keep = n > 0.35
        for i in range(len(V)):
            k = np.nonzero(keep[i])[0]
            if len(k):
                p = rng.choice(k, min(E.PER_IMG, len(k)), False)
                out.append(V[i, p] / n[i, p][:, None])
    return np.concatenate(out)


def codes(B, W, nonneg, L):
    a = np.zeros((len(B), len(W)), np.float32)
    s = 1.0 / max(L, 1e-6)
    for _ in range(ISTA):
        g = (B - a @ W) @ W.T
        a = a + s * g
        if nonneg:
            a = np.maximum(a - s * LAM, 0.0)
    return a


def train(P, seed, nonneg_a, nonneg_w):
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((N, P.shape[1]))) if nonneg_w else \
        rng.standard_normal((N, P.shape[1]))
    W /= np.linalg.norm(W, axis=1, keepdims=True) + EPS
    W = W.astype(np.float32)
    for ep in range(EPOCHS):
        order = rng.permutation(len(P))
        for s in range(0, len(order), BATCH):
            B = P[order[s:s + BATCH]].astype(np.float32)
            L = np.linalg.norm(W, 2) ** 2
            a = codes(B, W, nonneg_a, L)
            W = W + ETA * (a.T @ (B - a @ W)) / len(B)
            if nonneg_w:
                W = np.maximum(W, 0.0)
            W /= np.linalg.norm(W, axis=1, keepdims=True) + EPS
    return W, a


def match(A, Bm):
    """Greedy one-to-one pairing by |cosine|; mean of the matched values."""
    S = np.abs(A @ Bm.T)
    used, tot = set(), []
    for i in np.argsort(-S.max(1)):
        j = np.argsort(-S[i])
        j = next((x for x in j if x not in used), None)
        if j is None:
            break
        used.add(j); tot.append(S[i, j])
    return float(np.mean(tot))


def main():
    t0 = time.time()
    Xtr, ytr, _, _ = E.load()
    Pc = patches(Xtr, ytr, np.random.default_rng(1), True)
    Pr = patches(Xtr, ytr, np.random.default_rng(1), False)
    print(f"{len(Pc):,} centred patches, {len(Pr):,} raw patches, "
          f"{N} templates over 25 dims ({N/25:.0f}x overcomplete)\n", flush=True)

    r = np.random.default_rng(0)
    R1 = r.standard_normal((N, 25)); R1 /= np.linalg.norm(R1, axis=1, keepdims=True)
    R2 = r.standard_normal((N, 25)); R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
    chance = match(R1, R2)
    print(f"  chance line (two random sets of directions): {chance:.4f}\n", flush=True)

    conds = [("A raw, nonneg W and a", Pr, True, True),
             ("B centred, nonneg a", Pc, True, False),
             ("C centred, unconstrained", Pc, False, False)]
    res = {"chance": chance, "N": N}
    for name, P, na, nw in conds:
        W1, a1 = train(P, 10, na, nw)
        W2, _ = train(P, 99, na, nw)
        m = match(W1, W2)
        act = float((a1 > 1e-6).sum(1).mean())
        rec = float(np.linalg.norm(P[:2000] - codes(P[:2000].astype(np.float32), W1, na,
                                  np.linalg.norm(W1, 2) ** 2) @ W1, axis=1).mean())
        res[name] = {"match": m, "over_chance": m - chance,
                     "active_per_patch": act, "rebuild_error": rec}
        print(f"  {name:<26} match {m:.4f}  (chance {chance:.4f}, "
              f"+{m-chance:.4f})   active {act:.1f}/{N}   error {rec:.4f}", flush=True)

        order = np.argsort(-np.abs(W1).sum(1))
        tile = np.full((10 * 6 - 1, 10 * 6 - 1), np.nan)
        for i, j in enumerate(order[:100]):
            rr, cc = divmod(i, 10)
            tile[rr * 6:rr * 6 + 5, cc * 6:cc * 6 + 5] = W1[j].reshape(5, 5)
        plt.figure(figsize=(6, 6.4))
        v = np.nanmax(np.abs(tile))
        plt.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        plt.title(f"{name}\nmatch across two seeds {m:.3f} (chance {chance:.3f})",
                  fontsize=10)
        plt.xticks([]); plt.yticks([]); plt.tight_layout()
        plt.savefig(OUT / f"rotation_{name[0]}.png", dpi=130); plt.close()

    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "rotation.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
