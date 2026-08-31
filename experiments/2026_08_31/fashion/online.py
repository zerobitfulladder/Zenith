"""One sample at a time. No batches anywhere in the rule.

    for each arriving sample (x, y):
        every hypercolumn draws it and names it
        the best-graded one learns it          <- one geodesic step, one sample
        the one that would WIN THE READ GATE, if it names it wrongly, is
        pushed away from it -- but only if it has learned something within
        the last N samples

N is the only new number: how long expertise stays open to criticism. It is a
property of the architecture, not of the implementation -- there is no batch.

    python online.py                      stationary Fashion-MNIST
    python online.py --split              split MNIST, 0-4 then 5-9
    --window N   --no-consol   --no-repel
"""

import json, sys, time
from pathlib import Path
import numpy as np
from common import load, join, EPS, geo_step, CLASSES
from dopamine import geo_step_neg

OUT = Path(__file__).resolve().parent / "results"
H, K, ETA, GAMMA, LAM = 40, 36, 0.5, 0.3, 4.0
EPOCHS, SEED, N_IMG = 6, 0, 784
ETA_NEG, CAP_NEG, WINDOW = 0.25, np.pi / 16, 0.8
CAL_EPOCHS, CAL_LR = 6, 0.002
TAU = 0.001                      # per-sample decay of the win fraction
SPLIT = "--split" in sys.argv
CONSOL = "--no-consol" not in sys.argv
REPEL = "--no-repel" not in sys.argv
NWIN = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 200


def one(W, q):
    """Errors and label completions of every hypercolumn for a single vector."""
    S = W @ q                                     # (H, K)
    R = np.einsum("hk,hkd->hd", S, W)             # (H, D)
    e = np.linalg.norm(q[None, :N_IMG] - R[:, :N_IMG], axis=1) / max(
        np.linalg.norm(q[:N_IMG]), EPS)
    return e, R[:, N_IMG:]


def errors(W, X, chunk=2000):
    h, k, d = W.shape
    E = np.empty((len(X), h))
    for s in range(0, len(X), chunk):
        Q = join(X[s:s + chunk])
        S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
        R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
        E[s:s + chunk] = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG],
                                        axis=2) / np.maximum(
            np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return E


def stream(W, wins, state, X, y, rng, epochs):
    """Feed samples one by one."""
    f, since, seen = state["f"], state["since"], state["seen"]
    n_rep = 0
    for ep in range(epochs):
        for i in rng.permutation(len(y)):
            j = join(X[i:i + 1], y[i:i + 1])[0]
            q = join(X[i:i + 1])[0]
            e, L = one(W, q)
            conf = L[:, y[i]] / np.maximum(np.linalg.norm(L, axis=1), EPS)
            g = int((e + LAM * (1.0 - conf) - GAMMA * (1.0 / H - f)).argmin())
            W[g] = geo_step(W[g], j[None], ETA)
            wins[g, y[i]] += 1
            f *= (1.0 - TAU); f[g] += TAU
            since += 1; since[g] = 0
            seen += 1
            if REPEL and seen > len(y) // 2:          # a warm-up, in samples
                alive = wins.sum(1) > 0
                claim = np.where(alive, wins.argmax(1), -1)
                ea = np.where(alive, e, np.inf)
                r = int(ea.argmin())
                if claim[r] != y[i]:
                    same = np.where(claim == y[i])[0]
                    if len(same):
                        er = ea[same].min()
                        if ea[r] / max(er, EPS) > WINDOW and (
                                not CONSOL or since[r] <= NWIN):
                            W[r] = geo_step_neg(W[r], q[None], ETA_NEG, CAP_NEG)
                            n_rep += 1
        print(f"    epoch {ep+1}/{epochs}   pushes {n_rep}", flush=True)
    state.update(f=f, since=since, seen=seen)
    return n_rep


def calibrate(W, wins, X, y, rng, allowed):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.where(alive[None], errors(W, X), np.inf)
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
            c = int(same[s[same].argmin()])
            if allowed[w]:
                b[w] += CAL_LR
            if allowed[c]:
                b[c] -= CAL_LR
    return b


def accs(W, wins, Xte, yte, b):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.where(alive[None], errors(W, Xte), np.inf)
    pred = claim[(E + b[None]).argmin(1)]
    A, B = np.isin(yte, range(5)), np.isin(yte, range(5, 10))
    return (float((pred[A] == yte[A]).mean()), float((pred[B] == yte[B]).mean()),
            float((pred == yte).mean()))


def main():
    t0 = time.time()
    ds = "mnist" if SPLIT else "fashion_mnist"
    Xtr, ytr, Xte, yte = load(ds)
    rng = np.random.default_rng(SEED + 1)
    W = rng.standard_normal((H, K, N_IMG + 10))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    state = {"f": np.full(H, 1.0 / H), "since": np.zeros(H, np.int64), "seen": 0}
    tag = f"online_{'split' if SPLIT else 'fashion'}_" \
          f"{'consol' + str(NWIN) if CONSOL else 'plain'}{'' if REPEL else '_norepel'}"
    res = {"window": NWIN, "consolidation": CONSOL, "repulsion": REPEL}

    phases = [("1", list(range(5))), ("2", list(range(5, 10)))] if SPLIT \
        else [("all", list(range(10)))]
    b = np.zeros(H)
    for ph, digits in phases:
        idx = np.nonzero(np.isin(ytr, digits))[0]
        print(f"  phase {ph}: {len(idx)} samples", flush=True)
        before = wins.copy()
        n_rep = stream(W, wins, state, Xtr[idx], ytr[idx], rng, EPOCHS)
        raw = accs(W, wins, Xte, yte, np.zeros(H))
        allowed = np.ones(H, bool) if not CONSOL else (state["since"] <= NWIN)
        b = calibrate(W, wins, Xtr[idx], ytr[idx], np.random.default_rng(0), allowed)
        wa = accs(W, wins, Xte, yte, b)
        res[f"phase {ph}"] = {"raw": raw, "with_reluctance": wa, "pushes": n_rep,
                              "correctable": int(allowed.sum()),
                              "live": int((wins.sum(1) > 0).sum())}
        pw = (wins - before).sum(1)
        liveB = before.sum(1) > 0
        if ph == "2":
            res["to_previously_live"] = float(pw[liveB].sum() / max(pw.sum(), 1))
            res["changed_claim"] = int(sum(
                before[h].argmax() != wins[h].argmax() for h in np.where(liveB)[0]))
        print(f"  [phase {ph}] raw 0-4 {raw[0]:.4f}  5-9 {raw[1]:.4f}  "
              f"all {raw[2]:.4f}  |  +A 0-4 {wa[0]:.4f}  5-9 {wa[1]:.4f}  "
              f"all {wa[2]:.4f}   correctable {int(allowed.sum())}/{H}")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(OUT / f"{tag}.npz", W=W.astype(np.float32), wins=wins, b=b)
    print(f"\ndone in {time.time()-t0:.0f}s -> {tag}")


if __name__ == "__main__":
    main()
