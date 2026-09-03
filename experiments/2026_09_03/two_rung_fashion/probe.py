"""A linear probe at the end, like the paper's head, at three depths.

    L1 evidence   the stroke layer's per-cell table rows (160 numbers)   -- the 09-02 probe
    L1 contrast   the standardised contrast message (6400)               -- "1 layer + head"
    L2 contrast   the object layer's match profile, centred + ReLU (K2)  -- "2 layers + head"

Probe = softmax regression on standardised features, SGD + momentum, weight
decay, 100 epochs, final-epoch test accuracy (no early stopping). Counted
reads of the same layers alongside. Fashion 12k/3k, 2 seeds.

Usage:  uv run python probe.py [--k2 1600]
"""
import sys, time, json
import numpy as np
import cupy as cp
sys.argv += ["--contrast-std"]
import two_rung as TR
import rf_sweep as R

K2 = TR.K2
SEEDS = [7, 8]


def probe(Xtr, ytr, Xte, yte, epochs=100, lr=0.05, wd=1e-4, bs=256, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    D = Xtr.shape[1]; NL = 10
    W = cp.zeros((D, NL), cp.float32); b = cp.zeros(NL, cp.float32)
    vW = cp.zeros_like(W); vb = cp.zeros_like(b)
    Y = cp.eye(NL, dtype=cp.float32)[ytr]
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), bs):
            ids = cp.asarray(order[s:s + bs]); xb, yb = Xtr[ids], Y[ids]
            z = xb @ W + b; z -= z.max(1, keepdims=True)
            p = cp.exp(z); p /= p.sum(1, keepdims=True)
            g = (p - yb) / len(ids)
            gW = xb.T @ g + wd * W; gb = g.sum(0)
            vW = 0.9 * vW - lr * gW; vb = 0.9 * vb - lr * gb
            W += vW; b += vb
    return float(((Xte @ W + b).argmax(1) == yte).mean())


def l2_profile(W2, C):
    S = TR.sim2(C, W2)                                 # signed cosine (standardised inputs)
    return cp.maximum(S - S.mean(1, keepdims=True), 0.0)


def main():
    R.GRID = TR.GRID; R.BATCH = 128
    rig = R.Rig(TR.PS)
    Xtr, ytr, Xte, yte = R.E.load("fashion_mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    print(f"Fashion {len(ytr)}/{len(yte)}, L2 x{K2}, contrast-std message, seeds {SEEDS}\n")
    print(f"{'seed':<5}{'L1 table':>9}{'L2 counted':>11} | {'probe: L1 evidence':>19}{'L1 contrast':>13}{'L2 contrast':>13}")
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        W1 = TR.train_l1(rig, Xtr, seed)
        Ctr, itr, ktr = TR.encode(rig, W1, Xtr); Cte, ite, kte = TR.encode(rig, W1, Xte)
        l1acc = TR.l1_table_acc(rig, itr, ktr, ytr_g, ite, kte, yte_g)
        T1 = rig.table_from(rig.build_table(itr, ytr_g, ktr))
        Etr, Ete = TR.evidence_map(rig, T1, itr, ktr), TR.evidence_map(rig, T1, ite, kte)
        Ktr, Kte = TR.encode_contrast(rig, W1, Xtr), TR.encode_contrast(rig, W1, Xte)
        mu, sd = Ktr.mean(0, keepdims=True), Ktr.std(0, keepdims=True) + 1e-6
        Ktr, Kte = (Ktr - mu) / sd, (Kte - mu) / sd
        Ktr /= cp.linalg.norm(Ktr, axis=1, keepdims=True) + TR.EPS
        Kte /= cp.linalg.norm(Kte, axis=1, keepdims=True) + TR.EPS
        TR.SIGNED = True
        # object layer, as in two_rung (train_l2 returns metrics; we need W2 -> rerun its loop here)
        out, W2 = train_l2_with_W(Ktr, ytr_g, Kte, yte_g, seed)
        l2acc = out["recount"][2]
        p_ev = probe(Etr, ytr_g, Ete, yte_g, seed=seed)
        p_k = probe(Ktr, ytr_g, Kte, yte_g, seed=seed)
        p_l2 = probe(l2_profile(W2, Ktr), ytr_g, l2_profile(W2, Kte), yte_g, seed=seed)
        rows.append((l1acc, l2acc, p_ev, p_k, p_l2))
        print(f"{seed:<5}{l1acc:>9.4f}{l2acc:>11.4f} | {p_ev:>19.4f}{p_k:>13.4f}{p_l2:>13.4f}   ({time.time()-t0:.0f}s)", flush=True)
    m = np.mean(rows, 0)
    print(f"{'mean':<5}{m[0]:>9.4f}{m[1]:>11.4f} | {m[2]:>19.4f}{m[3]:>13.4f}{m[4]:>13.4f}")
    (TR.OUT / f"probe_k{K2}.json").write_text(json.dumps({"rows": rows, "mean": m.tolist()}, indent=1))


def train_l2_with_W(Ctr, ytr, Cte, yte, seed):
    """two_rung.train_l2, returning W2 as well (single phase)."""
    rng = np.random.default_rng(seed)
    D = Ctr.shape[1]; NL = TR.NL; EPS = TR.EPS
    W2 = cp.asarray(rng.standard_normal((K2, D)), cp.float32)
    W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N = cp.zeros((K2, NL)); hires = 0
    for ep in range(TR.L2_EPOCHS):
        order = np.arange(len(Ctr)); rng.shuffle(order)
        for s in range(0, len(order), TR.L2_BATCH):
            ids = cp.asarray(order[s:s + TR.L2_BATCH]); Cb, yb = Ctr[ids], ytr[ids]
            T = TR.table2(N)
            S = TR.sim2(Cb, W2); err = 1.0 - S
            w0 = err.argmin(1); q = TR.belief2(T, w0)
            score = err - TR.BETA_L * T[:, yb].T - TR.BETA_B * (q @ T.T)
            win = score.argmin(1)
            pur = TR.purity2(N); pred = q.argmax(1)
            wrong = (pred != yb) & (pur[win] >= 0.5)
            if int(wrong.sum()):
                tot = float(N.sum())
                imp = cp.zeros(K2) if tot <= 0 else ((N / tot) * T).sum(1)
                imp[win] = cp.inf
                for yy in cp.unique(yb[wrong]).tolist():
                    grp = wrong & (yb == yy); t = int(imp.argmin()); imp[t] = cp.inf
                    win[grp] = t; N[t] = 0; hires += 1
            eta = ((1.0 - TR.purity2(N)) / (1.0 - 1.0 / NL)).astype(cp.float32)
            N += cp.bincount(win * NL + yb, minlength=K2 * NL).reshape(K2, NL)
            cnt = cp.bincount(win, minlength=K2); live = cnt > 0
            sums = cp.zeros((K2, D), cp.float32)
            import cupyx; cupyx.scatter_add(sums, win, Cb)
            W2[live] += eta[live][:, None] * (sums[live] / cnt[live, None] - W2[live])
            W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    T_on = TR.table2(N)
    _, wtr = TR.predict2(W2, T_on, Ctr)
    T_re = TR.table2(cp.bincount(wtr * NL + ytr, minlength=K2 * NL).reshape(K2, NL).astype(cp.float64))
    p_re, _ = TR.predict2(W2, T_re, Cte)
    return {"recount": TR.acc_split(p_re, yte)}, W2


if __name__ == "__main__":
    main()
