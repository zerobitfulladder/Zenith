"""Two rungs on Fashion-MNIST: a stroke layer read by an object layer.

L1   9x9 patches, 400 templates, top-1 per position, cnt/n step, NO class
     pressure (the patch-layer stream rule: online table = recount), trained
     without labels on all 12k images for 3 epochs, then frozen.
pool the 400 one-hot winners into 4x4 cells: one value per (template, cell),
     the max match score (cos^2) where a template won more than once in a
     cell. A 6400-long sparse, graded, nonnegative code.
L2   400 object templates over that code, cosine matching, no centring.
     Today's object-layer machinery: label 0.25 + belief 0.5 in the
     competition, purity step, hire on error (misread + committed winner ->
     cheapest template, row wiped, one hire per class per batch).
read the L2 winner's row in the L2 table, plus the belief.

Protocol: joint 0-9 (the split was dropped: the question is stacking performance).
Reference logged alongside: the L1 per-cell table read on the same 4x4 cells.
Downward residual logged, not used: for each image, at how many cells the L2
winner's most-expected L1 identity is absent from the code.

Usage:  uv run python two_rung.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import rf_sweep as R

OUT = HERE / "results"
SMOKE = "--smoke" in sys.argv
PS, GRID = 9, 4
def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
K1, K2, NL = R.K, arg("--k2", 400), R.NL
TOPK = arg("--topk", 1)              # read the k nearest L2 templates' rows
STRICT = "--strict" in sys.argv      # hire only if the true class is below chance in the read
EVIDENCE = "--evidence" in sys.argv  # L2 input = L1 table's per-cell evidence map, not the raw code
CONTRAST = "--contrast" in sys.argv       # message = relu(match - mean over templates), max-pooled (SoftHebb's Triangle)
CSTD = "--contrast-std" in sys.argv       # ... standardised per feature, signed matching at L2
if CSTD:
    CONTRAST = True
TAG = (f"k{K2}_top{TOPK}" + ("_strict" if STRICT else "") + ("_ev" if EVIDENCE else "")
       + ("_contrast" if CONTRAST else "") + ("_std" if CSTD else ""))
L1_EPOCHS = 1 if SMOKE else 3
L2_EPOCHS = 2 if SMOKE else 20
L2_BATCH = 512
BETA_L, BETA_B = 0.25, 0.5
SEEDS = [7] if SMOKE else [7, 8]
ALPHA, EPS = R.ALPHA, R.EPS


# ----------------------------------------------------------------- L1
def train_l1(rig, X, seed):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K1, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True); W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    n = cp.zeros(K1)
    for ep in range(L1_EPOCHS):
        order = np.arange(len(X)); rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            ids = cp.asarray(order[s:s + R.BATCH])
            Q, keep = rig.patches(X[ids]); V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
            S = V @ W.T; win = (S ** 2).argmax(1)
            wk, Vk = win[k], V[k]
            cnt = cp.bincount(wk, minlength=K1); live = cnt >= rig.min_s
            sums = cp.zeros((K1, rig.dim), cp.float32); cupyx.scatter_add(sums, wk, Vk)
            n[live] += cnt[live]
            eta = cp.clip(cnt[live] / n[live], R.ETA_MIN, 1.0)[:, None]
            W[live] += eta * (sums[live] / cnt[live, None] - W[live])
            W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    return W


def encode(rig, W, X, chunk=256):
    """Pooled graded code (n, K1*gg) and the L1 winners (n, npos) for the reference read."""
    n = len(X); gg = rig.gg
    C = cp.zeros((n, K1 * gg), cp.float32)
    idx = cp.zeros((n, rig.npos), cp.int32); keepm = cp.zeros((n, rig.npos), bool)
    for a in range(0, n, chunk):
        Q, keep = rig.patches(X[a:a + chunk]); m = len(Q)
        V = Q.reshape(-1, rig.dim)
        S = (V @ W.T) ** 2
        win = S.argmax(1); sc = S[cp.arange(len(V)), win]
        k = keep.reshape(-1)
        rows = cp.repeat(cp.arange(a, a + m), rig.npos)
        cells = cp.tile(rig.cells, m)
        flat = rows * (K1 * gg) + win * gg + cells
        Cf = C.reshape(-1)
        cupyx.scatter_max(Cf, flat[k], sc[k])          # collide -> max
        idx[a:a + m] = win.reshape(m, -1); keepm[a:a + m] = keep
    return C, idx, keepm


def encode_contrast(rig, W, X, chunk=256):
    """SoftHebb's message: every template's match minus the mean over templates at that
    position, rectified, then max-pooled into the cells. (n, K1*gg), dense nonnegative."""
    n = len(X); gg = rig.gg
    C = cp.zeros((n, K1, gg), cp.float32)
    for a in range(0, n, chunk):
        Q, keep = rig.patches(X[a:a + chunk]); m = len(Q)
        V = Q.reshape(-1, rig.dim)
        S = (V @ W.T) ** 2                                       # (m*npos, K1)
        Sc = cp.maximum(S - S.mean(1, keepdims=True), 0.0)
        Sc *= keep.reshape(-1)[:, None]
        Sc = Sc.reshape(m, rig.npos, K1)
        for c in range(gg):
            pos = rig.cells == c
            C[a:a + m, :, c] = Sc[:, pos, :].max(1)
    return C.reshape(n, -1)


def l1_table_acc(rig, itr, ktr, ytr, ite, kte, yte):
    T = rig.table_from(rig.build_table(itr, ytr, ktr))
    return rig.accuracy(T, (ite, kte), yte)


def evidence_map(rig, T1, idx, keepm):
    """(n, gg*NL): per cell, the summed L1 table rows of the winners there."""
    n = len(idx)
    C = cp.tile(rig.cells, n).reshape(n, rig.npos)
    E = cp.zeros((n, rig.gg, NL), cp.float32)
    rows = T1[idx, C] * keepm[..., None]                       # (n, npos, NL)
    for c in range(rig.gg):
        E[:, c] = (rows * (C == c)[..., None]).sum(1)
    return E.reshape(n, -1)


# ----------------------------------------------------------------- L2
def table2(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K2)
    pm = (N.sum(1, keepdims=True) + ALPHA * NL) / (N.sum() + ALPHA * K2 * NL)
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def purity2(N):
    p = (N + ALPHA) / (N.sum(1, keepdims=True) + ALPHA * NL)
    return p.max(1)


def belief2(T, w0):
    ev = T[w0]
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


SIGNED = False


def sim2(C, W2):
    S = C @ W2.T
    return S if SIGNED else S ** 2            # signed cosine on standardised inputs; cos^2 otherwise


def read2(W2, T, C):
    """Summed rows of the TOPK nearest templates; returns (evidence, nearest)."""
    S = sim2(C, W2)
    if TOPK == 1:
        w0 = S.argmax(1)
        return T[w0], w0
    top = cp.argsort(-S, axis=1)[:, :TOPK]
    return T[top].sum(1), top[:, 0]


def predict2(W2, T, C):
    ev, w0 = read2(W2, T, C)
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    return (ev + cp.log(q + 1e-9)).argmax(1), w0


def acc_split(pred, y):
    old = y < 5
    return (float((pred[old] == y[old]).mean()), float((pred[~old] == y[~old]).mean()),
            float((pred == y).mean()))


def residual_diag(W2, w2, C, gg):
    """Share of cells where the L2 winner's most-expected L1 identity is absent."""
    E = W2[w2].reshape(len(w2), K1, gg)
    exp_id = E.argmax(1)                                          # (n, gg)
    present = C.reshape(len(w2), K1, gg) > 0
    hit = cp.take_along_axis(present, exp_id[:, None, :], 1)[:, 0, :]
    return 1.0 - hit.mean(1)                                      # (n,)


def train_l2(phases, Ctr_all, ytr_all, Cte, yte, seed):
    rng = np.random.default_rng(seed)
    D = Ctr_all.shape[1]
    W2 = cp.asarray(rng.standard_normal((K2, D)), cp.float32)
    if not SIGNED:
        W2 = cp.abs(W2)
    W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N = cp.zeros((K2, NL)); out = []; hires = 0
    for pi, sel in enumerate(phases):
        C, y = Ctr_all[sel], ytr_all[sel]
        for ep in range(L2_EPOCHS):
            order = np.arange(len(C)); rng.shuffle(order)
            for s in range(0, len(order), L2_BATCH):
                ids = cp.asarray(order[s:s + L2_BATCH]); Cb, yb = C[ids], y[ids]
                T = table2(N)
                S = sim2(Cb, W2); err = 1.0 - S
                w0 = err.argmin(1); q = belief2(T, w0)
                score = err - BETA_L * T[:, yb].T - BETA_B * (q @ T.T)
                win = score.argmin(1)
                # hire on error
                pur = purity2(N)
                if STRICT:
                    # the READ (top-k rows) puts the true class below chance: not even close
                    evr, _ = read2(W2, T, Cb)
                    z = (evr - evr.mean(1, keepdims=True)) / (evr.std(1, keepdims=True) + 1e-9)
                    qr = cp.exp(z - z.max(1, keepdims=True)); qr /= qr.sum(1, keepdims=True)
                    pred = evr.argmax(1)
                    wrong = (pred != yb) & (pur[win] >= 0.5) & (qr[cp.arange(len(yb)), yb] < 1.0 / NL)
                else:
                    pred = q.argmax(1)
                    wrong = (pred != yb) & (pur[win] >= 0.5)
                if int(wrong.sum()):
                    tot = float(N.sum())
                    imp = cp.zeros(K2) if tot <= 0 else ((N / tot) * T).sum(1)
                    imp[win] = cp.inf
                    for yy in cp.unique(yb[wrong]).tolist():
                        grp = wrong & (yb == yy); t = int(imp.argmin()); imp[t] = cp.inf
                        win[grp] = t; N[t] = 0; hires += 1
                eta = ((1.0 - purity2(N)) / (1.0 - 1.0 / NL)).astype(cp.float32)
                N += cp.bincount(win * NL + yb, minlength=K2 * NL).reshape(K2, NL)
                cnt = cp.bincount(win, minlength=K2); live = cnt > 0
                sums = cp.zeros((K2, D), cp.float32); cupyx.scatter_add(sums, win, Cb)
                W2[live] += eta[live][:, None] * (sums[live] / cnt[live, None] - W2[live])
                W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
        # phase-end evaluation
        T_on = table2(N)
        Ntr = cp.zeros((K2, NL))
        _, wtr = predict2(W2, T_on, Ctr_all)
        Ntr += cp.bincount(wtr * NL + ytr_all, minlength=K2 * NL).reshape(K2, NL)
        T_re = table2(Ntr)
        p_re, w2 = predict2(W2, T_re, Cte); p_on, _ = predict2(W2, T_on, Cte)
        rd = residual_diag(W2, w2, Cte, GG) if not SIGNED else cp.zeros(len(yte))
        out.append({"recount": acc_split(p_re, yte), "online": acc_split(p_on, yte),
                    "dead": int((N.sum(1) == 0).sum()), "hires": hires,
                    "resid_correct": float(rd[p_re == yte].mean()),
                    "resid_wrong": float(rd[p_re != yte].mean()) if int((p_re != yte).sum()) else 0.0})
    return out


def main():
    global GG
    OUT.mkdir(exist_ok=True)
    R.GRID = GRID; R.BATCH = 128
    rig = R.Rig(PS); GG = rig.gg
    Xtr, ytr, Xte, yte = R.E.load("fashion_mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    print(f"[{TAG}] Fashion {len(ytr)}/{len(yte)}, L1 {PS}x{PS} x{K1} pooled {GRID}x{GRID}, "
          f"L2 x{K2} over {'evidence map' if EVIDENCE else 'contrast message' if CONTRAST else 'raw code'}, read top-{TOPK}, "
          f"{'strict' if STRICT else 'plain'} hire, seeds {SEEDS}\n")
    print(f"{'seed':<5}{'L1 table':>9} | {'JOINT L2 recount':>17}{'online':>8}{'dead':>6}{'hires':>7}"
          f"{'res ok/bad':>12}")
    res = {}
    for seed in SEEDS:
        t0 = time.time()
        W1 = train_l1(rig, Xtr, seed)
        Ctr, itr, ktr = encode(rig, W1, Xtr); Cte, ite, kte = encode(rig, W1, Xte)
        l1acc = l1_table_acc(rig, itr, ktr, ytr_g, ite, kte, yte_g)
        global SIGNED
        if EVIDENCE:
            T1 = rig.table_from(rig.build_table(itr, ytr_g, ktr))
            Ctr, Cte = evidence_map(rig, T1, itr, ktr), evidence_map(rig, T1, ite, kte)
            mu, sd = Ctr.mean(0, keepdims=True), Ctr.std(0, keepdims=True) + 1e-6
            Ctr, Cte = (Ctr - mu) / sd, (Cte - mu) / sd
            SIGNED = True
        if CONTRAST:
            Ctr, Cte = encode_contrast(rig, W1, Xtr), encode_contrast(rig, W1, Xte)
            if CSTD:
                mu, sd = Ctr.mean(0, keepdims=True), Ctr.std(0, keepdims=True) + 1e-6
                Ctr, Cte = (Ctr - mu) / sd, (Cte - mu) / sd
                SIGNED = True
            Ctr /= cp.linalg.norm(Ctr, axis=1, keepdims=True) + EPS
            Cte /= cp.linalg.norm(Cte, axis=1, keepdims=True) + EPS
        J = train_l2([cp.ones(len(ytr), bool)], Ctr, ytr_g, Cte, yte_g, seed)
        res[seed] = {"l1_table": l1acc, "joint": J,
                     "code_bits": float((Ctr > 0).sum(1).mean())}
        j = J[-1]
        print(f"{seed:<5}{l1acc:>9.4f} | {j['recount'][2]:>17.4f}{j['online'][2]:>8.4f}{j['dead']:>6d}"
              f"{j['hires']:>7d}{j['resid_correct']:>6.2f}/{j['resid_wrong']:.2f}   ({time.time()-t0:.0f}s)",
              flush=True)
    (OUT / f"two_rung_{TAG}.json").write_text(json.dumps(res, indent=1))
    print(f"\nmean: L1 table {np.mean([res[s]['l1_table'] for s in SEEDS]):.4f}   "
          f"joint L2 {np.mean([res[s]['joint'][-1]['recount'][2] for s in SEEDS]):.4f}   "
          f"code bits on {np.mean([res[s]['code_bits'] for s in SEEDS]):.0f}/{K1 * rig.gg}")


if __name__ == "__main__":
    main()
