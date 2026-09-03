"""L2's residual guides allocation at L1. Both layers train together.

Per batch: L1 codes the images (top-1 per position, pooled 4x4, max on
collision); L2 competes and learns as in two_rung.py; then for every image L2
misreads with a committed winner, the RESIDUAL CELLS are found -- cells where
the L2 winner's most-expected L1 identity is absent from the code -- and the
patches at those positions are handed to the nearest L1 template that is
uncommitted at that cell (L1 per-cell table purity < 0.5). A new stroke
identity is allocated where the object layer needed a distinction. L1 then
takes its normal cnt/n step on all its winners, reassignments included.

Arms:  frozen      L1 frozen after 3 unsupervised epochs   (= two_rung.py)
       learns      L1 keeps learning during L2 training, no message
       residual    L1 keeps learning, L2's residual reassigns patches

Usage:  uv run python joint_rung.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import cupyx
import two_rung as TR
import rf_sweep as R

OUT = TR.OUT
K1, K2, NL, EPS = TR.K1, TR.K2, TR.NL, TR.EPS
BETA_L, BETA_B = TR.BETA_L, TR.BETA_B
SEEDS = TR.SEEDS
L2_EPOCHS, L2_BATCH = TR.L2_EPOCHS, TR.L2_BATCH
ARMS = ["frozen", "learns", "residual", "expected"]
if "--expected-only" in sys.argv:
    ARMS = ["expected"]
if TR.SMOKE:
    ARMS = ["expected"]


def code_batch(rig, W1, Xb):
    Q, keep = rig.patches(Xb); m = len(Q)
    V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
    S = (V @ W1.T) ** 2
    win = S.argmax(1); sc = S[cp.arange(len(V)), win]
    cells = cp.tile(rig.cells, m)
    C = cp.zeros(m * K1 * rig.gg, cp.float32)
    rows = cp.repeat(cp.arange(m), rig.npos)
    cupyx.scatter_max(C, (rows * K1 * rig.gg + win * rig.gg + cells)[k], sc[k])
    return C.reshape(m, -1), V, k, win, cells


def run(rig, arm, Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    W1 = TR.train_l1(rig, Xtr, seed)
    n1 = cp.zeros(K1) + 1.0                      # L1 counts (unknown from pretraining; start soft)
    N1 = cp.zeros((K1, rig.gg, NL))              # L1 per-cell table, counted online
    D = K1 * rig.gg
    W2 = cp.abs(cp.asarray(rng.standard_normal((K2, D)), cp.float32))
    W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N2 = cp.zeros((K2, NL)); hires2 = 0; hires1 = 0
    for ep in range(L2_EPOCHS):
        order = np.arange(len(Xtr)); rng.shuffle(order)
        for s in range(0, len(order), L2_BATCH):
            ids = cp.asarray(order[s:s + L2_BATCH]); yb = ytr[ids]; m = len(ids)
            Cb, V, k, win1, cells = code_batch(rig, W1, Xtr[ids])
            lab = cp.repeat(yb, rig.npos)
            # ---- L2 step (two_rung.py, top-1 read, plain hire)
            T = TR.table2(N2)
            S = Cb @ W2.T; err = 1.0 - S ** 2
            w0 = err.argmin(1); q = TR.belief2(T, w0)
            score = err - BETA_L * T[:, yb].T - BETA_B * (q @ T.T)
            win2 = score.argmin(1)
            pur2 = TR.purity2(N2); pred = q.argmax(1)
            wrong = (pred != yb) & (pur2[win2] >= 0.5)
            if int(wrong.sum()):
                tot = float(N2.sum())
                imp = cp.zeros(K2) if tot <= 0 else ((N2 / tot) * T).sum(1)
                imp[win2] = cp.inf
                for yy in cp.unique(yb[wrong]).tolist():
                    grp = wrong & (yb == yy); t = int(imp.argmin()); imp[t] = cp.inf
                    win2[grp] = t; N2[t] = 0; hires2 += 1
            eta2 = ((1.0 - TR.purity2(N2)) / (1.0 - 1.0 / NL)).astype(cp.float32)
            N2 += cp.bincount(win2 * NL + yb, minlength=K2 * NL).reshape(K2, NL)
            cnt2 = cp.bincount(win2, minlength=K2); live2 = cnt2 > 0
            sums2 = cp.zeros((K2, D), cp.float32); cupyx.scatter_add(sums2, win2, Cb)
            W2[live2] += eta2[live2][:, None] * (sums2[live2] / cnt2[live2, None] - W2[live2])
            W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
            # ---- the message down: residual cells of misread images -> L1 reassignment
            if arm in ("residual", "expected") and int(wrong.sum()):
                E = W2[win2].reshape(m, K1, rig.gg)
                exp_id = E.argmax(1)                                        # (m, gg)
                present = Cb.reshape(m, K1, rig.gg) > 0
                hit = cp.take_along_axis(present, exp_id[:, None, :], 1)[:, 0, :]
                resid_cell = (~hit) & wrong[:, None]                        # (m, gg)
                off = resid_cell[cp.repeat(cp.arange(m), rig.npos), cells] & k
                if arm == "expected" and int(off.sum()):
                    # the expected identity itself learns the patch it should have won
                    oi = cp.where(off)[0]
                    win1[oi] = exp_id[cp.repeat(cp.arange(m), rig.npos)[oi], cells[oi]]
                    hires1 += int(off.sum())
                elif int(off.sum()):
                    tot1 = N1.sum(2)
                    pur1 = ((N1 + R.ALPHA) / (tot1[..., None] + R.ALPHA * NL)).max(2)   # (K1, gg)
                    oi = cp.where(off)[0]
                    Sh = (V[oi] @ W1.T) ** 2
                    Sh = cp.where((pur1 < 0.5).T[cells[oi]], Sh, -1.0)
                    new = Sh.argmax(1); ok = Sh[cp.arange(len(oi)), new] > -1.0
                    win1[oi[ok]] = new[ok]; hires1 += int(ok.sum())
            # ---- L1 table and step
            N1 += cp.bincount((win1[k] * rig.gg + cells[k]) * NL + lab[k],
                              minlength=K1 * rig.gg * NL).reshape(N1.shape)
            if arm != "frozen":
                cnt1 = cp.bincount(win1[k], minlength=K1); live1 = cnt1 >= rig.min_s
                sums1 = cp.zeros((K1, rig.dim), cp.float32); cupyx.scatter_add(sums1, win1[k], V[k])
                n1[live1] += cnt1[live1]
                eta1 = cp.clip(cnt1[live1] / n1[live1], R.ETA_MIN, 1.0)[:, None]
                W1[live1] += eta1 * (sums1[live1] / cnt1[live1, None] - W1[live1])
                W1 /= cp.linalg.norm(W1, axis=1, keepdims=True) + EPS
    # ---- evaluate: final codes, L2 recount and online; L1 table reference
    Ctr, itr, ktr = TR.encode(rig, W1, Xtr); Cte, ite, kte = TR.encode(rig, W1, Xte)
    l1acc = TR.l1_table_acc(rig, itr, ktr, ytr, ite, kte, yte)
    T_on = TR.table2(N2)
    _, wtr = TR.predict2(W2, T_on, Ctr)
    T_re = TR.table2(cp.bincount(wtr * NL + ytr, minlength=K2 * NL).reshape(K2, NL).astype(cp.float64))
    p_re, w2 = TR.predict2(W2, T_re, Cte); p_on, _ = TR.predict2(W2, T_on, Cte)
    rd = TR.residual_diag(W2, w2, Cte, rig.gg)
    return {"l1_table": l1acc, "recount": float((p_re == yte).mean()), "online": float((p_on == yte).mean()),
            "dead2": int((N2.sum(1) == 0).sum()), "dead1": int((N1.sum((1, 2)) == 0).sum()),
            "hires2": hires2, "hires1": hires1,
            "resid_correct": float(rd[p_re == yte].mean()), "resid_wrong": float(rd[p_re != yte].mean())}


def main():
    OUT.mkdir(exist_ok=True)
    R.GRID = TR.GRID; R.BATCH = 128
    rig = R.Rig(TR.PS)
    Xtr, ytr, Xte, yte = R.E.load("fashion_mnist")
    if TR.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    print(f"Fashion {len(ytr)}/{len(yte)}, both layers training, seeds {SEEDS}\n")
    print(f"{'arm':<10}{'seed':>5}{'L1 table':>9}{'L2 recount':>11}{'online':>8}{'dead1':>6}{'dead2':>6}"
          f"{'hires1':>8}{'hires2':>7}{'res ok/bad':>12}")
    res = {}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            r = run(rig, arm, Xtr, ytr_g, Xte, yte_g, seed); rows.append(r)
            print(f"{arm:<10}{seed:>5}{r['l1_table']:>9.4f}{r['recount']:>11.4f}{r['online']:>8.4f}"
                  f"{r['dead1']:>6d}{r['dead2']:>6d}{r['hires1']:>8d}{r['hires2']:>7d}"
                  f"{r['resid_correct']:>6.2f}/{r['resid_wrong']:.2f}   ({time.time()-t0:.0f}s)", flush=True)
        res[arm] = rows
        print(f"{'':<10}{'mean':>5}{np.mean([r['l1_table'] for r in rows]):>9.4f}"
              f"{np.mean([r['recount'] for r in rows]):>11.4f}", flush=True)
    (OUT / ("joint_rung_expected.json" if "--expected-only" in sys.argv else "joint_rung.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
