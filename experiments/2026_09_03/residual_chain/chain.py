"""Winner, then residual winner: two-stage learning and a top-2 code.

The proposal: the winner learns from the patch (as now); then compete again on
what the winner could not explain, and only THAT winner learns from the
residual. Matching pursuit with winner-only at each stage. The table counts
both winners per position and reads both.

Arms (9x9 and 5x5 patches, MNIST 12k/3k, 3 epochs per phase, 2 seeds):

    champion        top-1 learn, belief 1.5, top-1 read      the reference
    top1 none       top-1 learn, no pressure, top-1 read     the patch-layer stream rule
    top2 read       top-1 learn, no pressure, read the 2 NEAREST templates   (reading alone)
    chain 0.0       residual winner always learns and is counted
    chain 0.5       residual winner only when |residual| > 0.5 of the patch (hire on surprise)

Stage-2 updates are sign-aligned to the residual (a residual direction has no
sign of its own). Everything else is rf_sweep's rig.

Usage:  uv run python chain.py [--smoke]
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
SIZES = [9] if SMOKE else [9, 5]
SEEDS = [7] if SMOKE else [7, 8]
EPOCHS = 1 if SMOKE else 3
K, NL = R.K, R.NL
ARMS = [("champion", "top1", 1.5, None), ("top1 none", "top1", 0.0, None),
        ("top2 read", "top2read", 0.0, None),
        ("chain 0.0", "chain", 0.0, 0.0), ("chain 0.5", "chain", 0.0, 0.5)]
if SMOKE:
    ARMS = [ARMS[1], ARMS[3]]


def two_stage(V, W, tau):
    """Stage-1 winner, residual, stage-2 winner, residual gate, aligned residual dir."""
    S = V @ W.T
    w1 = (S ** 2).argmax(1)
    i = cp.arange(len(V))
    s1 = S[i, w1]
    Rr = V - s1[:, None] * W[w1]
    rn = cp.linalg.norm(Rr, axis=1)
    Rn = Rr / cp.maximum(rn, R.EPS)[:, None]
    S2 = Rn @ W.T
    S2[i, w1] = 0.0
    w2 = (S2 ** 2).argmax(1)
    sgn = cp.sign(S2[i, w2]); sgn[sgn == 0] = 1.0
    return w1, w2, rn > tau, Rn * sgn[:, None], (S ** 2).argsort(1)[:, -2]


def code(rig, W, X, mode, tau, chunk=256):
    n = len(X)
    i1 = cp.zeros((n, rig.npos), cp.int32); i2 = cp.zeros((n, rig.npos), cp.int32)
    km = cp.zeros((n, rig.npos), bool); k2 = cp.zeros((n, rig.npos), bool)
    for a in range(0, n, chunk):
        Q, keep = rig.patches(X[a:a + chunk]); m = len(Q)
        V = Q.reshape(-1, rig.dim)
        w1, w2, g, _, second = two_stage(V, W, tau if tau is not None else 0.0)
        i1[a:a + m] = w1.reshape(m, -1); km[a:a + m] = keep
        if mode == "top2read":
            i2[a:a + m] = second.reshape(m, -1); k2[a:a + m] = keep
        elif mode == "chain":
            i2[a:a + m] = w2.reshape(m, -1); k2[a:a + m] = keep & g.reshape(m, -1)
    return i1, km, i2, k2


def build(rig, i1, km, i2, k2, y):
    n = len(i1)
    C = cp.tile(rig.cells, n).reshape(n, rig.npos)
    lab = cp.repeat(y, rig.npos).reshape(n, rig.npos)
    f1 = (i1[km] * rig.gg + C[km]) * NL + lab[km]
    f2 = (i2[k2] * rig.gg + C[k2]) * NL + lab[k2]
    N = cp.bincount(f1, minlength=K * rig.gg * NL)
    if f2.size:
        N = N + cp.bincount(f2, minlength=K * rig.gg * NL)
    return N.reshape(K, rig.gg, NL).astype(cp.float64)


def acc(rig, T, codes, y):
    i1, km, i2, k2 = codes
    n = len(i1)
    C = cp.tile(rig.cells, n).reshape(n, rig.npos)
    ev = (T[i1, C] * km[..., None]).sum(1) + (T[i2, C] * k2[..., None]).sum(1)
    pred = ev.argmax(1)
    old = y < 5
    return (float((pred[old] == y[old]).mean()), float((pred[~old] == y[~old]).mean()),
            float((pred == y).mean()))


def coherence(W):
    G = cp.abs(W @ W.T); G[cp.arange(K), cp.arange(K)] = 0
    return float(G.max(1).mean())


def run(rig, mode, beta, tau, phases, Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True); W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
    N = cp.zeros((K, rig.gg, NL)); n = cp.zeros(K)
    out = {"phase": []}; share2 = []
    for pi, (X, y) in enumerate(phases):
        for ep in range(EPOCHS):
            order = np.arange(len(X)); rng.shuffle(order)
            for s in range(0, len(order), R.BATCH):
                ids = cp.asarray(order[s:s + R.BATCH]); m = len(ids)
                Q, keep = rig.patches(X[ids]); V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
                cells = cp.tile(rig.cells, m); lab = cp.repeat(y[ids], rig.npos)
                T = rig.table_from(N)
                w1, w2, g, Rn, _ = two_stage(V, W, tau if tau is not None else 0.0)
                if beta > 0:
                    q = rig.belief(T, w1, k, cells, m)
                    win = rig.compete(V, W, T, q, cells, m)
                else:
                    win = w1
                N += cp.bincount((win[k] * rig.gg + cells[k]) * NL + lab[k],
                                 minlength=K * rig.gg * NL).reshape(N.shape)
                cnt = cp.bincount(win[k], minlength=K); live = cnt >= rig.min_s
                if int(live.sum()):
                    sums = cp.zeros((K, rig.dim), cp.float32); cupyx.scatter_add(sums, win[k], V[k])
                    n[live] += cnt[live]
                    eta = cp.clip(cnt[live] / n[live], R.ETA_MIN, 1.0)[:, None]
                    W[live] += eta * (sums[live] / cnt[live, None] - W[live])
                    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
                if mode == "chain":
                    k2 = k & g
                    share2.append(float(k2.sum() / max(int(k.sum()), 1)))
                    N += cp.bincount((w2[k2] * rig.gg + cells[k2]) * NL + lab[k2],
                                     minlength=K * rig.gg * NL).reshape(N.shape)
                    cnt2 = cp.bincount(w2[k2], minlength=K); live2 = cnt2 >= rig.min_s
                    if int(live2.sum()):
                        sums = cp.zeros((K, rig.dim), cp.float32)
                        cupyx.scatter_add(sums, w2[k2], Rn[k2])
                        n[live2] += cnt2[live2]
                        eta = cp.clip(cnt2[live2] / n[live2], R.ETA_MIN, 1.0)[:, None]
                        W[live2] += eta * (sums[live2] / cnt2[live2, None] - W[live2])
                        W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
        ctr = code(rig, W, Xtr, mode, tau); cte = code(rig, W, Xte, mode, tau)
        T_re = rig.table_from(build(rig, *ctr, ytr))
        out["phase"].append({"recount": acc(rig, T_re, cte, yte),
                             "online": acc(rig, rig.table_from(N), cte, yte)})
    out["dead"] = int((N.sum((1, 2)) == 0).sum()); out["coherence"] = coherence(W)
    out["share2"] = float(np.mean(share2)) if share2 else 0.0
    return out


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:2000], ytr[:2000], Xte[:500], yte[:500]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    joint = [(Xtr, ytr_g)]
    split = [(Xtr[A], ytr_g[A]), (Xtr[~A], ytr_g[~A]), (Xtr, ytr_g)]
    res = {}
    for ps in SIZES:
        R.BATCH = 128; rig = R.Rig(ps)
        print(f"== {ps}x{ps}, {rig.npos} positions, {EPOCHS} epochs/phase, seeds {SEEDS}")
        print(f"{'arm':<12}{'JOINT recount':>14}{'online':>8}{'dead':>6}{'coher':>7}{'stage2':>8} | "
              f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}")
        for tag, mode, beta, tau in ARMS:
            t0 = time.time()
            J = [run(rig, mode, beta, tau, joint, Xtr, ytr_g, Xte, yte_g, s) for s in SEEDS]
            S = [run(rig, mode, beta, tau, split, Xtr, ytr_g, Xte, yte_g, s) for s in SEEDS]
            res[f"{ps}|{tag}"] = {"joint": J, "split": S}
            jr = np.mean([j["phase"][-1]["recount"][2] for j in J]); jo = np.mean([j["phase"][-1]["online"][2] for j in J])
            bo = np.mean([s["phase"][1]["recount"][0] for s in S]); bn = np.mean([s["phase"][1]["recount"][1] for s in S])
            fa = np.mean([s["phase"][2]["recount"][2] for s in S]); fo = np.mean([s["phase"][2]["online"][2] for s in S])
            print(f"{tag:<12}{jr:>14.4f}{jo:>8.4f}{np.mean([j['dead'] for j in J]):>6.0f}"
                  f"{np.mean([j['coherence'] for j in J]):>7.3f}{np.mean([j['share2'] for j in J]):>8.2f} | "
                  f"{bo:>14.4f}/{bn:.4f}{fa:>8.4f}{fo:>8.4f}   ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "chain.json").write_text(json.dumps(res, indent=1))
        print()


if __name__ == "__main__":
    main()
