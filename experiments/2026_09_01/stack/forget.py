"""Split-MNIST on the champion: does the pressure-trained L1 forget?

Train the winner rig (400 templates, winner-take-all, pressure beta 1.5,
counted table) on digits 0-4 only, then continue the SAME weights and the SAME
accumulating table on digits 5-9 only, and measure what happened to the old
classes. The earlier split-MNIST result (sequential cost 0.31 pts) was on the
dense-experts design; this is the first run on the current champion, whose two
risk factors are new: the eta floor (templates always move at least 2% per
batch they win, so B-phase batches can drag A's specialists) and the pressure
bias (phase B's beliefs are computed against a table whose 0-4 rows go stale).

Three final tables, in decreasing honesty about what a continual system may do:

    online   the N accumulated across both phases, never rebuilt
             (contains phase-A counts taken while templates were still moving)
    seq      N_A rebuilt at the END of phase A + N_B rebuilt at the end of
             phase B -- each phase's statistics are clean for ITS templates,
             but A's rows describe templates as they stood before phase B
    oracle   full rebuild on all training data with the final templates
             (= replay; the upper bound continual learning is not allowed)

Controls: frozen (phase-A templates never touched by B, table counted on all
data -- what B training buys minus what it destroys) and joint (all ten classes
at once, the ceiling). Diagnostics: per-template drift split by phase-A usage,
winner churn on old data, and where phase B's wins landed (dead / light / heavy
phase-A templates -- "new classes go to dead experts" is the claim to check).
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
import experts as E
import gpu_stack as G

OUT = HERE / "results"
SMOKE = "--smoke" in sys.argv
if SMOKE:
    G.EPOCHS1 = 1
SEEDS = [7] if SMOKE else [7, 8, 9]
OLD = cp.asarray([0, 1, 2, 3, 4])
K1, NL, NPOS = G.K1, G.NL, G.SIDE * G.SIDE
GG = G.GRID1 * G.GRID1


def l1_train_cont(X, y, rng, W=None, N=None, n=None):
    """G.l1_train, verbatim, but able to CONTINUE from given weights, counts
    and step state -- phase B inherits everything phase A left behind."""
    if W is None:
        W = cp.asarray(rng.standard_normal((K1, 25)), cp.float32)
        W -= W.mean(1, keepdims=True)
        W /= cp.linalg.norm(W, axis=1, keepdims=True) + G.EPS
        N = cp.zeros((K1, GG, NL))
        n = cp.zeros(K1)
    T = G.table_from(N, K1)
    order = np.arange(len(X))
    for ep in range(G.EPOCHS1):
        rng.shuffle(order)
        for s in range(0, len(order), G.IMG_BATCH):
            ids = cp.asarray(order[s:s + G.IMG_BATCH]); m = len(ids)
            Q, keep, _ = G.patches(X[ids])
            V = Q.reshape(-1, 25); k = keep.reshape(-1)
            cells = cp.tile(G.C1, m)
            w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
            q = G.belief(T, w0, k, cells, m, NPOS)
            win = G.compete(V, W, T, q, cells, G.BETA, m, NPOS)
            lab = cp.repeat(y[ids], NPOS)
            flat = (win[k] * GG + cells[k]) * NL + lab[k]
            N += cp.bincount(flat, minlength=K1 * GG * NL).reshape(N.shape)
            T = G.table_from(N, K1)
            G.learn(W, V[k], win[k], n, K1)
    return W, N, n


def rebuild(W, X, y):
    i, m, k = G.l1_code(W, X)
    return G.build_table(i, G.C1, y, K1, G.GRID1, k)


def evaluate(T, codes, y):
    i, m, k = codes
    pred = G.score(T, i, G.C1, NPOS, k).argmax(1)
    old = y < 5
    return {"all": float((pred == y).mean()),
            "old": float((pred[old] == y[old]).mean()),
            "new": float((pred[~old] == y[~old]).mean())}


def per_class(T, codes, y):
    i, m, k = codes
    pred = G.score(T, i, G.C1, NPOS, k).argmax(1)
    return [float((pred[y == c] == c).mean()) for c in range(NL)]


def wins_of(W, X, chunk=512):
    i, m, k = G.l1_code(W, X, chunk)
    return cp.bincount(i[k], minlength=K1), (i, m, k)


def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    Xtr, ytr, Xte, yte = E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:4000], ytr[:4000], Xte[:800], yte[:800]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    XA, yA, XB, yB = Xtr[A], ytr_g[A], Xtr[~A], ytr_g[~A]
    log(f"split-MNIST on the champion: {int(A.sum())} images of 0-4, "
        f"{int((~A).sum())} of 5-9, test {len(yte)}")

    per_seed, diag = [], None
    for s in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(s)
        W, N, n = l1_train_cont(XA, yA, rng)
        W_A = W.copy()
        T_A, N_Ac = rebuild(W_A, XA, yA)
        codes_te_A = G.l1_code(W_A, Xte)
        after_A = evaluate(T_A, codes_te_A, yte_g)
        log(f"\n  seed {s}  after phase A: old {after_A['old']:.4f}  "
            f"({time.time()-t0:.0f}s)")

        W, N, n = l1_train_cont(XB, yB, rng, W, N, n)
        codes_te_B = G.l1_code(W, Xte)
        _, N_Bc = rebuild(W, XB, yB)
        finals = {
            "online": G.table_from(N, K1),
            "seq": G.table_from(N_Ac + N_Bc, K1),
            "oracle": rebuild(W, Xtr, ytr_g)[0],
        }
        row = {"seed": s, "after_A_old": after_A["old"]}
        for name, T in finals.items():
            row[name] = evaluate(T, codes_te_B, yte_g)
            log(f"    {name:<7s} old {row[name]['old']:.4f}  "
                f"new {row[name]['new']:.4f}  all {row[name]['all']:.4f}")

        T_fr, _ = rebuild(W_A, Xtr, ytr_g)
        row["frozen"] = evaluate(T_fr, codes_te_A, yte_g)
        log(f"    frozen  old {row['frozen']['old']:.4f}  "
            f"new {row['frozen']['new']:.4f}  all {row['frozen']['all']:.4f}")

        Wj, _, _ = l1_train_cont(Xtr, ytr_g, np.random.default_rng(s))
        Tj, _ = rebuild(Wj, Xtr, ytr_g)
        row["joint"] = evaluate(Tj, G.l1_code(Wj, Xte), yte_g)
        log(f"    joint   old {row['joint']['old']:.4f}  "
            f"new {row['joint']['new']:.4f}  all {row['joint']['all']:.4f}"
            f"   ({time.time()-t0:.0f}s)")

        if s == SEEDS[0]:
            row["per_class_seq"] = per_class(finals["seq"], codes_te_B, yte_g)
            winsA, (iA, _, kA) = wins_of(W_A, XA)
            winsB, _ = wins_of(W, XB)
            iA2, _, kA2 = G.l1_code(W, XA)
            both = kA & kA2
            churn = float((iA[both] != iA2[both]).mean())
            drift = 1.0 - cp.abs((W_A * W).sum(1))
            dead = winsA == 0
            med = float(cp.median(winsA[~dead])) if int((~dead).sum()) else 0
            light = (~dead) & (winsA <= med)
            heavy = winsA > med
            tot = float(winsB.sum())
            diag = {
                "old_winner_churn": churn,
                "drift_mean": {g: float(drift[m].mean()) if int(m.sum()) else None
                               for g, m in [("dead_in_A", dead),
                                            ("light_in_A", light),
                                            ("heavy_in_A", heavy)]},
                "B_win_share": {g: float(winsB[m].sum() / tot)
                                for g, m in [("dead_in_A", dead),
                                             ("light_in_A", light),
                                             ("heavy_in_A", heavy)]},
                "n_dead_in_A": int(dead.sum()),
            }
            log(f"    churn on old data {churn:.3f}   dead-in-A "
                f"{diag['n_dead_in_A']}/{K1}   B wins into "
                + " ".join(f"{g} {v:.2f}" for g, v in diag["B_win_share"].items()))
        per_seed.append(row)

    agg = {}
    for name in ("online", "seq", "oracle", "frozen", "joint"):
        for part in ("old", "new", "all"):
            v = [r[name][part] for r in per_seed]
            agg[f"{name}_{part}"] = {"mean": float(np.mean(v)),
                                     "std": float(np.std(v))}
    v = [r["after_A_old"] for r in per_seed]
    agg["after_A_old"] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    log(f"\n  over {len(SEEDS)} seed(s):")
    log(f"    old classes: after A {agg['after_A_old']['mean']:.4f}  ->  "
        f"seq {agg['seq_old']['mean']:.4f}+-{agg['seq_old']['std']:.4f}   "
        f"(oracle {agg['oracle_old']['mean']:.4f}, "
        f"joint {agg['joint_old']['mean']:.4f})")
    log(f"    forgetting: joint-vs-seq {agg['joint_old']['mean'] - agg['seq_old']['mean']:+.4f} on old; "
        f"table staleness share {agg['oracle_old']['mean'] - agg['seq_old']['mean']:+.4f}")

    tag = "_smoke" if SMOKE else ""
    res = {"per_seed": per_seed, "agg": agg, "diag": diag, "seeds": SEEDS}
    (OUT / f"forget_mnist{tag}.json").write_text(json.dumps(res, indent=2))
    (OUT / f"forget_mnist{tag}.log").write_text("\n".join(lines))
    log(f"\nwrote {OUT / f'forget_mnist{tag}.json'}")


if __name__ == "__main__":
    main()
