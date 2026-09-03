"""Split-MNIST on the champion rig with the WHOLE DIGIT as the receptive field.

The boundary-condition test for `2026_09_01/stack/forget.py`.

That run found the representation forgot nothing: after sequential training a
full recount of the table gave old classes 0.9736 against joint's 0.9734, even
though 54% of old-data winners had changed hands. The reading was that 5x5
stroke features are CLASS-AGNOSTIC, so phase B's rewrite was lateral -- the old
digits were still fully describable in the new vocabulary, and the only casualty
was the index-to-class table going stale (0.7152).

The obvious objection is that this is a statement about tiny generic features,
not about representations. So this run removes exactly one thing -- the small
receptive field -- and changes nothing else:

    patch rig    400 templates over 5x5 px, 576 positions, T[t, 6x6 cell, class]
    this         400 templates over 28x28 px, 1 position,  T[t, class]

A whole-digit template is a prototype of a digit. It cannot be class-agnostic:
if a template that was a 3 is dragged to an 8, the 3 is not renamed, it is gone.
So the prediction is that the recount does NOT rescue the old classes here, and
`oracle_old` falls well below `joint_old` -- which is what "the representation
was damaged" looks like, as opposed to "the bookkeeping went stale".

Everything else is gpu_stack.py verbatim: the competition rule 1-(V.W)^2, the
table bias at beta 1.5, winner-only learning with the eta floor at 0.02, the
log-ratio table, the five final tables and the same diagnostics.

TWO FORCED DEVIATIONS, both consequences of there being one position instead of
576, both recorded in the output json:

  MIN_S 4 -> 1    A template must win MIN_S items in a batch to learn. The patch
                  rig's batch held 128*331 = 42k patches, ~106 wins per template;
                  here a batch holds one item per image, so at 400 templates the
                  expectation is under one and MIN_S=4 would freeze learning
                  outright. The threshold was mini-batch hygiene, not a rule.
  batch 128 -> 512, epochs 3 -> 40
                  The patch rig got 576 winner-updates per image per epoch and
                  this gets 1. Epochs are swept and phase-A accuracy is reported
                  per epoch so the plateau is visible: a model that never learned
                  0-4 cannot demonstrate that it does not forget 0-4.

Usage:  uv run python whole.py [--smoke] [--beta 1.5] [--epochs 40] [--k 400]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2026_09_01" / "experts"))
import experts as E

OUT = HERE / "results"


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


SMOKE = "--smoke" in sys.argv
K = arg("--k", 400, int)
NL, EPS, ALPHA = 10, 1e-12, 1.0
D = 784                                  # the whole digit, one position
BETA = arg("--beta", 1.5)
EPOCHS = arg("--epochs", 3 if SMOKE else 40, int)
BATCH, ETA_MIN, MIN_S, FLOOR = 512, 0.02, 1, 0.05
SEEDS = [7] if SMOKE else [7, 8, 9]


def prep(Xb):
    """Centred, unit-length whole digits. Same prep as gpu_stack.patches, over
    784 numbers instead of 25, and with no position axis."""
    C = Xb - Xb.mean(-1, keepdims=True)
    n = cp.linalg.norm(C, axis=-1)
    return C / cp.maximum(n, EPS)[:, None], n > FLOOR, n


def table_from(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def belief(T, win, keep):
    """gpu_stack.belief with npos = 1: the single lookup IS the evidence."""
    ev = T[win, 0] * keep[:, None]
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def compete(V, W, T, q, beta):
    """gpu_stack.compete with one cell, so the einsum is a plain matmul."""
    err = 1.0 - (V @ W.T) ** 2
    if beta > 0:
        err = err - beta * (q @ T[:, 0, :].T)
    return err.argmin(1)


def learn(W, V, win, n):
    cnt = cp.bincount(win, minlength=K)
    sums = cp.zeros((K, D), cp.float32)
    cupyx.scatter_add(sums, win, V)
    live = cnt >= MIN_S
    n[live] += cnt[live]
    eta = cp.clip(cnt[live] / n[live], ETA_MIN, 1.0)[:, None]
    W[live] += eta * (sums[live] / cnt[live, None] - W[live])
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS


def train_cont(X, y, rng, W=None, N=None, n=None, epochs=EPOCHS, probe=None):
    """l1_train_cont, verbatim in structure: phase B inherits W, N and n."""
    if W is None:
        W = cp.asarray(rng.standard_normal((K, D)), cp.float32)
        W -= W.mean(1, keepdims=True)
        W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
        N = cp.zeros((K, 1, NL))
        n = cp.zeros(K)
    T = table_from(N)
    order = np.arange(len(X))
    curve = []
    for ep in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            ids = cp.asarray(order[s:s + BATCH])
            V, keep, _ = prep(X[ids])
            w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
            q = belief(T, w0, keep)
            win = compete(V, W, T, q, BETA)
            lab = y[ids]
            flat = win[keep] * NL + lab[keep]
            N += cp.bincount(flat, minlength=K * NL).reshape(N.shape)
            T = table_from(N)
            learn(W, V[keep], win[keep], n)
        if probe is not None and (ep + 1) % max(1, epochs // 10) == 0:
            Tp, _ = rebuild(W, X, y)
            curve.append([ep + 1, round(float(evaluate(Tp, code(W, probe[0]),
                                                       probe[1])[probe[2]]), 4)])
    return W, N, n, curve


def code(W, X, chunk=4096):
    idx = cp.zeros(len(X), cp.int32)
    keepm = cp.zeros(len(X), bool)
    for a in range(0, len(X), chunk):
        V, keep, _ = prep(X[a:a + chunk])
        idx[a:a + len(V)] = (1.0 - (V @ W.T) ** 2).argmin(1)
        keepm[a:a + len(V)] = keep
    return idx, keepm


def rebuild(W, X, y):
    return build_table(*code(W, X), y)


def build_table(win, keep, y):
    flat = win[keep] * NL + y[keep]
    N = cp.bincount(flat, minlength=K * NL).reshape(K, 1, NL).astype(cp.float64)
    return table_from(N), N


def score(T, codes):
    win, keep = codes
    return T[win, 0] * keep[:, None]


def evaluate(T, codes, y):
    pred = score(T, codes).argmax(1)
    old = y < 5
    return {"all": float((pred == y).mean()),
            "old": float((pred[old] == y[old]).mean()),
            "new": float((pred[~old] == y[~old]).mean())}


def per_class(T, codes, y):
    pred = score(T, codes).argmax(1)
    return [float((pred[y == c] == c).mean()) for c in range(NL)]


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
    log(f"split-MNIST, WHOLE DIGIT receptive field: {K} templates x {D} px, "
        f"1 position, beta {BETA}, {EPOCHS} epochs")
    log(f"  {int(A.sum())} images of 0-4, {int((~A).sum())} of 5-9, "
        f"test {len(yte)}")

    per_seed, diag, curve0 = [], None, None
    for s in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(s)
        probe = (Xte, yte_g, "old") if s == SEEDS[0] else None
        W, N, n, curve = train_cont(XA, yA, rng, probe=probe)
        W_A = W.copy()
        T_A, N_Ac = rebuild(W_A, XA, yA)
        codes_te_A = code(W_A, Xte)
        after_A = evaluate(T_A, codes_te_A, yte_g)
        if curve:
            curve0 = curve
            log("    phase-A learning curve (epoch, test 0-4): "
                + " ".join(f"{e}:{a:.4f}" for e, a in curve))
        log(f"\n  seed {s}  after phase A: old {after_A['old']:.4f}  "
            f"({time.time()-t0:.0f}s)")

        W, N, n, _ = train_cont(XB, yB, rng, W, N, n)
        codes_te_B = code(W, Xte)
        _, N_Bc = rebuild(W, XB, yB)
        finals = {
            "online": table_from(N),
            "seq": table_from(N_Ac + N_Bc),
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

        Wj, _, _, _ = train_cont(Xtr, ytr_g, np.random.default_rng(s))
        Tj, _ = rebuild(Wj, Xtr, ytr_g)
        row["joint"] = evaluate(Tj, code(Wj, Xte), yte_g)
        log(f"    joint   old {row['joint']['old']:.4f}  "
            f"new {row['joint']['new']:.4f}  all {row['joint']['all']:.4f}"
            f"   ({time.time()-t0:.0f}s)")

        if s == SEEDS[0]:
            row["per_class_oracle"] = per_class(finals["oracle"], codes_te_B, yte_g)
            iA, kA = code(W_A, XA)
            iA2, kA2 = code(W, XA)
            both = kA & kA2
            churn = float((iA[both] != iA2[both]).mean())
            winsA = cp.bincount(iA[kA], minlength=K)
            iB, kB = code(W, XB)
            winsB = cp.bincount(iB[kB], minlength=K)
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
                f"{diag['n_dead_in_A']}/{K}   B wins into "
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
    dmg = agg["joint_old"]["mean"] - agg["oracle_old"]["mean"]
    stale = agg["oracle_old"]["mean"] - agg["seq_old"]["mean"]
    log(f"    REPRESENTATION damage (joint - oracle): {dmg:+.4f}")
    log(f"    BOOKKEEPING staleness (oracle - seq):   {stale:+.4f}")
    log(f"    churn {diag['old_winner_churn']:.3f}")
    log(f"\n    patch rig for comparison: churn 0.542, damage +0.0002, "
        f"staleness +0.2583")

    tag = "_smoke" if SMOKE else ""
    tag += "" if BETA == 1.5 else f"_beta{BETA:g}"
    res = {"receptive_field": "whole 28x28, 1 position", "K": K, "D": D,
           "beta": BETA, "epochs": EPOCHS, "batch": BATCH, "min_s": MIN_S,
           "eta_min": ETA_MIN, "seeds": SEEDS,
           "deviations_from_patch_rig": {
               "MIN_S": "4 -> 1 (one item per image, not 331 patches)",
               "BATCH": "128 -> 512", "EPOCHS": f"3 -> {EPOCHS}"},
           "phase_A_curve": curve0,
           "representation_damage": dmg, "bookkeeping_staleness": stale,
           "patch_rig_reference": {"churn": 0.5418, "damage": 0.0002,
                                   "staleness": 0.2583, "oracle_old": 0.9736,
                                   "joint_old": 0.9734},
           "per_seed": per_seed, "agg": agg, "diag": diag}
    (OUT / f"whole_digit_mnist{tag}.json").write_text(json.dumps(res, indent=2))
    (OUT / f"whole_digit_mnist{tag}.log").write_text("\n".join(lines))
    log(f"\nwrote {OUT / f'whole_digit_mnist{tag}.json'}")


if __name__ == "__main__":
    main()
