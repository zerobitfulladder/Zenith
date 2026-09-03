"""Does reading the top-k templates instead of the single winner preserve 0-4?

The argument this tests, from looking at `templates_random.png`: after phase B
the 400 slots hold a mixture of 0-9, and there are MANY templates per digit. So
even when phase B converts some of the 0-slots into 6-slots, other 0-slots
survive. The class is stored redundantly. A top-1 read throws that redundancy
away -- if the single nearest template happens to be one that was converted, the
image is lost -- whereas summing the k nearest should ride over the conversions.

So: learning stays winner-only and the tally stays top-1 (nothing about training
changes); only the READ becomes a sum over the k nearest templates. That
isolates the question -- is the 0-4 information still present in the population
after phase B, and is top-1 simply failing to collect it?

Two tallies are reported at every probe, because they answer different questions:

    online     N as accumulated during training, never rebuilt. What a live
               system actually holds at that moment. This is the tally that
               feeds the beta pressure.
    recount    rebuilt from the current templates over all training data. What
               the templates COULD support if the bookkeeping were perfect.
               Upper bound; it is replay, and a continual system may not do it.

Outputs
  results/topk_curve.png    accuracy on 0-4 and 5-9 across both phases
  results/topk_sweep.png    accuracy vs k at the end of training
  results/topk.json

Usage:  uv run python topk.py [--beta-only 1.5]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import whole as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SEED = 7
KS = [1, 2, 3, 5, 10, 20, 50, 100]
KCURVE = [1, 10]
PROBE_EVERY = 20


def topk_idx(W, X, kk, chunk=4096):
    out = cp.zeros((len(X), kk), cp.int32)
    for a in range(0, len(X), chunk):
        Vb, _, _ = V.prep(X[a:a + chunk])
        err = 1.0 - (Vb @ W.T) ** 2
        out[a:a + len(Vb)] = cp.argpartition(err, kk - 1, axis=1)[:, :kk]
    return out


def acc_topk(T, idx, y, kk):
    """Sum the table rows of the kk nearest templates, then argmax."""
    pred = T[idx[:, :kk], 0].sum(1).argmax(1)
    old = y < 5
    return (float((pred[old] == y[old]).mean()),
            float((pred[~old] == y[~old]).mean()),
            float((pred == y).mean()))


def recount(W, X, y):
    """Top-1 counts from the current templates over all of X."""
    i, keep = V.code(W, X)
    flat = i[keep] * V.NL + y[keep]
    N = cp.bincount(flat, minlength=V.K * V.NL).reshape(V.K, 1, V.NL).astype(cp.float64)
    return V.table_from(N)


def probe(W, N, Xte, yte, Xtr, ytr, kmax):
    idx = topk_idx(W, Xte, kmax)
    T_on, T_re = V.table_from(N), recount(W, Xtr, ytr)
    return {f"{tag}_k{k}": acc_topk(T, idx, yte, k)
            for tag, T in (("online", T_on), ("recount", T_re))
            for k in KCURVE}


def run(beta, Xtr, ytr, Xte, yte, XA, yA, XB, yB):
    V.BETA = beta
    rng = np.random.default_rng(SEED)
    W = cp.asarray(rng.standard_normal((V.K, V.D)), cp.float32)
    W -= W.mean(1, keepdims=True)
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + V.EPS
    N = cp.zeros((V.K, 1, V.NL)); n = cp.zeros(V.K)
    curve, step = [], 0
    kmax = max(KCURVE)

    for phase, (X, y) in enumerate(((XA, yA), (XB, yB))):
        order = np.arange(len(X))
        for ep in range(V.EPOCHS):
            rng.shuffle(order)
            for s in range(0, len(order), V.BATCH):
                ids = cp.asarray(order[s:s + V.BATCH])
                Vb, keep, _ = V.prep(X[ids])
                T = V.table_from(N)
                w0 = (1.0 - (Vb @ W.T) ** 2).argmin(1)
                q = V.belief(T, w0, keep)
                win = V.compete(Vb, W, T, q, beta)
                lab = y[ids]
                N += cp.bincount(win[keep] * V.NL + lab[keep],
                                 minlength=V.K * V.NL).reshape(N.shape)
                V.learn(W, Vb[keep], win[keep], n)
                if step % PROBE_EVERY == 0:
                    curve.append({"step": step, "phase": phase,
                                  **probe(W, N, Xte, yte, Xtr, ytr, kmax)})
                step += 1
        print(f"    beta {beta} phase {'A' if phase == 0 else 'B'} done "
              f"({step} steps)", flush=True)
    curve.append({"step": step, "phase": 1,
                  **probe(W, N, Xte, yte, Xtr, ytr, kmax)})

    idx = topk_idx(W, Xte, max(KS))
    T_on, T_re = V.table_from(N), recount(W, Xtr, ytr)
    sweep = {tag: {k: acc_topk(T, idx, yte, k) for k in KS}
             for tag, T in (("online", T_on), ("recount", T_re))}
    return curve, sweep, step


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = V.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    XA, yA, XB, yB = Xtr[A], ytr_g[A], Xtr[~A], ytr_g[~A]

    betas = [float(sys.argv[sys.argv.index("--beta-only") + 1])] \
        if "--beta-only" in sys.argv else [1.5, 0.0]
    res = {}
    for b in betas:
        t0 = time.time()
        c, s, nstep = run(b, Xtr, ytr_g, Xte, yte_g, XA, yA, XB, yB)
        res[str(b)] = {"curve": c, "sweep": s, "steps": nstep}
        print(f"  beta {b} in {time.time()-t0:.0f}s", flush=True)

    # ---- curve figure -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
    for r, b in enumerate(betas):
        d = res[str(b)]
        x = [p["step"] for p in d["curve"]]
        switch = next(p["step"] for p in d["curve"] if p["phase"] == 1)
        for c, tag in enumerate(("online", "recount")):
            ax = axes[r, c]
            for k, ls in zip(KCURVE, ("--", "-")):
                for j, (cls, col) in enumerate((("0-4", "#1b6ca8"), ("5-9", "#c1121f"))):
                    ax.plot(x, [p[f"{tag}_k{k}"][j] for p in d["curve"]],
                            ls, color=col, lw=1.9 if k > 1 else 1.2,
                            alpha=1.0 if k > 1 else 0.65,
                            label=f"{cls}, top-{k}")
            ax.axvline(switch, color="k", lw=1, ls=":")
            ax.text(switch, 0.03, "  phase B starts (5-9 only)", fontsize=8, rotation=90)
            ax.set_title(f"beta {b}   —   {tag} tally", fontsize=10)
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, ncol=2, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("training batches (phase A then phase B)")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    fig.suptitle("Whole-digit rig: accuracy on the OLD classes (0-4) and NEW classes (5-9) "
                 "through both phases.\nLearning and tallying stay top-1; only the READ is a "
                 "sum over the k nearest templates.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "topk_curve.png", dpi=145)
    plt.close(fig)

    # ---- sweep figure -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for c, tag in enumerate(("online", "recount")):
        ax = axes[c]
        for b, style in zip(betas, ("-", "--")):
            sw = res[str(b)]["sweep"][tag]
            for j, (cls, col) in enumerate((("0-4", "#1b6ca8"), ("5-9", "#c1121f"),
                                            ("all", "#333333"))):
                ax.plot(KS, [sw[k][j] for k in KS], style, marker="o", ms=3,
                        color=col, label=f"beta {b}, {cls}")
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("k templates summed at read"); ax.grid(alpha=0.25)
        ax.set_title(f"{tag} tally", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test accuracy after both phases")
    axes[1].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Accuracy vs how many templates the read sums, after sequential training",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "topk_sweep.png", dpi=145)
    plt.close(fig)

    (OUT / "topk.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'':<12}{'k':>5}  " + "  ".join(f"{t:>22}" for t in ("online (0-4/5-9/all)",
                                                                  "recount (0-4/5-9/all)")))
    for b in betas:
        for k in KS:
            o = res[str(b)]["sweep"]["online"][k]
            r = res[str(b)]["sweep"]["recount"][k]
            print(f"beta {b:<7}{k:>5}  " +
                  f"{o[0]:.4f}/{o[1]:.4f}/{o[2]:.4f}".rjust(22) + "  " +
                  f"{r[0]:.4f}/{r[1]:.4f}/{r[2]:.4f}".rjust(22))
    print(f"\nwrote {OUT/'topk_curve.png'}, {OUT/'topk_sweep.png'}")


if __name__ == "__main__":
    main()
