"""Pursuit against settling, at matched sparsity. ~2 min."""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "place_code"))
from place_code import PlaceCode, unit_rows                  # noqa: E402
from sparse_column import SparseColumn                       # noqa: E402
from run_sparse_column import load, PATCH, NB, K, ETA, BATCH, EPOCHS  # noqa: E402

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
TARGETS = [2, 4, 8]
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def calibrate(col, U, target, iters=16):
    """lambda that makes the settled code as sparse as the pursuit's k."""
    lo, hi = 1e-4, 1.0
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        m = float((col.settle(U, mid, n_iter=48) > 0).sum(1).mean())
        lo, hi = (mid, hi) if m > target else (lo, mid)
    return float(np.sqrt(lo * hi))


def train(kind, U, target, rng):
    col = SparseColumn(K, U.shape[1], kmax=target, eta=ETA,
                       rng=np.random.default_rng(1))
    lam = None
    for ep in range(EPOCHS):
        idx = rng.permutation(len(U))
        if kind == "settle":
            col.learn(U[idx[:K]]) if col.n_boot < K else None
            lam = calibrate(col, U[idx[:1500]], target)
        for b in range(0, len(idx), BATCH):
            B = U[idx[b:b + BATCH]]
            col.learn(B) if kind == "pursue" else col.learn_settled(B, lam)
        col.revive(U[rng.choice(len(U), 2000, replace=False)])
    if kind == "settle":
        lam = calibrate(col, U[rng.choice(len(U), 1500, replace=False)], target)
    return col, lam


def score(col, U, P, pc, code):
    R = U - code @ col.W
    rec = float(np.sqrt((R * R).sum(1).mean() / (U * U).sum(1).mean()))
    pix = float(np.sqrt(((pc.decode(code @ col.W) - P) ** 2).mean()))
    return rec, pix, float((code > 0).sum(1).mean())


def main():
    P = load(); pc = PlaceCode(PATCH ** 2, nb=NB, lo=0., hi=1., halfw=1.5)
    Uall, _ = unit_rows(pc.encode(P))
    rng = np.random.default_rng(0)
    tr = rng.choice(len(Uall), 24000, replace=False)
    te = rng.choice(len(Uall), 4000, replace=False)
    U, Ute, Pte = Uall[tr], Uall[te], P[te]
    out = {"by_target": {}, "cross": {}, "timing": {}, "convergence": {}}
    cols = {}

    for tgt in TARGETS:
        row = {}
        for kind in ("pursue", "settle"):
            t0 = time.time()
            col, lam = train(kind, U, tgt, np.random.default_rng(0))
            C = col.pursue(Ute)[0] if kind == "pursue" else col.settle(Ute, lam)
            rec, pix, used = score(col, Ute, Pte, pc, C)
            sup = np.unique(C > 0, axis=0)
            row[kind] = {"rebuild": round(rec, 4), "pixel_rmse": round(pix, 4),
                         "mean_active": round(used, 2), "lam": lam,
                         "distinct_supports": int(len(sup)),
                         "train_s": round(time.time() - t0, 1)}
            cols[(tgt, kind)] = (col, lam)
        out["by_target"][tgt] = row
        print(f"target k={tgt}: " + "  ".join(
            f"{k}: rebuild {v['rebuild']:.3f} pixel {v['pixel_rmse']:.4f} "
            f"active {v['mean_active']:.2f}" for k, v in row.items()))

    # ---- same dictionary, both encoders: isolates encoder from learning --
    for tgt in TARGETS:
        col, _ = cols[(tgt, "pursue")]
        lam = calibrate(col, Ute[:1500], tgt)
        Cp, Cs = col.pursue(Ute)[0], col.settle(Ute, lam)
        rp, _, up = score(col, Ute, Pte, pc, Cp)
        rs, _, us = score(col, Ute, Pte, pc, Cs)
        agree = float(((Cp > 0) & (Cs > 0)).sum(1).mean() /
                      np.maximum((Cp > 0).sum(1).mean(), 1))
        out["cross"][tgt] = {"pursue_rebuild": round(rp, 4), "pursue_active": round(up, 2),
                             "settle_rebuild": round(rs, 4), "settle_active": round(us, 2),
                             "support_overlap": round(agree, 3)}
        print(f"  same templates, k={tgt}: pursuit {rp:.3f} ({up:.1f} on) vs "
              f"settle {rs:.3f} ({us:.1f} on), supports agree {agree:.2f}")

    # ---- convergence + timing ------------------------------------------ #
    col, lam = cols[(4, "settle")]
    for n_it in [4, 8, 16, 32, 64, 128]:
        C = col.settle(Ute[:2000], lam, n_iter=n_it)
        R = Ute[:2000] - C @ col.W
        out["convergence"][n_it] = {
            "rebuild": round(float(np.sqrt((R * R).sum(1).mean() /
                                           (Ute[:2000] * Ute[:2000]).sum(1).mean())), 4),
            "active": round(float((C > 0).sum(1).mean()), 2)}
    B = Ute[:2048]
    for tag, fn in [("pursue k=4", lambda: cols[(4, "pursue")][0].pursue(B)),
                    ("settle 16 it", lambda: col.settle(B, lam, n_iter=16)),
                    ("settle 64 it", lambda: col.settle(B, lam, n_iter=64))]:
        fn(); t = time.perf_counter(); fn(); dt = (time.perf_counter() - t) * 1e6 / len(B)
        out["timing"][tag] = round(dt, 2)
        print(f"  {tag:<14} {dt:6.2f} us/patch")

    # ---- stability ------------------------------------------------------ #
    stab, eps_l = {}, [0.01, 0.02, 0.05, 0.10, 0.20]
    Q = Pte[:1500]
    for tgt in TARGETS:
        for kind in ("pursue", "settle"):
            c, lm = cols[(tgt, kind)]
            enc = (lambda X: c.pursue(X)[0] > 0) if kind == "pursue" else \
                  (lambda X: c.settle(X, lm) > 0)
            S0 = enc(unit_rows(pc.encode(Q))[0])
            stab[f"{kind} k={tgt}"] = [
                float(((S0 & enc(unit_rows(pc.encode(np.clip(
                    Q + e * rng.standard_normal(Q.shape), 0, 1)))[0])).sum(1)
                    / np.maximum(S0.sum(1), 1)).mean()) for e in eps_l]
    out["stability"] = {"noise": eps_l, "kept": stab}
    print("  stability @5% noise: " + "  ".join(
        f"{k} {v[2]:.3f}" for k, v in stab.items()))

    figures(pc, cols, out, eps_l)
    (OUT / "compare.json").write_text(json.dumps(out, indent=2))


def figures(pc, cols, out, eps_l):
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    for kind, c in [("pursue", "#c1462d"), ("settle", "#1b6ca8")]:
        ax[0].plot(TARGETS, [out["by_target"][t][kind]["rebuild"] for t in TARGETS],
                   "o-", color=c, label=kind)
        ax[1].plot(TARGETS, [out["by_target"][t][kind]["pixel_rmse"] for t in TARGETS],
                   "o-", color=c, label=kind)
    ax[0].set_title("rebuild error"); ax[1].set_title("pixel RMSE")
    for a in ax[:2]:
        a.set_xlabel("templates active"); a.legend(fontsize=7); a.grid(alpha=0.25)
    it = sorted(int(i) for i in out["convergence"])
    ax[2].plot(it, [out["convergence"][str(i)]["rebuild"] if str(i) in out["convergence"]
                    else out["convergence"][i]["rebuild"] for i in it], "o-", color="#1b6ca8")
    ax[2].set_xscale("log"); ax[2].set_xlabel("settling iterations")
    ax[2].set_title("how long it needs to settle"); ax[2].grid(alpha=0.25)
    for kind, c in [("pursue", "#c1462d"), ("settle", "#1b6ca8")]:
        for t, ls in zip(TARGETS, ["-", "--", ":"]):
            ax[3].plot(eps_l, out["stability"]["kept"][f"{kind} k={t}"], ls,
                       color=c, marker="o", ms=3, label=f"{kind} k={t}")
    ax[3].set_xscale("log"); ax[3].set_xlabel("noise added to the patch")
    ax[3].set_title("support kept"); ax[3].legend(fontsize=5.5); ax[3].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "06_compare.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 17, figsize=(11, 1.8))
    for r, kind in enumerate(("pursue", "settle")):
        col = cols[(4, kind)][0]
        axes[r, 0].text(.5, .5, kind, ha="center", va="center", fontsize=8)
        axes[r, 0].axis("off")
        for j, t in enumerate(np.argsort(-col.wins)[:16]):
            axes[r, j + 1].imshow(pc.decode(col.W[t:t + 1])[0].reshape(PATCH, PATCH),
                                  cmap="gray_r", vmin=0, vmax=1)
            axes[r, j + 1].set_xticks([]); axes[r, j + 1].set_yticks([])
    fig.suptitle("templates at 4 active — pursuit above, settling below", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "07_templates_compare.png"); plt.close(fig)


if __name__ == "__main__":
    main()
