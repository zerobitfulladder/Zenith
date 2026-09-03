"""Winner by partial correlation: the least replaceable template wins.

Score(t) = the input's correlation with template t after everything the other
templates explain is removed (Gram inverse with a small ridge). The best score
wins, learns the WHOLE input with the purity step, and is counted; the read at
test time uses the same rule. Everything else is the best rig (label 0.25 +
belief 0.5, hire on error). 28x28 MNIST (784 dims > 400 templates, so the
leftover exists), joint and split, 2 seeds. Templates saved and drawn.

Usage:  uv run python loo_split.py [--smoke]
"""
import json, sys, time
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import purity_split as P
import rf_sweep as R

OUT = P.OUT
SEEDS = [7] if P.SMOKE else [7, 8]
P.EPOCHS = 2 if P.SMOKE else 20
ARMS = [("cosine winner", False), ("partial-corr winner", True)]
if P.SMOKE:
    ARMS = ARMS[1:]


def draw(store):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for i, (tag, (W, use, pur)) in enumerate(store.items()):
        for j, (ttl, order) in enumerate((("36 most used", np.argsort(-use)[:36]),
                                          ("36 random live", np.random.default_rng(0).permutation(np.where(use > 0)[0])[:36]))):
            tile = np.zeros((6 * 29, 6 * 29))
            for n, t in enumerate(order):
                r, c = divmod(n, 6); w = W[t].reshape(28, 28); v = np.abs(w).max() + 1e-9
                tile[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = w / v
            ax = axes[i, j]; ax.imshow(tile, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag}: {ttl}  (mean purity {pur[order].mean():.2f})", fontsize=10)
    fig.suptitle("28x28 templates after joint 0-9, seed 7", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "loo_templates.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(28); R.BATCH = 512
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if P.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    joint = [(Xtr, ytr_g, "0-9")]
    split = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    P.BETA_L, P.BETA_B, P.HIRE, P.CHASE, P.SOFT = 0.25, 0.5, True, False, None
    print(f"28x28 MNIST, {P.EPOCHS} epochs/phase, seeds {SEEDS}, ridge {P.LOO_RIDGE}\n")
    print(f"{'arm':<22}{'JOINT recount':>14}{'online':>8}{'dead':>6}{'purity':>8} | "
          f"{'SPLIT after 5-9 old/new':>24}{'final':>8}{'online':>8}")
    res, store = {}, {}
    for tag, loo in ARMS:
        P.LOO = loo
        t0 = time.time()
        J, S = [], []
        for s_ in SEEDS:
            c, _, sn = P.run(rig, "purity", "both", joint, Xtr, ytr_g, Xte, yte_g, s_)
            J.append(c)
            if s_ == SEEDS[0]:
                store[tag] = sn["final"]
                np.save(OUT / f"loo_templates_{'pc' if loo else 'cos'}.npy", sn["final"][0])
            S.append(P.run(rig, "purity", "both", split, Xtr, ytr_g, Xte, yte_g, s_)[0])
        e = [P.end_points(c) for c in S]
        m = lambda kk: np.mean([x[kk] for x in e])
        res[tag] = {"joint": J, "split": S}
        print(f"{tag:<22}{np.mean([c[-1]['recount'][2] for c in J]):>14.4f}"
              f"{np.mean([c[-1]['online'][2] for c in J]):>8.4f}{np.mean([c[-1]['n_dead'] for c in J]):>6.0f}"
              f"{np.mean([c[-1]['purity_live'] for c in J]):>8.3f} | "
              f"{m('B_old'):>14.4f}/{m('B_new'):.4f}{m('C_all'):>8.4f}{m('C_online'):>8.4f}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "loo_split.json").write_text(json.dumps(res, indent=1))
    draw(store)
    print(f"wrote {OUT / 'loo_templates.png'}")


if __name__ == "__main__":
    main()
