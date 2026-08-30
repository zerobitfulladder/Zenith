"""Can anything be built on a layer whose templates mean nothing?

The 2026-08-29 convolutional stack, run three ways. Only layer one
changes; layer two, the decode, the data and the hole are identical.

  dense         144 templates, all scoring, all learning from the
                leftover -- the layer that rebuilds its input almost
                optimally and whose templates are arbitrary
  dense-random  the same 144 templates, never trained. If this matches
                `dense`, layer one's learning contributed nothing and
                the stack is running on layer two alone.
  pca           144 fixed directions from one SVD of the patches, no
                learning at all. The dense rule converges to a basis of
                exactly this subspace, so this is what its learning is
                worth against a closed-form answer.
  wta           144 templates, winner-take-all -- the published
                08-29 layer one, at the same budget

The task: 1600 noisy pairs from a wave, a 1.6-wide hole cut out of
training entirely, and the stack has to put the curve back across it.

Run:  .venv/bin/python experiments/2026_08_30/dense_stack/run_dense_stack.py
"""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("regress", "two_layer", "conv_stack"):
    sys.path.insert(0, str(HERE.parents[1] / "2026_08_29" / _sub))

import pop_regress as P                                   # noqa: E402
import conv_stack as C                                    # noqa: E402
import two_layer as T                                     # noqa: E402
from dense_conv import DensePatchLayer, DenseCodeWindowLayer   # noqa: E402

OUT = HERE / "results"
K1, K2, TAPS2, WINDOWS = 144, 96, 51, 12000
GAP = (1.2, 2.8)
RM = np.array([True] * 4 + [False])          # reference taps, train == read

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ arms ---
def patch_matrix(l1, g, val, known):
    return np.array([l1._dense_vec(val[l1.taps_at(int(i))])[0]
                     for i in l1.valid(g, known)])


def build_l1(kind, g, val, known, seed=0):
    if kind == "pca":
        l1 = DensePatchLayer(k=K1, seed=seed, train_l1=False)
        X = patch_matrix(l1, g, val, known)
        l1.dense.W = np.linalg.svd(X, full_matrices=False)[2][:K1]
        return l1, True
    if kind == "wta":
        l1 = C.PatchLayer(k=K1, seed=seed)
        l1.train(g, val, known, n=20000, seed=seed)
        dense = False
    else:
        l1 = DensePatchLayer(k=K1, seed=seed, train_l1=(kind == "dense"))
        l1.train(g, val, known, n=20000, seed=seed)
        dense = True
    return l1, dense


def run_arm(kind, g, val, known, miss, seed=0):
    t0 = time.time()
    l1, dense = build_l1(kind, g, val, known, seed)

    ok = np.zeros(len(g), bool)
    ok[l1.valid(g, known)] = True
    codes = [None] * len(g)
    for i in range(len(g)):
        if ok[i]:
            c = l1.code(g, val, i, topm=8)
            codes[i] = (c[0], c[1])

    dim = K1 if dense else l1.hc.n_boot
    L2 = DenseCodeWindowLayer if dense else C.CodeWindowLayer
    l2 = L2(dim=dim, taps=TAPS2, step=1, k=K2, size=8192, seed=seed)
    anchors = l2.train(codes, ok, len(g), n=WINDOWS, seed=seed)

    centre = int(np.argmin(np.abs(g - 0.5 * (GAP[0] + GAP[1]))))
    if dense:
        shapes, winner, score = l2.complete_dense(codes, ok, centre, l1)
    else:
        ids, winner, score = l2.complete(codes, ok, centre)
        shapes = {p: l1.template_patch(j) for p, j in ids.items()}

    sol = C.stitch(shapes, l1, len(g),
                   anchor_vals=val[known], anchor_idx=np.nonzero(known)[0])
    pred = val.copy()
    for q, v in sol.items():
        if not known[q]:
            pred[q] = v

    err = rmse(pred[miss], P.target(g[miss]))
    print(f"  {kind:<13} completed {len(shapes):2d} codes from template "
          f"#{winner} (match {score:.3f})   RMSE {err:.3f}   "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return dict(kind=kind, l1=l1, l2=l2, codes=codes, ok=ok, pred=pred,
                rmse=err, winner=winner, score=score, dense=dense,
                n_completed=len(shapes), anchors=len(anchors))


# --------------------------------------------------------------- pictures ---
def save_vocabulary(arm, path):
    l1 = arm["l1"]
    fig, axes = plt.subplots(12, 12, figsize=(13, 13))
    off = l1.offsets(np.arange(2) * 0.1)
    for j, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.3); s.set_color("0.8")
        if j < K1:
            ax.plot(off, l1.template_patch(j), "-o", ms=1.6, lw=1.0, color="#1f77b4")
            ax.axhline(0, lw=0.4, color="0.75")
    fig.suptitle(f"{arm['kind']} — each layer-one template as the 5-sample "
                 f"patch it stands for", fontsize=12)
    fig.subplots_adjust(left=.006, right=.994, top=.965, bottom=.006,
                        wspace=.12, hspace=.12)
    fig.savefig(path, dpi=110); plt.close(fig)


def save_code_field(arm, g, path):
    codes, ok = arm["codes"], arm["ok"]
    F = np.full((K1, len(g)), np.nan)
    for i in range(len(g)):
        if ok[i]:
            idx, mag = codes[i]
            F[np.asarray(idx, int), i] = mag
    fig, ax = plt.subplots(figsize=(13, 4.4))
    m = np.nanmax(np.abs(F))
    ax.imshow(F, aspect="auto", cmap="bwr", vmin=-m, vmax=m,
              extent=[g[0], g[-1], K1, 0], interpolation="nearest")
    ax.axvspan(GAP[0], GAP[1], color="0.35", alpha=.35, lw=0)
    ax.set_xlabel("x"); ax.set_ylabel("layer-one template")
    ax.set_title(f"{arm['kind']} — what layer one says at every position "
                 f"(grey = the hole, no data)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)


def save_regeneration(g, val, known, arms, xtr, ytr, mlp_fn, path):
    miss = np.nonzero(~known)[0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    style = {"dense": ("#d62728", "-"), "dense-random": ("#ff7f0e", "--"),
             "pca": ("#1f77b4", "-."), "wta": ("#2ca02c", "-")}
    for ax, (lo, hi) in zip(axes, [(g[0], g[-1]), (GAP[0] - 1.4, GAP[1] + 1.4)]):
        sel = (xtr >= lo) & (xtr <= hi)
        ax.plot(xtr[sel], ytr[sel], ".", ms=2.2, color="0.72", label="training pairs")
        gg = np.linspace(lo, hi, 600)
        ax.plot(gg, P.target(gg), "-", lw=1.6, color="0.2", label="truth")
        ax.plot(gg, mlp_fn(gg), ":", lw=1.3, color="#9467bd", label="MLP (64-64)")
        for a in arms:
            c, ls = style[a["kind"]]
            ax.plot(g[miss], a["pred"][miss], ls, lw=2.0, color=c,
                    label=f"{a['kind']}  (RMSE {a['rmse']:.3f})")
        ax.axvspan(GAP[0], GAP[1], color="0.85", zorder=0)
        ax.set_xlim(lo, hi); ax.set_xlabel("x")
    axes[0].set_ylabel("y"); axes[0].legend(fontsize=8.5, loc="upper left")
    axes[1].set_title("the hole, close up", fontsize=11)
    axes[0].set_title("the stack has to put the curve back where it never "
                      "saw data", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_l2_templates(arm, g, path, n=8):
    l1, l2 = arm["l1"], arm["l2"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4))
    d = g[1] - g[0]
    for ax, j in zip(axes.ravel(), np.linspace(0, l2.hc.n_boot - 1, n).astype(int)):
        shapes = l2.template_shapes(int(j), l1)
        for t, s in enumerate(shapes):
            xs = (l2.taps_at(0)[t] - l2.T // 2) * d + l1.offsets(g)
            ax.plot(xs, s, "-", lw=0.8, alpha=.75)
        ax.set_title(f"template #{j}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{arm['kind']} — layer-two templates, each rebuilt through "
                 f"layer one", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ------------------------------------------------------------------- main ---
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P.configure(x_range=(-12, 12), y_range=(-6, 6), gap=GAP,
                target_name="wave_plus_trend")
    xtr, ytr, *_ = P.make_dataset(n_train=1600, seed=0)
    g = T.tap_grid(0.2)
    val, known = C.sample_raw(g, xtr, ytr)
    miss = np.nonzero(~known)[0]
    print(f"level 0: {known.sum()}/{len(g)} grid points have data; "
          f"{len(miss)} missing (x {g[miss[0]]:.1f}..{g[miss[-1]]:.1f})\n",
          flush=True)

    arms = [run_arm(k, g, val, known, miss) for k in
            ("dense", "dense-random", "pca", "wta")]

    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                       max_iter=20000, random_state=0).fit(xtr.reshape(-1, 1), ytr)
    l1w = C.PatchLayer(k=64, seed=0)
    l1w.train(g, val, known, n=20000, seed=0, ref_mask=RM)
    walk, _ = l1w.walk(g, val, known, ref_mask=RM)
    truth = P.target(g[miss])

    rows = [(f"{a['kind']} L1 + L2 over its codes", a["rmse"]) for a in arms]
    rows += [("conv L1 alone, walking", rmse(walk[miss], truth)),
             ("MLP (64-64, tanh)", rmse(mlp.predict(g[miss].reshape(-1, 1)), truth)),
             ("hold the value at the nearer rim",
              rmse(P.edge_hold(g[miss], GAP), truth)),
             ("published 08-29 wta stack (K1=64)", 0.086)]
    print("\n  RMSE across the hole")
    for n_, v in rows:
        print(f"  {n_:<40} {v:.3f}", flush=True)

    for a in arms:
        save_vocabulary(a, OUT / f"l1_vocabulary_{a['kind']}.png")
        save_code_field(a, g, OUT / f"l1_code_field_{a['kind']}.png")
        save_l2_templates(a, g, OUT / f"l2_templates_{a['kind']}.png")
    save_regeneration(g, val, known, arms, xtr, ytr,
                      lambda z: mlp.predict(np.asarray(z).reshape(-1, 1)),
                      OUT / "regeneration.png")

    # How much do two arms' codes agree, position by position? If the dense
    # layer learned anything, training must move this away from the random one.
    def field(a):
        F = np.zeros((K1, len(g)))
        for i in range(len(g)):
            if a["ok"][i]:
                idx, mag = a["codes"][i]
                F[np.asarray(idx, int), i] = mag
        return F[:, a["ok"]]
    fd, fr = field(arms[0]), field(arms[1])
    agree = float(np.mean([abs(np.corrcoef(fd[:, i], fr[:, i])[0, 1])
                           for i in range(fd.shape[1])]))

    metrics = {
        "config": {"k1": K1, "k2": K2, "l1_taps": 5, "l1_patch_width": 0.8,
                   "l2_taps": TAPS2, "l2_window_width": (TAPS2 - 1) * 0.1,
                   "windows": WINDOWS, "gap": list(GAP)},
        "level0": {"grid": len(g), "with_data": int(known.sum()),
                   "missing": int(len(miss))},
        "arms": {a["kind"]: {"rmse": round(a["rmse"], 4),
                             "l2_winner": a["winner"],
                             "l2_match": round(a["score"], 3),
                             "codes_completed": a["n_completed"],
                             "l2_templates": int(a["l2"].hc.n_boot)}
                 for a in arms},
        "gap_rmse": {n_: round(v, 4) for n_, v in rows},
        "dense_vs_random_code_agreement": round(agree, 4),
        "per_point": [{"x": round(float(g[q]), 1),
                       **{a["kind"]: round(float(a["pred"][q]), 3) for a in arms},
                       "truth": round(float(P.target(g[q])), 3)} for q in miss],
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\ndense vs untrained code agreement: {agree:.3f}")
    print(f"done in {metrics['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
