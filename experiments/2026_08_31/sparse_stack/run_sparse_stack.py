"""Layer one on a place code + sparse column, against the 0.57 / 0.93 board.

Same harness, same layer three, same judge, same rho. Only layers one and
two change. `python run_sparse_stack.py`
"""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2026_08_30" / "stack3"))
from stack3 import ConvStack, BoundLayer, unit_rows, PATCH, GRID, WIN, L2_GRID  # noqa
from sparse_stack import SparseStack, NoCenterBound                             # noqa

ROOT = HERE.parents[2]
OUT = HERE / "results"
K1, K2, K3 = 64, 128, 1024
ETA1, ETA3 = 0.5, 0.05
N_TRAIN, N_TEST, EPOCHS, EPOCHS3 = 20000, 5000, 2, 2
RHO, SEED = 0.6, 0
ARMS = [("dense", {}), ("wta", {}),
        ("sparse k=2", {"kmax": 2}), ("sparse k=4", {"kmax": 4}),
        ("sparse k=8", {"kmax": 8}),
        ("sparse k=4, settled read", {"kmax": 4, "read": "settle"}),
        ("sparse k=4, no L3 centring", {"kmax": 4, "l3": NoCenterBound})]


def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    if X.max() > 1.5:
        X = X / 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def onehot(y, n=10):
    O = np.zeros((len(y), n)); O[np.arange(len(y)), y] = 1.0
    return O


def norm_img(A):
    return unit_rows(A.reshape(len(A), -1))[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    judge = LogisticRegression(max_iter=2000, n_jobs=-1).fit(norm_img(Xtr), ytr)
    jacc = float(judge.score(norm_img(Xte), yte))
    print(f"train {len(Xtr)}  test {len(Xte)}  rho={RHO}  "
          f"judge (logistic on pixels) {jacc:.4f}\n")

    res, gens, reports = {}, {}, []
    for name, opt in ARMS:
        t = time.time()
        rng = np.random.default_rng(SEED)
        if name in ("dense", "wta"):
            cs = ConvStack(name, K1, K2, ETA1, rng)
        else:
            cs = SparseStack(K1, K2, opt["kmax"], ETA1, rng,
                             read=opt.get("read", "pursue"))
        cs.train(Xtr, EPOCHS, rng)
        Htr, Hte = cs.forward(Xtr), cs.forward(Xte)
        rec = cs.backward(Hte[:512], 512)
        fid = float(np.mean(np.linalg.norm(norm_img(rec) - norm_img(Xte[:512]), axis=1)))
        t12 = time.time() - t

        rng3 = np.random.default_rng(SEED + 1)
        cls = opt.get("l3", BoundLayer)
        l3 = cls(K3, Htr.shape[1], 10, RHO, ETA3, rng3)
        for _ in range(EPOCHS3):
            o = rng3.permutation(len(Htr))
            for s in range(0, len(o), 256):
                b = o[s:s + 256]
                l3.learn(Htr[b], onehot(ytr[b]), ytr[b])

        pred, _, _ = l3.classify(Hte)
        acc = float((pred == yte).mean())
        pur, used = l3.purity()
        g = {}
        for tag, m in (("top-1", 1), ("graded", 32)):
            code, _ = l3.generate(onehot(np.arange(10)), topm=m)
            imgs = cs.backward(code, 10)
            jl = judge.predict(norm_img(imgs))
            g[tag] = (float((jl == np.arange(10)).mean()), imgs, jl)
        gens[name] = g
        code_lit = float((Hte > 0).mean())
        res[name] = {"accuracy": acc, "purity": pur, "templates_used": used,
                     "gen_top1": g["top-1"][0], "gen_graded": g["graded"][0],
                     "recon_error": round(fid, 3), "l2_code_fraction_lit": round(code_lit, 4),
                     "seconds_l1l2": round(t12, 1), "seconds": round(time.time() - t, 1)}
        print(f"{name:<28} accuracy {acc:.4f}  purity {pur:.3f}  "
              f"gen top-1 {g['top-1'][0]:.1f} graded {g['graded'][0]:.1f}  "
              f"recon {fid:.3f}  lit {code_lit:.3f}  {res[name]['seconds']}s")
        if name.startswith("sparse k=4") and "settled" not in name and "centring" not in name:
            reports.append(f"=== {name}, rho={RHO} ===\n" +
                           classification_report(yte, pred, digits=4))
            fig, ax = plt.subplots(figsize=(5.6, 4.9))
            cm = confusion_matrix(yte, pred)
            ax.imshow(cm, cmap="Blues")
            for i in range(10):
                for j in range(10):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                                color="white" if cm[i, j] > cm.max() * .5 else "0.2")
            ax.set_xlabel("read off the winner"); ax.set_ylabel("true label")
            ax.set_title(f"{name} — accuracy {acc:.3f}", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / "confusion_sparse_k4.png", dpi=130)
            plt.close(fig)
        if isinstance(cs, SparseStack):
            T = cs.templates()
            fig, axes = plt.subplots(8, 8, figsize=(5, 5))
            for i, a in enumerate(axes.ravel()):
                a.imshow(T[i], cmap="gray_r", vmin=0, vmax=1)
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"{name} — layer-one templates", fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT / f"l1_templates_{name.replace(' ','_').replace('=','')}.png",
                        dpi=120)
            plt.close(fig)

    # ---- figures --------------------------------------------------------
    names = [n for n, _ in ARMS]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    cols = ["#888", "#888"] + ["#1b6ca8"] * 3 + ["#2f8f4e", "#7a4fa3"]
    for a, key, ttl in [(ax[0], "accuracy", "classification accuracy"),
                        (ax[1], "gen_graded", "generated digits the judge agrees with"),
                        (ax[2], "recon_error", "image -> code -> image error")]:
        a.barh(range(len(names)), [res[n][key] for n in names], color=cols)
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=7)
        a.invert_yaxis(); a.set_title(ttl, fontsize=9); a.grid(alpha=.25, axis="x")
        for i, n in enumerate(names):
            a.text(res[n][key], i, f" {res[n][key]:.3f}", va="center", fontsize=6.5)
    ax[0].axvline(0.9074, ls=":", color="k", lw=1)
    ax[0].text(0.9074, -0.6, " judge", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "01_board.png", dpi=130); plt.close(fig)

    show = [n for n in names if n in ("wta", "dense", "sparse k=4")]
    fig, axes = plt.subplots(2 * len(show), 10, figsize=(10, 1.15 * 2 * len(show)))
    for r, n in enumerate(show):
        for s, tag in enumerate(("top-1", "graded")):
            _, imgs, jl = gens[n][tag]
            for d in range(10):
                a = axes[2 * r + s, d]
                a.imshow(imgs[d], cmap="gray"); a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color("#2ca02c" if jl[d] == d else "#d62728")
                    sp.set_linewidth(1.8)
                if d == 0:
                    a.set_ylabel(f"{n}\n{tag}", fontsize=6, rotation=0,
                                 ha="right", va="center")
    fig.suptitle("the label alone, pushed back down to pixels "
                 "(green = judge agrees)", fontsize=10)
    fig.subplots_adjust(left=.13, right=.99, top=.9, bottom=.01, wspace=.05, hspace=.06)
    fig.savefig(OUT / "02_generated.png", dpi=135); plt.close(fig)

    (OUT / "classification_report.txt").write_text("\n\n".join(reports))
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"k1": K1, "k2": K2, "k3": K3, "rho": RHO, "nb": 8,
                    "train": N_TRAIN, "test": N_TEST, "epochs": EPOCHS,
                    "epochs_l3": EPOCHS3},
         "judge_accuracy": jacc, "results": res,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
