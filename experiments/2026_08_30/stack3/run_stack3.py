"""Three layers, a label concatenated at the top, read from both ends.

Trained on (image, label) pairs. Then asked two questions with the same
partial-cue read:

  classification  write the image half, read the label half back
  generation      write the label half, read the image half back, and
                  push it down through layer two and layer one to pixels

The label's share of layer three's code (`rho`) is swept, because with
10 label cells against 1152 image cells the label is 0.3% of the vector
if nothing is done about it, and the competition simply ignores it.

Both layer-one/two substrates from this folder are run: `dense` (all
templates score and learn) and `wta` (one winner). Layer three is
winner-take-all in both, because a partial-cue read needs a winner whose
other half can be read off.

Run:  .venv/bin/python experiments/2026_08_30/stack3/run_stack3.py
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
sys.path.insert(0, str(HERE))
from stack3 import (ConvStack, BoundLayer, unit_rows, SIDE, GRID, PATCH,
                    L2_GRID, WIN, EPS)                       # noqa: E402

ROOT = HERE.parents[2]
OUT = HERE / "results"
K1, K2, K3 = 64, 128, 1024
ETA1, ETA3 = 0.5, 0.05
N_TRAIN, N_TEST, EPOCHS, EPOCHS3 = 20000, 5000, 2, 2
RHOS = [0.1, 0.3, 0.6, 1.0]
SEED = 0


def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return (X[:N_TRAIN], y[:N_TRAIN],
            X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST])


def onehot(y, n=10):
    O = np.zeros((len(y), n))
    O[np.arange(len(y)), y] = 1.0
    return O


def norm_img(A):
    """Unit-norm images, so the judge is blind to overall contrast."""
    return unit_rows(A.reshape(len(A), -1))[0]


# ----------------------------------------------------------------- figures ---
def sheet(T, path, title, cmap="gray", cols=None):
    k = len(T)
    cols = cols or int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * .55, rows * .55))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        if i < k:
            t = T[i]
            m = np.abs(t).max() + EPS
            ax.imshow(t, cmap=cmap, vmin=-m if cmap == "bwr" else 0, vmax=m,
                      interpolation="nearest")
    fig.suptitle(title, fontsize=11, y=.998)
    fig.subplots_adjust(left=.003, right=.997, top=.96, bottom=.003,
                        wspace=.05, hspace=.05)
    fig.savefig(path, dpi=120); plt.close(fig)


def save_generated(G, judged, path, title):
    fig, axes = plt.subplots(len(G), 10, figsize=(11, 1.25 * len(G)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(G):
        for d in range(10):
            ax = axes[r, d]
            ax.imshow(imgs[d], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2ca02c" if judged[r][d] == d else "#d62728")
                s.set_linewidth(2.0)
            if r == 0:
                ax.set_title(str(d), fontsize=10)
            if d == 0:
                ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=.10, right=.995, top=.86, bottom=.01,
                        wspace=.05, hspace=.08)
    fig.savefig(path, dpi=135); plt.close(fig)


def save_confusion(cm, acc, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * .5 else "0.2")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("read off the winner"); ax.set_ylabel("true label")
    ax.set_title(f"{title} — accuracy {acc:.3f}", fontsize=11)
    fig.colorbar(im, fraction=.046); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def save_sweep(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for kind in res:
        r = res[kind]
        axes[0].plot(RHOS, [r[str(x)]["accuracy"] for x in RHOS], "-o", label=kind)
        axes[1].plot(RHOS, [r[str(x)]["generation_accuracy_top1"] for x in RHOS],
                     "-o", label=f"{kind} top-1")
        axes[1].plot(RHOS, [r[str(x)]["generation_accuracy_graded"] for x in RHOS],
                     "--s", label=f"{kind} graded")
        axes[2].plot(RHOS, [r[str(x)]["template_purity"] for x in RHOS], "-o",
                     label=kind)
    for ax, t in zip(axes, ["classification accuracy",
                            "generated digits a judge agrees with",
                            "class purity of a stored pair"]):
        ax.set_xlabel("rho — label energy as a fraction of image energy")
        ax.set_title(t, fontsize=10.5); ax.grid(alpha=.3, lw=.5)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# -------------------------------------------------------------------- main ---
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    print(f"train {len(Xtr)}  test {len(Xte)}   "
          f"L1 {GRID}x{GRID} of {PATCH}x{PATCH} -> k1={K1}   "
          f"L2 {L2_GRID}x{L2_GRID} of {WIN}x{WIN} -> k2={K2}   L3 k3={K3}")

    judge = LogisticRegression(max_iter=2000, n_jobs=-1).fit(norm_img(Xtr), ytr)
    print(f"judge (logistic on unit-norm pixels) test accuracy "
          f"{judge.score(norm_img(Xte), yte):.3f}\n")

    results, report_txt = {}, []
    for kind in ("dense", "wta"):
        rng = np.random.default_rng(SEED)
        cs = ConvStack(kind, K1, K2, ETA1, rng)
        t = time.time()
        cs.train(Xtr, EPOCHS, rng)
        Htr, Hte = cs.forward(Xtr), cs.forward(Xte)
        rec = cs.backward(Hte[:512], 512)
        fid = float(np.mean(np.linalg.norm(
            norm_img(rec) - norm_img(Xte[:512]), axis=1)))
        print(f"[{kind}] layers 1-2 trained in {time.time()-t:.0f}s; "
              f"image -> code -> image error {fid:.3f}")

        sheet(cs.l1.W.reshape(-1, PATCH, PATCH), OUT / f"l1_templates_{kind}.png",
              f"{kind} — layer-one templates (4x4 patches)", cmap="bwr")
        L2pix = cs.backward(np.eye(K2 * L2_GRID * L2_GRID)[
            [i * K2 + j for i in range(1) for j in range(min(K2, 64))]], 64)
        sheet(L2pix, OUT / f"l2_templates_{kind}.png",
              f"{kind} — layer-two templates, pushed back down to pixels")

        results[kind] = {}
        for rho in RHOS:
            rng3 = np.random.default_rng(SEED + 1)
            l3 = BoundLayer(K3, Htr.shape[1], 10, rho, ETA3, rng3)
            for _ in range(EPOCHS3):
                o = rng3.permutation(len(Htr))
                for s in range(0, len(o), 256):
                    b = o[s:s + 256]
                    l3.learn(Htr[b], onehot(ytr[b]), ytr[b])

            pred, _, _ = l3.classify(Hte)
            acc = float((pred == yte).mean())
            pur, used = l3.purity()

            gen_acc, gens, judged = {}, [], []
            for name, m in (("top-1", 1), ("graded", 32)):
                code, _ = l3.generate(onehot(np.arange(10)), topm=m)
                imgs = cs.backward(code, 10)
                jl = judge.predict(norm_img(imgs))
                gen_acc[name] = float((jl == np.arange(10)).mean())
                gens.append((name, imgs)); judged.append(jl)

            results[kind][str(rho)] = {
                "accuracy": acc, "template_purity": pur, "templates_used": used,
                "generation_accuracy_top1": gen_acc["top-1"],
                "generation_accuracy_graded": gen_acc["graded"],
                "recon_error": fid}
            print(f"  rho={rho:<4} accuracy {acc:.4f}   purity {pur:.3f}   "
                  f"generated ok  top-1 {gen_acc['top-1']:.1f}  "
                  f"graded {gen_acc['graded']:.1f}")

            if rho == 1.0 or (kind == "dense" and rho == 0.6):
                save_generated(gens, judged, OUT / f"generated_{kind}_rho{rho}.png",
                               f"{kind}, rho={rho} — the label alone, pushed "
                               f"back down to pixels (green = judge agrees)")
            if rho == 1.0:
                save_confusion(confusion_matrix(yte, pred), acc,
                               OUT / f"confusion_{kind}.png",
                               f"{kind}, rho=1.0 — image in, label read off the winner")
                report_txt.append(
                    f"=== {kind}, rho=1.0 — classification report "
                    f"({len(yte)} held-out digits) ===\n" +
                    classification_report(yte, pred, digits=4))
        print()

    save_sweep(results, OUT / "label_energy_sweep.png")
    (OUT / "classification_report.txt").write_text("\n\n".join(report_txt))
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"k1": K1, "k2": K2, "k3": K3, "train": N_TRAIN,
                    "test": N_TEST, "epochs": EPOCHS, "epochs_l3": EPOCHS3,
                    "rhos": RHOS},
         "judge_accuracy": float(judge.score(norm_img(Xte), yte)),
         "results": results, "seconds": round(time.time() - t0, 1)}, indent=2))
    print("\n".join(report_txt))
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
