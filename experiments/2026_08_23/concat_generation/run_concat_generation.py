"""Concat generation test: label concatenated at the top layer, generation
queried with the label alone (image channel zeroed).

L1: Zenith on pixels (top-1, bootstrap init, no feedback).
L2: a *free-competition* Zenith over the concatenation [L1 code ; lam * onehot],
    bootstrap init, top-1 WTA — the label is evidence, not a selector: the
    label<->content association is learned inside the templates, and template
    allocation per class is decided by competition (March's label-concat
    semantics, now one level up from pixels).

Generation: z = [zeros ; lam * onehot(y)] -> center+norm -> winner template ->
its code half, projected through W1, is the generated image. Also evaluated:
classification *without* labels (winner's stored label half is the prediction),
and templates-per-class allocation as a function of lam.

Run:  .venv/bin/python experiments/2026_08_23/concat_generation/run_concat_generation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import EPS, ClassTemplates, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
K1 = 100
K2 = 50                  # free-allocation top-layer templates (not 10!)
ETA1 = 0.04
ETA2 = 0.04
EPOCHS = 2
LAMBDAS = [0.25, 0.5, 1.0]   # label weight relative to the (unit-norm) code half
SEED = 42
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "concat_generation" / "results"
# ---------------------------------------------------------------------------


def load_data():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    X = X.reshape(len(X), -1)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def l1_code(x, l1):
    x_c = x - x.mean()
    n = np.linalg.norm(x_c)
    if n < EPS:
        return None, None
    x_hat = x_c / n
    c1 = l1.forward(x_hat)
    h_hat, h_norm = center_norm(np.maximum(c1, 0.0))
    return x_hat, (h_hat if h_norm > EPS else None)


def train(lam, Xtr, ytr):
    rng = np.random.default_rng(SEED)
    l1 = ZenithLayer(K1, Xtr.shape[1], ETA1, rng)
    l2 = ZenithLayer(K2, K1 + 10, ETA2, rng)
    l2_sel = ClassTemplates(10, K1, 0.10, rng)   # selector baseline on same L1

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc=f"lam={lam}", ncols=80):
            x_hat, h_hat = l1_code(x, l1)
            if x_hat is None:
                continue
            c1 = l1.forward(x_hat)
            l1.learn(x_hat, c1)
            if h_hat is None:
                continue

            label = np.zeros(10)
            label[y] = lam
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                c2 = l2.forward(z_hat)
                l2.learn(z_hat, c2)
            l2_sel.learn(h_hat, y)

    return l1, l2, l2_sel


def evaluate(lam, l1, l2, l2_sel, Xtr, ytr, Xte, yte):
    res = {}

    # Template-per-class allocation: each L2 template's dominant stored label.
    label_half = l2.W[:, K1:]
    owner = label_half.argmax(axis=1)
    strength = label_half.max(axis=1)
    alloc = np.bincount(owner[strength > 0], minlength=10)
    res["alloc"] = alloc

    # Label-only generation.
    gen_imgs, consistent = [], 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = lam
        z_hat, _ = center_norm(np.concatenate([np.zeros(K1), label]))
        winner = int(np.argmax(l2.forward(z_hat)))
        consistent += int(owner[winner] == j)
        gen_imgs.append(l2.W[winner, :K1] @ l1.W)
    res["gen_imgs"] = np.stack(gen_imgs)
    res["label_consistency"] = consistent / 10.0

    # Generation quality: correlation of each generated image with the class
    # mean image (both centered+normalized). Same score for the selector
    # baseline's generation from the same L1.
    def gen_score(imgs):
        scores = []
        for j in range(10):
            m = Xtr[ytr == j].mean(axis=0)
            m_hat, _ = center_norm(m)
            g_hat, _ = center_norm(imgs[j])
            scores.append(float(m_hat @ g_hat))
        return float(np.mean(scores))

    res["gen_corr"] = gen_score(res["gen_imgs"])
    res["gen_corr_selector"] = gen_score(l2_sel.W @ l1.W)

    # Classification WITHOUT labels: zero label half, winner's stored label
    # half is the prediction.
    correct = 0
    for x, y in zip(Xte, yte):
        _, h_hat = l1_code(x, l1)
        if h_hat is None:
            continue
        z_hat, _ = center_norm(np.concatenate([h_hat, np.zeros(10)]))
        winner = int(np.argmax(l2.forward(z_hat)))
        correct += int(owner[winner] == y)
    res["acc_no_label"] = correct / len(yte)

    return res


def img_grid(vectors, path, title):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        ax.imshow(vectors[j].reshape(28, 28), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    lines = [
        "# Concat generation test — label-only queries at the top layer",
        "",
        f"Config: TRAIN_N={TRAIN_N}, K1={K1}, K2={K2}, ETA1={ETA1}, ETA2={ETA2}, "
        f"EPOCHS={EPOCHS}, SEED={SEED}. L2 is a free-competition Zenith over "
        "[L1 code ; lam * onehot]; the selector baseline shares the same L1.",
        "",
        "| lam | label-only retrieval consistent | gen corr (concat) | gen corr (selector) | acc without label | templates per class |",
        "|---|---|---|---|---|---|",
    ]
    for lam in LAMBDAS:
        l1, l2, l2_sel = train(lam, Xtr, ytr)
        res = evaluate(lam, l1, l2, l2_sel, Xtr, ytr, Xte, yte)
        tag = str(lam).replace(".", "p")
        img_grid(res["gen_imgs"], OUTPUT_DIR / f"generation_labelonly_lam{tag}.png",
                 f"Label-only generation through concat L2 (lam={lam})")
        alloc = " ".join(str(a) for a in res["alloc"])
        lines.append(
            f"| {lam} | {res['label_consistency']:.1%} | {res['gen_corr']:.3f} | "
            f"{res['gen_corr_selector']:.3f} | {res['acc_no_label']:.4f} | {alloc} |")
        print(f"lam={lam}: consistency={res['label_consistency']:.1%} "
              f"gen={res['gen_corr']:.3f} (sel {res['gen_corr_selector']:.3f}) "
              f"acc_nolabel={res['acc_no_label']:.4f} alloc={alloc}")

    lines += [
        "",
        "'Templates per class' lists how many of the K2 templates store each",
        "digit 0-9 as their dominant label. Figures: generation_labelonly_lam*.png",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
