"""Gamma sweep for Variant A gain feedback on MNIST.

For each gamma (0 = no-feedback control), trains the identical two-layer
system on the identical data order from the identical init, then evaluates
with neutral gain (no label at test time). Produces figures and a report in
gamma_sweep/results/<variant>_<gain mode>/.

Run:  .venv/bin/python experiments/2026_08_23/gamma_sweep/run_experiment.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import (
    EPS,
    ClassTemplates,
    ZenithLayer,
    center_norm,
    class_gain_distinctness,
    encode_batch,
    gain_field,
    plasticity_gain,
)

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000          # training samples used to fit the linear probe
K1 = 100                # L1 templates
ETA1 = 0.04
ETA2 = 0.10
EPOCHS = 2
GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEED = 42
PAIRS = [(1, 7), (3, 5), (4, 9), (5, 8), (7, 9)]
COHERENCE_EVERY = 2000  # samples between gain-distinctness snapshots

# Overridable via env so past runs stay reproducible from the defaults:
#   VARIANT   "A": gain multiplies L1's input (the note's spec)
#             "B": gain multiplies the winner's learning step (V12 rule) —
#                  lives in template-index space, never touches what L1 sees
#   GAIN_MODE "prototype": source is W2[y]
#             "contrast":  source is W2[y] - mean(W2)
VARIANT = os.environ.get("GF_VARIANT", "A")
GAIN_MODE = os.environ.get("GF_GAIN_MODE", "prototype")
if VARIANT == "A":
    _DIR_NAME = f"A_{GAIN_MODE}"
else:
    _DIR_NAME = f"B_{GAIN_MODE}"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "gamma_sweep" / "results" / _DIR_NAME
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


def train_condition(gamma, Xtr, ytr):
    rng = np.random.default_rng(SEED)
    l1 = ZenithLayer(K1, Xtr.shape[1], ETA1, rng)
    l2 = ClassTemplates(10, K1, ETA2, rng)
    coherence = []

    step = 0
    for epoch in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc=f"gamma={gamma} epoch {epoch + 1}", ncols=80):
            x_c = x - x.mean()
            plast = None
            if gamma > 0.0:
                src = l2.W[y] - l2.W.mean(axis=0) if GAIN_MODE == "contrast" else l2.W[y]
                if VARIANT == "A":
                    x_c = gain_field(src, l1.W, gamma) * x_c
                else:
                    plast = plasticity_gain(src, gamma)
            n = np.linalg.norm(x_c)
            if n < EPS:
                continue
            x_hat = x_c / n

            c1 = l1.forward(x_hat)
            l1.learn(x_hat, c1, step_gain=plast)

            h_hat, h_norm = center_norm(np.maximum(c1, 0.0))
            if h_norm > EPS:
                l2.learn(h_hat, y)

            if step % COHERENCE_EVERY == 0:
                coherence.append((step, class_gain_distinctness(l2.W, l1.W)))
            step += 1

    return l1, l2, coherence


def evaluate(l1, l2, Xtr, ytr, Xte, yte):
    A1_te, C2_te = encode_batch(Xte, l1.W, l2.W)
    pred = C2_te.argmax(axis=1)
    acc = float((pred == yte).mean())

    conf = np.zeros((10, 10), dtype=np.int64)
    np.add.at(conf, (yte, pred), 1)

    margins = {}
    for a, b in PAIRS:
        ma = C2_te[yte == a, a] - C2_te[yte == a, b]
        mb = C2_te[yte == b, b] - C2_te[yte == b, a]
        margins[(a, b)] = float(np.concatenate([ma, mb]).mean())

    A1_tr, _ = encode_batch(Xtr[:PROBE_N], l1.W, l2.W)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(A1_tr, ytr[:PROBE_N])
    probe_acc = float(probe.score(A1_te, yte))

    Xc = Xte - Xte.mean(axis=1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + EPS
    top1_corr = float((Xc @ l1.W.T).max(axis=1).mean())

    p = l1.win_counts / l1.win_counts.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(l1.k))
    dead = int((l1.win_counts == 0).sum())

    return {
        "acc": acc, "conf": conf, "margins": margins, "probe_acc": probe_acc,
        "top1_corr": top1_corr, "usage_entropy": entropy, "dead": dead,
    }


def img_grid(vectors, path, title, cmap="gray", sym=False):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        im = vectors[j].reshape(28, 28)
        if sym:
            v = np.abs(im).max() + EPS
            ax.imshow(im, cmap="bwr", vmin=-v, vmax=v)
        else:
            ax.imshow(im, cmap=cmap)
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pair_diff_figure(l2, l1, gamma, path):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(3 * len(PAIRS), 3.4))
    for ax, (a, b) in zip(axes, PAIRS):
        d = (l2.W[a] - l2.W[b]) @ l1.W
        v = np.abs(d).max() + EPS
        ax.imshow(d.reshape(28, 28), cmap="bwr", vmin=-v, vmax=v)
        ax.set_title(f"g({a}) - g({b})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Where the downward projections disagree (gamma={gamma}) — red favors first class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    results = {}
    for gamma in GAMMAS:
        print(f"=== gamma = {gamma} ===")
        l1, l2, coherence = train_condition(gamma, Xtr, ytr)
        res = evaluate(l1, l2, Xtr, ytr, Xte, yte)
        res["coherence"] = coherence
        results[gamma] = res
        print(f"  acc={res['acc']:.4f}  probe={res['probe_acc']:.4f}  "
              f"top1corr={res['top1_corr']:.3f}  entropy={res['usage_entropy']:.3f}  dead={res['dead']}")

        tag = str(gamma).replace(".", "p")
        img_grid(l2.W @ l1.W, OUTPUT_DIR / f"generation_gamma{tag}.png",
                 f"Generated class images W2 @ W1 (gamma={gamma})")
        if gamma > 0 and VARIANT == "A":
            w2_bar = l2.W.mean(axis=0)
            fields = np.stack([
                gain_field(l2.W[j] - w2_bar if GAIN_MODE == "contrast" else l2.W[j], l1.W, gamma)
                for j in range(10)
            ])
            img_grid(fields, OUTPUT_DIR / f"gain_fields_gamma{tag}.png",
                     f"Final gain fields per class (gamma={gamma})", cmap="viridis")
        pair_diff_figure(l2, l1, gamma, OUTPUT_DIR / f"pair_diff_gamma{tag}.png")

    # ---- Summary figures --------------------------------------------------
    gs = list(results.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gs, [results[g]["acc"] for g in gs], "o-", label="pipeline accuracy (argmax L2)")
    ax.plot(gs, [results[g]["probe_acc"] for g in gs], "s-", label="linear probe on L1 codes")
    ax.set_xlabel("gamma (feedback strength)")
    ax.set_ylabel("test accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_accuracy.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for pair in PAIRS:
        ax.plot(gs, [results[g]["margins"][pair] for g in gs], "o-", label=f"{pair[0]} vs {pair[1]}")
    ax.set_xlabel("gamma (feedback strength)")
    ax.set_ylabel("mean margin  c2[true] - c2[rival]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_margins.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for g in gs:
        steps, vals = zip(*results[g]["coherence"])
        ax.plot(steps, vals, label=f"gamma={g}")
    ax.set_xlabel("training step")
    ax.set_ylabel("class-gain distinctness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coherence.png", dpi=110)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    lines = [
        f"# Variant {VARIANT} Gain Feedback — gamma sweep report (gain mode: {GAIN_MODE})",
        "",
        f"Config: TRAIN_N={TRAIN_N}, TEST_N={TEST_N}, K1={K1}, ETA1={ETA1}, "
        f"ETA2={ETA2}, EPOCHS={EPOCHS}, SEED={SEED}. Evaluation is always "
        "feedforward with neutral gain (no label at test time).",
        "",
        "## Headline table",
        "",
        "| gamma | pipeline acc | L1 probe acc | mean top-1 corr | L1 usage entropy | dead templates |",
        "|---|---|---|---|---|---|",
    ]
    for g in gs:
        r = results[g]
        lines.append(f"| {g} | {r['acc']:.4f} | {r['probe_acc']:.4f} | "
                     f"{r['top1_corr']:.3f} | {r['usage_entropy']:.3f} | {r['dead']} |")

    lines += ["", "## Confusable-pair margins (higher = better separated)", "",
              "| gamma | " + " | ".join(f"{a}v{b}" for a, b in PAIRS) + " |",
              "|---" * (len(PAIRS) + 1) + "|"]
    for g in gs:
        row = " | ".join(f"{results[g]['margins'][p]:.4f}" for p in PAIRS)
        lines.append(f"| {g} | {row} |")

    lines += ["", "## Confusion matrix (rows true, cols predicted)", ""]
    for g in gs:
        lines += [f"### gamma = {g}", "", "```",
                  str(results[g]["conf"]), "```", ""]

    lines += [
        "## Figures",
        "",
        "- `summary_accuracy.png` — both accuracies vs gamma",
        "- `summary_margins.png` — pair margins vs gamma",
        "- `coherence.png` — class-gain distinctness over training (self-bootstrap check)",
        "- `generation_gamma*.png` — the 10 downward projections W2 @ W1 as images",
        "- `gain_fields_gamma*.png` — final per-class gain fields",
        "- `pair_diff_gamma*.png` — where downward projections disagree for confusable pairs",
        "",
        "See README.md for the a-priori predictions and pass/fail criteria.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
