"""Attention-style unnormalized L1: argmax learning, scaled-softmax output.

Fixes both diseases of the sampled-softmax run: learning is argmax (the
only non-blurring rule), and the output is softmax(dots/tau) scaled by the
value magnitude relu(max dot) — attention weights times unnormalized
values, restoring the energy signal bare softmax destroyed. Tests the
open corner: argmax + no normalization has neither geometric nor
stochastic monopoly protection.

L1 drops mean-centering and L2 normalization entirely:
  logits  = W @ x_raw / tau          (raw patches, raw templates)
  learn   = sample ONE index from softmax(logits); blend it toward the
            patch: w += eta * (x - w)   (bounded; online k-means style)
  output  = softmax(logits)          (graded probabilities)
Generation reads the stored code hardened (argmax per position, dual-mode).
The top layer is the unchanged, validated concat Zenith (it centers and
normalizes its own input space as always). 2-layer batch-2 geometry:
4x4 windows, stride 2, 13x13 grid, K1=36, K2=200, lam=0.5.

Run:  .venv/bin/python experiments/2026_08_23/unnormalized_l1/run_attention_nonorm.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import EPS, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 4000
SIDE = 28
PATCH = 4
STRIDE = 2
K1 = 36
K2 = 200
ETA1 = 0.05
ETA2 = 0.04
EPOCHS = 2
LAM = 0.5
SEED = 42
TAUS = [0.5, 0.2]
INTENSITY_FLOOR = 0.10   # skip patches whose brightest pixel is below this —
                         # without it, uniform sampling on black patches slowly
                         # drags every template toward darkness
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "unnormalized_l1" / "results" / "attention"
# ---------------------------------------------------------------------------

POS = [(r, c) for r in range(0, SIDE - PATCH + 1, STRIDE)
       for c in range(0, SIDE - PATCH + 1, STRIDE)]
N_POS = len(POS)
CODE_DIM = N_POS * K1
FEATHER = np.outer([0.5, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 0.5])


def load_data():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


class SoftmaxUnit:
    """Unnormalized dictionary: raw-dot logits, softmax-sampled learning."""

    def __init__(self, k, dim, eta, tau, rng):
        self.k, self.dim, self.eta, self.tau = k, dim, eta, tau
        self.rng = rng
        self.W = np.zeros((k, dim))
        self.n_boot = 0
        self.win_counts = np.zeros(k, dtype=np.int64)

    def probs(self, x):
        z = self.W @ x / self.tau
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def learn(self, x):
        if self.n_boot < self.k:
            self.W[self.n_boot] = x + 0.01 * self.rng.standard_normal(self.dim)
            self.win_counts[self.n_boot] += 1
            self.n_boot += 1
            return
        # Argmax learning — the only non-blurring rule (sampled learning is
        # graded learning in expectation; measured twice). No geometric or
        # stochastic monopoly protection here: entropy is the open question.
        j = int(np.argmax(self.W @ x))
        self.W[j] += self.eta * (x - self.W[j])
        self.win_counts[j] += 1

    def message(self, x):
        """Attention-style output: softmax shape x value magnitude."""
        return self.probs(x) * max(float(np.max(self.W @ x)), 0.0)


def encode_image(x2d, unit, learning):
    code = np.zeros(CODE_DIM)
    for i, (r, c) in enumerate(POS):
        x = x2d[r:r + PATCH, c:c + PATCH].ravel()
        if x.max() < INTENSITY_FLOOR:
            continue
        if learning:
            unit.learn(x)
        code[i * K1:(i + 1) * K1] = unit.message(x)
    return code


def encode_batch(X2d, unit):
    N = len(X2d)
    code = np.zeros((N, CODE_DIM), dtype=np.float32)
    for pi, (r, c) in enumerate(POS):
        P = X2d[:, r:r + PATCH, c:c + PATCH].reshape(N, -1)
        ok = P.max(axis=1) >= INTENSITY_FLOOR
        D = P @ unit.W.T
        Z = D / unit.tau
        Z -= Z.max(axis=1, keepdims=True)
        Pr = np.exp(Z)
        Pr /= Pr.sum(axis=1, keepdims=True)
        mag = np.maximum(D.max(axis=1), 0.0)
        code[:, pi * K1:(pi + 1) * K1] = (Pr * mag[:, None] * ok[:, None]).astype(np.float32)
    return code


def render(code_half, unit):
    """Hardened read: per position argmax; patch = the raw template itself."""
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    for pi, (r, c) in enumerate(POS):
        seg = np.maximum(code_half[pi * K1:(pi + 1) * K1], 0.0)
        conf = float(seg.max())
        if conf <= 0.0:
            continue
        w = int(np.argmax(seg))
        num[r:r + PATCH, c:c + PATCH] += unit.W[w].reshape(PATCH, PATCH) * FEATHER * conf
        den[r:r + PATCH, c:c + PATCH] += FEATHER * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def run_tau(tau, Xtr, ytr, Xte, yte):
    rng = np.random.default_rng(SEED)
    unit = SoftmaxUnit(K1, PATCH * PATCH, ETA1, tau, rng)
    l2 = ZenithLayer(K2, CODE_DIM + 10, ETA2, rng)

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc=f"tau={tau}", ncols=80):
            code = encode_image(x, unit, learning=True)
            h_hat, h_norm = center_norm(code)
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = LAM
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                l2.learn(z_hat, l2.forward(z_hat))

    label_half = l2.W[:, CODE_DIM:]
    owner = label_half.argmax(axis=1)

    gen_imgs, consistent = [], 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), label]))
        winner = int(np.argmax(l2.forward(z_hat)))
        consistent += int(owner[winner] == j)
        gen_imgs.append(render(np.maximum(l2.W[winner, :CODE_DIM], 0.0), unit))

    Cte = encode_batch(Xte, unit)
    Ctr = encode_batch(Xtr[:PROBE_N], unit)
    H = Cte - Cte.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=np.float32)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ l2.W.T.astype(np.float32)
    acc_hard = float((owner[C2.argmax(axis=1)] == yte).mean())

    probe = LogisticRegression(max_iter=1000)
    probe.fit(Ctr, ytr[:PROBE_N])
    probe_acc = float(probe.score(Cte, yte))

    p = unit.win_counts / unit.win_counts.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(unit.k))
    dead = int((unit.win_counts == 0).sum())

    print(f"RESULT tau={tau} probe={probe_acc:.4f} hard={acc_hard:.4f} "
          f"entropy={entropy:.3f} dead={dead} consistent={consistent}/10")
    return unit, gen_imgs, {"tau": tau, "probe": probe_acc, "hard": acc_hard,
                            "entropy": entropy, "dead": dead, "consistent": consistent}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    rows = []
    for tau in TAUS:
        unit, gen_imgs, res = run_tau(tau, Xtr, ytr, Xte, yte)
        rows.append(res)
        tag = str(tau).replace(".", "p")

        fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
        for j, ax in enumerate(axes.flat):
            ax.imshow(gen_imgs[j], cmap="gray")
            ax.set_title(str(j), fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Label-only generation, softmax-sampled unnormalized L1 (tau={tau})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"generation_tau{tag}.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(6, 6, figsize=(7, 7.4))
        for t, ax in enumerate(axes.flat):
            ax.imshow(unit.W[t].reshape(PATCH, PATCH), cmap="gray")
            ax.axis("off")
        fig.suptitle(f"Unnormalized dictionary (tau={tau}) — raw patches, no centering")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"dictionary_tau{tag}.png", dpi=110)
        plt.close(fig)

    lines = [
        "# Softmax-sampled, unnormalized L1 (MNIST)",
        "",
        "Validated normalized 2-layer baselines: probe 0.9146, hard 0.743 (batch 2).",
        "",
        "| tau | probe | hard | usage entropy | dead | label-only consistent |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['tau']} | {r['probe']:.4f} | {r['hard']:.4f} | "
                     f"{r['entropy']:.3f} | {r['dead']} | {r['consistent']}/10 |")
    lines += ["", "Figures: generation_tau*.png, dictionary_tau*.png"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
