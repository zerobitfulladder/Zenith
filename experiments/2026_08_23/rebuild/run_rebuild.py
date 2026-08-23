"""The rebuild: local receptive fields + concat free-allocation top layer.

L1: a 7x7 grid of independent Zenith nodes ("hypercolumns"), each seeing one
non-overlapping 4x4 patch of the image. Each node is a whole-hypothesis
dictionary *of its window* — at this scale, strokes. Output per position is
top-1 sparse: one nonzero (the winner's correlation) out of K1.

L2: a free-competition Zenith over [concatenated sparse code ; lam * onehot]
— the concat top layer that won the earlier tests, now over parts.

Generation: label-only query -> winning template's code half is a
constellation of per-position stroke pointers -> each position's segment is
rendered through that node's templates -> assembled 28x28 image. Parts +
arrangement, generatively. No feedback anywhere (that chapter is closed).

Run:  .venv/bin/python experiments/2026_08_23/rebuild/run_rebuild.py
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

from gain_feedback import EPS, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000
SIDE = 28
PATCH = 4                    # non-overlapping 4x4 patches -> 7x7 grid
GRID = SIDE // PATCH         # 7
N_POS = GRID * GRID          # 49
K1 = 25                      # templates per position node
K2 = int(os.environ.get("GF_K2", "100"))   # free-allocation top-layer templates
ETA1 = 0.05
ETA2 = 0.04
EPOCHS = 2
LAMBDAS = [float(os.environ["GF_LAM"])] if os.environ.get("GF_LAM") else [0.5, 1.0]
SEED = 42
# Batch 1 fixes:
NORM_FLOOR = 0.15   # contrast floor: a patch below this centered norm is
                    # treated as blank — it neither activates nor becomes a
                    # template. Stops normalization from inflating smudges
                    # into single-spike vocabulary ("dotted paper" artifact).
_SUFFIX = f"_k{K2}" if K2 != 100 else ""
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "rebuild" / "results" / f"batch1{_SUFFIX}"
# ---------------------------------------------------------------------------

CODE_DIM = N_POS * K1


def load_data():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def patches_of(x2d):
    """Yield (position index, flattened 4x4 patch)."""
    for r in range(GRID):
        for c in range(GRID):
            yield r * GRID + c, x2d[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH].ravel()


def encode_l1(x2d, nodes, learning):
    """Sparse per-position top-1 code; optionally trains the nodes."""
    code = np.zeros(CODE_DIM)
    for i, p in patches_of(x2d):
        p_c = p - p.mean()
        n = np.linalg.norm(p_c)
        if n < NORM_FLOOR:
            continue
        p_hat = p_c / n
        c = nodes[i].forward(p_hat)
        if learning:
            w = nodes[i].learn(p_hat, c)
        else:
            w = int(np.argmax(c))
        code[i * K1 + w] = max(float(c[w]), 0.0)
    return code


def encode_batch_l1(X2d, nodes):
    """Vectorized inference-only codes for a batch of (N,28,28) images."""
    N = len(X2d)
    code = np.zeros((N, CODE_DIM))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            P = X2d[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH].reshape(N, -1)
            P = P - P.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(P, axis=1)
            ok = norms > NORM_FLOOR
            P[ok] /= norms[ok, None]
            C = P @ nodes[i].W.T
            wins = C.argmax(axis=1)
            vals = np.maximum(C[np.arange(N), wins], 0.0) * ok
            code[np.arange(N), i * K1 + wins] = vals
    return code


def render_code(code_half, nodes):
    """Project a top-layer code half back to pixel space, position by position."""
    img = np.zeros((SIDE, SIDE))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            # Only the active constellation renders — the slightly-negative
            # floor that mean-centering leaves on inactive entries is
            # bookkeeping, not content.
            seg = np.maximum(code_half[i * K1:(i + 1) * K1], 0.0)
            img[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = \
                (seg @ nodes[i].W).reshape(PATCH, PATCH)
    return img


def train(lam, Xtr, ytr):
    rng = np.random.default_rng(SEED)
    nodes = [ZenithLayer(K1, PATCH * PATCH, ETA1, rng) for _ in range(N_POS)]
    l2 = ZenithLayer(K2, CODE_DIM + 10, ETA2, rng)

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc=f"lam={lam}", ncols=80):
            code = encode_l1(x, nodes, learning=True)
            h_hat, h_norm = center_norm(code)
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = lam
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                l2.learn(z_hat, l2.forward(z_hat))
    return nodes, l2


def evaluate(lam, nodes, l2, Xtr, ytr, Xte, yte):
    res = {}
    label_half = l2.W[:, CODE_DIM:]
    owner = label_half.argmax(axis=1)
    res["alloc"] = np.bincount(owner, minlength=10)

    # Label-only generation through the hierarchy.
    gen_imgs, consistent = [], 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = lam
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), label]))
        winner = int(np.argmax(l2.forward(z_hat)))
        consistent += int(owner[winner] == j)
        gen_imgs.append(render_code(l2.W[winner, :CODE_DIM], nodes))
    res["gen_imgs"] = gen_imgs
    res["label_consistency"] = consistent / 10.0

    # Codes for classification and probe.
    Cte = encode_batch_l1(Xte, nodes)
    Ctr = encode_batch_l1(Xtr[:PROBE_N], nodes)

    Hte = Cte - Cte.mean(axis=1, keepdims=True)
    Hte /= np.linalg.norm(Hte, axis=1, keepdims=True) + EPS
    Z = np.concatenate([Hte, np.zeros((len(Hte), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ l2.W.T
    winners = C2.argmax(axis=1)
    res["acc_no_label"] = float((owner[winners] == yte).mean())
    # Soft label readout: every template votes its stored label half,
    # weighted by its (positive) correlation with the query code.
    votes = np.maximum(C2, 0.0) @ label_half
    res["acc_soft"] = float((votes.argmax(axis=1) == yte).mean())
    # Class-normalized votes: raw sums favor template-rich classes
    # (allocation is unequal), so divide by each class's template count.
    counts = np.maximum(np.bincount(owner, minlength=10), 1)
    res["acc_soft_norm"] = float(((votes / counts).argmax(axis=1) == yte).mean())
    # Top-k votes: only the 10 best-matching templates vote — local
    # consensus instead of a global sum over 100 mostly-irrelevant cells.
    kv = 10
    part = np.argpartition(-C2, kv, axis=1)[:, :kv]
    mask = np.zeros_like(C2)
    mask[np.arange(len(C2))[:, None], part] = 1.0
    votes_k = (np.maximum(C2, 0.0) * mask) @ label_half
    res["acc_soft_topk"] = float((votes_k.argmax(axis=1) == yte).mean())

    probe = LogisticRegression(max_iter=1000)
    probe.fit(Ctr, ytr[:PROBE_N])
    res["probe"] = float(probe.score(Cte, yte))

    # Per-position usage entropy (within-node monopoly check).
    ents = []
    for nd in nodes:
        tot = nd.win_counts.sum()
        if tot == 0:
            continue
        p = nd.win_counts / tot
        ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(nd.k)))
    res["mean_pos_entropy"] = float(np.mean(ents))

    # L1 reconstruction of a few test images (visual only).
    recon_pairs = []
    for x in Xte[:8]:
        code = encode_batch_l1(x[None], nodes)[0]
        recon_pairs.append((x, render_code(code, nodes)))
    res["recon_pairs"] = recon_pairs
    return res


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    lines = [
        "# Rebuild — local receptive fields + concat top layer (MNIST)",
        "",
        f"Config: {GRID}x{GRID} grid of {PATCH}x{PATCH} patches, K1={K1}/position, "
        f"K2={K2}, ETA1={ETA1}, ETA2={ETA2}, EPOCHS={EPOCHS}, TRAIN_N={TRAIN_N}, SEED={SEED}.",
        "Whole-image baselines from earlier runs: selector acc 0.699, concat "
        "acc 0.727, L1 probe 0.871.",
        "",
        "| lam | label-only consistent | acc hard | acc soft | acc soft norm | acc soft top10 | code probe | mean per-position entropy | templates per class |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for lam in LAMBDAS:
        nodes, l2 = train(lam, Xtr, ytr)
        res = evaluate(lam, nodes, l2, Xtr, ytr, Xte, yte)
        tag = str(lam).replace(".", "p")

        fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
        for j, ax in enumerate(axes.flat):
            ax.imshow(res["gen_imgs"][j], cmap="gray")
            ax.set_title(str(j), fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Label-only compositional generation (lam={lam})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"generation_labelonly_lam{tag}.png", dpi=110)
        plt.close(fig)

        if lam == LAMBDAS[0]:
            center = (GRID // 2) * GRID + GRID // 2
            fig, axes = plt.subplots(5, 5, figsize=(6, 6.4))
            for t, ax in enumerate(axes.flat):
                ax.imshow(nodes[center].W[t].reshape(PATCH, PATCH), cmap="bwr")
                ax.axis("off")
            fig.suptitle(f"Center node templates ({PATCH}x{PATCH})")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "l1_templates_center.png", dpi=110)
            plt.close(fig)

            fig, axes = plt.subplots(2, 8, figsize=(14, 4))
            for k, (orig, rec) in enumerate(res["recon_pairs"]):
                axes[0, k].imshow(orig, cmap="gray")
                axes[1, k].imshow(rec, cmap="gray")
                axes[0, k].axis("off")
                axes[1, k].axis("off")
            fig.suptitle("Test images (top) vs L1 winner-template reconstruction (bottom)")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "recon_test.png", dpi=110)
            plt.close(fig)

        alloc = " ".join(str(a) for a in res["alloc"])
        lines.append(f"| {lam} | {res['label_consistency']:.1%} | {res['acc_no_label']:.4f} | "
                     f"{res['acc_soft']:.4f} | {res['acc_soft_norm']:.4f} | {res['acc_soft_topk']:.4f} | "
                     f"{res['probe']:.4f} | {res['mean_pos_entropy']:.3f} | {alloc} |")
        print(f"RESULT lam={lam}: hard={res['acc_no_label']:.4f} soft={res['acc_soft']:.4f} "
              f"soft_norm={res['acc_soft_norm']:.4f} soft_top10={res['acc_soft_topk']:.4f} "
              f"probe={res['probe']:.4f} consistent={res['label_consistency']:.1%}")

    lines += ["", "Figures: generation_labelonly_lam*.png, l1_templates_center.png, recon_test.png"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
