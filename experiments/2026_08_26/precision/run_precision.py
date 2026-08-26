"""Weight precision: how many bits does a template actually need?

Loads the trained base-rig weights (base/results/weights.npz) and
quantizes every bank post-hoc to b bits per weight (per-template
symmetric uniform rounding; b=2 is ternary -1/0/+1), keeping one f32
scale per template (renormalization — runtime divisive normalization,
which the unit already assumes). Template mean needs no correction:
queries are mean-centered, so any template mean cancels in the match.

Arms: f32 reference, f16, 8, 6, 4, 3, 2 bits. Per arm: hard readout,
label-consistency, and the label-only generation row. No retraining.

Run:  .venv/bin/python experiments/2026_08_26/precision/run_precision.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "base"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_gpu_minibatch import DTYPE, Dict, XP_NAME, xp  # noqa: E402
from run_4layer_topk import ETA1, ETA2, ETATOP, KTOP, LAM, expand, load_data  # noqa: E402
from run_rich_palette_8x8 import G1, G2, S2, W1, W2, render8  # noqa: E402
from gain_feedback import center_norm  # noqa: E402
from run_base import CODE_DIM, EVAL_B, K1, K2, _harden_map, stack, to_np  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "precision" / "results" / "post_hoc"
BASE_WEIGHTS = ROOT / "experiments" / "2026_08_26" / "base" / "results" / "weights.npz"
ARMS = ["f32", "f16", 8, 6, 4, 3, 2]


def quantize(W, arm, split=None):
    """Per-row symmetric uniform quantization to `arm` bits, then per-row
    renormalization (one f32 scale per template).

    v2 codebook: the step is set from the 99.5th percentile of |w| (rare
    outliers clip) instead of the row max — max-scaling zeroed 100% of the
    top bank's code half at 4 bits (label channels are ~155x the typical
    weight). `split` quantizes [:split] and [split:] with separate scales
    (per-pathway gain for the top's code/label halves)."""
    if arm == "f32":
        Q = W.copy()
    elif arm == "f16":
        Q = W.astype(np.float16).astype(np.float32)
    else:
        levels = 2 ** (int(arm) - 1) - 1          # e.g. 4 bits -> +-7
        segs = [(0, split), (split, W.shape[1])] if split else [(0, W.shape[1])]
        Q = np.empty_like(W)
        for a, b in segs:
            seg = W[:, a:b]
            s = np.percentile(np.abs(seg), 99.5, axis=1, keepdims=True) / levels
            s = np.maximum(s, 1e-12)
            Q[:, a:b] = np.clip(np.round(seg / s), -levels, levels) * s
    return Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    wz = np.load(BASE_WEIGHTS)
    W1f, W2f, Wtf = wz["W1"], wz["W2"], wz["Wtop"]

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    rows = []
    fig, axes = plt.subplots(len(ARMS), 10, figsize=(10.5, 1.15 * len(ARMS) + 0.6))
    for ai, arm in enumerate(ARMS):
        W1q = quantize(W1f, arm)
        W2q = quantize(W2f, arm)
        Wtq = quantize(Wtf, arm, split=CODE_DIM)

        d1 = Dict(K1, W1 * W1, ETA1)
        d2 = Dict(K2, W2 * W2 * K1, ETA2)
        d1.W = xp.asarray(W1q, dtype=DTYPE)
        d2.W = xp.asarray(W2q, dtype=DTYPE)

        C2s = []
        for s in range(0, len(Xte), EVAL_B):
            xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
            C2s.append(to_np(stack(xb, d1, d2, False).reshape(-1, CODE_DIM)))
        C2te = np.concatenate(C2s)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()

        owner = np.argmax(Wtq[:, CODE_DIM:], axis=1)
        H = C2te - C2te.mean(axis=1, keepdims=True)
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
        Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
        Z -= Z.mean(axis=1, keepdims=True)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        hard = float((owner[(Z @ Wtq.T).argmax(axis=1)] == yte).mean())

        consistent = 0
        b2 = Bank(W2q)
        for j in range(10):
            lab = np.zeros(10)
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
            winner = int(np.argmax(Wtq @ z_hat))
            consistent += int(owner[winner] == j)
            c2 = np.maximum(Wtq[winner, :CODE_DIM], 0.0).reshape(G2, G2, K2)
            m1 = _harden_map(expand(_harden_map(c2), b2, W2, S2, (G1, G1, K1)))
            ax = axes[ai, j]
            ax.imshow(render8(m1, W1q), cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(arm))
        axes[ai, 0].axis("on")
        axes[ai, 0].set_xticks([])
        axes[ai, 0].set_yticks([])
        for spine in axes[ai, 0].spines.values():
            spine.set_visible(False)
        axes[ai, 0].set_ylabel(f"{arm}\nbits" if isinstance(arm, int) else arm,
                               fontsize=8, rotation=0, ha="right", va="center")

        rows.append((arm, hard, consistent))
        print(f"ARM {arm}: hard={hard:.4f} consistent={consistent}/10", flush=True)

    fig.suptitle("Label-only generation vs weight precision (base rig, no retraining)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_vs_bits.png", dpi=110)
    plt.close(fig)

    lines = [
        "# Weight precision sweep (base rig, post-training quantization)",
        "",
        "Per-template symmetric uniform rounding + one f32 scale per row",
        "(renormalization). f32 base reference: hard .9112, consistent 10/10.",
        "",
        "| bits | hard | consistent |",
        "|---|---|---|",
    ]
    for arm, hard, consistent in rows:
        lines.append(f"| {arm} | {hard:.4f} | {consistent}/10 |")
    lines += ["", "Figure: generation_vs_bits.png (one generation row per arm)."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
