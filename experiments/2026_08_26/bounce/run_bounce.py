"""Bouncing generation: top -> pixels -> top -> ... iterative resolution.

User's mechanism: generation as settling, like an associative network.
Label + empty code query the top once; the retrieved code renders down
to pixels; the pixels re-encode up through L1/L2; generate again from
that — bouncing top-to-bottom-to-top. Annealed resolution: start DENSE
(every template speaks), and each bounce keeps fewer templates per
position (k: dense -> 32 -> 8 -> 2 -> 1) so the architecture itself
eliminates candidates gradually instead of hardening in one step.

Arms:
  self    — bounce render/encode only; the top is consulted once at start.
  consult — each bounce re-queries the top with the re-encoded code +
            CLAMPED label, and blends the matched memory's stored code
            into the working code (associative anchor).

All on the saved base weights (base/results/weights.npz), CPU only.
Metrics: L2-code peakiness per bounce (does resolution rise?) and
self-classification of the final images (does the network recognize
its own drawings as the intended label?).

Run:  .venv/bin/python experiments/2026_08_26/bounce/run_bounce.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"           # CPU only; GPU is busy elsewhere

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_4layer_topk import LAM, NORM_FLOOR, expand  # noqa: E402
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    POS1,
    POS2,
    S2,
    W1,
    W2,
    render8,
)
from gain_feedback import center_norm  # noqa: E402

RESULTS = ROOT / "experiments" / "2026_08_26" / "base" / "results"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "bounce" / "results"
K1, K2 = 1024, 1024
CODE_DIM = G2 * G2 * K2
K_SCHEDULE = [None, 32, 8, 2, 1]        # None = dense (all templates speak)
BLEND = 0.5                             # consult arm: weight of the memory code


def _topk_map(m, k):
    if k is None:
        return np.maximum(m, 0.0)
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b_ in range(m.shape[1]):
            seg = np.maximum(m[a, b_], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b_, idx] = seg[idx]
    return o


def _encode_windows(field, positions, win, Wb):
    """Dense relu code of `field` (H, W, C) against bank rows."""
    D = win * win * field.shape[2]
    V = np.stack([field[r:r + win, c:c + win, :].reshape(D)
                  for r, c in positions])
    V = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(V, axis=1)
    V = V / np.maximum(n, 1e-9)[:, None]
    C = np.maximum(V @ Wb.T, 0.0) * (n > NORM_FLOOR)[:, None]
    g = int(np.sqrt(len(positions)))
    return C.reshape(g, g, -1)


def peakiness(c2):
    r = []
    for a in range(c2.shape[0]):
        for b_ in range(c2.shape[1]):
            seg = np.maximum(c2[a, b_], 0.0)
            s = seg.sum()
            if s > 0:
                r.append(seg.max() / s)
    return float(np.mean(r)) if r else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wz = np.load(RESULTS / "weights.npz")
    W1n, W2n, Wtn = wz["W1"], wz["W2"], wz["Wtop"]
    owner = np.argmax(Wtn[:, CODE_DIM:], axis=1)

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2 = Bank(W2n)

    def render_from_code(c2, k):
        m1 = _topk_map(expand(_topk_map(c2, k), b2, W2, S2, (G1, G1, K1)), k)
        return render8(m1, W1n)

    def encode_image(img):
        m1 = _encode_windows(img[..., None], POS1, W1, W1n)
        return _encode_windows(m1, POS2, W2, W2n)

    def top_match(c2, label=None):
        H, okh = center_norm(c2.reshape(-1))
        lab = np.zeros(10)
        if label is not None:
            lab[label] = LAM
        z, _ = center_norm(np.concatenate([H, lab]))
        return int(np.argmax(Wtn @ z))

    def memory_code(u):
        return np.maximum(Wtn[u, :CODE_DIM], 0.0).reshape(G2, G2, K2)

    def norm_flat(c2):
        f = c2.reshape(-1)
        return (f / (np.linalg.norm(f) + 1e-9)).reshape(c2.shape)

    report_rows = []
    for arm in ["self", "consult"]:
        fig, axes = plt.subplots(len(K_SCHEDULE), 10,
                                 figsize=(10.5, 1.15 * len(K_SCHEDULE) + 0.6))
        finals, peaks = [], np.zeros((len(K_SCHEDULE), 10))
        for j in range(10):
            c2 = memory_code(top_match(np.zeros((G2, G2, K2)), label=j))
            for bi, k in enumerate(K_SCHEDULE):
                if bi > 0:
                    c2 = encode_image(img)
                    if arm == "consult":
                        c_mem = memory_code(top_match(c2, label=j))
                        c2 = (1 - BLEND) * norm_flat(c2) + BLEND * norm_flat(c_mem)
                img = render_from_code(c2, k)
                peaks[bi, j] = peakiness(_topk_map(c2, None))
                ax = axes[bi, j]
                ax.imshow(img, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if j == 0:
                    ax.set_ylabel("dense" if k is None else f"k={k}",
                                  fontsize=8, rotation=0, ha="right", va="center")
            finals.append(img)

        self_cls = sum(int(owner[top_match(encode_image(finals[j]))] == j)
                       for j in range(10))
        report_rows.append((arm, self_cls, peaks.mean(axis=1)))
        fig.suptitle(f"Bouncing generation, {arm} arm — rows are bounces "
                     "(annealing k)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"bounce_{arm}.png", dpi=110)
        plt.close(fig)
        print(f"ARM {arm}: self-classification {self_cls}/10, "
              f"peakiness per bounce {np.round(peaks.mean(axis=1), 3)}",
              flush=True)

    lines = [
        "# Bouncing generation (iterative resolution), base weights",
        "",
        "One-shot baseline = the base rig's standard hardened generation.",
        "",
        "| arm | self-classification | peakiness per bounce |",
        "|---|---|---|",
    ]
    for arm, self_cls, pk in report_rows:
        lines.append(f"| {arm} | {self_cls}/10 | "
                     + " -> ".join(f"{p:.3f}" for p in pk) + " |")
    lines += ["", "Figures: bounce_self.png, bounce_consult.png "
              "(rows = bounces, k = dense,32,8,2,1)."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
