"""F3: inference-time searchlight — no training.

Render all 200 memories to pixels once; for LOW-MARGIN queries (ambiguous
at L4), re-score the top-3 candidate memories by pixel agreement between
the candidate's rendered digit and the actual input. Final score =
code_corr + w * pixel_corr (declared w=0.5; sweep reported as diagnostic).
Rigs: saved all-dense and all-top1 (allsparse/results weights).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/feedback_program/run_searchlight.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import numpy as np

from run_gpu_minibatch import Dict, DTYPE, XP_NAME, level_pass, xp  # noqa: E402
from run_4layer_topk import (  # noqa: E402
    ETA1,
    ETA2,
    ETA3,
    K2,
    K3,
    expand,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    G3,
    POS1,
    POS2,
    POS3,
    S2,
    S3,
    W1,
    W2,
    W3,
    render8,
    topk_mask,
)

DIR = ROOT / "experiments" / "2026_08_24" / "allsparse" / "results"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "feedback_program" / "results" / "searchlight"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
MARGIN_PCT = 20          # bottom 20% margins get the searchlight
TOPC = 3                 # candidates re-scored
W_DECLARED = 0.5


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def bank_dict(W, eta):
    d = Dict(W.shape[0], W.shape[1], eta)
    d.W = xp.asarray(W, dtype=DTYPE)
    d.n_boot = d.k
    return d


def run_rig(tag, Xte, yte):
    z = np.load(DIR / f"weights_K64_100_{tag}.npz")
    W1n, W2n, W3n, Wtn = z["W1"], z["W2"], z["W3"], z["Wtop"]
    d1, d2, d3 = bank_dict(W1n, ETA1), bank_dict(W2n, ETA2), bank_dict(W3n, ETA3)
    owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)

    # ---- render all 200 memories once -------------------------------------
    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b2, b3 = Bank(W2n), Bank(W3n)

    def _harden_map(m, k=1):
        o = np.zeros_like(m)
        for a in range(m.shape[0]):
            for b_ in range(m.shape[1]):
                seg = np.maximum(m[a, b_], 0.0)
                if seg.max() > 0:
                    idx = np.argsort(seg)[::-1][:k]
                    o[a, b_, idx] = seg[idx]
        return o

    renders = np.zeros((len(Wtn), 28 * 28))
    for i in range(len(Wtn)):
        c3 = np.maximum(Wtn[i, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
        m1 = _harden_map(expand(m2, b2, W2, S2, (G1, G1, K1)))
        renders[i] = render8(m1, W1n).ravel()
    renders -= renders.mean(axis=1, keepdims=True)
    renders /= np.linalg.norm(renders, axis=1, keepdims=True) + 1e-9

    # ---- encode test set (mode matches the rig's training) ----------------
    def stack(xb):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, False)
        if tag == "top1":
            m1 = topk_mask(m1, 1)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, False)
        if tag == "top1":
            m2 = topk_mask(m2, 1)
        m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, False)
        if tag == "top1":
            m3 = topk_mask(m3, 1)
        return m3.reshape(len(xb), -1)

    C3s = []
    for s in range(0, len(Xte), EVAL_B):
        xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
        C3s.append(to_np(stack(xb)))
    C3te = np.concatenate(C3s)
    if XP_NAME == "cupy":
        xp.get_default_memory_pool().free_all_blocks()

    H = C3te - C3te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    C = Z @ Wtn.T

    order = np.argsort(C, axis=1)
    top1_i = order[:, -1]
    margin = C[np.arange(len(C)), top1_i] - C[np.arange(len(C)), order[:, -2]]
    base_pred = owner[top1_i]
    base_acc = float((base_pred == yte).mean())

    thr = np.percentile(margin, MARGIN_PCT)
    low = margin <= thr

    # pixel agreement of the query with each of its top-3 candidates
    Q = Xte.reshape(len(Xte), -1).astype(np.float64)
    Q -= Q.mean(axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

    results = {}
    for w in [0.3, W_DECLARED, 1.0, "pixel-only"]:
        pred = base_pred.copy()
        for i in np.where(low)[0]:
            cands = order[i, -TOPC:][::-1]
            px = renders[cands] @ Q[i]
            if w == "pixel-only":
                score = px
            else:
                score = C[i, cands] + w * px
            pred[i] = owner[cands[int(np.argmax(score))]]
        acc = float((pred == yte).mean())
        acc_low = float((pred[low] == yte[low]).mean())
        results[str(w)] = (acc, acc_low)

    base_low = float((base_pred[low] == yte[low]).mean())
    return dict(tag=tag, base=base_acc, base_low=base_low,
                n_low=int(low.sum()), results=results)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, Xte, yte = load_data()

    lines = ["# F3: inference searchlight (no training; bottom-20% margins, top-3 candidates)", ""]
    for tag in ["dense", "top1"]:
        r = run_rig(tag, Xte, yte)
        lines += [
            f"## rig: all-{tag}",
            f"- baseline hard {r['base']:.4f}; low-margin subset "
            f"({r['n_low']} queries) baseline {r['base_low']:.4f}",
        ]
        for w, (acc, acc_low) in r["results"].items():
            lines.append(f"- w={w}: overall {acc:.4f}, low-margin {acc_low:.4f}")
        lines.append("")
        print("\n".join(lines[-6:]), flush=True)
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("report written")


if __name__ == "__main__":
    main()
