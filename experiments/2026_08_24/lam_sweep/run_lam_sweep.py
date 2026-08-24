"""F7: lambda sweep — label-half weight on the 8x8 dense champion.

lam in {0.25, 0.5, 0.75, 1.0}. Both directions measured: image->label
(hard readout, zero-label query) and label->image (label-only query:
consistency + retrieval margin). Predictions in README (F7).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/lam_sweep/run_lam_sweep.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import numpy as np

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    Dict,
    XP_NAME,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    EPOCHS,
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    K2,
    K3,
    KTOP,
    TRAIN_N,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    G1,
    G2,
    G3,
    POS1,
    POS2,
    POS3,
    W1,
    W2,
    W3,
    top_view,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "lam_sweep" / "results" / "sweep"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
LAMS = [0.25, 0.5, 0.75, 1.0]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def run_lam(lam, Xtr, ytr, Xte, yte):
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr[:TRAIN_N]] = lam
    labels_x = xp.asarray(labels_tr)

    def stack(xb, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
        m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, learning)
        return m3.reshape(len(xb), -1)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            Zd, okd = top_view(stack(xb, True), lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
                continue
            Ct = Zd @ top.W.T
            winners = xp.argmax(Ct, axis=1)
            cvals = xp.max(Ct, axis=1) * okd
            keep = cvals > 0
            top.update(Zd[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    C3s = []
    for s in range(0, len(Xte), EVAL_B):
        xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
        C3s.append(to_np(stack(xb, False)))
    C3te = np.concatenate(C3s)
    if XP_NAME == "cupy":
        xp.get_default_memory_pool().free_all_blocks()

    Wtn = to_np(top.W)
    owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)
    H = C3te - C3te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    hard = float((owner[(Z @ Wtn.T).argmax(axis=1)] == yte).mean())

    consistent = 0
    ret_margins = []
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = lam
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        c = Wtn @ z_hat
        w0 = int(np.argmax(c))
        consistent += int(owner[w0] == j)
        other = c[owner != owner[w0]]
        ret_margins.append(float(c[w0] - other.max()))

    return dict(lam=lam, train_s=train_s, hard=hard, consistent=consistent,
                ret_margin=float(np.mean(ret_margins)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    rows = []
    for lam in LAMS:
        r = run_lam(lam, Xtr, ytr, Xte, yte)
        rows.append(r)
        print(f"DONE lam={lam}: hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 "
              f"retrieval_margin={r['ret_margin']:.3f} train={r['train_s']:.0f}s",
              flush=True)

    lines = [
        "# F7: lambda sweep (8x8 dense champion, 2 epochs)",
        "",
        "| lam | label energy share | hard (image->label) | consistent | retrieval margin (label->image) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        share = r["lam"] ** 2 / (1 + r["lam"] ** 2)
        lines.append(f"| {r['lam']} | {share:.0%} | {r['hard']:.4f} | "
                     f"{r['consistent']}/10 | {r['ret_margin']:.3f} |")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
