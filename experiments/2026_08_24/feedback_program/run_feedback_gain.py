"""F2: gain-modulation retry on the spatial rig — class-prototype
multiplicative gain on the L4 message, training only.

Per-class EMA prototype P[c] of the L3 code; during training the code fed
to L4 becomes H * (1 + beta * P[c]/max(P[c])). No gain at query. Arms:
{dense, top1-all} x beta=0.5. Baselines: F1's gamma=0 arms (.9060/.7268).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/feedback_program/run_feedback_gain.py
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
from sklearn.linear_model import LogisticRegression

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
    LAM,
    PROBE_N,
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
    topk_mask,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "feedback_program" / "results" / "gain"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
BETA = 0.5
RHO = 0.05
ARMS = ["dense", "top1"]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def run_arm(mode, Xtr, ytr, Xte, yte, labels_x, ytr_x):
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    proto = xp.zeros((10, CODE3_DIM), dtype=DTYPE)

    def stack(xb, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        if mode == "top1":
            m1 = topk_mask(m1, 1)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
        if mode == "top1":
            m2 = topk_mask(m2, 1)
        m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, learning)
        if mode == "top1":
            m3 = topk_mask(m3, 1)
        return m3.reshape(len(xb), -1)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            yb = ytr_x[s:s + B]
            H = stack(xb, True)

            # update per-class prototypes (EMA over batch class means)
            for c in range(10):
                m = yb == c
                if bool(m.any()):
                    proto[c] = (1 - RHO) * proto[c] + RHO * H[m].mean(axis=0)

            # class-prototype multiplicative gain on the L4 message
            pmax = xp.maximum(proto.max(axis=1, keepdims=True), 1e-9)
            gain = 1.0 + BETA * (proto / pmax)[yb]
            Zd, okd = top_view(H * gain, lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
            if top.n_boot >= top.k:
                Ct = Zd @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cvals = xp.max(Ct, axis=1) * okd
                keep = cvals > 0
                top.update(Zd[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    def encode(X):
        C3s = []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            C3s.append(to_np(stack(xb, False)))
        return np.concatenate(C3s)

    C3tr = encode(Xtr[:PROBE_N])
    C3te = encode(Xte)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(C3tr, ytr[:PROBE_N])
    probe = float(clf.score(C3te, yte))

    Wtn = to_np(top.W)
    owner = np.argmax(Wtn[:, CODE3_DIM:], axis=1)
    H = C3te - C3te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    hard = float((owner[(Z @ Wtn.T).argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    return dict(mode=mode, train_s=train_s, probe=probe, hard=hard,
                consistent=consistent)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)
    ytr_x = xp.asarray(ytr[:TRAIN_N])

    rows = []
    for mode in ARMS:
        r = run_arm(mode, Xtr, ytr, Xte, yte, labels_x, ytr_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {mode} beta={BETA}: probeL3={r['probe']:.4f} "
              f"hard={r['hard']:.4f} consistent={r['consistent']}/10 "
              f"train={r['train_s']:.0f}s", flush=True)

    lines = [
        f"# F2: class-prototype gain on the L4 message (beta={BETA}, train only)",
        "",
        "| speech | probe L3 | hard | consistent |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['probe']:.4f} | {r['hard']:.4f} | "
                     f"{r['consistent']}/10 |")
    lines += ["", "No-gain baselines (F1 gamma=0 arms): dense .9060 / top1 .7268."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
