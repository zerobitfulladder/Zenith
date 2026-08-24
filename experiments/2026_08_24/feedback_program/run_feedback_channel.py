"""F1: label as a training-time channel at L2/L3 windows (feedback as content).

Each L2/L3 window gets gamma*onehot(label) appended during training
(matching, skeleton learning, and bootstrap all see it); at query the
label dims are zeros. Arms: {dense, top1-all} x {gamma 0, 0.35}.
Predictions in README (Feedback program, F1).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/feedback_program/run_feedback_channel.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    Dict,
    XP_NAME,
    _center_norm_rows,
    _skeleton,
    _windows,
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
    top_view,
    topk_mask,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "feedback_program" / "results" / "channel"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
ARMS = [("dense", 0.0), ("dense", 0.35), ("top1", 0.0), ("top1", 0.35)]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def level_pass_lab(prev_maps, dic, positions, win, kprev, g_out, learning, lab10):
    """level_pass with gamma*label appended to every window (lab10 may be zeros)."""
    N = len(prev_maps)
    P = len(positions)
    Vf = _windows(prev_maps, positions, win)                    # (N, P, D)
    lab_rows = xp.repeat(lab10, P, axis=0)                      # (N*P, 10)
    V = xp.concatenate([Vf.reshape(N * P, -1), lab_rows], axis=1)
    V_hat, ok = _center_norm_rows(V)

    if learning and dic.n_boot < dic.k:
        dic.bootstrap(V_hat[ok])

    C = V_hat @ dic.W.T
    code = xp.maximum(C, 0.0) * ok[:, None]

    if learning and dic.n_boot >= dic.k:
        SK = _skeleton(Vf.reshape(N * P, win * win, kprev))
        SKl = xp.concatenate([SK, lab_rows], axis=1)
        T_hat, ok_s = _center_norm_rows(SKl)
        Cs = T_hat @ dic.W.T
        winners = xp.argmax(Cs, axis=1)
        cvals = xp.max(Cs, axis=1) * ok_s
        keep = cvals > 0
        dic.update(T_hat[keep], winners[keep], cvals[keep])

    return code.reshape(N, g_out, g_out, dic.k)


def run_arm(mode, gamma, Xtr, ytr, Xte, yte, labels_x):
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1 + 10, ETA2)
    d3 = Dict(K3, W3 * W3 * K2 + 10, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    def stack(xb, lab10, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        if mode == "top1":
            m1 = topk_mask(m1, 1)
        m2 = level_pass_lab(m1, d2, POS2, W2, K1, G2, learning, lab10)
        if mode == "top1":
            m2 = topk_mask(m2, 1)
        m3 = level_pass_lab(m2, d3, POS3, W3, K2, G3, learning, lab10)
        if mode == "top1":
            m3 = topk_mask(m3, 1)
        return m3

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]                 # LAM-scaled onehot for L4
            lab10 = (gamma / LAM) * lb             # gamma-scaled onehot for L2/L3
            m3 = stack(xb, lab10, True)
            Zd, okd = top_view(m3.reshape(len(xb), -1), lb)
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
            zlab = xp.zeros((len(xb), 10), dtype=DTYPE)
            C3s.append(to_np(stack(xb, zlab, False).reshape(len(xb), -1)))
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
    C = Z @ Wtn.T
    pred = owner[C.argmax(axis=1)]
    hard = float((pred == yte).mean())

    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    pairs = []
    for a in range(10):
        for b_ in range(10):
            if a != b_:
                pairs.append((conf[a, b_], a, b_))
    pairs.sort(reverse=True)
    top_conf = ", ".join(f"{a}->{b_}:{n}" for n, a, b_ in pairs[:3])

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    np.savez(OUTPUT_DIR / f"weights_{mode}_g{gamma}.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)

    return dict(mode=mode, gamma=gamma, train_s=train_s, probe=probe,
                hard=hard, consistent=consistent, top_conf=top_conf)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for mode, gamma in ARMS:
        r = run_arm(mode, gamma, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {mode} g={gamma}: probeL3={r['probe']:.4f} hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 conf[{r['top_conf']}] "
              f"train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# F1: label channel at L2/L3 (8x8, K1=512; gamma at query = 0)",
        "",
        "| speech | gamma | probe L3 | hard | consistent | top confusions |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['gamma']} | {r['probe']:.4f} | "
                     f"{r['hard']:.4f} | {r['consistent']}/10 | {r['top_conf']} |")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
