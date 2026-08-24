"""F5: error-driven exposure — boosting curriculum, no architecture feedback.

Epoch 1 balanced; epochs 2-3 the network answers each batch (hard readout,
zero label, before the update) and per-class error EMAs steer batch
composition: wrong -> seen more, right -> seen less, floor keeps all
classes present, total exposure equal. Arms: {dense, top1} x {base, adapt}.
Predictions in README (F5).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/exposure/run_exposure.py
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
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    K2,
    K3,
    KTOP,
    LAM,
    PROBE_N,
    SEED,
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "exposure" / "results"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
N_EPOCHS = 3
RHO = 0.1          # error EMA
FLOOR = 0.3        # share of the batch always spread evenly
ARMS = [(rig, mode) for rig in ["dense", "top1"] for mode in ["base", "adapt"]]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def run_arm(rig, mode, Xtr, ytr, Xte, yte):
    rng = np.random.default_rng(SEED + 11)
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    idx_by_class = [np.where(ytr[:TRAIN_N] == c)[0] for c in range(10)]
    err = np.full(10, 0.5)

    def stack(xb, learning):
        m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
        if rig == "top1":
            m1 = topk_mask(m1, 1)
        m2 = level_pass(m1, d2, POS2, W2, K1, G2, True, learning)
        if rig == "top1":
            m2 = topk_mask(m2, 1)
        m3 = level_pass(m2, d3, POS3, W3, K2, G3, True, learning)
        if rig == "top1":
            m3 = topk_mask(m3, 1)
        return m3.reshape(len(xb), -1)

    n_batches = TRAIN_N // B
    t0 = time.time()
    for ep in range(N_EPOCHS):
        adaptive = mode == "adapt" and ep >= 1 and top.n_boot >= top.k
        perm = rng.permutation(TRAIN_N)
        for bi in range(n_batches):
            if adaptive:
                p = FLOOR / 10 + (1 - FLOOR) * err / err.sum()
                counts = rng.multinomial(B, p / p.sum())
                sel = np.concatenate([
                    rng.choice(idx_by_class[c], n, replace=True)
                    for c, n in enumerate(counts) if n > 0])
                rng.shuffle(sel)
            else:
                sel = perm[bi * B:(bi + 1) * B]
            xb = Xtr_x[xp.asarray(sel)]
            yb = ytr[sel]
            lb_np = np.zeros((len(sel), 10), dtype=np.float32)
            lb_np[np.arange(len(sel)), yb] = LAM
            lb = xp.asarray(lb_np)

            H = stack(xb, True)
            Zd, okd = top_view(H, lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
                continue

            if mode == "adapt":
                Z0, ok0 = top_view(H, xp.zeros_like(lb))
                owner = xp.argmax(top.W[:, CODE3_DIM:], axis=1)
                pred = to_np(owner[xp.argmax(Z0 @ top.W.T, axis=1)])
                wrong = pred != yb
                for c in range(10):
                    m = yb == c
                    if m.any():
                        err[c] = (1 - RHO) * err[c] + RHO * float(wrong[m].mean())

            Ct = Zd @ top.W.T
            winners = xp.argmax(Ct, axis=1)
            cvals = xp.max(Ct, axis=1) * okd
            keep = cvals > 0
            top.update(Zd[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    def encode(X):
        Cs = []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            Cs.append(to_np(stack(xb, False)))
        return np.concatenate(Cs)

    C3tr = encode(Xtr[:PROBE_N])
    C3te = encode(Xte)
    if XP_NAME == "cupy":
        xp.get_default_memory_pool().free_all_blocks()

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
    pred = owner[(Z @ Wtn.T).argmax(axis=1)]
    hard = float((pred == yte).mean())

    conf = np.zeros((10, 10), dtype=int)
    for t, p_ in zip(yte, pred):
        conf[t, p_] += 1
    pairs = sorted(((conf[a, b_], a, b_) for a in range(10) for b_ in range(10)
                    if a != b_), reverse=True)
    top_conf = ", ".join(f"{a}->{b_}:{n}" for n, a, b_ in pairs[:3])

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    final_w = ", ".join(f"{c}:{e:.2f}" for c, e in enumerate(err)) \
        if mode == "adapt" else "-"
    return dict(rig=rig, mode=mode, train_s=train_s, probe=probe, hard=hard,
                consistent=consistent, top_conf=top_conf, final_w=final_w)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    rows = []
    for rig, mode in ARMS:
        r = run_arm(rig, mode, Xtr, ytr, Xte, yte)
        rows.append(r)
        print(f"DONE {rig}/{mode}: probeL3={r['probe']:.4f} hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 conf[{r['top_conf']}] "
              f"errEMA[{r['final_w']}] train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# F5: error-driven exposure (3 epochs; adaptive from epoch 2)",
        "",
        "| rig | mode | probe L3 | hard | consistent | top confusions |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['rig']} | {r['mode']} | {r['probe']:.4f} | "
                     f"{r['hard']:.4f} | {r['consistent']}/10 | {r['top_conf']} |")
    lines += [""] + [f"final err EMA ({r['rig']}): {r['final_w']}"
                     for r in rows if r["mode"] == "adapt"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
