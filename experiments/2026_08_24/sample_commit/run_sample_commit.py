"""F4: sample-verify-commit — ambiguity resolution by sampled, verified,
committed attractors at L4.

Ambiguous instances (bottom-quartile L4 margin): sample one of the top-3
candidate memories (temperature softmax), verify by regeneration (its
cached rendered digit must explain the input at least as well as the
default winner's), and if verified THAT memory learns the instance.
Arms: {dense, top1} x {base, unsup, label}. Predictions in README (F4).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/sample_commit/run_sample_commit.py
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
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    K2,
    K3,
    KTOP,
    LAM,
    SEED,
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "sample_commit" / "results"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
N_EPOCHS = 3
TOPC = 3
TEMP = 0.03
AMB_PCT = 25
ARMS = [(rig, mode) for rig in ["dense", "top1"]
        for mode in ["base", "unsup", "label"]]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


class Bank:
    def __init__(self, Wb):
        self.W = Wb
        self.k = Wb.shape[0]


def _harden_map(m, k=1):
    o = np.zeros_like(m)
    for a in range(m.shape[0]):
        for b_ in range(m.shape[1]):
            seg = np.maximum(m[a, b_], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b_, idx] = seg[idx]
    return o


def render_all(d1, d2, d3, top):
    W1n, W2n, W3n, Wtn = map(to_np, (d1.W, d2.W, d3.W, top.W))
    b2, b3 = Bank(W2n), Bank(W3n)
    R = np.zeros((KTOP, 28 * 28))
    for i in range(KTOP):
        c3 = np.maximum(Wtn[i, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
        m1 = _harden_map(expand(m2, b2, W2, S2, (G1, G1, K1)))
        R[i] = render8(m1, W1n).ravel()
    R -= R.mean(axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
    return xp.asarray(R, dtype=DTYPE)


def run_arm(rig, mode, Xtr, ytr, Xte, yte, labels_x, Qtr_x):
    rng = np.random.default_rng(SEED + 7)
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    ytr_x = xp.asarray(ytr[:TRAIN_N])

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

    sampled_n = accepted_n = 0
    renders = None
    t0 = time.time()
    for ep in range(N_EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            yb = ytr_x[s:s + B]
            Qb = Qtr_x[s:s + B]
            H = stack(xb, True)
            Zd, okd = top_view(H, lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
                continue
            Ct = Zd @ top.W.T
            winners = xp.argmax(Ct, axis=1)
            wvals = xp.max(Ct, axis=1)

            use_sampler = mode != "base" and renders is not None
            if not use_sampler:
                cvals = wvals * okd
                keep = cvals > 0
                top.update(Zd[keep], winners[keep], cvals[keep])
                continue

            srt = xp.sort(Ct, axis=1)
            margins = srt[:, -1] - srt[:, -2]
            m_np = to_np(margins[okd])
            thr = float(np.percentile(m_np, AMB_PCT)) if len(m_np) else 0.0
            amb = okd & (margins <= thr) & (wvals > 0)

            chosen = winners.copy()
            if bool(amb.any()):
                idx_amb = xp.where(amb)[0]
                cand = xp.argsort(Ct[idx_amb], axis=1)[:, -TOPC:]      # (M,3)
                cv = xp.take_along_axis(Ct[idx_amb], cand, axis=1)
                g = xp.asarray(rng.gumbel(size=cv.shape).astype(np.float32))
                pick = xp.argmax(cv / TEMP + g, axis=1)
                samp = cand[xp.arange(len(idx_amb)), pick]

                fit_s = xp.sum(renders[samp] * Qb[idx_amb], axis=1)
                fit_w = xp.sum(renders[winners[idx_amb]] * Qb[idx_amb], axis=1)
                acc = fit_s >= fit_w
                if mode == "label":
                    owner = xp.argmax(top.W[:, CODE3_DIM:], axis=1)
                    acc = acc & (owner[samp] == yb[idx_amb])
                chosen[idx_amb] = xp.where(acc, samp, winners[idx_amb])
                sampled_n += int(len(idx_amb))
                accepted_n += int(to_np(acc.sum()))

            cvals = xp.take_along_axis(Ct, chosen[:, None], axis=1)[:, 0] * okd
            keep = cvals > 0
            top.update(Zd[keep], chosen[keep], cvals[keep])
        if mode != "base" and ep < N_EPOCHS - 1:
            renders = render_all(d1, d2, d3, top)
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    # ---- eval --------------------------------------------------------------
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
    C = Z @ Wtn.T
    srt = np.sort(C, axis=1)
    margins = srt[:, -1] - srt[:, -2]
    pred = owner[C.argmax(axis=1)]
    hard = float((pred == yte).mean())
    m20 = float(np.percentile(margins, 20))

    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    pairs = sorted(((conf[a, b_], a, b_) for a in range(10) for b_ in range(10)
                    if a != b_), reverse=True)
    top_conf = ", ".join(f"{a}->{b_}:{n}" for n, a, b_ in pairs[:3])

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    acc_rate = accepted_n / max(sampled_n, 1)
    return dict(rig=rig, mode=mode, train_s=train_s, hard=hard, m20=m20,
                consistent=consistent, top_conf=top_conf,
                sampled=sampled_n, acc_rate=acc_rate)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)
    Q = Xtr[:TRAIN_N].reshape(TRAIN_N, -1).astype(np.float32)
    Q -= Q.mean(axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    Qtr_x = xp.asarray(Q, dtype=DTYPE)

    rows = []
    for rig, mode in ARMS:
        r = run_arm(rig, mode, Xtr, ytr, Xte, yte, labels_x, Qtr_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {rig}/{mode}: hard={r['hard']:.4f} margin20={r['m20']:.4f} "
              f"consistent={r['consistent']}/10 sampled={r['sampled']} "
              f"accept={r['acc_rate']:.2f} conf[{r['top_conf']}] "
              f"train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# F4: sample-verify-commit (8x8; 3 epochs; sampler from epoch 2)",
        "",
        "| rig | mode | hard | margin p20 | consistent | accept rate | top confusions |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['rig']} | {r['mode']} | {r['hard']:.4f} | "
                     f"{r['m20']:.4f} | {r['consistent']}/10 | "
                     f"{r['acc_rate']:.2f} | {r['top_conf']} |")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
