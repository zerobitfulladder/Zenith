"""F6: corrective memory updates — LVQ-style "no, it's not a 7" at L4.

On a wrong zero-label answer: attract the best true-class memory toward
the instance (attract arm), and additionally repel the wrongly-winning
memory away from it at half rate (lvq arm). Dictionaries untouched.
Arms: {dense, top1} x {base, attract, lvq}. Predictions in README (F6).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/lvq/run_lvq.py
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

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    Dict,
    THETA_CAP,
    XP_NAME,
    _scatter_add,
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "lvq" / "results"
K1 = 512
CODE3_DIM = G3 * G3 * K3
EVAL_B = 125
N_EPOCHS = 3
ETA_NEG = ETATOP / 2
ARMS = [(rig, mode) for rig in ["dense", "top1"]
        for mode in ["base", "attract", "lvq"]]


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def repel(top, T_hat, winners, cvals):
    """Geodesic step AWAY from the targets (Dict.update with negated sin)."""
    if len(T_hat) == 0:
        return
    S = xp.zeros((top.k, top.dim), dtype=DTYPE)
    Csum = xp.zeros(top.k, dtype=DTYPE)
    _scatter_add(S, winners, T_hat)
    _scatter_add(Csum, winners, cvals)
    hit = Csum > 0
    if not bool(hit.any()):
        return
    mu = S[hit]
    mu = mu / (xp.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
    w = top.W[hit]
    cw = xp.sum(w * mu, axis=1, keepdims=True)
    tau = mu - cw * w
    tn = xp.linalg.norm(tau, axis=1, keepdims=True)
    theta = xp.minimum(ETA_NEG * Csum[hit], THETA_CAP)[:, None]
    w_new = xp.where(tn > 1e-9,
                     w * xp.cos(theta) - (tau / xp.maximum(tn, 1e-9)) * xp.sin(theta),
                     w)
    w_new = w_new - w_new.mean(axis=1, keepdims=True)
    w_new = w_new / (xp.linalg.norm(w_new, axis=1, keepdims=True) + 1e-9)
    top.W[hit] = w_new


def run_arm(rig, mode, Xtr, ytr, Xte, yte, labels_x):
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    ytr_np = ytr[:TRAIN_N]

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

    corrections = 0
    t0 = time.time()
    for ep in range(N_EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            yb = ytr_np[s:s + B]
            H = stack(xb, True)
            Zd, okd = top_view(H, lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
                continue

            # standard attract update (unchanged in all arms)
            Ct = Zd @ top.W.T
            winners = xp.argmax(Ct, axis=1)
            cvals = xp.max(Ct, axis=1) * okd
            keep = cvals > 0
            top.update(Zd[keep], winners[keep], cvals[keep])

            if mode == "base" or ep < 1:
                continue

            # corrective phase on zero-label errors
            Z0, ok0 = top_view(H, xp.zeros_like(lb))
            owner = xp.argmax(top.W[:, CODE3_DIM:], axis=1)
            C0 = Z0 @ top.W.T
            w0 = xp.argmax(C0, axis=1)
            yb_x = xp.asarray(yb)
            wrong = ok0 & (owner[w0] != yb_x)
            if not bool(wrong.any()):
                continue
            idx = xp.where(wrong)[0]
            corrections += int(len(idx))

            # attract: best TRUE-class memory steps toward the instance
            same = owner[None, :] == yb_x[idx][:, None]
            Cm = xp.where(same, C0[idx], -xp.inf)
            best_true = xp.argmax(Cm, axis=1)
            cv_t = xp.maximum(xp.take_along_axis(
                C0[idx], best_true[:, None], axis=1)[:, 0], 0.05)
            top.update(Zd[idx], best_true, cv_t)

            if mode == "lvq":
                cv_w = xp.take_along_axis(C0[idx], w0[idx][:, None], axis=1)[:, 0]
                pos = cv_w > 0
                repel(top, Z0[idx][pos], w0[idx][pos], cv_w[pos])
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

    if mode == "lvq":
        class Bank:
            def __init__(self, Wb):
                self.W = Wb
                self.k = Wb.shape[0]

        b2, b3 = Bank(to_np(d2.W)), Bank(to_np(d3.W))

        def _harden_map(m, k=1):
            o = np.zeros_like(m)
            for a in range(m.shape[0]):
                for b_ in range(m.shape[1]):
                    seg = np.maximum(m[a, b_], 0.0)
                    if seg.max() > 0:
                        i2 = np.argsort(seg)[::-1][:k]
                        o[a, b_, i2] = seg[i2]
            return o

        fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
        for j, ax in enumerate(axes.flat):
            lab = np.zeros(10)
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
            row = Wtn[int(np.argmax(Wtn @ z_hat))]
            c3 = np.maximum(row[:CODE3_DIM], 0.0).reshape(G3, G3, K3)
            m2r = _harden_map(expand(_harden_map(c3), b3, W3, S3, (G2, G2, K2)))
            m1r = _harden_map(expand(m2r, b2, W2, S2, (G1, G1, K1)))
            ax.imshow(render8(m1r, to_np(d1.W)), cmap="gray")
            ax.set_title(str(j), fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Generation — LVQ arm, {rig}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"generation_{rig}_lvq.png", dpi=110)
        plt.close(fig)
        np.savez(OUTPUT_DIR / f"weights_{rig}_lvq.npz",
                 W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W), Wtop=Wtn)

    return dict(rig=rig, mode=mode, train_s=train_s, hard=hard,
                consistent=consistent, top_conf=top_conf, corr=corrections)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    rows = []
    for rig, mode in ARMS:
        r = run_arm(rig, mode, Xtr, ytr, Xte, yte, labels_x)
        rows.append(r)
        print(f"DONE {rig}/{mode}: hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 corrections={r['corr']} "
              f"conf[{r['top_conf']}] train={r['train_s']:.0f}s", flush=True)

    lines = [
        "# F6: corrective memory updates (LVQ at L4; corrective from epoch 2)",
        "",
        "| rig | mode | hard | consistent | corrections | top confusions |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['rig']} | {r['mode']} | {r['hard']:.4f} | "
                     f"{r['consistent']}/10 | {r['corr']} | {r['top_conf']} |")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
