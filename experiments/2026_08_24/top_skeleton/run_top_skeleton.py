"""Twin-top experiment: should the archive learn from skeletons like L2/L3?

One lower stack (L1-L3, standard skeleton learning) trained once on the
stride-1 MNIST flagship geometry; two top dictionaries learn side by side
from the identical code stream — dense targets (control, current standard)
vs per-position-top-1 skeleton targets. Recognition queries stay dense for
both arms. Predictions in README.md before running.

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/top_skeleton/run_top_skeleton.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")   # stride-1 flagship geometry

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    Dict,
    POS1,
    POS2,
    POS3,
    XP_NAME,
    _center_norm_rows,
    _skeleton,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    CODE3_DIM,
    EPOCHS,
    ETA1,
    ETA2,
    ETA3,
    ETATOP,
    G1,
    G2,
    G3,
    K1,
    K2,
    K3,
    KTOP,
    LAM,
    TRAIN_N,
    W1_WIN,
    W2_WIN,
    W3_WIN,
    W2_STR,
    W3_STR,
    expand,
    load_data,
    render_pixels,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "top_skeleton" / "results"
EVAL_B = 250


def code_skeleton(m3):
    """(N, G3, G3, K3) -> per-position top-1 over channels, flat (N, CODE3_DIM)."""
    return _skeleton(m3.reshape(len(m3), G3 * G3, K3))


def top_view(H_flat, lb):
    """[center_norm(code) ; label] -> center_norm again; returns (Z, ok)."""
    Hh, okh = _center_norm_rows(H_flat)
    Z = xp.concatenate([Hh, lb], axis=1)
    Zh, okz = _center_norm_rows(Z)
    return Zh, okz & okh


def learn_top(top, Z, ok):
    if top.n_boot < top.k:
        top.bootstrap(Z[ok])
    if top.n_boot >= top.k:
        C = Z @ top.W.T
        winners = xp.argmax(C, axis=1)
        cvals = xp.max(C, axis=1) * ok
        keep = cvals > 0
        top.update(Z[keep], winners[keep], cvals[keep])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    d1 = Dict(K1, W1_WIN * W1_WIN, ETA1)
    d2 = Dict(K2, W2_WIN * W2_WIN * K1, ETA2)
    d3 = Dict(K3, W3_WIN * W3_WIN * K2, ETA3)
    top_dense = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    top_skel = Dict(KTOP, CODE3_DIM + 10, ETATOP)

    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m1 = level_pass(xb[..., None], d1, POS1, W1_WIN, 1, G1, False, True)
            m2 = level_pass(m1, d2, POS2, W2_WIN, K1, G2, True, True)
            m3 = level_pass(m2, d3, POS3, W3_WIN, K2, G3, True, True)

            Zd, okd = top_view(m3.reshape(len(xb), -1), lb)
            learn_top(top_dense, Zd, okd)

            Zs, oks = top_view(code_skeleton(m3), lb)
            learn_top(top_skel, Zs, oks)
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    # ---- Test-set encode (GPU, learning off) + hard readouts --------------
    Xte_x = xp.asarray(Xte, dtype=xp.float32)
    zero_lab = xp.zeros((EVAL_B, 10), dtype=DTYPE)
    correct = {"dense": 0, "skel": 0, "skel_skq": 0}
    n_eval = 0
    owner = {}
    for name, top in [("dense", top_dense), ("skel", top_skel)]:
        owner[name] = xp.asnumpy(xp.argmax(top.W[:, CODE3_DIM:], axis=1)) \
            if XP_NAME == "cupy" else np.argmax(top.W[:, CODE3_DIM:], axis=1)

    for s in range(0, len(Xte_x), EVAL_B):
        xb = Xte_x[s:s + EVAL_B]
        yb = yte[s:s + len(xb)]
        m1 = level_pass(xb[..., None], d1, POS1, W1_WIN, 1, G1, False, False)
        m2 = level_pass(m1, d2, POS2, W2_WIN, K1, G2, True, False)
        m3 = level_pass(m2, d3, POS3, W3_WIN, K2, G3, True, False)
        lb0 = zero_lab[:len(xb)]

        Zd, _ = top_view(m3.reshape(len(xb), -1), lb0)      # dense query
        Zsq, _ = top_view(code_skeleton(m3), lb0)           # skeleton query
        for name, top, Q in [("dense", top_dense, Zd),
                             ("skel", top_skel, Zd),
                             ("skel_skq", top_skel, Zsq)]:
            pick = xp.argmax(Q @ top.W.T, axis=1)
            pick = xp.asnumpy(pick) if XP_NAME == "cupy" else pick
            own = owner["dense" if name == "dense" else "skel"]
            correct[name] += int((own[pick] == yb).sum())
        n_eval += len(xb)

    hard = {k: v / n_eval for k, v in correct.items()}

    # ---- Memory statistics -------------------------------------------------
    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    W = {"dense": to_np(top_dense.W), "skel": to_np(top_skel.W)}
    wins = {"dense": to_np(top_dense.win_counts), "skel": to_np(top_skel.win_counts)}
    np.savez(OUTPUT_DIR / "weights.npz",
             W1=to_np(d1.W), W2=to_np(d2.W), W3=to_np(d3.W),
             Wtop_dense=W["dense"], Wtop_skel=W["skel"],
             wins_dense=wins["dense"], wins_skel=wins["skel"])

    stats = {}
    for name in ("dense", "skel"):
        rows = W[name]
        C = rows @ rows.T
        off = C[~np.eye(len(C), dtype=bool)]
        code = rows[:, :CODE3_DIM]
        code = code - code.mean(axis=1, keepdims=True)
        code = code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-9)
        Cc = code @ code.T
        offc = Cc[~np.eye(len(Cc), dtype=bool)]
        within = []
        own = owner[name]
        for j in range(10):
            idx = np.where(own == j)[0]
            if len(idx) > 1:
                sub = Cc[np.ix_(idx, idx)]
                within.append(sub[~np.eye(len(idx), dtype=bool)].mean())
        pk = []
        for t in range(len(rows)):
            m = np.maximum(rows[t, :CODE3_DIM], 0.0).reshape(G3 * G3, K3)
            ssum = m.sum(axis=1)
            keep = ssum > 0
            pk.extend((m.max(axis=1)[keep] / ssum[keep]).tolist())
        lab = np.zeros(10)
        consistent = 0
        for j in range(10):
            lab[:] = 0.0
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
            consistent += int(own[int(np.argmax(rows @ z_hat))] == j)
        stats[name] = dict(mean_cos=float(off.mean()), mean_abs=float(np.abs(off).mean()),
                           code_cos=float(offc.mean()), within=float(np.mean(within)),
                           peak=float(np.mean(pk)), consistent=consistent)

    # ---- Renders -----------------------------------------------------------
    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    b1, b2, b3 = Bank(to_np(d1.W)), Bank(to_np(d2.W)), Bank(to_np(d3.W))

    def _harden_map(m, k=1):
        o = np.zeros_like(m)
        for a in range(m.shape[0]):
            for b_ in range(m.shape[1]):
                seg = np.maximum(m[a, b_], 0.0)
                if seg.max() > 0:
                    idx = np.argsort(seg)[::-1][:k]
                    o[a, b_, idx] = seg[idx]
        return o

    def render_memory(row):
        c3 = np.maximum(row[:CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = _harden_map(expand(_harden_map(c3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
        m1 = _harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
        return render_pixels(m1, b1)

    for name in ("dense", "skel"):
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
        for j, ax in enumerate(axes.flat):
            lab = np.zeros(10)
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
            ax.imshow(render_memory(W[name][int(np.argmax(W[name] @ z_hat))]), cmap="gray")
            ax.set_title(str(j), fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Label-only generation, hardened read — {name} top")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"generation_{name}.png", dpi=110)
        plt.close(fig)

        order = np.argsort(wins[name])[::-1]
        tiers = [("top", order[:8]), ("median", order[len(order) // 2 - 4:len(order) // 2 + 4]),
                 ("low", order[-8:])]
        fig, axes = plt.subplots(3, 8, figsize=(13, 5.6))
        for r, (tier, idxs) in enumerate(tiers):
            for c, i in enumerate(idxs):
                axes[r, c].imshow(render_memory(W[name][i]), cmap="gray")
                axes[r, c].set_title(f"{tier} w{wins[name][i]}", fontsize=7)
                axes[r, c].axis("off")
        fig.suptitle(f"Memory gallery by rehearsal — {name} top "
                     "(win counts are batches-won, a proxy)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"gallery_{name}.png", dpi=110)
        plt.close(fig)

    # ---- Report ------------------------------------------------------------
    lines = [
        "# Twin-top: dense vs skeleton archive learning "
        f"({XP_NAME}, stride-1 flagship, train {train_s:.1f}s)",
        "",
        "| arm | hard (dense query) | consistent | mean cos | mean abs cos | code cos | within-class | top peak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("dense", "skel"):
        s_ = stats[name]
        lines.append(
            f"| {name} | {hard[name if name == 'dense' else 'skel']:.4f} | "
            f"{s_['consistent']}/10 | {s_['mean_cos']:+.3f} | {s_['mean_abs']:.3f} | "
            f"{s_['code_cos']:+.3f} | {s_['within']:+.3f} | {s_['peak']:.3f} |")
    lines += [
        "",
        f"Skeleton arm, skeleton query (secondary): hard {hard['skel_skq']:.4f}",
        "",
        "Reference (saved s1 flagship, dense top): mean cos +0.115, hard .849 "
        "(f64 CPU eval; this run's dense arm is the like-for-like f32 control).",
        "Figures: generation_{dense,skel}.png, gallery_{dense,skel}.png",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
