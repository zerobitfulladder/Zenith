"""Train WITH quantization (hidden-accumulator version, no freezing trap).

Every bank keeps fine f32 internal weights (the accumulator), but all
matching — competition, messages, the top's nearest-unit lookup — uses
the quantized EXPRESSED weights (per-template symmetric rounding to b
bits + per-row renormalization, same scheme as run_precision.py).
Updates apply the standard geodesic step to the internal weights, so
sub-notch refinements accumulate instead of rounding away; the
expressed weights are re-quantized after every update.

Rig = the base exactly (L1 8x8s1/1024, L2 3x3s2/1024, top 200, dense
communication, top-1 plasticity, skeleton L2 targets). Arms: 6, 4, 3,
2 bits. Compare against run_precision.py's post-hoc arms: the gap is
what learning-under-the-constraint buys.

Run:  .venv/bin/python experiments/2026_08_26/precision/run_qat.py
"""

import os
import sys
import time
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

from run_gpu_minibatch import (  # noqa: E402
    B,
    DTYPE,
    THETA_CAP,
    XP_NAME,
    _scatter_add,
    level_pass,
    xp,
)
from run_4layer_topk import (  # noqa: E402
    EPOCHS,
    ETA1,
    ETA2,
    ETATOP,
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
    POS1,
    POS2,
    S2,
    W1,
    W2,
    render8,
    top_view,
)
from gain_feedback import center_norm  # noqa: E402
from run_base import CODE_DIM, EVAL_B, K1, K2, _harden_map  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "precision" / "results" / "qat"
BASE_WEIGHTS = ROOT / "experiments" / "2026_08_26" / "base" / "results" / "weights.npz"
ARMS = [6, 4, 3, 2]


class QDict:
    """Dict with a fine internal accumulator and quantized expressed
    weights. Matching reads .W (quantized); learning moves .Ws (f32)."""

    def __init__(self, k, dim, eta, bits, split=None):
        self.k, self.dim, self.eta, self.bits = k, dim, eta, bits
        self.split = split
        self.Ws = xp.zeros((k, dim), dtype=DTYPE)   # internal, fine
        self.W = xp.zeros((k, dim), dtype=DTYPE)    # expressed, coarse
        self.n_boot = 0
        self.win_counts = xp.zeros(k, dtype=xp.int64)
        self._noise = np.random.default_rng(SEED)

    def _express(self):
        # v2 codebook (see run_precision.py): percentile-scaled step with
        # outlier clipping; the top's code/label halves get separate scales.
        levels = 2 ** (self.bits - 1) - 1
        segs = ([(0, self.split), (self.split, self.dim)]
                if self.split else [(0, self.dim)])
        Q = xp.empty_like(self.Ws)
        for a, b in segs:
            seg = self.Ws[:, a:b]
            s = xp.percentile(xp.abs(seg), 99.5, axis=1, keepdims=True) / levels
            s = xp.maximum(s, 1e-12)
            Q[:, a:b] = xp.clip(xp.rint(seg / s), -levels, levels) * s
        self.W = Q / (xp.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)

    def bootstrap(self, V_hat):
        need = self.k - self.n_boot
        if need <= 0 or len(V_hat) == 0:
            return
        take = min(need, len(V_hat))
        w = V_hat[:take] + xp.asarray(
            ((0.05 / np.sqrt(self.dim))
             * self._noise.standard_normal((take, self.dim))).astype(np.float32))
        w = w - w.mean(axis=1, keepdims=True)
        w = w / (xp.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        self.Ws[self.n_boot:self.n_boot + take] = w
        self.win_counts[self.n_boot:self.n_boot + take] += 1
        self.n_boot += take
        self._express()

    def update(self, T_hat, winners, cvals):
        if len(T_hat) == 0:
            return
        S = xp.zeros((self.k, self.dim), dtype=DTYPE)
        Csum = xp.zeros(self.k, dtype=DTYPE)
        _scatter_add(S, winners, T_hat)
        _scatter_add(Csum, winners, cvals)
        hit = Csum > 0
        if not bool(hit.any()):
            return
        mu = S[hit]
        mu = mu / (xp.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
        w = self.Ws[hit]
        cw = xp.sum(w * mu, axis=1, keepdims=True)
        tau = mu - cw * w
        tn = xp.linalg.norm(tau, axis=1, keepdims=True)
        theta = xp.minimum(self.eta * Csum[hit], THETA_CAP)[:, None]
        w_new = xp.where(
            tn > 1e-9,
            w * xp.cos(theta) + (tau / xp.maximum(tn, 1e-9)) * xp.sin(theta),
            w,
        )
        w_new = w_new - w_new.mean(axis=1, keepdims=True)
        w_new = w_new / (xp.linalg.norm(w_new, axis=1, keepdims=True) + 1e-9)
        self.Ws[hit] = w_new
        self.win_counts += (Csum > 0).astype(xp.int64)
        self._express()


def to_np(a):
    return a.get() if XP_NAME == "cupy" else a


def stack(xb, d1, d2, learning):
    m1 = level_pass(xb[..., None], d1, POS1, W1, 1, G1, False, learning)
    return level_pass(m1, d2, POS2, W2, K1, G2, True, learning)


def run_arm(bits, Xtr, ytr, Xte, yte, labels_x):
    d1 = QDict(K1, W1 * W1, ETA1, bits)
    d2 = QDict(K2, W2 * W2 * K1, ETA2, bits)
    top = QDict(KTOP, CODE_DIM + 10, ETATOP, bits, split=CODE_DIM)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m2 = stack(xb, d1, d2, True)
            Zd, okd = top_view(m2.reshape(len(xb), -1), lb)
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

    C2s = []
    for s in range(0, len(Xte), EVAL_B):
        xb = xp.asarray(Xte[s:s + EVAL_B], dtype=DTYPE)
        C2s.append(to_np(stack(xb, d1, d2, False).reshape(-1, CODE_DIM)))
    C2te = np.concatenate(C2s)

    W1q, W2q, Wtq = to_np(d1.W), to_np(d2.W), to_np(top.W)
    np.savez(OUTPUT_DIR / f"weights_{bits}bit.npz", W1=W1q, W2=W2q, Wtop=Wtq)

    owner = np.argmax(Wtq[:, CODE_DIM:], axis=1)
    H = C2te - C2te.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10), dtype=H.dtype)], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    hard = float((owner[(Z @ Wtq.T).argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtq @ z_hat))] == j)

    return dict(bits=bits, train_s=train_s, hard=hard, consistent=consistent,
                W1q=W1q, W2q=W2q, Wtq=Wtq, owner=owner)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    class Bank:
        def __init__(self, Wb):
            self.W = Wb
            self.k = Wb.shape[0]

    results = []
    for bits in ARMS:
        r = run_arm(bits, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        results.append(r)
        print(f"ARM {bits}bit: hard={r['hard']:.4f} "
              f"consistent={r['consistent']}/10 train={r['train_s']:.0f}s",
              flush=True)

    # Reference row: the f32 base weights, rendered the same way.
    wz = np.load(BASE_WEIGHTS)
    ref = dict(bits="f32", W1q=wz["W1"], W2q=wz["W2"], Wtq=wz["Wtop"],
               owner=np.argmax(wz["Wtop"][:, CODE_DIM:], axis=1))
    gen_rows = [ref] + results

    fig, axes = plt.subplots(len(gen_rows), 10,
                             figsize=(10.5, 1.15 * len(gen_rows) + 0.6))
    for ai, r in enumerate(gen_rows):
        b2 = Bank(r["W2q"])
        for j in range(10):
            lab = np.zeros(10)
            lab[j] = LAM
            z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), lab]))
            row = r["Wtq"][int(np.argmax(r["Wtq"] @ z_hat))]
            c2 = np.maximum(row[:CODE_DIM], 0.0).reshape(G2, G2, K2)
            m1 = _harden_map(expand(_harden_map(c2), b2, W2, S2, (G1, G1, K1)))
            ax = axes[ai, j]
            ax.imshow(render8(m1, r["W1q"]), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        lbl = r["bits"] if isinstance(r["bits"], str) else f"{r['bits']} bits"
        axes[ai, 0].set_ylabel(lbl, fontsize=8, rotation=0, ha="right",
                               va="center")
    fig.suptitle("Label-only generation — trained WITH quantization "
                 "(accumulator version)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_qat.png", dpi=110)
    plt.close(fig)

    lines = [
        "# Trained-with-quantization (hidden accumulator), base rig",
        "",
        "f32 base reference: hard .9112, consistent 10/10.",
        "Post-hoc comparison arms: precision/results/post_hoc/report.md.",
        "",
        "| bits | hard | consistent | train_s |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['bits']} | {r['hard']:.4f} | "
                     f"{r['consistent']}/10 | {r['train_s']:.0f} |")
    lines += ["", "Figure: generation_qat.png (f32 reference row on top)."]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
