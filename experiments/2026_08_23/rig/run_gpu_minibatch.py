"""GPU mini-batch trainer for the consolidated 4-layer sparselearn rig.

Mini-batch approximation of the online rule: B images are encoded against
frozen weights; each template then takes ONE vectorized geodesic rotation
toward the normalized mean of its batch targets, with
theta = clip(eta * sum(c_i), THETA_CAP). Communication is dense relu with
magnitudes; L2/L3 learning targets are per-position-top-1 skeleton views;
the top learns from [code ; lam*onehot]. Evaluation reuses the exact CPU
pipeline (run_4layer_topk) on the trained weights.

Env: GF_XP=cupy|numpy (default cupy), GF_B batch size (default 128).
Run:  .venv/bin/python experiments/2026_08_23/rig/run_gpu_minibatch.py
"""

import os
import time
from pathlib import Path

os.environ["GF_REPORT"] = "sparselearn"   # for the imported CPU eval pipeline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

XP_NAME = os.environ.get("GF_XP", "cupy")
if XP_NAME == "cupy":
    import cupy as xp
    import cupyx

    def _scatter_add(target, idx, vals):
        cupyx.scatter_add(target, idx, vals)
else:
    xp = np

    def _scatter_add(target, idx, vals):
        np.add.at(target, idx, vals)

from run_4layer_topk import (  # noqa: E402  (env must be set first)
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
    K_OUT1,
    K_OUT2,
    K_OUT3,
    LAM,
    NORM_FLOOR,
    PROBE_N,
    SEED,
    SIDE,
    TEST_N,
    TRAIN_N,
    W1_STR,
    W1_WIN,
    W2_STR,
    W2_WIN,
    W3_STR,
    W3_WIN,
    encode_over_batch,
    encode_pixels_batch,
    expand,
    render_pixels,
    load_data,
)

ROOT = Path(__file__).resolve().parents[3]
DTYPE = xp.float32   # GeForce runs f64 at 1/32 throughput; f32 verified
                     # against the f64 MNIST flagship numbers (see README)
B = int(os.environ.get("GF_B", "128"))
THETA_CAP = 0.3
_SUF = f"stride{W1_STR}"
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "gpu_minibatch" / "results" / _SUF

POS1 = [(r, c) for r in range(0, SIDE - W1_WIN + 1, W1_STR) for c in range(0, SIDE - W1_WIN + 1, W1_STR)]
POS2 = [(r, c) for r in range(0, G1 - W2_WIN + 1, W2_STR) for c in range(0, G1 - W2_WIN + 1, W2_STR)]
POS3 = [(r, c) for r in range(0, G2 - W3_WIN + 1, W3_STR) for c in range(0, G2 - W3_WIN + 1, W3_STR)]


class Dict:
    """A dictionary bank with bootstrap state, on xp."""

    def __init__(self, k, dim, eta):
        self.k, self.dim, self.eta = k, dim, eta
        self.W = xp.zeros((k, dim), dtype=DTYPE)
        self.n_boot = 0
        self.win_counts = xp.zeros(k, dtype=xp.int64)
        self._noise = np.random.default_rng(SEED)

    def bootstrap(self, V_hat):
        """Adopt rows of V_hat (valid, centered, normalized) until full."""
        need = self.k - self.n_boot
        if need <= 0 or len(V_hat) == 0:
            return
        take = min(need, len(V_hat))
        # Tie-breaking noise with FIXED total length (~5% of the unit signal),
        # regardless of dimension — per-component 0.01 was catastrophic at
        # high dim (norm ~2.0 at dim 40k: memories born two-thirds noise).
        w = V_hat[:take] + xp.asarray(
            ((0.05 / np.sqrt(self.dim))
             * self._noise.standard_normal((take, self.dim))).astype(np.float32))
        w = w - w.mean(axis=1, keepdims=True)
        w = w / (xp.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        self.W[self.n_boot:self.n_boot + take] = w
        self.win_counts[self.n_boot:self.n_boot + take] += 1
        self.n_boot += take

    def update(self, T_hat, winners, cvals):
        """One vectorized geodesic step per template toward the normalized
        mean of its batch targets; theta = clip(eta * sum(c), cap)."""
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
        w = self.W[hit]
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
        self.W[hit] = w_new
        self.win_counts += (Csum > 0).astype(xp.int64)


def _center_norm_rows(V):
    """Rows -> centered+normalized; returns (V_hat, valid mask)."""
    V = V - V.mean(axis=1, keepdims=True)
    n = xp.linalg.norm(V, axis=1)
    ok = n > NORM_FLOOR
    V = V / xp.maximum(n, 1e-9)[:, None]
    return V, ok


def _windows(maps, positions, win):
    """(N, G, G, K) -> (N, P, win*win*K) stacked window views."""
    parts = [maps[:, r:r + win, c:c + win, :].reshape(len(maps), -1)
             for r, c in positions]
    return xp.stack(parts, axis=1)


def _skeleton(Wn):
    """(M, P, Kprev) window blocks -> per-position top-1 skeleton, flattened."""
    am = xp.argmax(Wn, axis=2)
    vals = xp.take_along_axis(Wn, am[:, :, None], axis=2)
    sk = xp.zeros_like(Wn)
    xp.put_along_axis(sk, am[:, :, None], xp.maximum(vals, 0.0), axis=2)
    return sk.reshape(len(Wn), -1)


def level_pass(prev_maps, dic, positions, win, kprev, g_out, sparse_learn, learning):
    """One dictionary level, batched: returns code maps (N, g, g, K)."""
    N = len(prev_maps)
    P = len(positions)
    Vf = _windows(prev_maps, positions, win)                    # (N, P, D)
    V = Vf.reshape(N * P, -1)
    V_hat, ok = _center_norm_rows(V)

    if learning and dic.n_boot < dic.k:
        dic.bootstrap(V_hat[ok])

    C = V_hat @ dic.W.T                                          # (N*P, K)
    code = xp.maximum(C, 0.0) * ok[:, None]                      # dense relu message

    if learning and dic.n_boot >= dic.k:
        if sparse_learn:
            blocks = Vf.reshape(N * P, win * win, kprev)
            SK = _skeleton(blocks)
            T_hat, ok_s = _center_norm_rows(SK)
            Cs = T_hat @ dic.W.T
            winners = xp.argmax(Cs, axis=1)
            cvals = xp.max(Cs, axis=1) * ok_s
            keep = cvals > 0
            dic.update(T_hat[keep], winners[keep], cvals[keep])
        else:
            winners = xp.argmax(C, axis=1)
            cvals = xp.max(C, axis=1) * ok
            keep = cvals > 0
            dic.update(V_hat[keep], winners[keep], cvals[keep])

    return code.reshape(N, g_out, g_out, dic.k)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    d1 = Dict(K1, W1_WIN * W1_WIN, ETA1)
    d2 = Dict(K2, W2_WIN * W2_WIN * K1, ETA2)
    d3 = Dict(K3, W3_WIN * W3_WIN * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)

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

            H = m3.reshape(len(xb), -1)
            H, okh = _center_norm_rows(H)
            Z = xp.concatenate([H, lb], axis=1)
            Z, okz = _center_norm_rows(Z)
            okz = okz & okh
            if top.n_boot < top.k:
                top.bootstrap(Z[okz])
            if top.n_boot >= top.k:
                Ct = Z @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cvals = xp.max(Ct, axis=1) * okz
                keep = cvals > 0
                top.update(Z[keep], winners[keep], cvals[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    # ---- Transfer to numpy and evaluate with the exact CPU pipeline -------
    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    W1n, W2n, W3n, Wtn = map(to_np, (d1.W, d2.W, d3.W, top.W))
    np.savez(OUTPUT_DIR / "weights.npz", W1=W1n, W2=W2n, W3=W3n, Wtop=Wtn)

    class Bank:
        def __init__(self, W):
            self.W = W
            self.k = W.shape[0]

    b1, b2, b3 = Bank(W1n), Bank(W2n), Bank(W3n)
    M1te = encode_pixels_batch(Xte, b1, K_OUT1)
    M2te = encode_over_batch(M1te, b2, W2_WIN, W2_STR, G2, K_OUT2)
    M3te = encode_over_batch(M2te, b3, W3_WIN, W3_STR, G3, K_OUT3)
    M1tr = encode_pixels_batch(Xtr[:PROBE_N], b1, K_OUT1)
    M2tr = encode_over_batch(M1tr, b2, W2_WIN, W2_STR, G2, K_OUT2)
    M3tr = encode_over_batch(M2tr, b3, W3_WIN, W3_STR, G3, K_OUT3)

    probes = {}
    for name, tr, te in [("L1", M1tr, M1te), ("L2", M2tr, M2te), ("L3", M3tr, M3te)]:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(tr.reshape(len(tr), -1), ytr[:PROBE_N])
        probes[name] = float(clf.score(te.reshape(len(te), -1), yte))

    from gain_feedback import center_norm
    label_half = Wtn[:, CODE3_DIM:]
    owner = label_half.argmax(axis=1)
    H = M3te.reshape(len(M3te), -1)
    H = H - H.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    Z = np.concatenate([H, np.zeros((len(H), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    C2 = Z @ Wtn.T
    acc_hard = float((owner[C2.argmax(axis=1)] == yte).mean())

    consistent = 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        consistent += int(owner[int(np.argmax(Wtn @ z_hat))] == j)

    def peak(W, win, kprev):
        r = []
        for t in range(len(W)):
            w3 = np.maximum(W[t], 0.0).reshape(win * win, kprev)
            for p in range(win * win):
                ssum = w3[p].sum()
                if ssum > 0:
                    r.append(w3[p].max() / ssum)
        return float(np.mean(r))

    p2, p3 = peak(W2n, W2_WIN, K1), peak(W3n, W3_WIN, K2)

    print(f"RESULT xp={XP_NAME} B={B} train_s={train_s:.1f} "
          f"probeL1={probes['L1']:.4f} probeL2={probes['L2']:.4f} probeL3={probes['L3']:.4f} "
          f"hard={acc_hard:.4f} consistent={consistent}/10 peak2={p2:.3f} peak3={p3:.3f}")

    fig, axes = plt.subplots(4, 4, figsize=(6.5, 7))
    for u, ax in enumerate(axes.flat):
        c3 = np.zeros((G3, G3, K3))
        c3[G3 // 2, G3 // 2, u] = 1.0
        m2 = expand(c3, b3, W3_WIN, W3_STR, (G2, G2, K2))
        m1 = expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1))
        ax.imshow(render_pixels(m1, b1), cmap="gray")
        ax.axis("off")
    fig.suptitle(f"L3 parts, GPU mini-batch ({XP_NAME}, B={B})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l3_parts.png", dpi=110)
    plt.close(fig)

    def _harden_map(m, k=1):
        o = np.zeros_like(m)
        for a in range(m.shape[0]):
            for b_ in range(m.shape[1]):
                seg = np.maximum(m[a, b_], 0.0)
                if seg.max() > 0:
                    idx = np.argsort(seg)[::-1][:k]
                    o[a, b_, idx] = seg[idx]
        return o

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        winner = int(np.argmax(Wtn @ z_hat))
        c3 = np.maximum(Wtn[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = _harden_map(expand(_harden_map(c3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
        m1 = _harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
        ax.imshow(render_pixels(m1, b1), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Label-only generation, hardened read ({XP_NAME}, stride {W1_STR})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_hardened.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join([
        f"# GPU mini-batch trainer ({XP_NAME}, B={B})",
        "",
        f"Train wall-time: {train_s:.1f}s (CPU online baseline: ~1200s).",
        "CPU sparselearn baselines: probes .9528/.9592/.9590, hard .8748, "
        "peak .718/.534.",
        "",
        "| probe L1 | probe L2 | probe L3 | hard | consistent | peak2 | peak3 |",
        "|---|---|---|---|---|---|---|",
        f"| {probes['L1']:.4f} | {probes['L2']:.4f} | {probes['L3']:.4f} | "
        f"{acc_hard:.4f} | {consistent}/10 | {p2:.3f} | {p3:.3f} |",
        "",
        "Figure: l3_parts.png. Weights: weights.npz",
    ]) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
