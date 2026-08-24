"""F8: what should ambiguity do to plasticity?

Three opposed answers to one question, on the 8x8/K1=512 dense champion:
  base     - argmax, step from c_max                      (today's rule)
  samp t   - winner sampled from softmax(C/t)             (ambiguity randomizes)
  margin   - argmax, step from (c_max - c_2nd)            (ambiguity suppresses)
  margin_x - argmax, step from c_max * (c_max - c_2nd)

Every arm's step sizes are rescaled per batch to the same mean as c_max,
so the arms differ in HOW learning is distributed, not how much of it
there is. Predictions in README (F8).

Run:  GF_W1_STR=1 .venv/bin/python experiments/2026_08_24/ambiguity/run_ambiguity.py
      GF_SCOPE=all  - apply the rule at L1+L2+L3 instead of L1 only
      GF_ARMS=base,margin  - subset of arms
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
    THETA_CAP,
    Dict,
    XP_NAME,
    _center_norm_rows,
    _scatter_add,
    _skeleton,
    _windows,
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
    SEED,
    TRAIN_N,
    expand,
    load_data,
)
from run_rich_palette_8x8 import (  # noqa: E402
    CODE3_DIM,
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
    to_np,
    top_view,
)
from gain_feedback import center_norm  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_24" / "ambiguity" / "results"
K1 = 512
EVAL_B = 250
SCOPE = os.environ.get("GF_SCOPE", "l1")
RUNSEED = int(os.environ.get("GF_RUNSEED", "0"))   # 0 = canonical order
FAST = os.environ.get("GF_FAST") == "1"            # skip probes + gallery

ARMS = [("base", 0.0), ("samp", 0.01), ("samp", 0.02), ("samp", 0.05),
        ("samp", 0.1), ("margin", 0.0), ("margin_x", 0.0),
        ("dir_cmax", 0.0), ("dir_margin", 0.0), ("dir_marginx", 0.0)]

# Rules that reweight the target BLEND (where a template moves) rather
# than the step size (how far). Necessary because eta*Csum is clipped at
# THETA_CAP for ~95% of updates, which makes every cvals-based rule inert.
DIR_RULES = {"dir_cmax", "dir_margin", "dir_marginx"}

if XP_NAME == "cupy":
    GEN = xp.random.default_rng(SEED)
else:
    GEN = np.random.default_rng(SEED)


def arm_name(rule, tau):
    return f"{rule}_t{tau}" if rule == "samp" else rule


def _runner_up(C, cmax):
    """c_max - c_2nd. Mask-and-max beats a full partition on a
    (batch*positions, K) array by roughly 2x."""
    w = xp.argmax(C, axis=1)[:, None]
    Cm = C.copy()
    xp.put_along_axis(Cm, w, xp.asarray(-xp.inf, dtype=C.dtype), axis=1)
    return xp.maximum(cmax - xp.max(Cm, axis=1), 0.0)


def select(C, rule, tau, ok):
    """(winners, step_vals, blend_weights) under one plasticity rule.

    step_vals set theta (how far the template rotates) and are rescaled so
    their mean over the kept set equals the mean of c_max — the control
    that stops a rule from merely being a slower learning rate.
    blend_weights set the target blend (WHERE it rotates to); ones = the
    unweighted mean Dict.update normally takes.
    """
    cmax = xp.max(C, axis=1)
    ones = xp.ones_like(cmax)

    if rule == "base":
        return xp.argmax(C, axis=1), cmax * ok, ones

    if rule in DIR_RULES:
        # Step size identical to base; only the target blend changes.
        w = xp.argmax(C, axis=1)
        if rule == "dir_cmax":
            wd = cmax
        else:
            marg = _runner_up(C, cmax)
            wd = marg if rule == "dir_margin" else cmax * marg
        return w, cmax * ok, xp.maximum(wd, 0.0) * ok

    if rule == "samp":
        # Gumbel-max: argmax(C/tau + G) is an exact draw from softmax(C/tau).
        U = GEN.random(size=C.shape, dtype=DTYPE)
        G = -xp.log(-xp.log(U + 1e-20) + 1e-20)
        w = xp.argmax(C * (1.0 / tau) + G, axis=1)
        v = xp.take_along_axis(C, w[:, None], axis=1)[:, 0]
    else:
        w = xp.argmax(C, axis=1)
        marg = _runner_up(C, cmax)
        v = marg if rule == "margin" else cmax * marg

    v = v * ok
    keep = (v > 0) & (cmax > 0)
    if bool(keep.any()):
        v = v * (float(cmax[keep].mean()) / max(float(v[keep].mean()), 1e-9))
    return w, v, ones


def update_blend(dic, T_hat, winners, cvals, wdir):
    """Dict.update with a WEIGHTED target blend.

    Identical to Dict.update except each target contributes to its
    template's mean in proportion to wdir. The blend is normalized, so
    only the RELATIVE weights matter — step size stays exactly base.
    """
    if len(T_hat) == 0:
        return
    S = xp.zeros((dic.k, dic.dim), dtype=DTYPE)
    Csum = xp.zeros(dic.k, dtype=DTYPE)
    _scatter_add(S, winners, T_hat * wdir[:, None])
    _scatter_add(Csum, winners, cvals)
    hit = Csum > 0
    if not bool(hit.any()):
        return
    mu = S[hit]
    mu = mu / (xp.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
    w = dic.W[hit]
    cw = xp.sum(w * mu, axis=1, keepdims=True)
    tau_v = mu - cw * w
    tn = xp.linalg.norm(tau_v, axis=1, keepdims=True)
    theta = xp.minimum(dic.eta * Csum[hit], THETA_CAP)[:, None]
    w_new = xp.where(tn > 1e-9,
                     w * xp.cos(theta) + (tau_v / xp.maximum(tn, 1e-9)) * xp.sin(theta),
                     w)
    w_new = w_new - w_new.mean(axis=1, keepdims=True)
    w_new = w_new / (xp.linalg.norm(w_new, axis=1, keepdims=True) + 1e-9)
    dic.W[hit] = w_new
    dic.win_counts += (Csum > 0).astype(xp.int64)


def level_pass_rule(prev_maps, dic, positions, win, kprev, g_out,
                    sparse_learn, learning, rule, tau, usage=None):
    """level_pass with a swappable plasticity rule. Message is unchanged
    dense relu — only WHO learns and HOW MUCH differs across arms."""
    N = len(prev_maps)
    P = len(positions)
    Vf = _windows(prev_maps, positions, win)
    V = Vf.reshape(N * P, -1)
    V_hat, ok = _center_norm_rows(V)

    if learning and dic.n_boot < dic.k:
        dic.bootstrap(V_hat[ok])

    C = V_hat @ dic.W.T
    code = xp.maximum(C, 0.0) * ok[:, None]

    if learning and dic.n_boot >= dic.k:
        if sparse_learn:
            blocks = Vf.reshape(N * P, win * win, kprev)
            T_hat, ok_s = _center_norm_rows(_skeleton(blocks))
            Cl, okl = T_hat @ dic.W.T, ok_s
        else:
            T_hat, Cl, okl = V_hat, C, ok
        winners, cvals, wdir = select(Cl, rule, tau, okl)
        keep = cvals > 0
        if usage is not None and bool(keep.any()):
            usage += xp.bincount(winners[keep], minlength=dic.k)
        if rule in DIR_RULES:
            update_blend(dic, T_hat[keep], winners[keep],
                         cvals[keep], wdir[keep])
        else:
            dic.update(T_hat[keep], winners[keep], cvals[keep])

    return code.reshape(N, g_out, g_out, dic.k)


def peakiness(W, win, kprev):
    r = []
    for t in range(len(W)):
        w3 = np.maximum(W[t], 0.0).reshape(win * win, kprev)
        s = w3.sum(axis=1)
        m = w3.max(axis=1)
        r.append(m[s > 0] / s[s > 0])
    return float(np.concatenate(r).mean())


def run_arm(rule, tau, Xtr, ytr, Xte, yte, labels_x):
    name = arm_name(rule, tau)
    d1 = Dict(K1, W1 * W1, ETA1)
    d2 = Dict(K2, W2 * W2 * K1, ETA2)
    d3 = Dict(K3, W3 * W3 * K2, ETA3)
    top = Dict(KTOP, CODE3_DIM + 10, ETATOP)
    Xtr_x = xp.asarray(Xtr, dtype=DTYPE)
    usage = xp.zeros(K1, dtype=xp.int64)
    up = (rule, tau) if SCOPE == "all" else ("base", 0.0)

    t0 = time.time()
    for _ in range(EPOCHS):
        for s in range(0, TRAIN_N, B):
            xb = Xtr_x[s:s + B]
            lb = labels_x[s:s + B]
            m1 = level_pass_rule(xb[..., None], d1, POS1, W1, 1, G1,
                                 False, True, rule, tau, usage)
            m2 = level_pass_rule(m1, d2, POS2, W2, K1, G2, True, True, *up)
            m3 = level_pass_rule(m2, d3, POS3, W3, K2, G3, True, True, *up)
            Zd, okd = top_view(m3.reshape(len(xb), -1), lb)
            if top.n_boot < top.k:
                top.bootstrap(Zd[okd])
            if top.n_boot >= top.k:
                Ct = Zd @ top.W.T
                cv = xp.max(Ct, axis=1) * okd
                keep = cv > 0
                top.update(Zd[keep], xp.argmax(Ct, axis=1)[keep], cv[keep])
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    def encode(X):
        C2s, C3s = [], []
        for s in range(0, len(X), EVAL_B):
            xb = xp.asarray(X[s:s + EVAL_B], dtype=DTYPE)
            m1 = level_pass_rule(xb[..., None], d1, POS1, W1, 1, G1,
                                 False, False, rule, tau)
            m2 = level_pass_rule(m1, d2, POS2, W2, K1, G2, True, False, *up)
            m3 = level_pass_rule(m2, d3, POS3, W3, K2, G3, True, False, *up)
            C2s.append(to_np(m2.reshape(len(xb), -1)))
            C3s.append(to_np(m3.reshape(len(xb), -1)))
        return np.concatenate(C2s), np.concatenate(C3s)

    probes = {"L2": float("nan"), "L3": float("nan")}
    if FAST:
        _, C3te = encode(Xte)
    else:
        C2tr, C3tr = encode(Xtr[:PROBE_N])
        C2te, C3te = encode(Xte)
        for nm, tr, te in [("L2", C2tr, C2te), ("L3", C3tr, C3te)]:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(tr, ytr[:PROBE_N])
            probes[nm] = float(clf.score(te, yte))

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

    W1n, W2n, W3n = to_np(d1.W), to_np(d2.W), to_np(d3.W)
    Cc = W1n @ W1n.T
    off = np.abs(Cc[~np.eye(len(Cc), dtype=bool)])
    crowd = float(off.mean())
    clones = float((off > 0.95).mean() * 100.0)

    xb = xp.asarray(Xte[:EVAL_B], dtype=DTYPE)
    Vh, ok = _center_norm_rows(_windows(xb[..., None], POS1, W1)
                               .reshape(-1, W1 * W1))
    Cm = to_np(Vh @ d1.W.T)[to_np(ok)]
    Cm.sort(axis=1)
    margin = float(np.median(Cm[:, -1] - Cm[:, -2]))

    u = to_np(usage).astype(np.float64)
    dead = int((u == 0).sum())
    p = u[u > 0] / u.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(K1))

    class Bank:
        def __init__(self, Wb):
            self.W, self.k = Wb, Wb.shape[0]

    b2, b3 = Bank(W2n), Bank(W3n)

    def harden(m, k=1):
        o = np.zeros_like(m)
        for a in range(m.shape[0]):
            for b_ in range(m.shape[1]):
                seg = np.maximum(m[a, b_], 0.0)
                if seg.max() > 0:
                    idx = np.argsort(seg)[::-1][:k]
                    o[a, b_, idx] = seg[idx]
        return o

    if FAST:
        return dict(arm=name, probeL2=probes["L2"], probeL3=probes["L3"],
                    hard=hard, consistent=consistent,
                    peak2=peakiness(W2n, W2, K1), peak3=peakiness(W3n, W3, K2),
                    crowd=crowd, clones=clones, margin=margin,
                    entropy=entropy, dead=dead, train_s=train_s)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        row = Wtn[int(np.argmax(Wtn @ z_hat))]
        c3 = np.maximum(row[:CODE3_DIM], 0.0).reshape(G3, G3, K3)
        m2 = harden(expand(harden(c3), b3, W3, S3, (G2, G2, K2)))
        m1 = harden(expand(m2, b2, W2, S2, (G1, G1, K1)))
        ax.imshow(render8(m1, W1n), cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Generation — plasticity rule {name} (scope {SCOPE})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"generation_{name}_{SCOPE}.png", dpi=110)
    plt.close(fig)

    return dict(arm=name, probeL2=probes["L2"], probeL3=probes["L3"],
                hard=hard, consistent=consistent,
                peak2=peakiness(W2n, W2, K1), peak3=peakiness(W3n, W3, K2),
                crowd=crowd, clones=clones, margin=margin,
                entropy=entropy, dead=dead, train_s=train_s)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    if RUNSEED:
        # Real seed variation: online learning depends on presentation
        # order. (The bootstrap tie-break noise alone barely moves.)
        perm = np.random.default_rng(RUNSEED).permutation(len(Xtr))
        Xtr, ytr = Xtr[perm], ytr[perm]
    labels_tr = np.zeros((TRAIN_N, 10), dtype=np.float32)
    labels_tr[np.arange(TRAIN_N), ytr] = LAM
    labels_x = xp.asarray(labels_tr)

    only = os.environ.get("GF_ARMS")
    arms = [a for a in ARMS if not only or arm_name(*a) in only.split(",")]
    rows = []
    for rule, tau in arms:
        r = run_arm(rule, tau, Xtr, ytr, Xte, yte, labels_x)
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(f"DONE {r['arm']}: probeL3={r['probeL3']:.4f} "
              f"hard={r['hard']:.4f} peak2={r['peak2']:.3f} "
              f"peak3={r['peak3']:.3f} crowd={r['crowd']:.3f} "
              f"margin={r['margin']:.3f} H={r['entropy']:.3f} "
              f"dead={r['dead']} {r['train_s']:.0f}s", flush=True)

    lines = [
        f"# F8: ambiguity in plasticity (8x8, K1=512, scope={SCOPE}, "
        f"{EPOCHS} epochs, seed {SEED})",
        "",
        "| arm | probe L2 | probe L3 | hard | consistent | peak2 | peak3 "
        "| L1 crowd | clones% | top1-2 margin | usage H | dead | train_s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['probeL2']:.4f} | {r['probeL3']:.4f} | "
            f"{r['hard']:.4f} | {r['consistent']}/10 | {r['peak2']:.3f} | "
            f"{r['peak3']:.3f} | {r['crowd']:.3f} | {r['clones']:.2f} | "
            f"{r['margin']:.3f} | {r['entropy']:.3f} | {r['dead']} | "
            f"{r['train_s']:.0f} |")
    lines += ["",
              "Step sizes rate-matched across arms (mean step = mean c_max).",
              "Reference (dense@512, same rig): hard .9094, margin 0.038.",
              "clones% = share of off-diagonal L1 template pairs above 0.95."]
    tag = f"{SCOPE}_s{RUNSEED}" if RUNSEED else SCOPE
    (OUTPUT_DIR / f"report_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
