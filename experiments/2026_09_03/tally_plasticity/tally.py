"""Plasticity set by the tally alone. Full 0-9, no phases.

The champion's step size is blind: eta = clip(cnt/n, 0.02, 1) is the same rule
for a template carrying the classifier and for one that is noise, and it never
reaches zero, so templates drift forever. This replaces it outright. Nothing
about how far a template moves depends on how often it has won -- only on how
much it is worth to the tally:

    imp(t) = SUM over cells,classes  P(t,c,y) * T[t,c,y]     its share of the
                                                             information the code
                                                             carries about the label
    r(t)   = fraction of templates with STRICTLY lower imp    ties -> everyone at 0
    eta_t  = ETA_MAX * (1 - r(t))

One knob, ETA_MAX. Two properties come free. With an empty table every T entry
is 0, so every importance is 0, every rank is 0, and everything is fully plastic
-- "at first nobody knows anything, so everyone learns". And a template that
becomes informative and then stops winning keeps its counts, keeps its
importance, and stays frozen: a fossil, which is the preserved memory you want.

THE RISK this run exists to find. A template must sit still long enough to
accumulate coherent counts before it can look informative, but it only sits
still once it IS informative. If ETA_MAX is too high everything churns, every
template's counts smear across wherever it has been, no importance ever rises
and nothing freezes. That is a race, and ETA_MAX is what sets it -- hence the
sweep rather than a single value.

THE CONTROL that decides whether any of this means anything. Freezing SOME
templates and not others will change results by itself. `shuffled` applies the
identical multiset of plasticities through a fixed random permutation of the
template indices, so the same number of templates are equally rigid -- just the
wrong ones. If `tally` does not beat `shuffled`, the ordering is doing nothing
and the finding is only "a mix of rigid and plastic templates helps".

Everything else is the champion rig untouched: 5x5 patches, 576 positions, 400
templates, per-cell counted table T[t, 6x6 cell, class], beta 1.5, read by
summing one lookup per position.

Usage:  uv run python tally.py [--smoke] [--seeds 7,8,9]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2026_09_01" / "experts"))
import experts as E

OUT = HERE / "results"


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


SMOKE = "--smoke" in sys.argv
PS, SIDE = 5, 24
NPOS = SIDE * SIDE
K, NL, GRID, GG = 400, 10, 6, 36
EPS, ALPHA, BETA = 1e-12, 1.0, 1.5
BATCH, ETA_MIN_BASE, MIN_S, FLOOR = 128, 0.02, 4, 0.05
EPOCHS = 1 if SMOKE else 3
PROBE_EVERY = 5 if SMOKE else 20
ETA_MAXES = [0.1, 0.3] if SMOKE else [0.1, 0.2, 0.3, 0.5, 1.0]

PIDX = cp.asarray((((np.arange(SIDE)[:, None, None, None] + np.arange(PS)[None, None, :, None]) * 28
                    + (np.arange(SIDE)[None, :, None, None] + np.arange(PS)[None, None, None, :]))
                   ).reshape(NPOS, PS * PS))


def cellmap(n, g):
    a = np.arange(n * n)
    return cp.asarray(((a // n) * g // n) * g + ((a % n) * g // n))


CELLS = cellmap(SIDE, GRID)


def patches(Xb):
    P = Xb[:, PIDX]
    C = P - P.mean(-1, keepdims=True)
    nn = cp.linalg.norm(C, axis=-1)
    return C / cp.maximum(nn, EPS)[..., None], nn > FLOOR


def table_from(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def importance(N, T):
    """Each template's share of the information the code carries about the label."""
    tot = float(N.sum())
    if tot <= 0:
        return cp.zeros(K)
    return ((N / tot) * T).sum(axis=(1, 2))


def plasticity(imp, eta_max):
    """eta = eta_max * (1 - rank). Rank counts templates STRICTLY below, so an
    all-equal importance vector gives rank 0 and full plasticity to everyone."""
    r = (imp[None, :] < imp[:, None]).sum(1) / K
    return (eta_max * (1.0 - r)).astype(cp.float32)


def belief(T, win, keep, cells, m):
    ev = (T[win, cells] * keep[:, None]).reshape(m, NPOS, NL).sum(1)
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def compete(V, W, T, q, cells, m):
    err = 1.0 - (V @ W.T) ** 2
    bias = cp.einsum('ik,ihk->ih', cp.repeat(q, NPOS, 0),
                     cp.ascontiguousarray(T.transpose(1, 0, 2))[cells])
    return (err - BETA * bias).argmin(1)


def learn(W, V, win, eta_vec):
    cnt = cp.bincount(win, minlength=K)
    sums = cp.zeros((K, PS * PS), cp.float32)
    cupyx.scatter_add(sums, win, V)
    live = cnt >= MIN_S
    if int(live.sum()) == 0:
        return cnt
    e = eta_vec[live][:, None]
    W[live] += e * (sums[live] / cnt[live, None] - W[live])
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    return cnt


def code(W, X, chunk=256):
    idx = cp.zeros((len(X), NPOS), cp.int32)
    keepm = cp.zeros((len(X), NPOS), bool)
    for a in range(0, len(X), chunk):
        Q, keep = patches(X[a:a + chunk])
        m = len(Q)
        idx[a:a + m] = (1.0 - (Q.reshape(-1, PS * PS) @ W.T) ** 2).argmin(1).reshape(m, -1)
        keepm[a:a + m] = keep
    return idx, keepm


def build_table(win, y, valid):
    C = cp.tile(CELLS, len(win)).reshape(len(win), NPOS)
    lab = cp.repeat(y, NPOS).reshape(len(win), NPOS)
    flat = (win[valid] * GG + C[valid]) * NL + lab[valid]
    return cp.bincount(flat, minlength=K * GG * NL).reshape(K, GG, NL).astype(cp.float64)


def accuracy(T, codes, y):
    win, keep = codes
    C = cp.tile(CELLS, len(win)).reshape(len(win), NPOS)
    return float(((T[win, C] * keep[..., None]).sum(1).argmax(1) == y).mean())


def run(arm, eta_max, Xtr, ytr, Xte, yte, seed=7):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K, PS * PS)), cp.float32)
    W -= W.mean(1, keepdims=True)
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    N = cp.zeros((K, GG, NL)); n = cp.zeros(K)
    perm = cp.asarray(np.random.default_rng(seed + 1000).permutation(K))
    curve, snaps, step = [], [], 0

    for ep in range(EPOCHS):
        order = np.arange(len(Xtr)); rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            ids = cp.asarray(order[s:s + BATCH]); m = len(ids)
            Q, keep = patches(Xtr[ids])
            V = Q.reshape(-1, PS * PS); k = keep.reshape(-1)
            cells = cp.tile(CELLS, m)
            T = table_from(N)
            w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
            q = belief(T, w0, k, cells, m)
            win = compete(V, W, T, q, cells, m)
            lab = cp.repeat(ytr[ids], NPOS)
            flat = (win[k] * GG + cells[k]) * NL + lab[k]
            N += cp.bincount(flat, minlength=K * GG * NL).reshape(N.shape)

            imp = importance(N, T)
            if arm == "frozen_all":
                eta = cp.zeros(K, cp.float32)          # templates never learn at all
            elif arm == "flat":
                eta = cp.full(K, eta_max, cp.float32)  # no annealing, no gating
            elif arm == "base":
                cnt = cp.bincount(win[k], minlength=K)
                n += cnt
                eta = cp.clip(cnt / cp.maximum(n, 1.0), ETA_MIN_BASE, 1.0).astype(cp.float32)
            else:
                eta = plasticity(imp, eta_max)
                if arm == "shuffled":
                    eta = eta[perm]
            cnt = learn(W, V[k], win[k], eta)

            if step % PROBE_EVERY == 0:
                itr, ktr = code(W, Xtr)
                T_re = table_from(build_table(itr, ytr, ktr))
                codes_te = code(W, Xte)
                top = cp.argsort(-imp)[:K // 10]
                frozen = eta < ETA_MIN_BASE
                curve.append({
                    "step": step,
                    "recount": accuracy(T_re, codes_te, yte),
                    "online": accuracy(table_from(N), codes_te, yte),
                    "imp_mean": float(imp.mean()), "imp_max": float(imp.max()),
                    "imp_top10_share": float(imp[top].sum() / (imp.sum() + 1e-12)),
                    "eta_mean": float(eta.mean()),
                    "n_frozen": int(frozen.sum()),
                    "win_share_frozen": float(cnt[frozen].sum() / cp.maximum(cnt.sum(), 1)),
                    "n_dead": int((N.sum((1, 2)) == 0).sum()),
                })
                snaps.append((step, cp.asnumpy(W)))
            step += 1

    itr, ktr = code(W, Xtr)
    T_re = table_from(build_table(itr, ytr, ktr))
    codes_te = code(W, Xte)
    final = {"recount": accuracy(T_re, codes_te, yte),
             "online": accuracy(table_from(N), codes_te, yte)}
    imp = importance(N, table_from(N))
    snaps.append((step, cp.asnumpy(W)))
    return curve, final, cp.asnumpy(W), cp.asnumpy(imp), snaps


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    print(f"full 0-9: {len(ytr)} train / {len(yte)} test, {PS}x{PS} patches, "
          f"{NPOS} positions, {K} templates, table {K}x{GG}x{NL}, beta {BETA}\n", flush=True)

    jobs = ([("base", None), ("frozen_all", None)]
            + [("flat", e) for e in ETA_MAXES]
            + [("tally", e) for e in ETA_MAXES]
            + [("shuffled", e) for e in ETA_MAXES])
    res, store = {}, {}
    for arm, em in jobs:
        tag = arm if em is None else f"{arm}_{em:g}"
        t0 = time.time()
        c, f, W, imp, snaps = run(arm, em, Xtr, ytr_g, Xte, yte_g)
        res[tag] = {"arm": arm, "eta_max": em, "curve": c, "final": f}
        store[tag] = (W, imp, snaps)
        last = c[-1]
        print(f"  {tag:<14s} recount {f['recount']:.4f}  online {f['online']:.4f}  "
              f"gap {f['recount']-f['online']:+.4f}   frozen {last['n_frozen']:>3d}/{K}  "
              f"dead {last['n_dead']:>3d}  win-share-frozen {last['win_share_frozen']:.3f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "tally.json").write_text(json.dumps(res, indent=2))
    draw(res, store)


def draw(res, store):
    tags = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap("viridis")
    for tag in tags:
        c = res[tag]["curve"]; x = [p["step"] for p in c]
        arm, em = res[tag]["arm"], res[tag]["eta_max"]
        col = "k" if em is None else cmap(ETA_MAXES.index(em) / max(1, len(ETA_MAXES) - 1))
        ls = {"base": "-", "frozen_all": "-.", "tally": "-",
              "flat": "--", "shuffled": ":"}[arm]
        lw = 2.0 if em is None else 1.5
        axes[0].plot(x, [p["recount"] for p in c], ls, color=col, lw=lw, label=tag)
        axes[1].plot(x, [p["recount"] - p["online"] for p in c], ls, color=col, lw=lw)
        axes[2].plot(x, [p["n_frozen"] for p in c], ls, color=col, lw=lw)
    for ax, t, yl in ((axes[0], "recount accuracy", "test accuracy"),
                      (axes[1], "staleness  (recount − online)", "gap"),
                      (axes[2], "templates effectively frozen (eta < 0.02)", "count")):
        ax.set_title(t, fontsize=10); ax.set_ylabel(yl, fontsize=9)
        ax.set_xlabel("training batches"); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Plasticity from the tally alone — full 0-9, 5x5 patches", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "curves.png", dpi=145); plt.close(fig)

    def montage(W, path, title, order=None, mark=None):
        idx = np.arange(len(W)) if order is None else order
        side = int(np.ceil(np.sqrt(len(idx))))
        fig, ax = plt.subplots(side, side, figsize=(side * 0.34, side * 0.34))
        for j, a in enumerate(ax.ravel()):
            a.set_xticks([]); a.set_yticks([])
            if j < len(idx):
                w = W[idx[j]]; v = float(np.abs(w).max())
                a.imshow(w.reshape(PS, PS), cmap="RdBu_r", vmin=-v, vmax=v)
                if mark is not None and mark[idx[j]]:
                    for sp in a.spines.values():
                        sp.set_color("crimson"); sp.set_linewidth(1.4)
            else:
                a.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(wspace=0.06, hspace=0.06, top=0.955)
        fig.savefig(path, dpi=135, bbox_inches="tight"); plt.close(fig)

    best = max([t for t in tags if res[t]["arm"] == "tally"],
               key=lambda t: res[t]["final"]["recount"])
    for tag in ("base", best):
        W, imp, _ = store[tag]
        montage(W, OUT / f"templates_final_{tag}.png",
                f"all {K} templates, {tag}  (sorted by importance, most first)",
                order=np.argsort(-imp))

    W, imp, _ = store[best]
    o = np.argsort(-imp)
    fig, axes = plt.subplots(2, 16, figsize=(16 * 0.62, 2 * 0.85))
    for c in range(16):
        for r, sel in enumerate((o[:16], o[-16:])):
            w = W[sel[c]]; v = float(np.abs(w).max())
            axes[r, c].imshow(w.reshape(PS, PS), cmap="RdBu_r", vmin=-v, vmax=v)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("most\nimportant\n(frozen)", fontsize=7, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("least\nimportant\n(plastic)", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"{best}: the templates the tally froze vs the ones it left plastic", fontsize=10)
    fig.subplots_adjust(wspace=0.06, hspace=0.08, top=0.80, left=0.10)
    fig.savefig(OUT / "templates_by_importance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    _, imp, snaps = store[best]
    o = np.argsort(-imp)
    cols = np.concatenate([o[:8], o[len(o) // 2 - 4:len(o) // 2 + 4], o[-8:]])
    pick = [snaps[int(i)] for i in np.linspace(0, len(snaps) - 1, 6)]
    fig, axes = plt.subplots(6, len(cols), figsize=(len(cols) * 0.6, 6 * 0.7))
    for r, (st, Wm) in enumerate(pick):
        for c, i in enumerate(cols):
            w = Wm[i]; v = float(np.abs(w).max())
            axes[r, c].imshow(w.reshape(PS, PS), cmap="RdBu_r", vmin=-v, vmax=v)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"step {st}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"{best}: templates through training.  left 8 = most important (should stop "
                 f"moving),\nmiddle 8 = median,  right 8 = least important (should keep churning)",
                 fontsize=9)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.86, left=0.07)
    fig.savefig(OUT / "templates_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  best tally arm: {best}")
    print(f"  wrote curves.png, templates_final_*.png, templates_by_importance.png, "
          f"templates_time.png", flush=True)


if __name__ == "__main__":
    main()
