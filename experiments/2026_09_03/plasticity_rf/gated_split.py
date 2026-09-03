"""Tally-driven plasticity where templates are actually worth protecting.

`rf_sweep.py` measured what template learning buys at each receptive field:

    5x5 +0.0065   9x9 +0.0110   13x13 +0.0083   17x17 -0.0015
    21x21 -0.0113   25x25 +0.0512   28x28 +0.3835

Everything from 5 to 21 pixels is flat or negative -- random filters plus
position-indexed counting is already a strong code, so no rule about which
templates may move can matter there. That is why `../tally_plasticity` at 5x5
could not differentiate anything. Only at 25x25 and 28x28, where positions stop
being shared and each template has to BE a digit, is there anything at stake.

So the gate is tested at 28x28, on a split, which is the setting it was designed
for. `../whole_digit` established the stakes: at beta 0 the whole-digit rig loses
14.5 points to phase B dragging its templates, with 57% of them converted into
other digits. That is the hole this mechanism is supposed to fill.

    eta_t = ETA_MAX * (1 - rank of importance(t))
    importance(t) = SUM over cells,classes  P(t,c,y) * T[t,c,y]

Arms, at beta 0 (where drift is unopposed, so the gate acts alone) and beta 1.5
(where routing already protects, so there is less room):

    base       eta = clip(cnt/n, 0.02, 1)                the champion's rule
    tally      eta = ETA_MAX * (1 - rank(importance))    the proposal
    shuffled   identical plasticities, wrong templates   the control

Phases: 0-4, then 5-9, then all ten. Accuracy on old and new classes tracked
throughout, against the running tally and against a full recount.

Usage:  uv run python gated_split.py [--ps 28] [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rf_sweep as R

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


SMOKE = "--smoke" in sys.argv
PS = arg("--ps", 28, int)
R.BATCH = 512                      # one win per image, so batches must be larger
EPOCHS = 2 if SMOKE else 20
ETA_MAXES = [0.1] if SMOKE else [0.05, 0.1, 0.3]
BETAS = [0.0] if SMOKE else [0.0, 1.5]
PROBE_EVERY = 10
K, NL = R.K, R.NL


def importance(N, T):
    tot = float(N.sum())
    return cp.zeros(K) if tot <= 0 else ((N / tot) * T).sum(axis=(1, 2))


def plasticity(imp, eta_max):
    r = (imp[None, :] < imp[:, None]).sum(1) / K
    return (eta_max * (1.0 - r)).astype(cp.float32)


def split_acc(rig, T, codes, y):
    win, keep = codes
    C = cp.tile(rig.cells, len(win)).reshape(len(win), rig.npos)
    pred = (T[win, C] * keep[..., None]).sum(1).argmax(1)
    old = y < 5
    return (float((pred[old] == y[old]).mean()),
            float((pred[~old] == y[~old]).mean()),
            float((pred == y).mean()))


def run(rig, arm, eta_max, beta, phases, Xtr, ytr, Xte, yte, seed=7):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True)
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
    N = cp.zeros((K, rig.gg, NL)); n = cp.zeros(K)
    perm = cp.asarray(np.random.default_rng(seed + 1000).permutation(K))
    curve, snaps, bounds, step = [], [], [], 0

    for pi, (X, y, _) in enumerate(phases):
        bounds.append(step)
        for ep in range(EPOCHS):
            order = np.arange(len(X)); rng.shuffle(order)
            for s in range(0, len(order), R.BATCH):
                ids = cp.asarray(order[s:s + R.BATCH]); m = len(ids)
                Q, keep = rig.patches(X[ids])
                V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
                cells = cp.tile(rig.cells, m)
                T = rig.table_from(N)
                w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
                if beta > 0:
                    q = rig.belief(T, w0, k, cells, m)
                    err = 1.0 - (V @ W.T) ** 2
                    bias = cp.einsum('ik,ihk->ih', cp.repeat(q, rig.npos, 0),
                                     cp.ascontiguousarray(T.transpose(1, 0, 2))[cells])
                    win = (err - beta * bias).argmin(1)
                else:
                    win = w0
                lab = cp.repeat(y[ids], rig.npos)
                flat = (win[k] * rig.gg + cells[k]) * NL + lab[k]
                N += cp.bincount(flat, minlength=K * rig.gg * NL).reshape(N.shape)

                imp = importance(N, T)
                cnt = cp.bincount(win[k], minlength=K)
                if arm == "base":
                    n += cnt
                    eta = cp.clip(cnt / cp.maximum(n, 1.0), R.ETA_MIN, 1.0).astype(cp.float32)
                else:
                    eta = plasticity(imp, eta_max)
                    if arm == "shuffled":
                        eta = eta[perm]
                live = cnt >= rig.min_s
                if int(live.sum()):
                    sums = cp.zeros((K, rig.dim), cp.float32)
                    cupyx.scatter_add(sums, win[k], V[k])
                    W[live] += eta[live][:, None] * (sums[live] / cnt[live, None] - W[live])
                    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS

                if step % PROBE_EVERY == 0:
                    itr, ktr = rig.code(W, Xtr)
                    T_re = rig.table_from(rig.build_table(itr, ytr, ktr))
                    ct = rig.code(W, Xte)
                    curve.append({"step": step, "phase": pi,
                                  "recount": split_acc(rig, T_re, ct, yte),
                                  "online": split_acc(rig, rig.table_from(N), ct, yte),
                                  "n_frozen": int((eta < R.ETA_MIN).sum()),
                                  "win_share_frozen":
                                      float(cnt[eta < R.ETA_MIN].sum() / cp.maximum(cnt.sum(), 1))})
                    snaps.append((step, cp.asnumpy(W)))
                step += 1
        curve.append({**curve[-1], "step": step, "phase": pi})
    itr, ktr = rig.code(W, Xtr)
    T_re = rig.table_from(rig.build_table(itr, ytr, ktr))
    ct = rig.code(W, Xte)
    snaps.append((step, cp.asnumpy(W)))
    return (curve, bounds, snaps,
            {"recount": split_acc(rig, T_re, ct, yte),
             "online": split_acc(rig, rig.table_from(N), ct, yte)})


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    phases = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    print(f"{PS}x{PS}, {rig.npos} position(s), {K} templates, min_s {rig.min_s}, "
          f"{EPOCHS} epochs/phase, batch {R.BATCH}")
    print(f"phases {int(A.sum())} / {int((~A).sum())} / {len(ytr)}, test {len(yte)}\n")

    jobs = [("base", None)] + [(a, e) for e in ETA_MAXES for a in ("tally", "shuffled")]
    res, store = {}, {}
    print(f"{'':<22}{'recount 0-4/5-9/all':>28}{'online all':>12}{'frozen':>8}{'ws-frz':>8}")
    for beta in BETAS:
        for arm, em in jobs:
            tag = f"b{beta:g}_{arm}" + ("" if em is None else f"_{em:g}")
            t0 = time.time()
            c, b, sn, f = run(rig, arm, em, beta, phases, Xtr, ytr_g, Xte, yte_g)
            res[tag] = {"beta": beta, "arm": arm, "eta_max": em,
                        "curve": c, "bounds": b, "final": f}
            store[tag] = sn
            r = f["recount"]
            print(f"  {tag:<20s}" + f"{r[0]:.4f}/{r[1]:.4f}/{r[2]:.4f}".rjust(28)
                  + f"{f['online'][2]:>12.4f}{c[-1]['n_frozen']:>8d}"
                  + f"{c[-1]['win_share_frozen']:>8.3f}   ({time.time()-t0:.0f}s)", flush=True)
    (OUT / f"gated_split_{PS}.json").write_text(json.dumps(res, indent=2))
    draw(res, store)


def draw(res, store):
    for beta in BETAS:
        tags = [t for t in res if res[t]["beta"] == beta]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        cmap = plt.get_cmap("viridis")
        for t in tags:
            c = res[t]["curve"]; x = [p["step"] for p in c]
            arm, em = res[t]["arm"], res[t]["eta_max"]
            col = "k" if em is None else cmap(ETA_MAXES.index(em) / max(1, len(ETA_MAXES) - 1))
            ls = {"base": "-", "tally": "-", "shuffled": ":"}[arm]
            lw = 2.2 if arm == "base" else 1.5
            axes[0].plot(x, [p["recount"][0] for p in c], ls, color=col, lw=lw, label=t)
            axes[1].plot(x, [p["recount"][2] for p in c], ls, color=col, lw=lw)
        for ax, ttl in ((axes[0], "OLD classes (0-4), recount"),
                        (axes[1], "all ten, recount")):
            for bb, nm in zip(res[tags[0]]["bounds"], ("0-4", "5-9", "0-9")):
                ax.axvline(bb, color="k", lw=0.8, ls=":")
                ax.text(bb + 3, 0.05, nm, fontsize=8)
            ax.set_title(ttl, fontsize=10); ax.set_xlabel("training batches")
            ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=6, ncol=2, loc="lower left")
        fig.suptitle(f"{PS}x{PS} templates, beta {beta}: does tally-gated plasticity protect "
                     f"the old classes?", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / f"gated_split_b{beta:g}.png", dpi=145); plt.close(fig)

    for beta in BETAS:
        base = f"b{beta:g}_base"
        best = max([t for t in res if res[t]["beta"] == beta and res[t]["arm"] == "tally"],
                   key=lambda t: res[t]["final"]["recount"][0])
        fig, axes = plt.subplots(2, 12, figsize=(12 * 0.72, 2 * 0.95))
        for r, tag in enumerate((base, best)):
            sn = store[tag]
            pick = [sn[int(i)] for i in np.linspace(0, len(sn) - 1, 12)]
            for c, (st, Wm) in enumerate(pick):
                w = Wm[0]; v = np.abs(w).max() + 1e-9
                axes[r, c].imshow(w.reshape(PS, PS), cmap="RdBu_r", vmin=-v, vmax=v)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(f"{st}", fontsize=6)
            axes[r, 0].set_ylabel(tag.replace(f"b{beta:g}_", ""), fontsize=7,
                                  rotation=0, ha="right", va="center")
        fig.suptitle(f"beta {beta}: one template's life through 0-4 -> 5-9 -> 0-9", fontsize=10)
        fig.subplots_adjust(wspace=0.06, top=0.78, left=0.13)
        fig.savefig(OUT / f"one_template_b{beta:g}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"\nwrote gated_split_b*.png, one_template_b*.png")


if __name__ == "__main__":
    main()
