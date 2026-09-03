"""How much is template learning worth, as a function of receptive field?

`tally_plasticity` was run at 5x5 and could not differentiate anything, because
at 5x5 training the vocabulary is worth 0.7 points -- there was nothing for a
plasticity rule to win. This measures that number directly across patch sizes so
the next experiment is run somewhere it can matter.

Two arms at each size, everything else the champion:

    base         eta = clip(cnt/n, 0.02, 1)     templates learn normally
    frozen_all   eta = 0                        templates stay at random init,
                                                only the table counts

The gap between them IS what template learning buys at that receptive field.
Where the gap is near zero, no rule about which templates may move can matter.

Usage:  uv run python rf_sweep.py
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
K, NL, GRID, GG = 400, 10, 6, 36
EPS, ALPHA, BETA = 1e-12, 1.0, 1.5
BATCH, ETA_MIN, FLOOR = 128, 0.02, 0.05
EPOCHS = 3
SIZES = [5, 9, 13, 17, 21, 25, 28]
SEEDS = [7, 8]


class Rig:
    """Everything that depends on the patch size."""

    def __init__(self, ps):
        self.ps = ps
        self.side = 28 - ps + 1
        self.npos = self.side * self.side
        self.dim = ps * ps
        a, b = np.arange(self.side), np.arange(ps)
        self.pidx = cp.asarray((((a[:, None, None, None] + b[None, None, :, None]) * 28
                                 + (a[None, :, None, None] + b[None, None, None, :]))
                                ).reshape(self.npos, self.dim))
        g = min(GRID, self.side)
        self.grid, self.gg = g, g * g
        i = np.arange(self.npos)
        self.cells = cp.asarray(((i // self.side) * g // self.side) * g
                                + ((i % self.side) * g // self.side))
        # a template must win MIN_S patches in a batch to learn; with few
        # positions per image that threshold would freeze learning outright
        self.min_s = 4 if BATCH * self.npos / K >= 20 else 1

    def patches(self, Xb):
        P = Xb[:, self.pidx]
        C = P - P.mean(-1, keepdims=True)
        n = cp.linalg.norm(C, axis=-1)
        return C / cp.maximum(n, EPS)[..., None], n > FLOOR

    def table_from(self, N):
        pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
        pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
              / (N.sum((0, 2), keepdims=True) + ALPHA * K * NL))
        return (cp.log(pc) - cp.log(pm)).astype(cp.float32)

    def belief(self, T, win, keep, cells, m):
        ev = (T[win, cells] * keep[:, None]).reshape(m, self.npos, NL).sum(1)
        z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
        q = cp.exp(z - z.max(1, keepdims=True))
        return q / q.sum(1, keepdims=True)

    def compete(self, V, W, T, q, cells, m):
        err = 1.0 - (V @ W.T) ** 2
        bias = cp.einsum('ik,ihk->ih', cp.repeat(q, self.npos, 0),
                         cp.ascontiguousarray(T.transpose(1, 0, 2))[cells])
        return (err - BETA * bias).argmin(1)

    def code(self, W, X, chunk=256):
        idx = cp.zeros((len(X), self.npos), cp.int32)
        keepm = cp.zeros((len(X), self.npos), bool)
        for a in range(0, len(X), chunk):
            Q, keep = self.patches(X[a:a + chunk])
            m = len(Q)
            idx[a:a + m] = (1.0 - (Q.reshape(-1, self.dim) @ W.T) ** 2).argmin(1).reshape(m, -1)
            keepm[a:a + m] = keep
        return idx, keepm

    def build_table(self, win, y, valid):
        C = cp.tile(self.cells, len(win)).reshape(len(win), self.npos)
        lab = cp.repeat(y, self.npos).reshape(len(win), self.npos)
        flat = (win[valid] * self.gg + C[valid]) * NL + lab[valid]
        return cp.bincount(flat, minlength=K * self.gg * NL
                           ).reshape(K, self.gg, NL).astype(cp.float64)

    def accuracy(self, T, codes, y):
        win, keep = codes
        C = cp.tile(self.cells, len(win)).reshape(len(win), self.npos)
        return float(((T[win, C] * keep[..., None]).sum(1).argmax(1) == y).mean())


def run(rig, learns, Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True)
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    N = cp.zeros((K, rig.gg, NL)); n = cp.zeros(K)
    for ep in range(EPOCHS):
        order = np.arange(len(Xtr)); rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            ids = cp.asarray(order[s:s + BATCH]); m = len(ids)
            Q, keep = rig.patches(Xtr[ids])
            V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
            cells = cp.tile(rig.cells, m)
            T = rig.table_from(N)
            w0 = (1.0 - (V @ W.T) ** 2).argmin(1)
            q = rig.belief(T, w0, k, cells, m)
            win = rig.compete(V, W, T, q, cells, m)
            lab = cp.repeat(ytr[ids], rig.npos)
            flat = (win[k] * rig.gg + cells[k]) * NL + lab[k]
            N += cp.bincount(flat, minlength=K * rig.gg * NL).reshape(N.shape)
            if learns:
                wk, Vk = win[k], V[k]
                cnt = cp.bincount(wk, minlength=K)
                sums = cp.zeros((K, rig.dim), cp.float32)
                cupyx.scatter_add(sums, wk, Vk)
                live = cnt >= rig.min_s
                n[live] += cnt[live]
                eta = cp.clip(cnt[live] / n[live], ETA_MIN, 1.0)[:, None]
                W[live] += eta * (sums[live] / cnt[live, None] - W[live])
                W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    itr, ktr = rig.code(W, Xtr)
    T_re = rig.table_from(rig.build_table(itr, ytr, ktr))
    return rig.accuracy(T_re, rig.code(W, Xte), yte), cp.asnumpy(W)


def main():
    OUT.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)

    print(f"{'patch':>6}{'pos':>6}{'wins/tmpl':>11}{'min_s':>7}  "
          f"{'trained':>18}{'never learned':>18}{'learning is worth':>19}")
    rows, store = [], {}
    for ps in SIZES:
        rig = Rig(ps)
        t0 = time.time()
        tr = [run(rig, True, Xtr, ytr_g, Xte, yte_g, s) for s in SEEDS]
        fr = [run(rig, False, Xtr, ytr_g, Xte, yte_g, s) for s in SEEDS]
        a_tr = np.array([x[0] for x in tr]); a_fr = np.array([x[0] for x in fr])
        store[ps] = (tr[0][1], fr[0][1])
        rows.append({"ps": ps, "npos": rig.npos, "min_s": rig.min_s,
                     "trained": a_tr.mean(), "trained_std": a_tr.std(),
                     "frozen": a_fr.mean(), "frozen_std": a_fr.std(),
                     "worth": a_tr.mean() - a_fr.mean()})
        r = rows[-1]
        print(f"{ps:>4}x{ps:<2}{rig.npos:>6}{BATCH*rig.npos/K:>11.0f}{rig.min_s:>7}  "
              f"{r['trained']:.4f}+-{r['trained_std']:.4f}   "
              f"{r['frozen']:.4f}+-{r['frozen_std']:.4f}   "
              f"{r['worth']:>+13.4f}   ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "rf_sweep.json").write_text(json.dumps(rows, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = [r["ps"] for r in rows]
    axes[0].errorbar(x, [r["trained"] for r in rows], [r["trained_std"] for r in rows],
                     marker="o", color="#1b6ca8", label="templates learn")
    axes[0].errorbar(x, [r["frozen"] for r in rows], [r["frozen_std"] for r in rows],
                     marker="s", color="#c1121f", label="never learn (random filters)")
    axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=9)
    axes[0].set_title("accuracy vs receptive field", fontsize=10)
    axes[1].plot(x, [r["worth"] for r in rows], marker="o", color="#333333")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("trained − random")
    axes[1].set_title("what template learning is worth", fontsize=10)
    for ax in axes:
        ax.set_xlabel("patch side (pixels)"); ax.set_xticks(x); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Where do templates start to matter?  MNIST 0-9, 400 templates, counted table",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "rf_sweep.png", dpi=145); plt.close(fig)

    fig, axes = plt.subplots(2, len(SIZES), figsize=(len(SIZES) * 1.5, 3.4))
    for j, ps in enumerate(SIZES):
        for i, W in enumerate(store[ps]):
            g = int(np.ceil(np.sqrt(36)))
            tile = np.zeros((g * (ps + 1), g * (ps + 1)))
            for t in range(36):
                r, c = divmod(t, g)
                w = W[t].reshape(ps, ps); v = np.abs(w).max() + 1e-9
                tile[r * (ps + 1):r * (ps + 1) + ps, c * (ps + 1):c * (ps + 1) + ps] = w / v
            axes[i, j].imshow(tile, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[0, j].set_title(f"{ps}x{ps}", fontsize=9)
    axes[0, 0].set_ylabel("learned", fontsize=9)
    axes[1, 0].set_ylabel("random", fontsize=9)
    fig.suptitle("36 templates at each receptive field", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "rf_templates.png", dpi=145); plt.close(fig)
    print(f"\nwrote {OUT/'rf_sweep.png'}, {OUT/'rf_templates.png'}")


if __name__ == "__main__":
    main()
