"""Letting the class talk back down, and watching the winners change.

The bottom round of a 6 and the bottom round of a 0 are the same shape, so the
expert down there cannot tell them apart. Summing evidence already CLASSIFIES
such an image correctly (0.9423) -- the top of the digit settles it. But the
bottom patch never finds out, so two 6s keep producing different winners down
there and the code stays unstable: 0.540 within class against 0.406 between.

So the class is fed back. No new weights: the counted table is

    T[expert, cell, class] = log P(expert | cell, class) / P(expert | cell)

Forwards it turns a winner into evidence for a class. BACKWARDS it turns a
belief about the class into a preference over which expert should be winning at
that cell. Same numbers, read the other way.

    1  pick winners from the ink alone
    2  sum T over positions          -> class belief q
    3  bias every expert's score by  BETA * sum_c q_c T[h, cell, c]
    4  re-pick winners, back to 2

Two things have to be watched together, because one of them is easy to fake.
Injecting a class belief into the code will ALWAYS make same-class codes look
more alike -- we put it there. Stability is only real if accuracy holds up
while it happens. If the gap widens and accuracy falls, we have built a machine
that talks itself into things.
"""

import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import experts as E
import vote as V

OUT = Path(__file__).resolve().parent / "results"
SIDE, GRID, NL = V.SIDE, V.GRID, 10
ROUNDS = 5
BETAS = [0.0, 0.05, 0.15, 0.4, 1.0]
NSIM = 600
B_ = SIDE // GRID
CELL = ((np.arange(SIDE * SIDE) // SIDE) // B_) * GRID + ((np.arange(SIDE * SIDE) % SIDE) // B_)


def fits(W, X, chunk=128):
    """(n, H, 576) how well each expert rebuilds each patch, and the flat mask."""
    h = len(W)
    F = np.zeros((len(X), h, SIDE * SIDE), np.float32)
    KP = np.zeros((len(X), SIDE * SIDE), bool)
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        B = E.join(Q.reshape(-1, E.PS * E.PS))
        S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
        R = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
        err = ((B[None] - R)[:, :, :E.PS * E.PS] ** 2).sum(-1)
        F[a:a + len(Q)] = (1.0 - err).reshape(h, len(Q), -1).transpose(1, 0, 2)
        KP[a:a + len(Q)] = keep.reshape(len(Q), -1)
    return F, KP


def settle(F, KP, T, beta, rounds=ROUNDS):
    """Returns winners and class scores after each round."""
    Tc = T[:, CELL, :]                                    # (H, 576, 10)
    hist = []
    bias = np.zeros_like(F)
    for r in range(rounds):
        win = (F + bias).argmax(1)                        # (n, 576)
        s = np.zeros((len(F), NL))
        for i in range(len(F)):
            p = np.nonzero(KP[i])[0]
            s[i] = Tc[win[i, p], p].sum(0)
        hist.append((np.where(KP, win, -1), s.copy()))
        z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        bias = beta * np.einsum('nc,hpc->nhp', q, Tc, optimize=True).astype(np.float32)
    return hist


def code(win, h):
    n = len(win)
    M = np.zeros((n, SIDE * SIDE, h), np.float32)
    r, c = np.nonzero(win >= 0)
    M[r, c, win[r, c]] = 1.0
    return M.reshape(n, GRID, B_, GRID, B_, h).max(axis=(2, 4)).reshape(n, -1)


def simstats(C, y):
    n = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    S = n @ n.T
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    w, b = float(S[same & off].mean()), float(S[~same].mean())
    return w, b, w - b, S


def main():
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()

    itr = np.concatenate([  # winners on the training set, for the table
        np.where(kp, f.argmax(1), -1) for f, kp in
        [fits(W, Xtr[a:a + 1000]) for a in range(0, len(Xtr), 1000)]])
    T = V.llr_table(itr, ytr, h, by_cell=True)
    print(f"table {T.shape}, range [{T.min():.2f}, {T.max():.2f}]", flush=True)

    F, KP = fits(W, Xte)
    sub = np.concatenate([np.nonzero(yte == c)[0][:NSIM // NL] for c in range(NL)])
    res = {}
    for beta in BETAS:
        hist = settle(F, KP, T, beta)
        rows = []
        for r, (win, s) in enumerate(hist):
            acc = float((s.argmax(1) == yte).mean())
            w, b, g, _ = simstats(code(win[sub], h), yte[sub])
            ch = float((win != hist[0][0]).mean())
            rows.append({"round": r, "acc": acc, "within": w, "between": b,
                         "gap": g, "changed": ch})
            print(f"  beta {beta:<5} round {r}  acc {acc:.4f}  within {w:.4f}  "
                  f"between {b:.4f}  GAP {g:+.4f}  moved {ch*100:5.1f}%", flush=True)
        res[str(beta)] = rows
        print(flush=True)

    best = max(BETAS[1:], key=lambda b: res[str(b)][-1]["gap"] + res[str(b)][-1]["acc"])
    res["chosen_beta"] = best
    (OUT / "settle.json").write_text(json.dumps(res, indent=2))

    # ---- the gif ---------------------------------------------------------
    pick = [int(np.nonzero(yte == c)[0][0]) for c in [0, 6, 4, 9, 3, 5]]
    hist = settle(F[pick], KP[pick], T, best, rounds=ROUNDS)
    fig, ax = plt.subplots(2, len(pick), figsize=(2.0 * len(pick), 4.6),
                           gridspec_kw={"height_ratios": [2.2, 1.0]})
    ims, bars = [], []
    for j, p in enumerate(pick):
        ax[0, j].imshow(Xte[p], cmap="gray_r", alpha=0.25,
                        extent=(0, SIDE, SIDE, 0))
        ims.append(ax[0, j].imshow(np.ma.masked_less(hist[0][0][j].reshape(SIDE, SIDE), 0),
                                   cmap="tab20", vmin=0, vmax=h - 1, interpolation="nearest"))
        ax[0, j].set_title(f"true {yte[p]}", fontsize=9); ax[0, j].axis("off")
        bars.append(ax[1, j].bar(range(NL), np.zeros(NL), color="tab:blue"))
        ax[1, j].set_xticks(range(NL)); ax[1, j].tick_params(labelsize=6)
        ax[1, j].set_yticks([])
    ttl = fig.suptitle("", fontsize=12)

    def upd(r):
        win, s = hist[r]
        z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        for j in range(len(pick)):
            ims[j].set_data(np.ma.masked_less(win[j].reshape(SIDE, SIDE), 0))
            for c, bar in enumerate(bars[j]):
                bar.set_height(max(z[j, c], 0))
                bar.set_color("tab:red" if c == z[j].argmax() else "tab:blue")
            ax[1, j].set_ylim(0, max(z.max(), 1e-6) * 1.05)
        moved = (win != hist[0][0]).mean() * 100
        ttl.set_text(f"settling round {r}   beta={best}   "
                     f"{moved:.1f}% of positions have changed winner")
        return ims + [ttl]

    FuncAnimation(fig, upd, frames=ROUNDS, blit=False).save(
        OUT / "settling.gif", writer=PillowWriter(fps=1.2))
    print(f"\nchosen beta {best} -> results/settling.gif")


if __name__ == "__main__":
    main()
