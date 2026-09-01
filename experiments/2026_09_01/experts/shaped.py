"""The class decides once, then reshapes the map. The map never re-decides.

This afternoon's loop recomputed the class from the settled winners every round,
so a wrong first guess biased the winners, which biased the next guess. Errors
compounded and accuracy slid 0.9423 -> 0.8650.

Cutting that feedback path costs nothing and fixes it:

    class    from the feedforward sum only          0.9423, and it is optimal --
                                                    with one label per image the
                                                    sum IS the exact posterior
    code     one round of bias from that belief,    and never read back

The bias is the counted table read backwards: how much does this expert
associate with the believed class, at this cell. It is added to the ink fit, not
substituted for it, so a badly-fitting expert still cannot win. Only the
positions where the ink was nearly indifferent actually flip -- which is exactly
where the bottom of a 6 and the bottom of a 0 live.

Reported: the class accuracy (fixed by construction) and the code's
within-minus-between separation, before and after.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, vote as V, settle as S

OUT = Path(__file__).resolve().parent / "results"
SIDE, GRID, NL = V.SIDE, V.GRID, 10
BETAS = [0.15, 0.4, 1.0]
BETA_FIG = 1.0
CLASSES, NPER, NSIM = [0, 1, 4, 9], 5, 600


def belief(s):
    z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
    q = np.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def forward(F, KP, Tc):
    """Feedforward winners and class scores. Independent of beta -- computed once."""
    win0 = np.where(KP, F.argmax(1), -1)
    ev = Tc[F.argmax(1), np.arange(F.shape[2])[None, :]]        # (n, P, 10)
    return win0, (ev * KP[..., None]).sum(1)


def reshape(F, KP, Tc, s, beta):
    """One round of bias from a FROZEN belief. The map is never read back."""
    bias = beta * np.einsum('nc,hpc->nhp', belief(s), Tc, optimize=True).astype(np.float32)
    return np.where(KP, (F + bias).argmax(1), -1)


def main():
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    h = len(W)
    Xtr, ytr, Xte, yte = E.load()
    itr = np.concatenate([np.where(kp, f.argmax(1), -1) for f, kp in
                          [S.fits(W, Xtr[a:a + 1000]) for a in range(0, len(Xtr), 1000)]])
    T = V.llr_table(itr, ytr, h, True)

    sub = np.concatenate([np.nonzero(yte == c)[0][:NSIM // NL] for c in range(NL)])
    Tc = T[:, S.CELL, :]
    res = {}
    chunks = [(a, *S.fits(W, Xte[a:a + 500])) for a in range(0, len(Xte), 500)]
    fwd = [forward(F, KP, Tc) for _, F, KP in chunks]
    acc = np.concatenate([s.argmax(1) for _, s in fwd])
    C0 = np.concatenate([S.code(w0, h) for w0, _ in fwd])
    for beta in BETAS:
        codes1, moved = [], []
        for (_, F, KP), (w0, s) in zip(chunks, fwd):
            w1 = reshape(F, KP, Tc, s, beta)
            codes1.append(S.code(w1, h)); moved.append((w0 != w1).mean())
        C1 = np.concatenate(codes1)
        w_0, b_0, g0, _ = S.simstats(C0[sub], yte[sub])
        w_1, b_1, g1, _ = S.simstats(C1[sub], yte[sub])
        res[str(beta)] = {"accuracy": float((acc == yte).mean()),
                          "gap_feedforward": g0, "gap_shaped": g1,
                          "within_shaped": w_1, "between_shaped": b_1,
                          "moved": float(np.mean(moved))}
        print(f"  beta {beta:<5} accuracy {res[str(beta)]['accuracy']:.4f}   "
              f"gap {g0:+.4f} -> {g1:+.4f}   within {w_1:.4f} between {b_1:.4f}   "
              f"moved {np.mean(moved)*100:.1f}%", flush=True)
    (OUT / "shaped.json").write_text(json.dumps(res, indent=2))

    # ---------- the figure ----------
    pick = np.concatenate([np.nonzero(yte == c)[0][:NPER] for c in CLASSES])
    F, KP = S.fits(W, Xte[pick])
    w0, sf = forward(F, KP, Tc)
    w1 = reshape(F, KP, Tc, sf, BETA_FIG)
    lab, n = yte[pick], len(pick)
    P0, P1 = S.code(w0, h), S.code(w1, h)
    _, _, g0, M0 = S.simstats(P0, lab)
    _, _, g1, M1 = S.simstats(P1, lab)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.6, 2.3, 2.3, 3.0, 0.12], hspace=0.33)
    top = gs[0].subgridspec(3, n, hspace=0.05, wspace=0.05)
    for i in range(n):
        for r, im in ((0, Xte[pick[i]]), (1, w0[i].reshape(SIDE, SIDE)),
                      (2, w1[i].reshape(SIDE, SIDE))):
            a = fig.add_subplot(top[r, i])
            if r == 0:
                a.imshow(im, cmap="gray_r")
                if i % NPER == 0:
                    a.set_title(f"class {lab[i]}", fontsize=9, color="tab:blue")
            else:
                a.imshow(np.ma.masked_less(im, 0), cmap="tab20", vmin=0, vmax=h - 1,
                         interpolation="nearest")
            a.axis("off")
    for y, t in ((0.905, "image"), (0.845, "winners\nfeedforward"), (0.785, "winners\nSHAPED")):
        fig.text(0.005, y, t, fontsize=8, rotation=90, va="center")

    for ax, M, t in ((fig.add_subplot(gs[1]), P0, "pooled code, FEEDFORWARD"),
                     (fig.add_subplot(gs[2]), P1, f"pooled code, SHAPED (beta={BETA_FIG})")):
        ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(t, fontsize=10); ax.set_xticks([])
        ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=7)
        for y in range(NPER, n, NPER):
            ax.axhline(y - 0.5, color="cyan", lw=1.0)

    bot = gs[3].subgridspec(1, 2, wspace=0.25)
    for j, (M, t, g) in enumerate(((M0, "similarity, FEEDFORWARD", g0),
                                   (M1, "similarity, SHAPED", g1))):
        a = fig.add_subplot(bot[0, j])
        im = a.imshow(M, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"{t}   within-minus-between = {g:+.3f}", fontsize=10)
        a.set_xticks(range(n)); a.set_xticklabels(lab, fontsize=6)
        a.set_yticks(range(n)); a.set_yticklabels(lab, fontsize=6)
        for k in range(NPER, n, NPER):
            a.axhline(k - .5, color="w", lw=.8); a.axvline(k - .5, color="w", lw=.8)
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.text(0.5, 0.075, f"class decided by the feedforward sum only "
             f"(0.9423, unchanged); the map is reshaped once and never re-read",
             ha="center", fontsize=9, style="italic")
    plt.savefig(OUT / "sparse_patterns_shaped.png", dpi=125, bbox_inches="tight")
    print("\n-> results/sparse_patterns_shaped.png")


if __name__ == "__main__":
    main()
