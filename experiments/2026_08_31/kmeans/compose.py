"""Can today's stack compose after all -- in the MAP space rather than the index?

The 2026 gain-feedback rig invented a woman with a mustache by arithmetic on a
graded code: base_female + beta * mustache_direction, then rendered. Today's
message is an index and has no arithmetic. But the L1 MAP is graded -- every
position carries a magnitude per channel -- so the same trick is available one
level down:

    base  = mean fine-L1 map over females
    delta = mean map over mustached faces - mean map over the rest
    comp  = relu(base + beta * ||base|| * delta_hat)     rendered through L1

If a mustache appears on a female base as beta rises, the ability lives in the
code being GRADED, and identity-only messaging is what gave it up.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import celeba as C

OUT = Path(__file__).resolve().parent / "results"
DS, EPS = 2, 1e-12
DN = C.SIDE // DS


def mean_map(idx, mag, sel, dpos, k1):
    """Mean half-resolution L1 map over a subset of faces."""
    acc = np.zeros((DN * DN, k1), np.float64)
    n = 0
    for a in range(0, len(idx), 2000):
        sl = slice(a, a + 2000)
        s = sel[sl]
        if not s.any():
            continue
        ii, mm = idx[sl][s], mag[sl][s]
        r, c = np.nonzero(ii >= 0)
        np.add.at(acc, (dpos[c], ii[r, c]), mm[r, c])
        n += int(s.sum())
    return acc / max(n, 1), n


def main():
    t0 = time.time()
    z = np.load(OUT / "celeba.npz")
    W1 = z["W1"].astype(np.float64)
    k1 = len(W1)
    Xtr, Atr, Xte, Ate, names = C.load()
    itr, mtr = C.l1_map(W1, Xtr)
    pos = np.arange(C.SIDE * C.SIDE)
    dpos = (pos // C.SIDE // DS) * DN + (pos % C.SIDE // DS)

    def paint(M):
        sel, mag = M.argmax(1), M.max(1)
        img, ct = np.zeros((48, 48)), np.zeros((48, 48))
        for p in range(DN * DN):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, DN)
            r, c = r * DS, c * DS
            img[r:r + C.PS, c:c + C.PS] += W1[sel[p]].reshape(C.PS, C.PS) * mag[p]
            ct[r:r + C.PS, c:c + C.PS] += 1
        return img / np.maximum(ct, 1)

    A = Atr
    male, must, glass = (names.index("Male"), names.index("Mustache"),
                         names.index("Eyeglasses"))
    base_f, n_f = mean_map(itr, mtr, A[:, male] < 0, dpos, k1)
    base_m, n_m = mean_map(itr, mtr, A[:, male] > 0, dpos, k1)
    pos_mu, n_mu = mean_map(itr, mtr, A[:, must] > 0, dpos, k1)
    neg_mu, _ = mean_map(itr, mtr, A[:, must] < 0, dpos, k1)
    pos_gl, n_gl = mean_map(itr, mtr, A[:, glass] > 0, dpos, k1)
    neg_gl, _ = mean_map(itr, mtr, A[:, glass] < 0, dpos, k1)
    d_mu = pos_mu - neg_mu
    d_gl = pos_gl - neg_gl
    print(f"females {n_f}, males {n_m}, mustached {n_mu}, glasses {n_gl}")

    betas = [0.0, 0.3, 0.6, 1.0, 1.5]
    rows = [("female + mustache", base_f, d_mu),
            ("male + mustache (a sanity check — this combination exists)", base_m, d_mu),
            ("female + eyeglasses", base_f, d_gl)]
    fig, axes = plt.subplots(len(rows) + 1, len(betas), figsize=(2.0 * len(betas), 2.2 * (len(rows) + 1)))
    for r, (title, base, d) in enumerate(rows):
        bn = np.linalg.norm(base)
        dh = d / (np.linalg.norm(d) + EPS)
        for cix, b in enumerate(betas):
            axes[r][cix].imshow(paint(np.maximum(base + b * bn * dh, 0.0)), cmap="gray")
            axes[r][cix].set_xticks([]); axes[r][cix].set_yticks([])
            if r == 0:
                axes[r][cix].set_title(f"beta = {b}", fontsize=8)
        axes[r][0].set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center")
    # the delta on its own, and the real group means, for reference
    for cix, (M, t) in enumerate([(np.maximum(d_mu, 0), "mustache delta (positive part)"),
                                  (pos_mu, "real mustached mean"),
                                  (base_f, "real female mean"),
                                  (base_m, "real male mean"),
                                  (np.maximum(d_gl, 0), "glasses delta")]):
        axes[-1][cix].imshow(paint(M), cmap="gray")
        axes[-1][cix].set_xticks([]); axes[-1][cix].set_yticks([])
        axes[-1][cix].set_title(t, fontsize=6)
    axes[-1][0].set_ylabel("reference", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("composition by arithmetic on the GRADED L1 map "
                 "(0 women with mustaches exist in the data)", fontsize=10)
    fig.subplots_adjust(left=.22, right=.995, top=.93, bottom=.01,
                        wspace=.05, hspace=.12)
    fig.savefig(OUT / "celeba_compose.png", dpi=150); plt.close(fig)
    (OUT / "compose.json").write_text(json.dumps(
        {"n_female": n_f, "n_male": n_m, "n_mustache": n_mu, "n_glasses": n_gl,
         "betas": betas, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> celeba_compose.png")


if __name__ == "__main__":
    main()
