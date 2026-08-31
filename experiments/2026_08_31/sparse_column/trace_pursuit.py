"""Watch the pursuit run on one real patch, step by step."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent / "place_code")]
from place_code import PlaceCode, unit_rows                      # noqa
from sparse_column import SparseColumn                           # noqa
from run_sparse_column import load, train                        # noqa

KMAX = 4


def main():
    P = load(); pc = PlaceCode(16, nb=8, lo=0., hi=1., halfw=1.5)
    U, _ = unit_rows(pc.encode(P))
    rng = np.random.default_rng(0)
    col = train(SparseColumn(64, 128, kmax=KMAX, eta=0.5,
                             rng=np.random.default_rng(1)), U[rng.choice(len(U), 24000, False)])

    picks = [i for i in rng.choice(len(P), 40, False) if P[i].sum() > 3][:2]
    fig, axes = plt.subplots(2 * len(picks), KMAX + 1, figsize=(9.5, 2.4 * 2 * len(picks)))
    for row, ix in enumerate(picks):
        x = U[ix:ix + 1][0]
        r = x.copy()
        a, b = axes[2 * row], axes[2 * row + 1]
        a[0].imshow(pc.decode(x[None])[0].reshape(4, 4), cmap="gray_r", vmin=0, vmax=1)
        a[0].set_title("the patch", fontsize=8)
        b[0].imshow(x.reshape(16, 8), cmap="magma", aspect="auto")
        b[0].set_ylabel("its place code\n16 pixels x 8 cells", fontsize=6)
        print(f"\npatch {ix}:  ||x|| = 1.000")
        for t in range(KMAX):
            s = col.W @ r
            i = int(s.argmax()); c = float(s[i])
            tau = r - c * col.W[i]
            a[t + 1].imshow(pc.decode(col.W[i:i + 1])[0].reshape(4, 4),
                            cmap="gray_r", vmin=0, vmax=1)
            a[t + 1].set_title(f"step {t+1}: template {i}\ntakes {c:.3f}", fontsize=8)
            b[t + 1].imshow(tau.reshape(16, 8), cmap="magma", aspect="auto")
            b[t + 1].set_xlabel(f"what's left: {np.linalg.norm(tau):.3f}", fontsize=7)
            print(f"  step {t+1}: template {i:>2} scores {c:.3f}  "
                  f"-> residual {np.linalg.norm(tau):.3f}   "
                  f"(check: sqrt(prev^2 - c^2) = "
                  f"{np.sqrt(max(np.linalg.norm(r)**2 - c*c, 0)):.3f})")
            r = tau
        for ax in list(a) + list(b):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("matching pursuit: pick the best match, subtract it, repeat on what's left",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(H / "results" / "08_pursuit_trace.png", dpi=130)
    print("\n-> results/08_pursuit_trace.png")


if __name__ == "__main__":
    main()
