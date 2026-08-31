"""Layer-two templates, pushed back down to pixels.

A layer-two template is a pattern over 3x3 layer-one codes, so it has no
picture of its own. Setting one code to a lone 1 at the centre position and
running the whole down path -- l2.decode, unwindow, l1.decode, place-code
centroid, unpatch -- draws what that template stands for.
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent/"sparse_column"), str(H.parent/"place_code"),
                str(H.parents[1]/"2026_08_30"/"stack3")]
from stack3 import EPS, L2_GRID, PATCH                                # noqa
from sparse_stack import SparseStack                                  # noqa
from run_sparse_stack import load, K1, K2, ETA1, EPOCHS, SEED         # noqa

CENTRE = 4                                     # of the 9 layer-two positions


def main():
    Xtr, _, _, _ = load()
    rng = np.random.default_rng(SEED)
    cs = SparseStack(K1, K2, 4, ETA1, rng)
    cs.train(Xtr, EPOCHS, rng)

    T1 = cs.templates()                                   # (64, 4, 4)
    fig, ax = plt.subplots(8, 8, figsize=(5.2, 5.2))
    order1 = np.argsort(-cs.l1.wins)
    for i, a in enumerate(ax.ravel()):
        a.imshow(T1[order1[i]], cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("layer one — 64 templates as 4x4 patches, most-used first", fontsize=10)
    fig.tight_layout(); fig.savefig(H/"results"/"l1_templates.png", dpi=125); plt.close(fig)

    code = np.zeros((K2, L2_GRID * L2_GRID * K2))
    for j in range(K2):
        code[j, CENTRE * K2 + j] = 1.0
    T2 = cs.backward(code, K2)                            # (128, 28, 28)
    order2 = np.argsort(-cs.l2.wins)
    fig, ax = plt.subplots(8, 16, figsize=(11, 5.8))
    for i, a in enumerate(ax.ravel()):
        t = T2[order2[i]]
        a.imshow(t, cmap="gray_r", vmin=0, vmax=max(t.max(), EPS))
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("layer two — 128 templates at the centre position, pushed all the "
                 "way down to pixels, most-used first", fontsize=10)
    fig.tight_layout(); fig.savefig(H/"results"/"l2_templates.png", dpi=125); plt.close(fig)

    m = (T2.reshape(K2, -1) > 0.02 * T2.max()).sum(1)
    print(f"layer one: {len(T1)} templates, 4x4 each")
    print(f"layer two: {K2} templates; each covers {m.mean():.0f} pixels on "
          f"average (min {m.min()}, max {m.max()}) of the 12x12=144 its window spans")
    print(f"usage share, layer two: top template {cs.l2.wins.max()/cs.l2.wins.sum():.3f}, "
          f"dead {(cs.l2.wins == 0).sum()}")
    print("-> results/l1_templates.png, results/l2_templates.png")


if __name__ == "__main__":
    main()
