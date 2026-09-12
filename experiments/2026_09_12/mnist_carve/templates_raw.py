"""What a template actually IS, with no decoding to pixels.

A light-level template is a set of slots. A slot is (position, cluster): "at
patch 12, cluster 7". Nothing else. An upper template is a set of template
numbers. This prints them that way, and draws each light-level template as the
49 x 20 bit-grid it literally is -- one row per patch position, one column per
cluster, a dot where the template has a member.

Small quick fit so it does not wait on the main run.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mnist import (G, INK, K, NCODE, NL, CarveMNIST, encode,  # noqa: E402
                   load, patches)

N, SEED = 2500, 0


def describe(eng, t, dep, indent=""):
    ms = eng.members[t]
    lab = [int(m - NCODE) for m in ms if NCODE <= m < NL]
    codes = [int(m) for m in ms if m < NCODE]
    subs = [int(m - NL) for m in ms if m >= NL]
    head = f"{indent}T#{t:<4} depth {dep[t]}  fired {int(eng.fired[t]):>4}  "
    parts = []
    if codes:
        parts.append(f"{len(codes)} code slots: " + ", ".join(
            f"(pos {c // K:>2} = row {c // K // G} col {c // K % G}, cluster {c % K:>2})" for c in codes))
    if subs:
        parts.append(f"member templates {subs}")
    print(head + "  +  ".join(parts) + (f"   + LABEL {lab[0]}" if lab else ""))
    for s_ in subs:
        describe(eng, s_, dep, indent + "        ")


def main():
    Xtr, ytr, _, _ = load(N, 10, SEED)
    Ptr = patches(Xtr)
    km = KMeans(K, n_init=3, random_state=SEED).fit(Ptr[Ptr.sum(2) > INK])
    V = encode(Ptr, km)
    V[np.arange(N), NCODE + ytr] = True
    eng = CarveMNIST(NL, M=400, min_frac=0.6, warmup=500, min_age=200).fit(V, log_every=0)
    dep = eng.depth()
    print(f"{eng.V} templates; by depth {dict(zip(*[list(map(int, a)) for a in np.unique(dep, return_counts=True)]))}\n")

    print("=== the ten most-fired templates, as they are ===")
    for t in np.argsort(-eng.fired[:eng.V])[:10]:
        describe(eng, int(t), dep)

    print("\n=== every template of depth >= 2 (up to twelve), unfolded ===")
    deep = [t for t in np.argsort(-dep) if dep[t] >= 2][:12]
    for t in deep:
        describe(eng, int(t), dep)
        print()

    # ---- the bit-grids: 49 rows (positions) x 20 columns (clusters)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = list(np.argsort(-eng.fired[:eng.V])[:12]) + [int(t) for t in deep[:6]]
    fig, axes = plt.subplots(2, 9, figsize=(17, 8))
    for ax, t in zip(axes.ravel(), show):
        grid = np.zeros((G * G, K))
        v = np.zeros(eng.nT, bool)
        v[eng.NL + t] = True
        L = eng.expand(v)                                  # down to slots, not pixels
        for c in np.flatnonzero(L[:NCODE]):
            grid[c // K, c % K] = 1
        lab = np.flatnonzero(L[NCODE:NL])
        ax.imshow(grid, cmap="Greys", aspect="auto", interpolation="nearest")
        ax.set_title(f"T#{t}  d{dep[t]}" + (f"  label {lab[0]}" if len(lab) else ""), fontsize=8)
        ax.set_xlabel("cluster 0..19", fontsize=7)
        ax.set_ylabel("patch position 0..48", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("templates as they are: one row per patch position, one column per cluster, "
                 "a mark where the template has a slot\n(top row: most-fired; bottom right: deepest, "
                 "unfolded to their slots)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "results/templates_raw.png", dpi=110)
    print("wrote results/templates_raw.png")


if __name__ == "__main__":
    main()
